"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Every meaningful node writes an AuditEntry. Interrupt nodes record both the
pending request and the human response. A pending entry can appear again when
the interrupted node restarts on resume; posting lives in downstream nodes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

import httpx
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)
from exercises.exercise_3_escalation import handle_interrupt
from exercises.review_helpers import (
    analysis_messages, checkpoint_serializer, make_console, structured_review,
    synthesis_messages, validate_answers, validate_approval,
)


console = make_console()
AGENT_ID = "pr-review-agent@v0.1"


class AuditedReviewState(ReviewState, total=False):
    """Keep posting errors in the checkpoint without changing the supplied schema."""

    post_error: str | None


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(
        thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry,
    )


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = await asyncio.to_thread(fetch_pr, state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = structured_review(get_llm())
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke(analysis_messages(state))
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="analyze", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="route", confidence=c,
        risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision],
        reason=f"Route to {decision}: {state['analysis'].confidence_reasoning}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="human_approval", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="pending",
        reason="Waiting for approve / reject / edit",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    update = validate_approval(resp)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="human_approval", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), reviewer_id=os.environ.get("GITHUB_USER"),
        decision=update["human_choice"], reason=update["human_feedback"] or "Reviewer responded",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return update


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> dict:
    try:
        body = _render_comment_body(state)
        post_review_comment(state["pr_url"], body)
    except Exception as error:
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            try:
                message = error.response.json().get("message", error.response.reason_phrase)
            except (ValueError, AttributeError):
                message = error.response.reason_phrase
            detail = f"GitHub HTTP {status}: {message}"
            if status == 403:
                detail += " Check that your classic PAT has the public_repo scope."
        else:
            detail = f"{type(error).__name__}: {error}"
        # Never include configured credentials in UI, terminal or persisted logs.
        for name in ("GITHUB_TOKEN", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
            secret = os.environ.get(name)
            if secret:
                detail = detail.replace(secret, "[redacted]")
        console.print("Post failed: " + detail, markup=False)
        return {"final_action": "commit_failed", "posted_comment_body": None, "post_error": detail}
    console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
    return {"final_action": "committed", "posted_comment_body": body, "post_error": None}


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        result = await asyncio.to_thread(_post, state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        result = {"final_action": "rejected", "posted_comment_body": None, "post_error": None}
    a = state["analysis"]
    decision = "escalate" if state.get("escalation_answers") else state.get("human_choice", "pending")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="commit", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision=decision,
        reviewer_id=os.environ.get("GITHUB_USER"),
        reason=f"{result['final_action']}: {result.get('post_error') or state.get('human_feedback') or a.summary}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return result


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    result = await asyncio.to_thread(_post, state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="auto_approve", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="auto",
        reason=f"{result['final_action']}: {result.get('post_error') or a.confidence_reasoning}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    if result["final_action"] == "committed":
        result["final_action"] = "auto_committed"
    return result


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="escalate", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="pending",
        reason="Waiting for answers: " + " | ".join(questions),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    answers = validate_answers(answers, questions)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="escalate", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate", reason="\n".join(f"Q: {q}\nA: {a}" for q, a in answers.items()),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    llm = structured_review(get_llm())
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke(synthesis_messages(state))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="synthesize", confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence), decision="escalate",
        reason=refined.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    checkpointer.serde = checkpoint_serializer()
    g = StateGraph(AuditedReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


async def invoke_session(app, pr_url: str, thread_id: str, resume_value=None):
    """Start once, restore a paused/completed session, or resume its interrupt."""
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.aget_state(cfg)
    if snapshot.values:
        if snapshot.values["pr_url"] != pr_url:
            raise ValueError("This thread belongs to a different PR")
        pending = tuple(item for task in snapshot.tasks for item in task.interrupts)
        if resume_value is not None:
            if not pending:
                raise ValueError("This session is not waiting for a reviewer")
            payload = pending[0].value
            if payload["kind"] == "approval_request":
                validate_approval(resume_value)
            else:
                validate_answers(resume_value, payload["questions"])
            return await app.ainvoke(Command(resume=resume_value), cfg)
        if pending:
            return {**snapshot.values, "__interrupt__": pending}
        if snapshot.next:
            return await app.ainvoke(None, cfg)
        return dict(snapshot.values)
    if resume_value is not None:
        raise ValueError("Review session not found")
    return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        result = await invoke_session(app, pr_url, thread_id)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await invoke_session(app, pr_url, thread_id, handle_interrupt(payload))

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
