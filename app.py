"""Exercise 5: run with `uv run streamlit run app.py`."""

from __future__ import annotations

import asyncio
import uuid
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from common.db import db_conn, db_path
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph, invoke_session


DEFAULT_PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"


async def recent_sessions() -> list[dict]:
    """The replay CLI prints a Rich table; the UI needs the underlying rows."""
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url, MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3
                                            WHEN 'med' THEN 2 ELSE 1 END)
                     WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
        """) as cur:
            return [dict(row) for row in await cur.fetchall()]


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Each UI action opens a fresh connection and restores the same checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        return await invoke_session(build_graph(cp), pr_url, thread_id, resume_value)


async def load_session(thread_id: str):
    """Read an existing session without invoking nodes or making API calls."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        snapshot = await build_graph(cp).aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise ValueError("Session not found. Start a new review or choose a recent session.")
    pending = tuple(item for task in snapshot.tasks for item in task.interrupts)
    result = dict(snapshot.values)
    if pending:
        result["__interrupt__"] = pending
    return result


def store_result(result: dict) -> None:
    pending = result.get("__interrupt__")
    st.session_state.interrupt_payload = pending[0].value if pending else None
    st.session_state.final = None if pending else result


def render_approval_card(payload: dict) -> dict | None:
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{comment.get('line') or '?'}`"
            f" — {comment['body']}"
        )
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_area(
        "Feedback (optional)", key=f"approval_feedback_{st.session_state.thread_id}",
    )
    st.caption("Approve posts this review to GitHub. Reject and Edit end this review without posting.")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + "; ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            question: st.text_area(question, key=f"answer_{st.session_state.thread_id}_{index}")
            for index, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        if any(not answer.strip() for answer in answers.values()):
            st.warning("Please answer every question before submitting.")
        else:
            return {question: answer.strip() for question, answer in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action")
    if action in {"auto_committed", "committed"}:
        st.success("Review comment posted to GitHub.")
        analysis = final["analysis"]
        st.metric("Confidence", f"{analysis.confidence:.0%}")
        st.markdown(analysis.summary)
        st.link_button("View comments on GitHub", final["pr_url"])
        if final.get("posted_comment_body"):
            with st.expander("Posted review"):
                st.markdown(final["posted_comment_body"])
    elif action == "rejected":
        st.warning("Review ended without posting a comment.")
    elif action == "commit_failed":
        st.error("Could not post the review. Check the GitHub token and repository permissions.")
        if final.get("post_error"):
            st.error(final["post_error"])
    else:
        st.info("This session has unfinished work. Continue it from its saved checkpoint.")
        if st.button("Continue session"):
            try:
                store_result(asyncio.run(run_graph(final["pr_url"], st.session_state.thread_id)))
                st.rerun()
            except Exception as error:
                st.error(str(error))


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    st.title("HITL PR Review Agent")
    for key, value in {
        "thread_id": st.query_params.get("thread"), "pr_url": DEFAULT_PR_URL,
        "interrupt_payload": None, "final": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Session state survives UI reruns; the URL thread restores browser refreshes.
    if (st.session_state.thread_id and st.session_state.final is None
            and st.session_state.interrupt_payload is None):
        try:
            result = asyncio.run(load_session(st.session_state.thread_id))
            st.session_state.pr_url = result["pr_url"]
            store_result(result)
        except Exception as error:
            st.warning(str(error))
            st.session_state.thread_id = None
            st.query_params.clear()

    with st.sidebar:
        st.header("Recent sessions")
        try:
            rows = asyncio.run(recent_sessions())
            if rows:
                event = st.dataframe(
                    rows, hide_index=True, on_select="rerun", selection_mode="single-row",
                    key=f"sessions_{st.session_state.thread_id}",
                )
                if event.selection.rows:
                    selected = rows[event.selection.rows[0]]
                    if selected["thread_id"] != st.session_state.thread_id:
                        result = asyncio.run(load_session(selected["thread_id"]))
                        st.session_state.thread_id = selected["thread_id"]
                        st.session_state.pr_url = result["pr_url"]
                        st.query_params["thread"] = selected["thread_id"]
                        store_result(result)
                        st.rerun()
            else:
                st.caption("No review sessions yet.")
        except Exception as error:
            st.error(f"Could not load sessions: {error}")

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url or DEFAULT_PR_URL,
            help="Demo PR #1 is prefilled. Change /1 to /2 to try the escalation demo.",
        )
        submitted = st.form_submit_button("Run review")

    if submitted:
        try:
            pr_url = pr_url.strip()
            if not pr_url:
                raise ValueError("PR URL is empty. Paste a GitHub pull request URL before running the review.")
            parsed = urlparse(pr_url)
            if parsed.scheme != "https" or parsed.netloc != "github.com":
                raise ValueError("Enter an HTTPS GitHub pull request URL.")
            owner, repo, number = parse_pr_url(pr_url)
            pr_url = f"https://github.com/{owner}/{repo}/pull/{number}"
            st.session_state.pr_url = pr_url
            st.session_state.thread_id = str(uuid.uuid4())
            st.query_params["thread"] = st.session_state.thread_id
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            with st.spinner("Fetching PR + asking the LLM..."):
                store_result(asyncio.run(run_graph(pr_url, st.session_state.thread_id)))
            st.rerun()
        except Exception as error:
            st.error(str(error))

    payload = st.session_state.interrupt_payload
    if payload is not None:
        if payload["kind"] == "approval_request":
            answer = render_approval_card(payload)
        elif payload["kind"] == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error(f"Unknown interrupt kind: {payload['kind']}")
            answer = None
        if answer is not None:
            try:
                with st.spinner("Resuming..."):
                    result = asyncio.run(run_graph(
                        st.session_state.pr_url, st.session_state.thread_id, answer,
                    ))
                store_result(result)
                st.rerun()
            except Exception as error:
                st.error(str(error))

    if st.session_state.final is not None:
        render_final(st.session_state.final)
    if st.session_state.thread_id:
        st.caption(f"Session: {st.session_state.thread_id}")


if __name__ == "__main__":
    main()
