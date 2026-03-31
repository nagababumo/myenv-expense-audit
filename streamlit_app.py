import json
import re
import time
import streamlit as st
from expense_audit_env.data import build_report
from expense_audit_env.models import ExpenseAction, ExpenseEntry, ExpenseReport, ExpenseState
from expense_audit_env.server.environment import ExpenseAuditEnvironment

try:
    import baseline_gemini as llm_baseline
except Exception:
    llm_baseline = None


def generate_dynamic_report() -> ExpenseReport:
    if llm_baseline is None:
        raise RuntimeError("LLM baseline module is not available.")

    provider, model = llm_baseline.get_model_and_provider(None)
    prompt = (
        "You are a synthetic corporate expense report generator.\n"
        "Output valid JSON only.\n"
        "Create a single expense entry object using keys: expense_id, amount, category, requires_receipt, receipt_present, receipt_text, metadata, expected_decision, reason.\n"
        "metadata must include employee_id, date, merchant, report_id.\n"
        "Use only the categories travel, meals, lodging, office, or misc.\n"
        "Use a valid amount number and a valid decision: approve, reject, or flag.\n"
        "If receipt_present is false, receipt_text should be null."
    )
    if provider == "openai":
        raw_output = llm_baseline.call_openai(prompt, model, temperature=0.2)
    else:
        raw_output = llm_baseline.call_gemini(prompt, model, temperature=0.2)

    payload = llm_baseline.parse_json_text(raw_output)
    if not isinstance(payload, dict):
        raise ValueError("Generated output is not a JSON object.")

    if "entries" in payload and "report_id" in payload:
        if "spending_limits" not in payload:
            payload["spending_limits"] = build_report("easy").spending_limits.model_dump()
        if "employee_id" not in payload:
            payload["employee_id"] = payload.get("entries", [{}])[0].get("metadata", {}).get("employee_id", "E-0000")
        if "policy_name" not in payload:
            payload["policy_name"] = "LLM generated expense report"
        return ExpenseReport.model_validate(payload)

    entry = ExpenseEntry.model_validate(payload)
    return ExpenseReport(
        report_id=f"R-LLM-{int(time.time())}",
        employee_id=entry.metadata.employee_id,
        policy_name="LLM generated expense report",
        entries=[entry],
        spending_limits=build_report("easy").spending_limits,
    )


TASKS = ["easy", "medium", "hard"]
DECISIONS = ["approve", "reject", "flag"]
PLAYER_MODES = ["Manual", "LLM agent"]

st.set_page_config(
    page_title="Expense Audit Demo",
    page_icon="🧾",
    layout="centered",
)

st.title("Expense Report Auditing Demo")
st.markdown(
    """
    This demo shows how to use the expense auditing environment.

    1. Select a task difficulty.
    2. Click **Start / Reset task**.
    3. Review the current expense entry.
    4. Choose `approve`, `reject`, or `flag`.
    5. Click **Submit Decision** to advance.
    """
)

if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.observation = None
    st.session_state.task_id = TASKS[0]
    st.session_state.history = []
    st.session_state.last_decision = DECISIONS[0]
    st.session_state.comment = ""
    st.session_state.player_mode = PLAYER_MODES[0]
    st.session_state.llm_status = ""

selected_task = st.selectbox(
    "Select task difficulty",
    TASKS,
    index=TASKS.index(st.session_state.task_id),
)

st.sidebar.header("Player options")
st.session_state.player_mode = st.sidebar.radio(
    "Mode",
    PLAYER_MODES,
    index=PLAYER_MODES.index(st.session_state.player_mode),
)
st.session_state.llm_status = "OK" if llm_baseline is not None else "LLM module not available"

if st.sidebar.button("Generate task with LLM"):
    if llm_baseline is None:
        st.sidebar.error("LLM support is not available. Install dependencies and set your API key.")
    else:
        try:
            report = generate_dynamic_report()
            env = ExpenseAuditEnvironment(st.session_state.task_id)
            env._state = ExpenseState(report=report, current_index=0, task_id="llm-generated")
            st.session_state.env = env
            st.session_state.observation = env._observation_for(
                report.entries[0], reward=0.0, done=False, feedback="Generated task loaded."
            )
            st.sidebar.success("Generated a new task successfully.")
        except Exception as exc:
            st.sidebar.error(f"Could not generate task: {exc}")

st.sidebar.caption("Requires OPENAI_API_KEY or GEMINI_API_KEY configured in environment.")

if selected_task != st.session_state.task_id:
    st.session_state.task_id = selected_task

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Start / Reset task"):
        st.session_state.env = ExpenseAuditEnvironment(st.session_state.task_id)
        st.session_state.observation = st.session_state.env.reset()
        st.session_state.history = []
        st.session_state.last_decision = DECISIONS[0]
        st.session_state.comment = ""

with col2:
    st.write("**Current task:**", st.session_state.task_id)
    if st.session_state.env is not None:
        state = st.session_state.env.state
        st.write("**Progress:**", f"{state.current_index}/{len(state.report.entries)} entries")
        st.write("**Cumulative reward:**", f"{state.cumulative_reward:.2f}")
        st.write("**Done:**", state.done)

if st.session_state.observation is None:
    st.info("Click **Start / Reset task** to load the first expense for the selected task.")
    st.stop()

obs = st.session_state.observation
st.subheader("Current Expense")

st.write(
    {
        "expense_id": obs.expense_id,
        "amount": f"${obs.amount:.2f}",
        "category": obs.category,
        "requires_receipt": obs.requires_receipt,
        "receipt_text": obs.receipt_text or "(none)",
        "position": f"{obs.position + 1} / {obs.total_entries}",
        "policy_hint": obs.policy_hint,
    }
)

with st.expander("Expense metadata"):
    st.write(obs.metadata.model_dump())

with st.expander("Spending limits"):
    st.write(obs.spending_limits.model_dump())

st.write("**Last feedback:**", obs.last_feedback or "None")
st.write("**Status:**", obs.status)
st.write("**Reward from last step:**", f"{obs.reward:.2f}")

st.subheader("Take an action")
selected_decision = st.radio("Decision", DECISIONS, index=DECISIONS.index(st.session_state.last_decision))
comment = st.text_input("Comment (optional)", value=st.session_state.comment)

if st.button("Submit Decision"):
    action = ExpenseAction(
        decision=selected_decision,
        index=obs.position,
        comment=comment,
    )
    try:
        observation = st.session_state.env.step(action)
        st.session_state.observation = observation
        st.session_state.history = st.session_state.env.state.decisions.copy()
        st.session_state.last_decision = selected_decision
        st.session_state.comment = comment
        if observation.done:
            st.success("Episode complete! All expenses have been processed.")
        else:
            st.info("Decision submitted. Review the next expense.")
    except Exception as exc:
        st.error(f"Error: {exc}")

if st.session_state.player_mode == "LLM agent":
    st.markdown("---")
    st.subheader("LLM agent actions")
    st.write("Use the LLM agent to choose the next decision for the current expense.")
    if st.button("Run LLM decision"):
        if llm_baseline is None:
            st.error("LLM support is not available. Install dependencies and set your API key.")
        else:
            try:
                provider, model = llm_baseline.get_model_and_provider(None)
                decision_payload = llm_baseline.decide(obs, provider, model, 0.0)
                action = ExpenseAction(
                    decision=decision_payload["decision"],
                    index=obs.position,
                    comment=decision_payload.get("comment", ""),
                )
                observation = st.session_state.env.step(action)
                st.session_state.observation = observation
                st.session_state.history = st.session_state.env.state.decisions.copy()
                st.session_state.last_decision = action.decision
                st.session_state.comment = action.comment
                if observation.done:
                    st.success("Episode complete! All expenses have been processed.")
                else:
                    st.info("LLM decision applied. Review the next expense.")
            except Exception as exc:
                st.error(f"LLM error: {exc}")

st.subheader("Episode summary")
state = st.session_state.env.state
st.write(
    {
        "task_id": state.task_id,
        "current_index": state.current_index,
        "total_entries": len(state.report.entries),
        "cumulative_reward": f"{state.cumulative_reward:.2f}",
        "done": state.done,
    }
)

if state.decisions:
    st.subheader("Decision history")
    st.json(state.decisions)
else:
    st.info("No decisions submitted yet.")

st.sidebar.header("Sample test actions")
st.sidebar.markdown(
    """
    **Easy**
    - `reject`
    
    **Medium**
    - `approve`, `reject`, `approve`, `reject`
    
    **Hard**
    - `flag`, `approve`, `reject`, `flag`, `flag`
    
    Use the sidebar to guide your decisions as you step through each task.
    """
)

st.sidebar.header("HTTP API examples")
st.sidebar.code(
    '''
POST /reset/easy

POST /step
{
  "action": {
    "decision": "reject",
    "index": 0,
    "comment": "Missing receipt"
  },
  "timeout_s": 30
}
    '''
)
