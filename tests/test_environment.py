from expense_audit_env.baseline import RuleBasedAuditor
from expense_audit_env.data import build_report
from expense_audit_env.grader import grade_task
from expense_audit_env.models import ExpenseAction
from expense_audit_env.server.environment import ExpenseAuditEnvironment


def test_easy_reset_and_step():
    env = ExpenseAuditEnvironment("easy")
    obs = env.reset()
    assert obs.expense_id == "EXP-001"
    assert not obs.done

    next_obs = env.step(ExpenseAction(decision="reject"))
    assert next_obs.done is True
    assert env.state.done is True
    assert env.state.cumulative_reward > 0


def test_step_auto_resets_when_needed():
    env = ExpenseAuditEnvironment("easy")
    first_step_obs = env.step(ExpenseAction(decision="approve"))
    assert env.state is not None
    assert first_step_obs is not None
    assert env.state.current_index == 1


def test_grader_medium():
    report = build_report("medium")
    preds = RuleBasedAuditor().predict_report(report)
    score = grade_task("medium", preds, report)
    assert 0.0 <= score <= 1.0
    assert score > 0.0
