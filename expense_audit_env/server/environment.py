from __future__ import annotations

from typing import Any

from openenv.core.env_server import Environment

from ..data import build_report
from ..models import ExpenseAction, ExpenseObservation, ExpenseState, StepRecord
from ..policy import evaluate_expense


class ExpenseAuditEnvironment(Environment):
    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self.task_id = task_id.lower().strip()
        self._state: ExpenseState | None = None

    def _current_entry(self):
        if self._state is None:
            raise RuntimeError("Environment not reset.")
        if self._state.current_index >= len(self._state.report.entries):
            return None
        return self._state.report.entries[self._state.current_index]

    def _observation_for(
        self,
        entry,
        *,
        reward: float = 0.0,
        done: bool = False,
        feedback: str = "",
    ) -> ExpenseObservation:
        report = self._state.report
        idx = min(self._state.current_index, len(report.entries) - 1)
        return ExpenseObservation(
            expense_id=entry.expense_id,
            amount=entry.amount,
            category=entry.category,
            requires_receipt=entry.requires_receipt,
            receipt_text=entry.receipt_text,
            spending_limits=report.spending_limits,
            metadata=entry.metadata,
            position=idx,
            total_entries=len(report.entries),
            reward=reward,
            done=done,
            status="done" if done else "pending",
            policy_hint=entry.reason,
            last_feedback=feedback,
        )

    def reset(self) -> ExpenseObservation:
        report = build_report(self.task_id)
        self._state = ExpenseState(report=report, current_index=0, task_id=self.task_id)
        first = self._current_entry()
        return self._observation_for(first, reward=0.0, done=False, feedback="Episode started.")

    def step(self, action: ExpenseAction) -> ExpenseObservation:
        if self._state is None:
            self.reset()

        entry = self._current_entry()
        if entry is None:
            return ExpenseObservation(
                expense_id="",
                amount=0.0,
                category="",
                requires_receipt=False,
                receipt_text=None,
                spending_limits=self._state.report.spending_limits,
                metadata=self._state.report.entries[-1].metadata,
                position=len(self._state.report.entries),
                total_entries=len(self._state.report.entries),
                reward=0.0,
                done=True,
                status="done",
                policy_hint="Episode already complete.",
                last_feedback="Episode already complete.",
            )

        evaluation = evaluate_expense(entry, action.decision, self._state.report.spending_limits)
        record = StepRecord(
            expense_id=entry.expense_id,
            decision=action.decision,
            gold_decision=evaluation.gold,
            reward=evaluation.reward,
            correct=evaluation.correct,
            feedback=evaluation.feedback,
        )

        self._state.decisions.append(record.model_dump())
        self._state.cumulative_reward += evaluation.reward
        self._state.last_feedback = evaluation.feedback
        self._state.last_reward = evaluation.reward
        self._state.current_index += 1
        self._state.done = self._state.current_index >= len(self._state.report.entries)

        if self._state.done:
            return ExpenseObservation(
                expense_id=entry.expense_id,
                amount=entry.amount,
                category=entry.category,
                requires_receipt=entry.requires_receipt,
                receipt_text=entry.receipt_text,
                spending_limits=self._state.report.spending_limits,
                metadata=entry.metadata,
                position=len(self._state.report.entries) - 1,
                total_entries=len(self._state.report.entries),
                reward=evaluation.reward,
                done=True,
                status="done",
                policy_hint=entry.reason,
                last_feedback=evaluation.feedback,
            )

        next_entry = self._current_entry()
        return self._observation_for(
            next_entry,
            reward=evaluation.reward,
            done=False,
            feedback=evaluation.feedback,
        )

    @property
    def state(self) -> ExpenseState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    def state_payload(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Call reset() before state_payload().")
        return self._state.model_dump()