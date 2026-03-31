'''from __future__ import annotations

from openenv.core import EnvClient, StepResult

from .data import build_report
from .models import ExpenseAction, ExpenseObservation, ExpenseState


class ExpenseAuditEnvClient(EnvClient[ExpenseAction, ExpenseObservation, ExpenseState]):
    def _step_payload(self, action: ExpenseAction) -> dict:
        return {
            "decision": action.decision,
            "index": action.index,
            "comment": action.comment,
        }

    def _parse_result(self, payload: dict) -> StepResult[ExpenseObservation]:
        obs = ExpenseObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> ExpenseState:
        # Fallback state parser that keeps the client usable even if extra fields appear.
        report_id = payload.get("report", {}).get("report_id", "")
        task_hint = payload.get("task_id", "easy")
        try:
            report = build_report(task_hint)
        except Exception:
            from .models import ExpenseReport, ExpenseEntry, ExpenseMetadata, SpendingLimits

            report_payload = payload.get("report", {})
            limits_payload = report_payload.get("spending_limits", {})
            limits = SpendingLimits(**limits_payload)
            entries = []
            for item in report_payload.get("entries", []):
                item = dict(item)
                item["metadata"] = ExpenseMetadata(**item["metadata"])
                entries.append(ExpenseEntry(**item))
            report = ExpenseReport(
                report_id=report_payload.get("report_id", report_id),
                employee_id=report_payload.get("employee_id", ""),
                policy_name=report_payload.get("policy_name", ""),
                entries=entries,
                spending_limits=limits,
                trip_window=tuple(report_payload["trip_window"]) if report_payload.get("trip_window") else None,
                total_budget=report_payload.get("total_budget"),
            )

        return ExpenseState(
            report=report,
            current_index=payload.get("current_index", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            decisions=payload.get("decisions", []),
            last_feedback=payload.get("last_feedback", ""),
            last_reward=payload.get("last_reward", 0.0),
            task_id=payload.get("task_id", ""),
        )
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import requests
from openenv.core import EnvClient

from .data import build_report
from .models import ExpenseAction, ExpenseObservation, ExpenseState

TObservation = TypeVar("TObservation")


@dataclass
class StepResult(Generic[TObservation]):
    observation: TObservation
    reward: float
    done: bool
    info: dict | None = None


class ExpenseAuditEnvClient(EnvClient[ExpenseAction, ExpenseObservation, ExpenseState]):
    def _step_payload(self, action: ExpenseAction) -> dict:
        return {
            "decision": action.decision,
            "index": action.index,
            "comment": action.comment,
        }

    def _parse_result(self, payload: dict) -> StepResult[ExpenseObservation]:
        obs = ExpenseObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict) -> ExpenseState:
        report_id = payload.get("report", {}).get("report_id", "")
        task_hint = payload.get("task_id", "easy")

        try:
            report = build_report(task_hint)
        except Exception:
            from .models import ExpenseEntry, ExpenseReport, ExpenseMetadata, SpendingLimits

            report_payload = payload.get("report", {})
            limits_payload = report_payload.get("spending_limits", {})
            limits = SpendingLimits(**limits_payload)

            entries = []
            for item in report_payload.get("entries", []):
                item = dict(item)
                item["metadata"] = ExpenseMetadata(**item["metadata"])
                entries.append(ExpenseEntry(**item))

            report = ExpenseReport(
                report_id=report_payload.get("report_id", report_id),
                employee_id=report_payload.get("employee_id", ""),
                policy_name=report_payload.get("policy_name", ""),
                entries=entries,
                spending_limits=limits,
                trip_window=tuple(report_payload["trip_window"]) if report_payload.get("trip_window") else None,
                total_budget=report_payload.get("total_budget"),
            )

        return ExpenseState(
            report=report,
            current_index=payload.get("current_index", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            decisions=payload.get("decisions", []),
            last_feedback=payload.get("last_feedback", ""),
            last_reward=payload.get("last_reward", 0.0),
            task_id=payload.get("task_id", ""),
        )