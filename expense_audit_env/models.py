from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "reject", "flag"]


class SpendingLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_category: dict[str, float] = Field(default_factory=dict)
    daily_limit: float = 0.0
    require_receipt_over: float = 0.0
    flag_margin_ratio: float = 0.9


class ExpenseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str
    date: str
    merchant: str
    report_id: str = ""
    travel_trip_id: str | None = None
    location: str | None = None
    currency: str = "USD"
    notes: str = ""


class ExpenseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expense_id: str
    amount: float
    category: str
    requires_receipt: bool
    receipt_text: str | None = None
    receipt_present: bool = False
    metadata: ExpenseMetadata = Field(
        default_factory=lambda: ExpenseMetadata(employee_id="", date="", merchant="")
    )
    weight: float = 1.0
    soft_review: bool = False
    duplicate_of: str | None = None
    combined_group: str | None = None
    expected_decision: Decision = "approve"
    reason: str = ""


class ExpensePolicyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str
    spending_limits: SpendingLimits
    trip_window: tuple[str, str] | None = None
    total_budget: float | None = None


class ExpenseReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    employee_id: str
    policy_name: str
    entries: list[ExpenseEntry]
    spending_limits: SpendingLimits
    trip_window: tuple[str, str] | None = None
    total_budget: float | None = None


class ExpenseAction(Action):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    index: int | None = None
    comment: str = ""


class ExpenseObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    expense_id: str
    amount: float
    category: str
    requires_receipt: bool
    receipt_text: str | None = None
    spending_limits: SpendingLimits
    metadata: ExpenseMetadata
    position: int
    total_entries: int
    status: str = "pending"
    policy_hint: str = ""
    last_feedback: str = ""
    allowed_decisions: tuple[Decision, ...] = ("approve", "reject", "flag")


class ExpenseState(State):
    model_config = ConfigDict(extra="forbid")

    report: ExpenseReport
    current_index: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    last_feedback: str = ""
    last_reward: float = 0.0
    task_id: str = ""


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expense_id: str
    decision: Decision
    gold_decision: Decision
    reward: float
    correct: bool
    feedback: str