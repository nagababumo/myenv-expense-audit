from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, ExpenseEntry, SpendingLimits


@dataclass
class Evaluation:
    gold: Decision
    reward: float
    feedback: str
    correct: bool
    severity: str


def _receipt_required(entry: ExpenseEntry, limits: SpendingLimits) -> bool:
    return entry.requires_receipt or entry.amount >= limits.require_receipt_over


def evaluate_expense(entry: ExpenseEntry, decision: Decision, limits: SpendingLimits) -> Evaluation:
    category_limit = limits.per_category.get(entry.category, limits.daily_limit or entry.amount)
    receipt_needed = _receipt_required(entry, limits)

    valid = True
    reasons: list[str] = []

    if entry.amount > category_limit:
        valid = False
        reasons.append(f"amount {entry.amount:.2f} exceeds {entry.category} limit {category_limit:.2f}")

    if receipt_needed and not entry.receipt_present:
        valid = False
        reasons.append("required receipt is missing")

    if entry.duplicate_of is not None:
        valid = False
        reasons.append("duplicate or split receipt pattern")

    if entry.soft_review and entry.amount >= limits.flag_margin_ratio * category_limit:
        gold: Decision = "flag"
        severity = "soft-review"
    else:
        gold = "approve" if valid else "reject"
        severity = "hard-rule" if not valid else "clean"

    if entry.soft_review and entry.duplicate_of is not None:
        gold = "flag"
        severity = "soft-review"

    correct = decision == gold

    if correct:
        if gold == "flag":
            reward = 0.6
            feedback = "Correct: manual review was appropriate."
        elif gold == "approve":
            reward = 0.8
            feedback = "Correct: expense complies with policy."
        else:
            reward = 0.8
            feedback = "Correct: expense should be rejected."
    else:
        if gold == "approve" and decision == "reject":
            reward = -0.6
            feedback = "Overly strict: a valid expense was rejected."
        elif gold == "approve" and decision == "flag":
            reward = -0.2
            feedback = "Unnecessary flag: valid expense should have been approved."
        elif gold == "reject" and decision == "approve":
            reward = -1.5
            feedback = "Critical error: invalid expense was approved."
        elif gold == "reject" and decision == "flag":
            reward = -0.2
            feedback = "Cautious but incorrect: invalid expense should have been rejected."
        elif gold == "flag" and decision == "approve":
            reward = -1.0
            feedback = "Missed anomaly: item needed manual review."
        elif gold == "flag" and decision == "reject":
            reward = -0.3
            feedback = "Too harsh: item needed manual review, not outright rejection."
        else:
            reward = -0.5
            feedback = "Incorrect decision."

    if severity == "soft-review" and decision == "approve":
        reward -= 0.2
        if correct:
            feedback = "Correct category, but approval on a borderline item is slightly risky."

    return Evaluation(gold=gold, reward=reward, feedback=feedback, correct=correct, severity=severity)
