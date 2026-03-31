from __future__ import annotations

from .models import Decision, ExpenseEntry, ExpenseReport
from .policy import evaluate_expense


class RuleBasedAuditor:
    def decide(self, entry: ExpenseEntry, report: ExpenseReport) -> Decision:
        evaluation = evaluate_expense(entry, "approve", report.spending_limits)
        if evaluation.gold == "flag":
            return "flag"
        if evaluation.gold == "reject":
            return "reject"
        return "approve"

    def predict_report(self, report: ExpenseReport) -> list[Decision]:
        return [self.decide(entry, report) for entry in report.entries]
