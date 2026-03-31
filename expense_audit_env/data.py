from __future__ import annotations

from .models import ExpenseEntry, ExpenseMetadata, ExpenseReport, SpendingLimits


def _base_limits() -> SpendingLimits:
    return SpendingLimits(
        per_category={
            "travel": 300.0,
            "meals": 60.0,
            "lodging": 220.0,
            "office": 80.0,
            "misc": 40.0,
        },
        daily_limit=400.0,
        require_receipt_over=25.0,
        flag_margin_ratio=0.9,
    )


def _entry(
    expense_id: str,
    amount: float,
    category: str,
    requires_receipt: bool,
    employee_id: str = "E-1024",
    date: str = "2026-03-21",
    merchant: str = "Generic Vendor",
    report_id: str = "R-1001",
    receipt_present: bool = True,
    receipt_text: str | None = None,
    weight: float = 1.0,
    soft_review: bool = False,
    duplicate_of: str | None = None,
    combined_group: str | None = None,
    expected_decision: str = "approve",
    reason: str = "",
) -> ExpenseEntry:
    return ExpenseEntry(
        expense_id=expense_id,
        amount=amount,
        category=category,
        requires_receipt=requires_receipt,
        receipt_present=receipt_present,
        receipt_text=receipt_text,
        metadata=ExpenseMetadata(
            employee_id=employee_id,
            date=date,
            merchant=merchant,
            report_id=report_id,
            notes=reason,
        ),
        weight=weight,
        soft_review=soft_review,
        duplicate_of=duplicate_of,
        combined_group=combined_group,
        expected_decision=expected_decision,  # type: ignore[arg-type]
        reason=reason,
    )


def build_report(task_id: str) -> ExpenseReport:
    task_id = task_id.lower().strip()
    limits = _base_limits()

    if task_id == "easy":
        entries = [
            _entry(
                "EXP-001",
                amount=124.50,
                category="travel",
                requires_receipt=True,
                receipt_present=False,
                receipt_text=None,
                expected_decision="reject",
                reason="Missing receipt for a receipt-required travel expense.",
            )
        ]
        return ExpenseReport(
            report_id="R-EASY-001",
            employee_id="E-1024",
            policy_name="Standard Travel Policy",
            entries=entries,
            spending_limits=limits,
            trip_window=("2026-03-20", "2026-03-25"),
            total_budget=500.0,
        )

    if task_id == "medium":
        entries = [
            _entry(
                "EXP-101",
                amount=18.20,
                category="meals",
                requires_receipt=False,
                expected_decision="approve",
                reason="Under meal limit and no receipt required.",
            ),
            _entry(
                "EXP-102",
                amount=74.00,
                category="meals",
                requires_receipt=False,
                expected_decision="reject",
                reason="Meals exceed per-category meal limit.",
            ),
            _entry(
                "EXP-103",
                amount=220.00,
                category="lodging",
                requires_receipt=True,
                receipt_present=True,
                receipt_text="Hotel folio #88311",
                expected_decision="approve",
                reason="Lodging is within limit and receipt is attached.",
            ),
            _entry(
                "EXP-104",
                amount=39.99,
                category="office",
                requires_receipt=True,
                receipt_present=False,
                receipt_text=None,
                expected_decision="reject",
                reason="Office expense is above receipt threshold and missing receipt.",
            ),
        ]
        return ExpenseReport(
            report_id="R-MED-002",
            employee_id="E-2048",
            policy_name="Department Travel and Meals Policy",
            entries=entries,
            spending_limits=limits,
            trip_window=("2026-03-18", "2026-03-22"),
            total_budget=700.0,
        )

    if task_id == "hard":
        entries = [
            _entry(
                "EXP-201",
                amount=287.00,
                category="travel",
                requires_receipt=True,
                receipt_present=True,
                receipt_text="Airline itinerary and receipt",
                expected_decision="flag",
                soft_review=True,
                reason="Close to the travel limit and should be manually reviewed.",
            ),
            _entry(
                "EXP-202",
                amount=34.00,
                category="meals",
                requires_receipt=False,
                expected_decision="approve",
                reason="Normal meal expense under limit.",
            ),
            _entry(
                "EXP-203",
                amount=165.00,
                category="travel",
                requires_receipt=True,
                receipt_present=False,
                receipt_text=None,
                expected_decision="reject",
                reason="Travel expense missing a receipt.",
            ),
            _entry(
                "EXP-204",
                amount=92.00,
                category="office",
                requires_receipt=True,
                receipt_present=True,
                receipt_text="Split receipt part 1",
                duplicate_of="EXP-205",
                combined_group="split-1",
                expected_decision="flag",
                soft_review=True,
                reason="Possible receipt splitting across two office items.",
            ),
            _entry(
                "EXP-205",
                amount=95.00,
                category="office",
                requires_receipt=True,
                receipt_present=True,
                receipt_text="Split receipt part 2",
                duplicate_of="EXP-204",
                combined_group="split-1",
                expected_decision="flag",
                soft_review=True,
                reason="Possible receipt splitting across two office items.",
            ),
        ]
        return ExpenseReport(
            report_id="R-HARD-003",
            employee_id="E-4096",
            policy_name="Advanced Audit and Exception Policy",
            entries=entries,
            spending_limits=limits,
            trip_window=("2026-03-16", "2026-03-24"),
            total_budget=900.0,
        )

    raise ValueError(f"Unknown task_id: {task_id}")


def available_tasks() -> list[str]:
    return ["easy", "medium", "hard"]
