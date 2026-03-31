from __future__ import annotations

from typing import Iterable

from .models import Decision, ExpenseReport


def _weighted_accuracy(predictions: list[Decision], gold: list[Decision], weights: list[float]) -> float:
    total = sum(weights) if weights else 0.0
    if total == 0:
        return 0.0
    score = 0.0
    for pred, truth, weight in zip(predictions, gold, weights):
        if pred == truth:
            score += weight
    return score / total


def _asymmetric_score(predictions: list[Decision], gold: list[Decision], weights: list[float]) -> float:
    total = sum(weights) if weights else 0.0
    if total == 0:
        return 0.0

    penalty = 0.0
    for pred, truth, weight in zip(predictions, gold, weights):
        if pred == truth:
            continue
        if truth == "reject" and pred == "approve":
            penalty += 1.5 * weight
        elif truth == "approve" and pred == "reject":
            penalty += 0.6 * weight
        elif truth == "flag" and pred == "approve":
            penalty += 1.0 * weight
        else:
            penalty += 0.2 * weight

    score = max(0.0, 1.0 - penalty / (1.5 * total))
    return round(score, 6)


def grade_task(task_id: str, predictions: Iterable[Decision], report: ExpenseReport) -> float:
    preds = list(predictions)
    gold = [entry.expected_decision for entry in report.entries]
    weights = [entry.weight for entry in report.entries]

    if len(preds) != len(gold):
        return 0.0

    task_id = task_id.lower().strip()
    if task_id == "easy":
        return 1.0 if preds == gold else 0.0
    if task_id == "medium":
        return round(_weighted_accuracy(preds, gold, weights), 6)
    if task_id == "hard":
        return _asymmetric_score(preds, gold, weights)
    raise ValueError(f"Unknown task_id: {task_id}")
