from __future__ import annotations

import os

from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app

from ..baseline import RuleBasedAuditor
from ..data import available_tasks, build_report
from ..grader import grade_task
from ..models import ExpenseAction, ExpenseObservation
from .environment import ExpenseAuditEnvironment

TASK_ID = os.environ.get("TASK_ID", "easy")

# OpenEnv manages the environment instance for /reset, /step, /state.
app: FastAPI = create_fastapi_app(
    ExpenseAuditEnvironment,
    ExpenseAction,
    ExpenseObservation,
)


@app.get("/")
def root():
    return {
        "name": "Expense Report Auditing Environment",
        "task_id": TASK_ID,
        "tasks": available_tasks(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    items = []
    for task_id in available_tasks():
        report = build_report(task_id)
        items.append(
            {
                "task_id": task_id,
                "report_id": report.report_id,
                "entries": len(report.entries),
                "policy_name": report.policy_name,
            }
        )
    return items


@app.post("/baseline/{task_id}")
def baseline(task_id: str):
    task_id = task_id.lower().strip()
    report = build_report(task_id)
    predictions = RuleBasedAuditor().predict_report(report)
    score = grade_task(task_id, predictions, report)
    return {"task_id": task_id, "predictions": predictions, "score": score}


@app.post("/grade/{task_id}")
def grade(task_id: str, predictions: list[str]):
    task_id = task_id.lower().strip()
    report = build_report(task_id)
    score = grade_task(task_id, predictions, report)
    return {"task_id": task_id, "score": score}