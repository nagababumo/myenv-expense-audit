---
---
title: Expense Report Auditing Environment
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Expense Report Auditing Environment (OpenEnv)

A deterministic OpenEnv environment for auditing synthetic expense reports.

## What this environment does

The agent reviews expense entries one by one and chooses one of three actions:

- `approve` for valid expenses
- `reject` for clear violations
- `flag` for manual review

The environment provides structured observations with policy context, tracks stepwise reward, and exposes a deterministic grader for `easy`, `medium`, and `hard` tasks.

## Project structure

```text
expense_audit_openenv/
├── README.md
├── Dockerfile
├── app.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── expense_audit_env/
│   ├── __init__.py
│   ├── models.py
│   ├── data.py
│   ├── policy.py
│   ├── grader.py
│   ├── baseline.py
│   ├── client.py
│   └── server/
│       ├── __init__.py
│       ├── environment.py
│       └── app.py
├── tasks/
│   ├── easy.json
│   ├── medium.json
│   └── hard.json
└── tests/
    └── test_environment.py
```

## Quick start

### 1) Local setup

Install the local dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) Run locally with Docker

```bash
docker build -t expense-audit-openenv .
```

Run the FastAPI environment:

```bash
docker run --rm -p 7860:7860 expense-audit-openenv
```

Run the Streamlit UI locally:

```bash
streamlit run streamlit_app.py
```

Or run it from Docker:

```bash
docker run --rm -p 8501:7860 -e APP_MODE=streamlit expense-audit-openenv
```

Then open the app in your browser at:

- `http://localhost:8501`

### Streamlit app usage

The Streamlit UI provides a simple interactive front end for the expense audit environment.

1. Select a task difficulty: `easy`, `medium`, or `hard`.
2. Click `Start / Reset task` to begin a new episode.
3. Review the current expense entry and metadata.
4. Choose one of the three decisions: `approve`, `reject`, or `flag`.
5. Add an optional comment and click `Submit Decision` to advance to the next expense.

### Streamlit features

- Task selection: switch between easy, medium, and hard task fixtures.
- Episode control: reset the environment and start over at any time.
- Expense details: view the current expense, receipt text, spending limits, and metadata.
- Decision history: see the list of submitted actions and rewards.
- LLM agent mode: switch to `LLM agent` in the sidebar and let the Gemini/OpenAI baseline propose the next action.
- Generate task with LLM: create a synthetic expense report dynamically using the loaded LLM model.

### What to expect in the Streamlit app

- Progress bar: current index and total entries for the selected task.
- Rewards: live cumulative reward updates after each decision.
- Feedback: last feedback text from the environment on each step.
- Done state: the app indicates when the episode is complete.

Then open:

- `GET /health`
- `GET /docs`
- `GET /tasks`
- `POST /reset/{task_id}`
- `POST /baseline/{task_id}`

### Baseline inference script

Run the inference baseline locally with:

```bash
python inference.py --task all
```

Set these environment variables before running:

- `OPENAI_API_KEY` or `HF_TOKEN` (preferred)
- `API_BASE_URL` (optional, for custom OpenAI-compatible endpoints)
- `MODEL_NAME` (optional, default: `gpt-4o-mini`)

You can also create a `.env` file from `.env.example` and place your keys there.

Run the baseline inside Docker:

```bash
docker run --rm -e OPENAI_API_KEY="your_openai_api_key" \
  -e MODEL_NAME="gpt-4o-mini" expense-audit-openenv python inference.py --task all
```

If you want to keep the legacy Gemini script, you can still run:

```bash
python baseline_gemini.py --task all
```

### 2) Use the environment with OpenEnv

The environment follows the OpenEnv pattern:

- define `Action`, `Observation`, and `State`
- implement `reset()`, `step(action)`, and `state()`
- serve it behind FastAPI
- package it in Docker for Hugging Face Spaces

OpenEnv’s docs describe this flow, including `create_fastapi_app(...)`, Docker packaging, and the `EnvClient` pattern. citeturn596173view0turn325773view1turn777575search1

### 3) Hugging Face Spaces

This repo is ready for a Docker Space because the README has `sdk: docker` and `app_port: 7860`, which is the documented Docker Spaces setup. citeturn596173view1

## File roles

`expense_audit_env/models.py`  
Typed dataclasses for actions, observations, expense entries, policy context, and episode state.

`expense_audit_env/data.py`  
Synthetic task generation and fixed task fixtures.

`expense_audit_env/policy.py`  
Deterministic policy evaluation and reward shaping.

`expense_audit_env/server/environment.py`  
The OpenEnv environment implementation with `reset`, `step`, and `state`.

`expense_audit_env/server/app.py`  
FastAPI app creation plus utility endpoints.

`expense_audit_env/grader.py`  
Deterministic scoring for `easy`, `medium`, and `hard`.

`expense_audit_env/baseline.py`  
A simple rule-based baseline that can run end-to-end.

`baseline_gemini.py`  
A Gemini/OpenAI inference baseline for the expense audit environment.

`expense_audit_env/client.py`  
An OpenEnv-style `EnvClient` implementation.

## Task summary

### Easy
One clear decision. Score is binary: correct or incorrect.

### Medium
Multiple independent entries with weighted accuracy.

### Hard
Mixed valid, invalid, and borderline entries. Wrong approvals are penalized more heavily than cautious flags.

The OpenEnv docs also describe the client-side pattern and the project layout used for custom environments, including `models.py`, `server/`, `client.py`, `openenv.yaml`, and Docker packaging. citeturn325773view2turn777575search1
