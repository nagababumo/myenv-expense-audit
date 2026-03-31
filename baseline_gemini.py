import argparse
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from expense_audit_env.data import available_tasks, build_report
from expense_audit_env.grader import grade_task
from expense_audit_env.models import ExpenseAction
from expense_audit_env.server.environment import ExpenseAuditEnvironment

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert corporate expense auditor. "
    "Review the expense entry and choose one of exactly three decisions: approve, reject, or flag. "
    "Respond with valid JSON only, using the keys 'decision' and 'comment'. "
    "Do not include any additional text outside the JSON object."
)

ALLOWED_DECISIONS = {"approve", "reject", "flag"}


def get_provider() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GENAI_API_KEY")
    ):
        return "gemini"
    raise RuntimeError(
        "No API key found. Set OPENAI_API_KEY or GEMINI_API_KEY / GOOGLE_API_KEY / GENAI_API_KEY."
    )


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?\n", "", text)
    text = re.sub(r"\n```", "", text)
    return text


def parse_json_text(text: str) -> dict[str, Any]:
    text = normalize_text(text)

    # Try to extract the first JSON object from the response.
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse JSON from model response: {text}")


def extract_decision(raw: str) -> str:
    raw = raw.lower()
    for decision in ALLOWED_DECISIONS:
        if decision in raw:
            return decision
    raise ValueError(f"No valid decision found in response: {raw}")


def parse_response(output: str) -> dict[str, str]:
    try:
        payload = parse_json_text(output)
        decision = str(payload.get("decision", "")).strip().lower()
        comment = str(payload.get("comment", "")).strip()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError("Invalid decision in parsed JSON")
        return {"decision": decision, "comment": comment}
    except ValueError:
        decision = extract_decision(output)
        return {"decision": decision, "comment": output.strip()}


def format_observation(obs: Any) -> str:
    lines = [
        f"Expense ID: {obs.expense_id}",
        f"Amount: ${obs.amount:.2f}",
        f"Category: {obs.category}",
        f"Requires receipt: {obs.requires_receipt}",
        f"Receipt present: {obs.receipt_text is not None}",
    ]
    if obs.receipt_text:
        lines.append(f"Receipt text: {obs.receipt_text}")

    lines.extend(
        [
            f"Policy hint: {obs.policy_hint}",
            f"Position: {obs.position + 1}/{obs.total_entries}",
            "Spending limits:",
        ]
    )
    for category, limit in obs.spending_limits.per_category.items():
        lines.append(f"  {category}: {limit:.2f}")
    lines.append(f"Daily limit: {obs.spending_limits.daily_limit:.2f}")
    lines.append(f"Receipt required threshold: ${obs.spending_limits.require_receipt_over:.2f}")
    lines.append(f"Flag margin ratio: {obs.spending_limits.flag_margin_ratio:.2f}")
    lines.append("Metadata:")
    metadata = obs.metadata.model_dump()
    for key, value in metadata.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def build_prompt(obs: Any) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current expense observation:\n{format_observation(obs)}\n\n"
        "Choose exactly one of approve, reject, or flag. "
        "Explain your choice briefly in the comment field. "
        "Output only JSON, for example: {\"decision\": \"reject\", \"comment\": \"Missing receipt\"}."
    )


def call_openai(prompt: str, model: str, temperature: float) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=250,
    )

    if hasattr(response, "output_text") and response.output_text is not None:
        return response.output_text

    output = []
    raw_output = getattr(response, "output", None)
    if raw_output is None:
        return str(response)

    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        for item in raw_output:
            if isinstance(item, str):
                output.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    output.append(content)
                elif isinstance(content, list):
                    for segment in content:
                        if isinstance(segment, str):
                            output.append(segment)
                        elif isinstance(segment, dict):
                            output.append(str(segment.get("text", "")))
    return "".join(output).strip() or str(response)


def call_gemini(prompt: str, model: str, temperature: float) -> str:
    from google import genai
    from google.genai import types

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY / GENAI_API_KEY to use Gemini.")

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    config = types.GenerateContentConfig(temperature=temperature)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    if getattr(response, "text", None):
        return response.text

    if getattr(response, "output_text", None):
        return response.output_text

    raw_output = getattr(response, "output", None)
    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        segments: list[str] = []
        for item in raw_output:
            if isinstance(item, str):
                segments.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    segments.append(content)
                elif isinstance(content, list):
                    for segment in content:
                        if isinstance(segment, str):
                            segments.append(segment)
                        elif isinstance(segment, dict):
                            segments.append(str(segment.get("text", "")))
        return "".join(segments).strip()

    return str(response)

def get_model_and_provider(model_arg: str | None = None) -> tuple[str, str]:
    provider = get_provider()
    if model_arg:
        return provider, model_arg
    if provider == "openai":
        return provider, "gpt-4o-mini"
    return provider, "gemini-3.1-flash-lite-preview"


def decide(obs: Any, provider: str, model: str, temperature: float) -> dict[str, str]:
    prompt = build_prompt(obs)
    if provider == "openai":
        raw = call_openai(prompt, model, temperature)
    else:
        raw = call_gemini(prompt, model, temperature)
    return parse_response(raw)


def run_task(task_id: str, provider: str, model: str, temperature: float) -> tuple[list[str], float]:
    env = ExpenseAuditEnvironment(task_id)
    observation = env.reset()
    decisions: list[str] = []

    print(f"\n--- Running task {task_id} ---")
    while True:
        decision_payload = decide(observation, provider, model, temperature)
        decision = decision_payload["decision"]
        comment = decision_payload.get("comment", "")
        action = ExpenseAction(decision=decision, index=observation.position, comment=comment)
        observation = env.step(action)
        decisions.append(decision)

        print(f"Expense {action.index + 1}/{observation.total_entries}: {action.decision} - {comment}")
        print(f"  reward={observation.reward:.2f}, done={observation.done}, feedback={observation.last_feedback}\n")

        if observation.done:
            break

    report = build_report(task_id)
    score = grade_task(task_id, decisions, report)
    return decisions, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Gemini/OpenAI baseline on the expense audit environment.")
    parser.add_argument("--task", default="all", help="Task to run: easy, medium, hard, or all")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. If omitted, uses default for the provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    args = parser.parse_args()

    provider, model = get_model_and_provider(args.model)
    print(f"Using provider={provider}, model={model}, temperature={args.temperature}")

    tasks = [args.task] if args.task != "all" else available_tasks()
    scores = {}

    for task in tasks:
        decisions, score = run_task(task, provider, model, args.temperature)
        scores[task] = score

    print("\n=== Baseline scores ===")
    for task, score in scores.items():
        print(f"{task}: {score:.6f}")

    average = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"Average score: {average:.6f}")


if __name__ == "__main__":
    main()
