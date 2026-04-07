import argparse
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from expense_audit_env.data import available_tasks, build_report
from expense_audit_env.grader import grade_task
from expense_audit_env.models import ExpenseAction
from expense_audit_env.server.environment import ExpenseAuditEnvironment

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SYSTEM_PROMPT = (
    "You are an expert corporate expense auditor.\n"
    "Review the expense entry and choose exactly one of approve, reject, or flag.\n"
    "Respond using valid JSON only, with keys 'decision' and 'comment'.\n"
    "Example: {\"decision\": \"reject\", \"comment\": \"Missing receipt.\"}."
)

ALLOWED_DECISIONS = {"approve", "reject", "flag"}


def get_api_key() -> str:
    return OPENAI_API_KEY or GOOGLE_API_KEY or HF_TOKEN or ""


def get_provider() -> str:
    if GOOGLE_API_KEY:
        return "google"
    if OPENAI_API_KEY:
        return "openai"
    if HF_TOKEN:
        return "huggingface"
    raise RuntimeError(
        "Set OPENAI_API_KEY, GOOGLE_API_KEY, or HF_TOKEN in your environment before running inference.py"
    )


def get_model_name() -> str:
    default_openai = "gpt-4o-mini"
    default_google = "gemini-1.5-flash"
    provider = get_provider()
    if MODEL_NAME and MODEL_NAME != "<your-active-model-name>":
        return MODEL_NAME
    return default_openai if provider == "openai" else default_google


def map_model_for_provider(model: str, provider: str) -> str:
    """Map model names for Google's OpenAI-compatible API."""
    if provider == "google":
        # Google's OpenAI-compatible API might use different model identifiers
        model_mapping = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-3.1-flash-lite-preview": "gemini-1.5-flash",  # Fallback
            "gpt-4o-mini": "gemini-1.5-flash",  # Fallback to Gemini
        }
        return model_mapping.get(model, model)
    return model



def get_api_base_url() -> str | None:
    provider = get_provider()
    if provider == "google":
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    if API_BASE_URL and API_BASE_URL != "<your-active-api-base-url>":
        return API_BASE_URL
    return None


def env_summary() -> dict[str, str]:
    return {
        "OPENAI_API_KEY_set": str(bool(OPENAI_API_KEY)),
        "GOOGLE_API_KEY_set": str(bool(GOOGLE_API_KEY)),
        "HF_TOKEN_set": str(bool(HF_TOKEN)),
        "API_BASE_URL_set": str(bool(API_BASE_URL and API_BASE_URL != "<your-active-api-base-url>")),
        "MODEL_NAME": MODEL_NAME if MODEL_NAME != "<your-active-model-name>" else "<default>",
        "LOCAL_IMAGE_NAME_set": str(bool(LOCAL_IMAGE_NAME)),
    }


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?\n", "", text)
    text = re.sub(r"\n```", "", text)
    return text


def parse_json_text(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty response from model")
    text = normalize_text(text)
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON from model response: {text}") from exc


def extract_decision(raw: str) -> str:
    raw = raw.lower()
    for decision in ALLOWED_DECISIONS:
        if decision in raw:
            return decision
    raise ValueError(f"No valid decision found in response: {raw}")


def parse_response_text(output: str) -> dict[str, str]:
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
        f"Receipt text: {obs.receipt_text or 'None'}",
        f"Policy hint: {obs.policy_hint}",
        f"Position: {obs.position + 1}/{obs.total_entries}",
    ]
    lines.append("Spending limits:")
    for category, limit in obs.spending_limits.per_category.items():
        lines.append(f"  {category}: {limit:.2f}")
    lines.append(f"  daily_limit: {obs.spending_limits.daily_limit:.2f}")
    lines.append(f"  require_receipt_over: {obs.spending_limits.require_receipt_over:.2f}")
    lines.append(f"  flag_margin_ratio: {obs.spending_limits.flag_margin_ratio:.2f}")
    lines.append("Metadata:")
    for key, value in obs.metadata.model_dump().items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def build_prompt(obs: Any) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current expense observation:\n{format_observation(obs)}\n\n"
        "Output only valid JSON with keys 'decision' and 'comment'."
    )


def create_client(provider: str) -> Any:
    if provider == "openai":
        return OpenAI(api_key=OPENAI_API_KEY)

    if provider == "google":
        return OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )

    if provider == "huggingface":
        return OpenAI(
            api_key=HF_TOKEN,
            base_url="https://router.huggingface.co/v1",
        )

    raise RuntimeError(f"Unknown provider: {provider}")


def extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text is not None:
        return response.output_text
    raw_output = getattr(response, "output", None)
    if raw_output is None:
        return str(response)
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        parts: list[str] = []
        for item in raw_output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for segment in content:
                        if isinstance(segment, str):
                            parts.append(segment)
                        elif isinstance(segment, dict):
                            parts.append(str(segment.get("text", "")))
        return "".join(parts).strip()
    return str(response)


def call_model(client: Any, prompt: str, model: str, temperature: float, provider: str) -> str:
    mapped_model = map_model_for_provider(model, provider)
    try:
        response = client.chat.completions.create(
            model=mapped_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=250,
        )
        if not response or not getattr(response, "choices", None):
            raise ValueError("Empty response from chat completion")
        return response.choices[0].message.content.strip()

    except Exception as e:
        status_code = getattr(e, "status_code", None)
        body = getattr(getattr(e, "response", None), "text", None)
        raise ValueError(f"API call failed: {type(e).__name__} status={status_code} body={body}") from e


def decide(obs: Any, client: Any, model: str, temperature: float, provider: str) -> dict[str, str]:
    prompt = build_prompt(obs)
    raw = call_model(client, prompt, model, temperature, provider)
    return parse_response_text(raw)


def run_task(task_id: str, client: Any, model: str, temperature: float, provider: str) -> tuple[list[str], float]:
    env = ExpenseAuditEnvironment(task_id)
    observation = env.reset()
    decisions: list[str] = []

    print(f"\n--- Running task {task_id} ---")
    while True:
        choice = decide(observation, client, model, temperature, provider)
        action = ExpenseAction(
            decision=choice["decision"],
            index=observation.position,
            comment=choice.get("comment", ""),
        )
        observation = env.step(action)
        decisions.append(action.decision)
        print(
            f"Expense {action.index + 1}/{observation.total_entries}: {action.decision} - {action.comment}"
        )
        print(
            f"  reward={observation.reward:.2f}, done={observation.done}, feedback={observation.last_feedback}"
        )
        if observation.done:
            break

    report = build_report(task_id)
    score = grade_task(task_id, decisions, report)
    return decisions, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible inference baseline.")
    parser.add_argument(
        "--task",
        default="all",
        help="Task to run: easy, medium, hard, or all",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. If omitted, uses MODEL_NAME env var or default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    args = parser.parse_args()

    print("ENV summary:", env_summary())
    provider = get_provider()
    client = create_client(provider)
    model = args.model or get_model_name()
    print("START inference")
    print(f"Using provider={provider}, model={model}, temperature={args.temperature}")

    tasks = [args.task] if args.task != "all" else available_tasks()
    scores = {}
    for task in tasks:
        print(f"STEP {task} start")
        decisions, score = run_task(task, client, model, args.temperature, provider)
        scores[task] = score
        print(f"STEP {task} complete score={score:.6f}")
    print("\n=== Baseline scores ===")
    for task, score in scores.items():
        print(f"{task}: {score:.6f}")
    average = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"Average score: {average:.6f}")
    print("END inference")


if __name__ == "__main__":
    main()
