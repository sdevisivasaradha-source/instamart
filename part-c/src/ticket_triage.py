"""Part C: privacy-safe, validated, and routed support-ticket triage."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_CATEGORIES = {"billing", "delivery", "fraud", "technical"}
MAX_INPUT_TOKENS = 400
MAX_RESPONSE_TOKENS = 40
PII_PATTERNS = {
    "email": (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    "phone": (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    "order_id": (re.compile(r"\bORD-\d{6}\b", re.IGNORECASE), "[ORDER_ID]"),
}
CATEGORY_KEYWORDS = {
    "fraud": {"unauthorized", "fraud", "hacked", "stolen"},
    "technical": {"app", "crash", "login", "otp", "error", "freezes"},
    "billing": {
        "charged",
        "invoice",
        "bill",
        "refund",
        "deducted",
        "payment",
    },
    "delivery": {"arrived", "rider", "delivery", "delivered", "late"},
}
ZERO_SHOT_PROMPT = (
    "Classify the support ticket as billing, delivery, technical, or fraud. "
    "Return JSON with category and reason."
)
FEW_SHOT_PROMPT = (
    ZERO_SHOT_PROMPT
    + "\nExamples:\n"
    "Unauthorized transaction -> fraud\n"
    "Order arrived late -> delivery\n"
    "App crashes at login -> technical\n"
    "Charged twice -> billing"
)
PROMPTS = {"zero_shot": ZERO_SHOT_PROMPT, "few_shot": FEW_SHOT_PROMPT}


class TransientModelError(RuntimeError):
    """Retryable simulated provider error used by offline mock mode."""

    def __init__(
        self, message: str, consumed_tokens: int = 0, retry_count: int = 0
    ) -> None:
        super().__init__(message)
        self.consumed_tokens = consumed_tokens
        self.retry_count = retry_count


class TokenBudgetError(RuntimeError):
    """Raised before a model attempt that could exceed the hard budget."""

    def __init__(
        self, message: str, consumed_tokens: int = 0, retry_count: int = 0
    ) -> None:
        super().__init__(message)
        self.consumed_tokens = consumed_tokens
        self.retry_count = retry_count


def estimate_tokens(text: str) -> int:
    """Use a deterministic conservative token estimate for guardrails."""
    return math.ceil(len((text or "").split()) * 1.3)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Mask known PII patterns before any model receives ticket text."""
    safe_text = text
    counts = {}
    for name, (pattern, replacement) in PII_PATTERNS.items():
        safe_text, count = pattern.subn(replacement, safe_text)
        counts[name] = count
    return safe_text, counts


@dataclass
class OfflineTriageModel:
    """Deterministic model simulator with controlled failure modes."""

    calls: int = 0
    successful_calls: int = 0

    def classify(self, safe_text: str, prompt_mode: str = "few_shot") -> str:
        self.calls += 1
        if self.calls % 9 == 0:
            raise TransientModelError("simulated temporary provider failure")
        self.successful_calls += 1
        words = set(re.findall(r"[a-z]+", safe_text.lower()))
        category_order = (
            ("fraud", "technical", "billing", "delivery")
            if prompt_mode == "few_shot"
            else ("billing", "fraud", "technical", "delivery")
        )
        category = next(
            (
                name
                for name in category_order
                if words & CATEGORY_KEYWORDS[name]
            ),
            "technical",
        )
        if self.successful_calls % 17 == 0:
            return json.dumps(
                {"category": "shipping", "reason": "invalid category simulation"}
            )
        if self.successful_calls % 13 == 0:
            return f"The request is best classified as {category}."
        return json.dumps({"category": category, "reason": "keyword evidence"})


def call_with_retries(
    model: OfflineTriageModel,
    safe_text: str,
    prompt_mode: str,
    available_tokens: int,
    max_attempts: int = 4,
) -> tuple[str, int, int]:
    """Retry transient failures while accounting for every model attempt."""
    request_tokens = estimate_tokens(safe_text) + estimate_tokens(PROMPTS[prompt_mode])
    consumed_tokens = 0
    for attempt in range(1, max_attempts + 1):
        if (
            consumed_tokens + request_tokens + MAX_RESPONSE_TOKENS
            > available_tokens
        ):
            raise TokenBudgetError(
                "insufficient budget for bounded model response",
                consumed_tokens=consumed_tokens,
                retry_count=attempt - 1,
            )
        consumed_tokens += request_tokens
        try:
            content = model.classify(safe_text, prompt_mode)
            consumed_tokens += estimate_tokens(content)
            return content, attempt - 1, consumed_tokens
        except TransientModelError as error:
            if attempt == max_attempts:
                raise TransientModelError(
                    "transient model failure after maximum retries",
                    consumed_tokens=consumed_tokens,
                    retry_count=attempt - 1,
                ) from error
    raise AssertionError("retry loop exited unexpectedly")


def validate_model_output(content: str) -> tuple[str, str]:
    """Accept allow-listed JSON or an unambiguous category in fallback text."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        categories = {
            word
            for word in re.findall(r"[a-z]+", (content or "").lower())
            if word in ALLOWED_CATEGORIES
        }
        if len(categories) == 1:
            return categories.pop(), "classified_keyword_fallback"
        return "", "flagged_invalid_output"

    category = payload.get("category") if isinstance(payload, dict) else None
    if category in ALLOWED_CATEGORIES:
        return str(category), "classified_json"
    return "", "flagged_invalid_output"


def route_ticket(
    status: str, category: str, is_premium: bool
) -> tuple[str, str]:
    """Apply routing and priority policy after validated classification."""
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "transient_failure",
    }:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_triage(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = OUTPUT_DIR,
    token_budget: int = 100_000,
    model: OfflineTriageModel | None = None,
    segment_path: Path = OUTPUT_DIR / "customer_segments.csv",
    prompt_mode: str = "few_shot",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Classify, evaluate, and route every ticket without exposing PII."""
    if prompt_mode not in PROMPTS:
        raise ValueError(f"prompt_mode must be one of {sorted(PROMPTS)}")
    ticket_path = processed_dir / "tickets.csv"
    if not ticket_path.is_file():
        raise FileNotFoundError(
            f"Missing {ticket_path}. Run `python -m src.data_pipeline`."
        )
    if not segment_path.is_file():
        raise FileNotFoundError(
            f"Missing {segment_path}. Run `python -m src.customer_intelligence`."
        )

    tickets = pd.read_csv(ticket_path, keep_default_na=False)
    segments = pd.read_csv(segment_path)
    premium_ids = set(
        segments.loc[segments["is_premium"].astype(str).str.lower().eq("true"), "customer_id"]
    )
    classifier = model or OfflineTriageModel()
    token_total = 0
    retry_total = 0
    redaction_totals = Counter({name: 0 for name in PII_PATTERNS})
    rows = []

    for ticket in tickets.itertuples(index=False):
        text = str(ticket.text)
        status = ""
        category = ""
        tokens_used = 0
        retry_count = 0
        redaction_count = 0
        if not text.strip():
            status = "skipped_empty"
        elif estimate_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            safe_text, counts = redact_pii(text)
            redaction_totals.update(counts)
            redaction_count = sum(counts.values())
            if any(pattern.search(safe_text) for pattern, _ in PII_PATTERNS.values()):
                raise AssertionError("PII remained after redaction")
            if token_total >= token_budget:
                status = "budget_exhausted"
            else:
                try:
                    content, retry_count, tokens_used = call_with_retries(
                        classifier,
                        safe_text,
                        prompt_mode,
                        available_tokens=token_budget - token_total,
                    )
                    token_total += tokens_used
                    retry_total += retry_count
                    category, status = validate_model_output(content)
                except TokenBudgetError as error:
                    tokens_used = error.consumed_tokens
                    retry_count = error.retry_count
                    token_total += tokens_used
                    retry_total += retry_count
                    status = "budget_exhausted"
                except TransientModelError as error:
                    tokens_used = error.consumed_tokens
                    retry_count = error.retry_count
                    token_total += tokens_used
                    retry_total += retry_count
                    status = "transient_failure"

        route, priority = route_ticket(
            status, category, str(ticket.customer_id) in premium_ids
        )
        rows.append(
            {
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "true_category": ticket.true_category,
                "prompt_mode": prompt_mode,
                "status": status,
                "predicted_category": category,
                "priority": priority,
                "route": route,
                "tokens_used": tokens_used,
                "retry_count": retry_count,
                "pii_items_redacted": redaction_count,
            }
        )

    result = pd.DataFrame(rows)
    evaluable = result[
        result["status"].isin({"classified_json", "classified_keyword_fallback"})
    ]
    summary = {
        "tickets": len(result),
        "prompt_mode": prompt_mode,
        "classified": len(evaluable),
        "accuracy_on_classified": round(
            float(
                evaluable["predicted_category"].eq(
                    evaluable["true_category"]
                ).mean()
            ),
            4,
        ),
        "token_budget": token_budget,
        "tokens_used": token_total,
        "retries": retry_total,
        "redactions": dict(redaction_totals),
        "status_counts": {
            str(key): int(value) for key, value in result["status"].value_counts().items()
        },
        "route_counts": {
            str(key): int(value) for key, value in result["route"].value_counts().items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        "triage_results.csv"
        if prompt_mode == "few_shot"
        else "triage_zero_shot_results.csv"
    )
    summary_path = output_dir / (
        "triage_summary.json"
        if prompt_mode == "few_shot"
        else "triage_zero_shot_summary.json"
    )
    result.to_csv(result_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result, summary


def compare_prompt_strategies(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = OUTPUT_DIR,
    token_budget: int = 100_000,
    segment_path: Path = OUTPUT_DIR / "customer_segments.csv",
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Run and persist equivalent zero-shot and few-shot evaluations."""
    summaries = {}
    rows = []
    for prompt_mode in ("zero_shot", "few_shot"):
        _, summary = run_triage(
            processed_dir=processed_dir,
            output_dir=output_dir,
            token_budget=token_budget,
            model=OfflineTriageModel(),
            segment_path=segment_path,
            prompt_mode=prompt_mode,
        )
        summaries[prompt_mode] = summary
        rows.append(
            {
                "prompt_mode": prompt_mode,
                "classified": summary["classified"],
                "accuracy_on_classified": summary["accuracy_on_classified"],
                "human_review": summary["route_counts"].get("human_review", 0),
                "tokens_used": summary["tokens_used"],
                "retries": summary["retries"],
            }
        )
    comparison = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "triage_prompt_comparison.csv", index=False)
    (output_dir / "triage_prompt_comparison.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return comparison, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--token-budget", type=int, default=100_000)
    args = parser.parse_args()
    comparison, summary = compare_prompt_strategies(
        args.processed_dir, args.output_dir, args.token_budget
    )
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
