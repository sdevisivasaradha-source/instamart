import re
from pathlib import Path

from src.ticket_triage import (
    OfflineTriageModel,
    TransientModelError,
    compare_prompt_strategies,
    redact_pii,
    run_triage,
    validate_model_output,
)


class AlwaysFailModel(OfflineTriageModel):
    def classify(self, safe_text: str, prompt_mode: str = "few_shot") -> str:
        raise TransientModelError("provider unavailable")


def test_redacts_email_phone_and_order_id() -> None:
    text = "Email me@example.com, call 9876543210 about ord-123456."
    safe, counts = redact_pii(text)

    assert counts == {"email": 1, "phone": 1, "order_id": 1}
    assert safe == "Email [EMAIL], call [PHONE] about [ORDER_ID]."


def test_output_validation_is_strict_but_has_unambiguous_fallback() -> None:
    assert validate_model_output('{"category":"fraud","reason":"x"}') == (
        "fraud",
        "classified_json",
    )
    assert validate_model_output("This is a delivery issue.") == (
        "delivery",
        "classified_keyword_fallback",
    )
    assert validate_model_output('{"category":"shipping"}') == (
        "",
        "flagged_invalid_output",
    )
    assert validate_model_output("billing or delivery") == (
        "",
        "flagged_invalid_output",
    )


def test_triage_applies_guardrails_routing_and_retries(tmp_path: Path) -> None:
    result, summary = run_triage(
        output_dir=tmp_path,
        model=OfflineTriageModel(),
    )

    assert len(result) == 164
    assert summary["status_counts"]["skipped_empty"] == 4
    assert summary["status_counts"]["flagged_long"] == 4
    assert summary["retries"] > 0
    assert summary["redactions"] == {"email": 55, "phone": 91, "order_id": 132}
    assert summary["accuracy_on_classified"] == 1.0
    assert result.loc[
        result["predicted_category"].eq("fraud"), "route"
    ].eq("security_team").all()
    assert (tmp_path / "triage_results.csv").is_file()
    assert (tmp_path / "triage_summary.json").is_file()

    output_text = (tmp_path / "triage_results.csv").read_text(encoding="utf-8")
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", output_text)
    assert not re.search(r"\b[6-9]\d{9}\b", output_text)
    assert not re.search(r"\bORD-\d{6}\b", output_text, re.IGNORECASE)


def test_budget_exhaustion_routes_to_human(tmp_path: Path) -> None:
    result, summary = run_triage(output_dir=tmp_path, token_budget=80)

    assert summary["status_counts"]["budget_exhausted"] > 0
    assert result.loc[
        result["status"].eq("budget_exhausted"), "route"
    ].eq("human_review").all()
    assert summary["tokens_used"] <= summary["token_budget"]


def test_failed_retry_attempts_are_charged_to_budget(tmp_path: Path) -> None:
    result, summary = run_triage(
        output_dir=tmp_path,
        token_budget=100,
        model=OfflineTriageModel(calls=8),
    )

    exhausted = result[result["status"].eq("budget_exhausted")]
    assert not exhausted.empty
    assert exhausted["tokens_used"].max() > 0
    assert summary["tokens_used"] > 0
    assert summary["tokens_used"] <= 100


def test_exhausted_retries_are_charged_to_budget(tmp_path: Path) -> None:
    result, summary = run_triage(
        output_dir=tmp_path,
        token_budget=500,
        model=AlwaysFailModel(),
    )

    failed = result[result["status"].eq("transient_failure")]
    assert not failed.empty
    assert failed.iloc[0]["tokens_used"] > 0
    assert failed.iloc[0]["retry_count"] == 3
    assert summary["tokens_used"] <= 500


def test_zero_shot_and_few_shot_are_compared(tmp_path: Path) -> None:
    comparison, summaries = compare_prompt_strategies(output_dir=tmp_path)

    assert comparison["prompt_mode"].tolist() == ["zero_shot", "few_shot"]
    assert set(summaries) == {"zero_shot", "few_shot"}
    assert summaries["few_shot"]["accuracy_on_classified"] >= summaries[
        "zero_shot"
    ]["accuracy_on_classified"]
    assert (tmp_path / "triage_prompt_comparison.csv").is_file()
    assert (tmp_path / "triage_prompt_comparison.json").is_file()
