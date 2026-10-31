import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.refund_agent import (
    RefundLedger,
    calculate_refund,
    evaluate_refund_agent,
    run_refund_agent,
)


def test_refund_calculation_enforces_policy() -> None:
    assert calculate_refund(
        {
            "status": "completed",
            "order_amount": 500,
            "delivery_minutes": 46,
            "order_ts": "2025-09-01 10:00:00",
        },
        datetime(2025, 9, 2, tzinfo=timezone.utc),
    ) == {
        "eligible": True,
        "refund_inr": 500.0,
        "reason": "late_delivery_full_refund",
        "approval": "automatic",
    }
    assert (
        calculate_refund(
            {
                "status": "completed",
                "order_amount": 1500,
                "delivery_minutes": 60,
                "order_ts": "2025-09-01 10:00:00",
            },
            datetime(2025, 9, 2, tzinfo=timezone.utc),
        )["approval"]
        == "human_required"
    )
    assert (
        calculate_refund(
            {
                "status": "cancelled",
                "order_amount": 500,
                "delivery_minutes": None,
                "order_ts": "2025-09-01 10:00:00",
            },
            datetime(2025, 9, 2, tzinfo=timezone.utc),
        )["eligible"]
        is False
    )
    assert (
        calculate_refund(
            {
                "status": "completed",
                "order_amount": 500,
                "delivery_minutes": 60,
                "order_ts": "2025-09-01 10:00:00",
            },
            datetime(2025, 9, 1, 10, 30, tzinfo=timezone.utc),
        )["reason"]
        == "request_before_delivery"
    )


def test_agent_covers_refund_escalation_denial_and_info(tmp_path: Path) -> None:
    evaluation, summary = evaluate_refund_agent(output_dir=tmp_path)

    assert summary["scenarios"] == 9
    assert summary["accuracy"] == 1.0
    assert evaluation["matches"].all()
    assert set(evaluation["actual_status"]) == {
        "DENIED",
        "ESCALATED",
        "NEEDS_INFO",
        "AUTH_REQUIRED",
        "NOT_FOUND_OR_FORBIDDEN",
        "REFUNDED",
    }


def test_ledger_is_idempotent(tmp_path: Path) -> None:
    ledger = RefundLedger(tmp_path / "ledger.db")
    first = ledger.record("issue_refund", "ORD-123456", 500)
    second = ledger.record("issue_refund", "ORD-123456", 500)

    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    with sqlite3.connect(tmp_path / "ledger.db") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM refund_actions"
        ).fetchone()[0] == 1


def test_agent_enforces_step_and_token_limits(tmp_path: Path) -> None:
    ledger = RefundLedger(tmp_path / "ledger.db")
    step_limited = run_refund_agent(
        "Refund order ORD-100048",
        customer_id="C0007",
        ledger=ledger,
        max_steps=1,
    )
    terminal_at_limit = run_refund_agent(
        "Refund order ORD-100001",
        customer_id="C0001",
        ledger=ledger,
        max_steps=3,
    )
    token_limited = run_refund_agent(
        "Refund order ORD-100048 " + "word " * 100,
        customer_id="C0007",
        ledger=ledger,
        token_limit=20,
    )

    assert step_limited["status"] == "FALLBACK_MAX_STEPS"
    assert terminal_at_limit["status"] == "DENIED"
    assert token_limited["status"] == "FALLBACK_TOKEN_LIMIT"
    assert token_limited["tokens_used"] <= 20


def test_token_limit_is_checked_before_refund_side_effect(tmp_path: Path) -> None:
    ledger = RefundLedger(tmp_path / "ledger.db")
    result = run_refund_agent(
        "Refund order ORD-100048",
        customer_id="C0007",
        ledger=ledger,
        token_limit=118,
        request_time=datetime(2025, 4, 3, tzinfo=timezone.utc),
    )

    assert result["status"] == "FALLBACK_TOKEN_LIMIT"
    with sqlite3.connect(tmp_path / "ledger.db") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM refund_actions"
        ).fetchone()[0] == 0
