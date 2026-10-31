"""Part D: bounded database-backed refund tools and workflow agent."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "outputs"
ORDER_PATTERN = re.compile(r"\bORD-\d{6}\b")


def get_order(
    order_id: str, customer_id: str, db_path: Path = DB_PATH
) -> dict[str, object]:
    """Return refund fields only when the authenticated customer owns the order."""
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT order_id, customer_id, status, order_amount,
                   delivery_minutes, order_ts
            FROM orders WHERE order_id = ? AND customer_id = ?
            """,
            (order_id, customer_id),
        ).fetchone()
    if row is None:
        return {"error": "order_not_found_or_forbidden", "order_id": order_id}
    return {
        "order_id": str(row[0]),
        "customer_id": str(row[1]),
        "status": str(row[2]),
        "order_amount": float(row[3]),
        "delivery_minutes": None if row[4] is None else int(row[4]),
        "order_ts": str(row[5]),
    }


def calculate_refund(
    order: dict[str, object], request_time: datetime
) -> dict[str, object]:
    """Apply refund eligibility and approval thresholds deterministically."""
    if order.get("error"):
        return {"eligible": False, "refund_inr": 0.0, "reason": "order_not_found"}
    if order["status"] != "completed":
        return {
            "eligible": False,
            "refund_inr": 0.0,
            "reason": "only_completed_orders_are_eligible",
        }
    delivery = order["delivery_minutes"]
    if delivery is None or int(delivery) <= 45:
        return {
            "eligible": False,
            "refund_inr": 0.0,
            "reason": "delivery_not_over_45_minutes",
        }
    order_time = datetime.fromisoformat(str(order["order_ts"])).replace(
        tzinfo=timezone.utc
    )
    delivery_time = order_time + timedelta(minutes=int(delivery))
    if request_time.astimezone(timezone.utc) < delivery_time:
        return {
            "eligible": False,
            "refund_inr": 0.0,
            "reason": "request_before_delivery",
        }
    if request_time.astimezone(timezone.utc) > delivery_time + timedelta(days=7):
        return {
            "eligible": False,
            "refund_inr": 0.0,
            "reason": "request_outside_7_day_window",
        }
    amount = round(float(order["order_amount"]), 2)
    return {
        "eligible": True,
        "refund_inr": amount,
        "reason": "late_delivery_full_refund",
        "approval": "automatic" if amount <= 1000 else "human_required",
    }


@dataclass
class RefundLedger:
    """Transactional ledger with a unique action/order idempotency key."""

    path: Path = STATE_DIR / "refund_ledger.db"
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS refund_actions (
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    refund_inr REAL NOT NULL,
                    UNIQUE(action, order_id)
                )
                """
            )
        self._initialized = True

    def record(
        self, action: str, order_id: str, refund_inr: float
    ) -> dict[str, object]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "order_id": order_id,
            "refund_inr": round(float(refund_inr), 2),
        }
        with sqlite3.connect(self.path, timeout=30) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO refund_actions
                    (timestamp, action, order_id, refund_inr)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    action,
                    order_id,
                    record["refund_inr"],
                ),
            )
            stored = connection.execute(
                """
                SELECT timestamp, action, order_id, refund_inr
                FROM refund_actions WHERE action = ? AND order_id = ?
                """,
                (action, order_id),
            ).fetchone()
        if stored is None:
            raise AssertionError("refund ledger record was not persisted")
        return {
            "timestamp": str(stored[0]),
            "action": str(stored[1]),
            "order_id": str(stored[2]),
            "refund_inr": float(stored[3]),
            "idempotent_replay": cursor.rowcount == 0,
        }


def issue_refund(
    ledger: RefundLedger, order_id: str, refund_inr: float
) -> dict[str, object]:
    return ledger.record("issue_refund", order_id, refund_inr)


def escalate_to_human(
    ledger: RefundLedger, order_id: str, refund_inr: float
) -> dict[str, object]:
    return ledger.record("escalate_to_human", order_id, refund_inr)


def run_refund_agent(
    user_message: str,
    customer_id: str | None = None,
    db_path: Path = DB_PATH,
    ledger: RefundLedger | None = None,
    rag: PolicyRAG | None = None,
    max_steps: int = 6,
    token_limit: int = 2_000,
    checkpoint_path: Path | None = None,
    request_time: datetime | None = None,
) -> dict[str, object]:
    """Run a bounded state-machine agent over explicit refund tools."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    estimated_tokens = len(user_message.split()) * 2 + 50
    if estimated_tokens > token_limit:
        return {
            "status": "FALLBACK_TOKEN_LIMIT",
            "final_answer": "Token limit reached; hand off to a human agent.",
            "steps": [],
            "tokens_used": 0,
        }

    def early_result(status: str, final_answer: str) -> dict[str, object]:
        answer_tokens = max(1, int(len(final_answer.split()) * 1.3))
        if estimated_tokens + answer_tokens > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "Token limit reached; hand off to a human agent.",
                "steps": [],
                "tokens_used": estimated_tokens,
            }
        return {
            "status": status,
            "final_answer": final_answer,
            "steps": [],
            "tokens_used": estimated_tokens + answer_tokens,
        }

    if not customer_id:
        return early_result(
            "AUTH_REQUIRED", "Authenticated customer identity is required."
        )
    decision_time = request_time or datetime.now(timezone.utc)
    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)

    match = ORDER_PATTERN.search(user_message)
    if not match:
        return early_result(
            "NEEDS_INFO", "Please provide an order ID in the form ORD-123456."
        )
    order_id = match.group(0)
    tools = ledger or RefundLedger()
    retriever = rag or PolicyRAG()
    steps = []
    tokens_used = estimated_tokens

    def step_token_cost(tool: str, observation: dict[str, object]) -> int:
        return max(
            1,
            int(
                len(
                    json.dumps(
                        {"tool": tool, "observation": observation}, default=str
                    ).split()
                )
                * 1.3
            ),
        )

    def answer_token_cost(final_answer: str) -> int:
        return max(1, int(len(final_answer.split()) * 1.3))

    def add_step(tool: str, observation: dict[str, object]) -> bool:
        nonlocal tokens_used
        step_tokens = step_token_cost(tool, observation)
        if tokens_used + step_tokens > token_limit:
            return False
        tokens_used += step_tokens
        steps.append({"tool": tool, "observation": observation})
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "order_id": order_id,
                        "step_number": len(steps),
                        "steps": steps,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return True

    def final_result(status: str, final_answer: str) -> dict[str, object]:
        answer_tokens = answer_token_cost(final_answer)
        if tokens_used + answer_tokens > token_limit:
            return token_limit_result()
        return {
            "status": status,
            "final_answer": final_answer,
            "steps": steps,
            "tokens_used": tokens_used + answer_tokens,
        }

    def step_limit_result() -> dict[str, object]:
        return {
            "status": "FALLBACK_MAX_STEPS",
            "final_answer": "Step limit reached; hand off to a human agent.",
            "steps": steps,
            "tokens_used": tokens_used,
        }

    def token_limit_result() -> dict[str, object]:
        return {
            "status": "FALLBACK_TOKEN_LIMIT",
            "final_answer": "Token limit reached; hand off to a human agent.",
            "steps": steps,
            "tokens_used": tokens_used,
        }

    order = get_order(order_id, customer_id, db_path)
    if not add_step("get_order", order):
        return token_limit_result()
    if order.get("error"):
        return final_result(
            "NOT_FOUND_OR_FORBIDDEN",
            "The order was not found for the authenticated customer.",
        )
    if len(steps) >= max_steps:
        return step_limit_result()

    policy = retriever.answer("late delivery refund approval policy")
    policy_observation = {
        "answer": policy["answer"],
        "sources": policy["sources"],
    }
    if not add_step("search_policy", policy_observation):
        return token_limit_result()
    if len(steps) >= max_steps:
        return step_limit_result()

    decision = calculate_refund(order, decision_time)
    if not add_step("calculate_refund", decision):
        return token_limit_result()
    if not decision["eligible"]:
        return final_result("DENIED", f"Refund denied: {decision['reason']}.")
    if len(steps) >= max_steps:
        return step_limit_result()

    if decision["approval"] == "human_required":
        status = "ESCALATED"
        final_answer = (
            f"Refund of INR {decision['refund_inr']:.2f} requires human approval."
        )
        tool_name = "escalate_to_human"
    else:
        status = "REFUNDED"
        final_answer = (
            f"Refund of INR {decision['refund_inr']:.2f} was issued to the "
            "original payment method."
        )
        tool_name = "issue_refund"
    provisional_observation = {
        "timestamp": "timestamp",
        "action": tool_name,
        "order_id": order_id,
        "refund_inr": float(decision["refund_inr"]),
        "idempotent_replay": False,
    }
    required_tokens = step_token_cost(
        tool_name, provisional_observation
    ) + answer_token_cost(final_answer)
    if tokens_used + required_tokens > token_limit:
        return token_limit_result()
    if tool_name == "escalate_to_human":
        observation = escalate_to_human(
            tools, order_id, float(decision["refund_inr"])
        )
    else:
        observation = issue_refund(
            tools, order_id, float(decision["refund_inr"])
        )
    if not add_step(tool_name, observation):
        raise AssertionError("reserved side-effect observation exceeded token budget")
    return final_result(status, final_answer)


def _scenario_orders(db_path: Path) -> dict[str, dict[str, str]]:
    queries = {
        "automatic_refund": """
            SELECT order_id, customer_id, order_ts, delivery_minutes FROM orders
            WHERE status='completed' AND delivery_minutes > 45
              AND order_amount <= 1000
            ORDER BY order_id LIMIT 1
        """,
        "human_approval": """
            SELECT order_id, customer_id, order_ts, delivery_minutes FROM orders
            WHERE status='completed' AND delivery_minutes > 45
              AND order_amount > 1000
            ORDER BY order_id LIMIT 1
        """,
        "on_time_denial": """
            SELECT order_id, customer_id, order_ts, delivery_minutes FROM orders
            WHERE status='completed' AND delivery_minutes <= 45
            ORDER BY order_id LIMIT 1
        """,
        "cancelled_denial": """
            SELECT order_id, customer_id, order_ts, delivery_minutes FROM orders
            WHERE status='cancelled'
            ORDER BY order_id LIMIT 1
        """,
    }
    with sqlite3.connect(db_path) as connection:
        orders = {
            name: connection.execute(query).fetchone()
            for name, query in queries.items()
        }
    missing = [name for name, row in orders.items() if row is None]
    if missing:
        raise AssertionError(f"Missing refund evaluation scenarios: {missing}")
    return {
        name: {
            "order_id": str(row[0]),
            "customer_id": str(row[1]),
            "order_ts": str(row[2]),
            "delivery_minutes": str(row[3]) if row[3] is not None else "",
        }
        for name, row in orders.items()
    }


def _valid_request_time(order: dict[str, str], days_after_delivery: int = 1) -> datetime:
    order_time = datetime.fromisoformat(order["order_ts"]).replace(tzinfo=timezone.utc)
    delivery_minutes = int(order["delivery_minutes"] or 0)
    return order_time + timedelta(
        minutes=delivery_minutes, days=days_after_delivery
    )


def evaluate_refund_agent(
    db_path: Path = DB_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run representative refund outcomes and persist deterministic evidence."""
    orders = _scenario_orders(db_path)
    scenarios = [
        ("automatic_refund", orders["automatic_refund"], "REFUNDED", 1),
        ("human_approval", orders["human_approval"], "ESCALATED", 1),
        ("on_time_denial", orders["on_time_denial"], "DENIED", 1),
        ("cancelled_denial", orders["cancelled_denial"], "DENIED", 1),
        ("expired_request", orders["automatic_refund"], "DENIED", 8),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_ledger = RefundLedger(output_dir / ".refund_evaluation_ledger.db")
    if evaluation_ledger.path.exists():
        evaluation_ledger.path.unlink()
        evaluation_ledger = RefundLedger(evaluation_ledger.path)
    rows = []
    for scenario, order, expected_status, days_after in scenarios:
        message = f"Refund order {order['order_id']}"
        result = run_refund_agent(
            message,
            customer_id=order["customer_id"],
            db_path=db_path,
            ledger=evaluation_ledger,
            request_time=_valid_request_time(order, days_after),
        )
        rows.append(
            {
                "scenario": scenario,
                "order_id": order["order_id"],
                "expected_status": expected_status,
                "actual_status": result["status"],
                "matches": result["status"] == expected_status,
                "tool_sequence": ">".join(
                    step["tool"] for step in result["steps"]
                ),
                "final_answer": result["final_answer"],
            }
        )
    exceptional = [
        (
            "not_found",
            "Refund order ORD-999999",
            orders["automatic_refund"]["customer_id"],
            "NOT_FOUND_OR_FORBIDDEN",
        ),
        (
            "wrong_customer",
            f"Refund order {orders['automatic_refund']['order_id']}",
            "C9999",
            "NOT_FOUND_OR_FORBIDDEN",
        ),
        (
            "missing_order_id",
            "Please refund my late order",
            orders["automatic_refund"]["customer_id"],
            "NEEDS_INFO",
        ),
        (
            "missing_identity",
            f"Refund order {orders['automatic_refund']['order_id']}",
            None,
            "AUTH_REQUIRED",
        ),
    ]
    for scenario, message, customer_id, expected_status in exceptional:
        result = run_refund_agent(
            message,
            customer_id=customer_id,
            db_path=db_path,
            ledger=evaluation_ledger,
            request_time=_valid_request_time(orders["automatic_refund"]),
        )
        match = ORDER_PATTERN.search(message)
        rows.append(
            {
                "scenario": scenario,
                "order_id": match.group(0) if match else "",
                "expected_status": expected_status,
                "actual_status": result["status"],
                "matches": result["status"] == expected_status,
                "tool_sequence": ">".join(
                    step["tool"] for step in result["steps"]
                ),
                "final_answer": result["final_answer"],
            }
        )
    evaluation = pd.DataFrame(rows)
    summary = {
        "scenarios": len(evaluation),
        "passed": int(evaluation["matches"].sum()),
        "accuracy": round(float(evaluation["matches"].mean()), 4),
        "max_steps": 6,
        "automatic_refund_limit_inr": 1000,
        "late_delivery_threshold_minutes": 45,
    }
    evaluation_path = output_dir / "refund_agent_evaluation.csv"
    summary_path = output_dir / "refund_agent_summary.json"
    evaluation.to_csv(evaluation_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    evaluation_ledger.path.unlink(missing_ok=True)
    return evaluation, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    _, summary = evaluate_refund_agent(args.db_path, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
