import pytest
from fastapi.testclient import TestClient

from backend.app import FRONTEND_DIST, app

client = TestClient(app)


def test_health_reports_mock_mode_and_policy_count() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "mode": "mock",
        "policy_documents": 5,
    }


def test_policy_endpoint_returns_grounded_answer_and_sources() -> None:
    response = client.post(
        "/api/policy/query",
        json={"question": "Who approves a refund above INR 1000?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["sources"] == ["refund_policy.txt"]
    assert "human support lead" in payload["answer"]
    assert payload["retrieved"][0]["score"] > 0


def test_policy_endpoint_falls_back_for_unsupported_question() -> None:
    response = client.post(
        "/api/policy/query",
        json={"question": "Who won the cricket world cup final?"},
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []


def test_refund_endpoint_requires_server_issued_session() -> None:
    missing = client.post("/api/refunds/query", json={"message": "Refund order ORD-100048"})
    invalid = client.post(
        "/api/refunds/query",
        headers={"Authorization": "Bearer invalid"},
        json={"message": "Refund order ORD-100048"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_refund_endpoint_allows_authorization_cors_header() -> None:
    response = client.options(
        "/api/refunds/query",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_refund_endpoint_does_not_expose_another_customers_order() -> None:
    session = client.post(
        "/api/demo/session", json={"profile": "on_time_denial"}
    ).json()
    response = client.post(
        "/api/refunds/query",
        headers={"Authorization": f"Bearer {session['token']}"},
        json={"message": "Refund order ORD-100048"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NOT_FOUND_OR_FORBIDDEN"
    assert payload["steps"][0]["observation"]["error"] == (
        "order_not_found_or_forbidden"
    )


def test_demo_sessions_can_show_refund_and_escalation() -> None:
    expected = {
        "automatic_refund": "REFUNDED",
        "human_approval": "ESCALATED",
        "on_time_denial": "DENIED",
    }
    for profile, status in expected.items():
        session = client.post(
            "/api/demo/session", json={"profile": profile}
        ).json()
        response = client.post(
            "/api/refunds/query",
            headers={"Authorization": f"Bearer {session['token']}"},
            json={"message": session["suggested_message"]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == status


@pytest.mark.skipif(
    not FRONTEND_DIST.is_dir(),
    reason="Run `npm run build --prefix frontend` to create the production UI.",
)
def test_fastapi_serves_the_built_react_application() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Instamart Support Copilot" in response.text
