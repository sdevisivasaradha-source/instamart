from pathlib import Path

from src.policy_rag import FALLBACK_ANSWER, PolicyRAG, run_policy_rag


def test_retrieval_returns_scored_policy_chunks() -> None:
    rag = PolicyRAG()
    hits = rag.retrieve("refund above INR 1000 approval", top_k=3)

    assert len(hits) == 3
    assert hits[0]["source"] == "refund_policy.txt"
    assert hits[0]["score"] > 0
    assert all({"source", "chunk_index", "text", "score"} <= set(hit) for hit in hits)


def test_answers_are_grounded_and_unsupported_questions_fallback() -> None:
    rag = PolicyRAG()
    answer = rag.answer("Who approves a refund above INR 1000?")
    unsupported = rag.answer("Who won the cricket world cup final?")

    assert answer["grounded"] is True
    assert answer["sources"] == ["refund_policy.txt"]
    assert "human support lead" in answer["answer"]
    assert unsupported["answer"] == FALLBACK_ANSWER
    assert unsupported["sources"] == []


def test_evaluation_meets_full_fixture_accuracy(tmp_path: Path) -> None:
    result = run_policy_rag(output_dir=tmp_path)

    assert result["summary"]["questions"] == 14
    assert result["summary"]["source_accuracy"] == 1.0
    assert result["summary"]["fact_accuracy"] == 1.0
    assert result["summary"]["fallback_accuracy"] == 1.0
    assert all(path.is_file() for path in result["paths"].values())
