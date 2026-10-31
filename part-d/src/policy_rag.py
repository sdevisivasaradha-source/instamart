"""Part D: local policy retrieval with grounded answers and evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"
EVALUATION_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"
FALLBACK_ANSWER = "I could not find that information in the policy documents."
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "that",
    "to",
    "what",
    "when",
    "which",
    "who",
    "why",
}


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


class PolicyRAG:
    """TF-IDF retrieval over overlapping local policy chunks."""

    def __init__(
        self,
        policy_dir: Path = POLICY_DIR,
        chunk_size: int = 55,
        overlap: int = 12,
        fallback_threshold: float = 0.08,
    ) -> None:
        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10 words")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        paths = sorted(policy_dir.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"No policy documents found under {policy_dir}")
        self.fallback_threshold = fallback_threshold
        self.chunks = []
        step = chunk_size - overlap
        for path in paths:
            words = path.read_text(encoding="utf-8").strip().split()
            for chunk_index, start in enumerate(range(0, len(words), step)):
                chunk_words = words[start : start + chunk_size]
                self.chunks.append(
                    {
                        "source": path.name,
                        "chunk_index": chunk_index,
                        "text": " ".join(chunk_words),
                    }
                )
                if start + chunk_size >= len(words):
                    break
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(
            chunk["text"] for chunk in self.chunks
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        """Return top policy chunks with stable scores and source metadata."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        indexes = scores.argsort()[::-1][:top_k]
        return [
            {
                **self.chunks[int(index)],
                "score": round(float(scores[int(index)]), 6),
            }
            for index in indexes
        ]

    def answer(self, question: str, top_k: int = 3) -> dict[str, object]:
        """Extract the best grounded sentence or return an explicit fallback."""
        hits = self.retrieve(question, top_k)
        if not hits or float(hits[0]["score"]) < self.fallback_threshold:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "retrieved": hits,
                "grounded": False,
            }

        question_terms = _terms(question)
        candidates = []
        for hit in hits:
            sentences = re.split(r"(?<=[.!?])\s+", str(hit["text"]))
            for sentence in sentences:
                overlap = question_terms & _terms(sentence)
                candidates.append(
                    (
                        len(overlap),
                        float(hit["score"]),
                        str(hit["source"]),
                        sentence.strip(),
                    )
                )
        overlap_count, _, source, sentence = max(candidates)
        if overlap_count == 0:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "retrieved": hits,
                "grounded": False,
            }
        return {
            "answer": sentence,
            "sources": [source],
            "retrieved": hits,
            "grounded": True,
        }


def evaluate_policy_rag(
    rag: PolicyRAG,
    evaluation_path: Path = EVALUATION_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate expected source retrieval, fact grounding, and fallback behavior."""
    questions = pd.read_csv(evaluation_path, keep_default_na=False)
    rows = []
    for question in questions.itertuples(index=False):
        result = rag.answer(str(question.question))
        expects_fallback = question.expected_doc == "NONE"
        source_match = (
            expects_fallback
            and not result["sources"]
            or question.expected_doc in result["sources"]
        )
        fact_match = (
            expects_fallback
            and result["answer"] == FALLBACK_ANSWER
            or str(question.expected_fact).lower() in str(result["answer"]).lower()
        )
        rows.append(
            {
                "question": question.question,
                "expected_doc": question.expected_doc,
                "answer": result["answer"],
                "sources": ",".join(result["sources"]),
                "source_match": bool(source_match),
                "fact_match": bool(fact_match),
                "fallback": result["answer"] == FALLBACK_ANSWER,
                "top_score": result["retrieved"][0]["score"]
                if result["retrieved"]
                else 0.0,
            }
        )
    evaluation = pd.DataFrame(rows)
    supported = evaluation["expected_doc"].ne("NONE")
    unsupported = ~supported
    summary = {
        "questions": len(evaluation),
        "supported_questions": int(supported.sum()),
        "unsupported_questions": int(unsupported.sum()),
        "source_accuracy": round(
            float(evaluation.loc[supported, "source_match"].mean()), 4
        ),
        "fact_accuracy": round(
            float(evaluation.loc[supported, "fact_match"].mean()), 4
        ),
        "fallback_accuracy": round(
            float(evaluation.loc[unsupported, "fallback"].mean()), 4
        ),
        "chunks": len(rag.chunks),
        "documents": len({chunk["source"] for chunk in rag.chunks}),
    }
    return evaluation, summary


def run_policy_rag(
    policy_dir: Path = POLICY_DIR,
    evaluation_path: Path = EVALUATION_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    """Build, evaluate, and persist local policy RAG evidence."""
    rag = PolicyRAG(policy_dir)
    evaluation, summary = evaluate_policy_rag(rag, evaluation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path_out = output_dir / "policy_rag_evaluation.csv"
    summary_path = output_dir / "policy_rag_summary.json"
    chunks_path = output_dir / "policy_chunks.csv"
    evaluation.to_csv(evaluation_path_out, index=False)
    pd.DataFrame(rag.chunks).to_csv(chunks_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "rag": rag,
        "evaluation": evaluation,
        "summary": summary,
        "paths": {
            "evaluation": evaluation_path_out,
            "summary": summary_path,
            "chunks": chunks_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--evaluation-path", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run_policy_rag(
        args.policy_dir, args.evaluation_path, args.output_dir
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
