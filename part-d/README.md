# Instamart Customer Copilot - Part D

## Overview

Part D combines policy retrieval-augmented generation with a bounded refund agent. A FastAPI backend exposes both workflows, and a React interface displays grounded policy evidence and authenticated refund decisions.

The instructor teaching kit is preserved unchanged under `instructor_reference/`. The implementation runs in deterministic mock mode without hosted-model credentials.

## Methodology

1. Split five approved policy documents into searchable chunks.
2. Retrieve relevant chunks with TF-IDF similarity.
3. Answer only from retrieved evidence and fall back for unsupported questions.
4. Authenticate fixed synthetic demo identities with opaque server-issued sessions.
5. Enforce order ownership, delivery state, lateness, refund-window, value, step, and token limits.
6. Use transactional SQLite idempotency and escalate high-value cases to human approval.
7. Serve the RAG and refund workflows through FastAPI and React.

## Results

- 12 indexed chunks across five policy documents.
- 100% source/fact accuracy on 12 supported evaluation questions.
- 100% fallback accuracy on two unsupported questions.
- All nine refund-agent evaluation scenarios pass.
- Cross-customer orders are hidden.
- Automatic refund, human escalation, and denial paths are demonstrated in the UI.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.policy_rag
python -m src.refund_agent
pytest -q
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The executed analysis is in `notebooks/04_support_copilot.ipynb`.

## Conclusion

Part D provides source-grounded answers and a tightly constrained refund workflow while preserving authentication, ownership, human approval, and policy boundaries.
