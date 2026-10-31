# Instamart Customer Copilot

This repository contains a complete offline implementation of capstone Parts A-D: reproducible synthetic data, EDA and cleaning, SQLite validation, customer segmentation and lapse prediction, privacy-safe ticket triage, policy RAG, a bounded refund agent, and an integrated React/FastAPI co-pilot interface.

## Instructor reference

The supplied teaching kit is preserved unchanged under `instructor_reference/`. Its `generate_data.py` is also copied byte-for-byte to the repository root so the required command can run from this directory. Reference code is retained for instruction and comparison; it is not presented as original project implementation.

## Setup

Python 3.11 is recommended for later stages. SQLite is included in the Python standard library.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` for the planned Pandas/NumPy/scikit-learn workflow, LangChain/FastAPI backend, and pytest-based tests. The React frontend is intentionally not initialized; that belongs to Stage 7.

## Model configuration

Copy `.env.example` to `.env` when configuration is needed. `LLM_MODE=mock` requires no API credentials. A future real-model integration must set `LLM_MODE=real` and replace `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` with the chosen provider, model identifier, and secret API key. Never commit `.env` or credentials.

## Generate data

From the repository root:

```bash
python generate_data.py
```

The unchanged deterministic generator writes CSV exports to `data/raw/` and policy documents to `data/policies/`. See [`docs/data_inventory.md`](docs/data_inventory.md) for exact generated results, schemas, and relationships.

## Run Stage 3 EDA

Generate the tracked EDA evidence:

```bash
python -m src.eda
```

Open the executed assignment notebook:

```bash
jupyter lab notebooks/01_raw_data_eda.ipynb
```

The notebook and `outputs/eda_report.md` show dataset shapes, inferred data types, blank counts and percentages, column value counts with percentages, duplicate patterns, raw city/gender variations, and domain-based outlier flags. Stage 3 is diagnostic only: it does not modify `data/raw/`. Standardization, imputation, SQLite loading, and Pandas/SQL validation belong to the next stage.

### Stage 3 results

- Customers: 1,200 rows × 5 columns; 24 raw city labels representing six cities, 10 nonblank gender labels, 61 blank ages, 30 blank genders, and six domain-invalid ages.
- Orders: 13,084 rows × 9 columns; 12,309 distinct order IDs, 119 exact duplicate rows, 775 excess repeated-ID rows, 120 nonnumeric amounts, 106 negative delivery durations, and 2,541 blank delivery durations.
- Tickets: 164 rows × 5 columns; eight blank order IDs, four blank texts, and four texts longer than 1,000 characters.

## Run cleaning and SQLite validation

Complete the remaining data-foundation tasks:

```bash
python -m src.data_pipeline
jupyter lab notebooks/02_data_cleaning_sql_validation.ipynb
```

The pipeline writes four clean CSVs to `data/processed/`, creates the local `data/instamart.db`, and writes cleaning and validation evidence under `outputs/`. The database is reproducible local state and is intentionally ignored by Git.

Cleaning reduces 24 city spellings to six standard cities and 11 gender values (including blank) to `Male`, `Female`, or `Unknown`. It imputes 67 invalid/blank ages, resolves 13,084 raw order rows to 12,309 unique final orders, repairs 120 nonnumeric amounts, and removes or imputes 106 negative delivery durations. All nine cross-check metrics match between Pandas and SQL.

## Run Parts A-D

Run the complete offline project:

```bash
python run_project.py
```

Assignment notebooks:

1. `notebooks/01_raw_data_eda.ipynb` — raw-data exploration.
2. `notebooks/02_data_cleaning_sql_validation.ipynb` — cleaning and SQLite proof.
3. `notebooks/03_customer_intelligence.ipynb` — RFM segments and lapse model.
4. `notebooks/04_support_copilot.ipynb` — privacy-safe triage, policy RAG, and refund agent.

Part B produces four RFM segments, K=2–8 diagnostics, PCA evidence, two tuned lapse models, a persisted selected model, and retention recommendations. Part C explicitly compares zero-shot and few-shot prompts, redacts PII before deterministic mock-model calls, applies input/token/retry/output guardrails, and routes fraud, Premium, unresolved, and standard tickets. Part D indexes all five policy documents, evaluates grounded answers and unsupported fallbacks, and tests nine bounded refund scenarios including ownership, authentication, and the seven-day request window.

Mock mode is the implemented and tested default: it uses no API key or network call. A future real-provider adapter must preserve the same redaction, schema validation, budget, source-grounding, and human-approval boundaries.

## Run the React co-pilot

Start the API:

```bash
source .venv/bin/activate
# The API creates data/instamart.db from tracked CSVs if it is missing.
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the React app:

```bash
cd frontend
# Node.js 20+ is required.
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Policy mode displays grounded answers, source files, retrieved chunks, and scores. Refund mode uses an opaque server-issued token for one of three fixed synthetic demo identities, hides orders owned by other identities, and displays the bounded agent's tool steps. Demo sessions use server-controlled historical decision times so refund, escalation, and denial can all be demonstrated without allowing clients to change policy dates. Replace this teaching-only session endpoint with institutional SSO/OAuth in production. The API documentation is available at `http://127.0.0.1:8000/docs`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Generated source CSV exports |
| `data/policies/` | Generated policy documents |
| `data/processed/` | Clean customers, products, orders, and ticket CSVs |
| `src/` | Reusable EDA and data-cleaning modules |
| `sql/` | SQLite validation queries |
| `tests/` | Automated EDA, cleaning, and database tests |
| `outputs/` | Tracked EDA, cleaning, and SQL-validation evidence |
| `notebooks/` | Executed assignment notebooks, beginning with Stage 3 EDA |
| `backend/` | FastAPI endpoints for health, policy RAG, and refund queries |
| `frontend/` | Responsive React/Vite co-pilot interface |
| `docs/` | Project documentation and generated-data inventory |
| `instructor_reference/` | Exact unchanged snapshot of the supplied teaching kit |
| `models/` | Persisted lapse-model artifact |

