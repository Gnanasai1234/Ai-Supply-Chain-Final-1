# AI-Powered Supply Chain Disruption Assistant

A local decision-support system for comparing new supply-chain disruption events against 100,000 historical case records. A FastAPI backend powers a multi-node LangGraph analysis workflow, and a React/Vite frontend delivers interactive dashboards and visualizations.

---

## Project Overview

### Problem Statement

Global supply chains face frequent, unpredictable disruptions—ranging from factory incidents and cyber attacks to port congestion, labor strikes, and natural disasters. Operations and procurement teams urgently require empirical evidence from prior disruption precedents when triaging new events. Relying purely on intuition leads to delayed responses, suboptimal mitigations, and elevated financial losses.

### Business Objective

Reduce the time needed to identify the right disruption response—from hours to minutes—by instantly surfacing historically similar disruption precedents, providing machine-learning-based impact predictions, and generating evidence-grounded recommendations with explicit human-review safeguards.

### Proposed Solution

The system combines structured MySQL persistence, ChromaDB semantic vector retrieval, BM25 lexical search, four trained scikit-learn Random Forest models, and a 10-node sequential LangGraph workflow. A FastAPI backend exposes RESTful endpoints to a React/Vite web application. Recommendations are grounded in retrieved evidence. An optional LLM (Gemini, OpenAI via OpenRouter, or Anthropic) synthesizes explanatory narratives; a deterministic historical-analysis fallback guarantees uninterrupted offline operation.

---

## Main Application Capabilities

- **Hybrid Retrieval**: Combines ChromaDB cosine-distance similarity (weight 0.7) and BM25 keyword scores (weight 0.3), with SQL metadata filtering by industry, region, tier, backup status, and severity.
- **Relational Persistence**: MySQL stores 100,000 structured disruption records, user accounts, analysis sessions, recommendations, predictions, feedback, model versions, and evaluation results.
- **Predictive Machine Learning**: Four saved scikit-learn Random Forest models predict production impact (%), financial loss (USD), full recovery time (days), and permanent supplier change risk, with feature importance.
- **10-Node LangGraph Decision Workflow**: Classifies requests, retrieves cases, runs predictive models, analyzes response strategies and recovery durations, computes resilience and priority scores, synthesizes evidence-grounded recommendations, and validates evidence coverage.
- **What-If Disruption Simulator**: Empirically compares four strategic mitigation paths (Alternative Supplier, Expedited Shipping, Inventory Buffer, Production Rerouting) using historical recovery days, production impact, revenue loss, effectiveness, advantages, disadvantages, and trade-offs.
- **Supplier Risk Intelligence**: Aggregates multi-factor vulnerability scores across industries, tiers, and regions with interactive charts.
- **Secure Authentication and User Isolation**: Scrypt password hashing with per-user salt, 24-hour cryptographically secure opaque bearer tokens, and user-isolated session histories.
- **Feedback and Drift Monitoring**: Users log actual observed outcomes; the system tracks MAE drift against model predictions and provides retraining guidance.
- **RAGAS Quality Assessment**: Online RAG evaluation measuring faithfulness, answer relevancy, context precision, and context recall for every analysis run.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.11+, FastAPI, Pydantic, Uvicorn |
| Database | MySQL 8.0+, SQLAlchemy, PyMySQL |
| Vector Database | ChromaDB (on-disk cosine similarity) |
| Embedding Model | BAAI/bge-small-en-v1.5 (Sentence Transformers, 384 dimensions) |
| Lexical Search | rank-bm25 |
| Orchestration | LangGraph (10-node sequential StateGraph) |
| Machine Learning | scikit-learn (Random Forest), joblib, pandas, NumPy |
| LLM Integration | Google Gemini, Anthropic Claude, OpenAI via OpenRouter (optional) |
| RAG Evaluation | RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) |
| Frontend | React 18, Vite, Axios, Recharts, Lucide React, CSS3 |

---

## System Functionality

### 1. User Authentication and Session Handling

Users register with a name, email, and password (scrypt-hashed with a cryptographic salt). Login issues a 64-character opaque bearer token valid for 24 hours, stored as a SHA-256 hash in MySQL. All analysis history, feedback, and evaluation records are partitioned by `user_id`. Input descriptions are validated by regex guards that reject prompt-injection attempts.

### 2. Supply-Chain Disruption Analysis

Users submit a disruption event including a free-text description, disruption type, industry, supplier tier, region, size, backup availability, and severity rating. The 10-node LangGraph workflow then: classifies the event, investigates retrieval filters, retrieves historical cases, runs ML predictions, analyzes response strategies, computes recovery distributions, evaluates resilience, computes a priority score, generates a grounded recommendation, and validates evidence sufficiency.

### 3. Historical Incident Retrieval

The HybridRetriever combines ChromaDB cosine-distance scores and BM25 keyword scores, applies SQL metadata filters, and returns the top-K most similar historical cases enriched with structured MySQL outcome data. Each retrieved case includes disruption attributes, production impact, revenue loss, response type, recovery days, and supplier change status.

### 4. Machine Learning Predictions

Four Random Forest models are loaded from disk at startup. Given initial event features (disruption type, industry, tier, region, size, backup availability, severity), they predict: production impact percentage, revenue loss in USD, full recovery days, and probability of permanent supplier change. SHAP-based TreeExplainer feature importances are returned alongside predictions.

### 5. Recommendations and Recovery Analysis

The recommendation node synthesizes an evidence-backed mitigation strategy. If an LLM is configured, it generates a structured JSON explanation grounded in retrieved evidence. Otherwise, a deterministic rule engine selects the most frequently observed response strategy in matching historical cases. The validation node enforces that the recommended strategy appears in at least 3 historical precedents; fewer cases trigger a human-review flag.

### 6. What-If Simulation

Users can compare four strategic mitigation paths (Alternative Supplier, Expedited Shipping, Inventory Buffer, Production Rerouting) using aggregated statistics from historically similar cases, including median recovery days, production impact, revenue loss, and effectiveness ratings.

### 7. Feedback Collection and Evaluation

Users submit actual observed outcomes (response type used, recovery days, production impact, revenue loss) after an event resolves. The feedback system calculates MAE drift between predicted and actual values, and provides automated retraining recommendations when drift thresholds are exceeded.

### 8. Analysis History and Analytics

All analyses are persisted to MySQL with full request/response data, enabling paginated, filterable history views and population-level analytics covering disruption counts by type, industry, region, severity, response strategy medians, and recovery time distributions.

---

## Installation and Environment Setup

### Prerequisites

- Python 3.11 or 3.12 (Python 3.13 is also supported)
- MySQL 8.0+ server running locally
- Node.js 18+ and npm for the React/Vite frontend
- Internet access for the initial Sentence Transformer model download (approximately 130 MB, cached after first use)

### Step 1: Create a Python virtual environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The root `requirements.txt` installs matplotlib and python-pptx for documentation generation scripts. Backend Python dependencies are in `backend/requirements.txt`.

### Step 2: Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

### Step 3: Configure environment variables

Copy the example environment file and update values:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with your credentials. Do not expose actual secrets. Required variables:

```
DATABASE_URL=mysql+pymysql://root:your_password@127.0.0.1:3306/supplychain
LLM_PROVIDER=demo
```

To enable LLM synthesis (optional), set one of:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-3.5-flash-lite
```

Or for Anthropic:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key
LLM_MODEL=claude-sonnet-4-5
```

If neither is configured, `LLM_PROVIDER=demo` runs entirely offline using the deterministic historical-analysis fallback.

### Step 4: Create the MySQL database

```sql
CREATE DATABASE supplychain CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### Step 5: Initialize data and models

Run each offline preparation step once from the repository root:

```powershell
python scripts/inspect_dataset.py
python scripts/ingest_dataset.py
python scripts/build_vector_index.py
python scripts/build_bm25_index.py
python scripts/train_models.py
```

This sequence validates and ingests the CSV dataset into MySQL, generates persistent ChromaDB and BM25 search indexes, and trains the four Random Forest estimators. Model metrics are saved to `evaluation/model_metrics.json`.

### Step 6: Start the backend

```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

The FastAPI interactive Swagger UI is available at `http://127.0.0.1:8000/docs`.

### Step 7: Start the frontend

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite serves the user interface at `http://localhost:5173`.

---

## Usage Guide

1. Open `http://localhost:5173` in a browser.
2. Register a new account or log in with an existing account.
3. Navigate to **Disruption Analysis** in the sidebar.
4. Complete the analysis form: enter an event description, select the disruption type, industry, supplier tier, region, size, backup availability, and severity rating (1–5).
5. Submit the form. The system runs the full 10-node LangGraph workflow, retrieves similar historical cases, generates ML predictions, and returns a grounded recommendation.
6. Review the **Similar Cases** panel to inspect retrieved historical precedents.
7. Use the **Predictions** tab to view estimated production impact, revenue loss, recovery time, and supplier change probability.
8. Use the **Recommendations** tab to review the recommended mitigation strategy with evidence citations.
9. Navigate to **What-If Analysis** to compare four strategic response paths side-by-side.
10. Navigate to **Supplier Risk** to view aggregated risk dashboards by industry, tier, and region.
11. Submit feedback on completed events via the **Feedback** page to contribute to model drift monitoring.
12. Review system-level analytics on the **Analytics** page, including disruption counts, response type distributions, and recovery histograms.

---

## Sample Queries

The following sample queries illustrate the types of inputs and expected response categories. Outputs below are illustrative and describe the type of system response; they are not verified execution results.

| Query / Scenario | Expected System Response |
|---|---|
| Critical port congestion at a European hub delaying micro-controller delivery, severity 4, Electronics, Tier 1, Europe | High priority score (>75), hybrid retrieval of similar port disruptions, predicted production impact ~30%, recommendation to activate alternative supplier with cited historical evidence |
| Minor customs inspection delay for packaging components, severity 1, Consumer Goods, Tier 3, North America | Low priority classification (<45), expedited shipping or safety stock buffer recommendation, human review flag set to false |
| High-severity cyber attack encrypting a regional Tier 2 packaging plant, severity 5, Pharmaceuticals, Asia-Pacific | Critical classification, urgent alternative supplier recommendation, high predicted revenue loss, human review required |
| Unprecedented rare event with no historical precedent in the dataset (novel disruption type or region combination) | System retrieves fewer than 3 cases, triggers fallback, explicitly flags insufficient historical precedent, mandates human review without fabricating citations |
| Natural disaster disrupting an Aerospace Tier 1 supplier in South America, severity 4 | Medium-high priority, retrieval of similar natural disaster cases, prediction of extended recovery duration, combined strategy recommendation |

---

## Evaluation

### Offline Machine Learning Metrics

Measured on a held-out test split of 15,000 cases from the 100,000-record dataset. Metrics are recorded in `evaluation/model_metrics.json` (training timestamp: 2026-10-05T20:28:32Z).

| Prediction Task | Algorithm | MAE | RMSE | R² / F1 |
|---|---|---|---|---|
| Production impact (%) | Random Forest Regressor | 7.99 pp | 9.92 | R² 0.684 |
| Revenue loss (USD) | Random Forest Regressor | $1,774,595 | $4,419,908 | R² 0.374 |
| Full recovery time (days) | Random Forest Regressor | 31.74 days | 45.21 | R² 0.369 |
| Permanent supplier change | Random Forest Classifier | — | — | F1 0.401, Accuracy 0.640, ROC-AUC 0.635 |

### RAGAS Online Evaluation

Every disruption analysis run computes RAGAS faithfulness, answer relevancy, context precision, and context recall in real time. Results are persisted to MySQL and accessible via `GET /api/evaluation`.

A 10-test RAGAS benchmark report is available at `evaluation/reports/latest_evaluation.json`. Summary: 9/10 tests passed, 1 failed (TC-04: low-severity logistics delay — answer relevancy 0.20), overall pass rate 90.0%, average score 84.6%.

### Offline Retrieval Benchmark

`python evaluation/evaluate.py` runs an offline benchmark using `evaluation/cases.json`. Results (5 test cases, top-K=10): Context Precision@10 = 1.00, Labeled Recall@10 = 1.00, Mean Reciprocal Rank = 1.00, Recommendation Support Rate = 1.00, Evidence Consistency Rate = 1.00, Empty-Result Handling = 1.00. Overall status: PASS.

See `evaluation/reports/latest_offline_evaluation.json` for full details and `evaluation/model_metrics.json` for model training metrics.

---

## Troubleshooting

| Issue | Likely Cause | Resolution |
|---|---|---|
| `SQLAlchemyError` on startup | MySQL not running or credentials wrong | Start MySQL; verify `DATABASE_URL` in `.env` |
| `HTTP 503` on `/api/disruptions/analyze` | Retrieval indexes not built | Run `build_vector_index.py` and `build_bm25_index.py` |
| `HTTP 503 Model not trained` | ML model `.joblib` files missing | Run `python scripts/train_models.py` |
| Slow first request after startup | ChromaDB and embedding model still loading | Wait ~30 seconds; subsequent requests are fast |
| Frontend shows `Network Error` | Backend not running or CORS mismatch | Confirm backend is on port 8000; check `ALLOWED_ORIGINS` in `.env` |
| LLM errors or timeouts | API key invalid or provider rate-limited | Set `LLM_PROVIDER=demo` for offline fallback |

---

## Repository Structure

```text
AI Supply chain 1/
├── .env                          Environment variables (not in version control)
├── .gitignore                    Git ignore rules
├── requirements.txt              Root-level Python dependencies (matplotlib, python-pptx)
├── README.md                     Root project README
│
├── backend/                      FastAPI application
│   ├── app/
│   │   ├── main.py               API routes and startup lifecycle
│   │   ├── agents/workflow.py    LangGraph 10-node StateGraph
│   │   ├── rag/hybrid_retriever.py  Chroma + BM25 hybrid search
│   │   ├── ml/predictor.py       Random Forest inference and feature importance
│   │   ├── services/             Analysis, supplier risk, feedback analytics
│   │   ├── models.py             SQLAlchemy ORM models
│   │   ├── schemas.py            Pydantic request/response schemas
│   │   ├── security.py           Scrypt hashing and bearer token management
│   │   ├── config.py             Settings loaded from .env
│   │   ├── database.py           SQLAlchemy engine and session factory
│   │   └── evaluation_service.py RAGAS quality evaluation
│   ├── data/
│   │   ├── supply_chain_disruption_recovery.csv   100,000-record dataset
│   │   └── chroma/               ChromaDB vector index (generated by script)
│   ├── ml/models/                Trained .joblib model files (generated by script)
│   └── requirements.txt          Backend Python dependencies
│
├── frontend/                     React 18 / Vite web application
│   ├── src/                      JSX components, CSS, API client
│   └── package.json              Frontend npm dependencies
│
├── scripts/                      Offline preparation and documentation scripts
│   ├── inspect_dataset.py        Profiles the CSV dataset
│   ├── ingest_dataset.py         Cleans and ingests CSV into MySQL
│   ├── build_vector_index.py     Builds ChromaDB vector index
│   ├── build_bm25_index.py       Builds BM25 keyword index
│   ├── train_models.py           Trains and saves the four Random Forest models
│   ├── capstone_pdf_builder.py   Generates capstone documentation PDFs
│   └── generate_presentation.py  Generates the capstone presentation PPTX
│
├── evaluation/                   Evaluation scripts and recorded results
│   ├── evaluate.py               Offline retrieval benchmark
│   ├── run_evaluation.py         Online RAGAS/DeepEval evaluation runner
│   ├── model_metrics.json        Recorded ML training and test metrics
│   ├── dataset_profile.json      Dataset schema and distribution profile
│   ├── cases.json                Offline benchmark test case definitions
│   └── reports/                  Evaluation output reports (JSON)
│
├── tests/                        Automated API and integration test suite
│
├── docs/                         Capstone documentation
│   ├── API_Documentation.pdf     Complete API reference
│   ├── Dataset_Details.pdf       Dataset documentation
│   └── Evaluation_Report.pdf     Evaluation methodology and results
│
├── design/                       Design documentation
│   └── Design_Document.pdf       System design document
│
└── architecture/                 System architecture diagram
    └── Architecture_Diagram.pdf  System architecture diagram PDF
```

---

## Capstone Deliverables

| Deliverable | Location |
|---|---|
| Design Document | design/Design_Document.pdf |
| API Documentation | docs/API_Documentation.pdf |
| Dataset Details | docs/Dataset_Details.pdf |
| Evaluation Report | docs/Evaluation_Report.pdf |
| Architecture Diagram | architecture/Architecture_Diagram.pdf |
| Presentation | Project_Presentation.pptx |
