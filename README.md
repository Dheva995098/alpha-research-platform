# Alpha Research Platform - Phase 6: Full Research Dashboard

Implementation of foundational infrastructure, deterministic alpha generation, quota-aware simulation orchestration, ML ranking, filtering, and an operational dashboard for WorldQuant BRAIN research automation.

## Project Structure

```
alpha-research-platform/
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── brain_api.py          # BRAIN API wrapper + session manager
│   │   └── data_fields.py        # 125k fields schema + validators
│   ├── models.py                 # SQLAlchemy ORM (Account, Simulation, Result, etc.)
│   ├── config.py                 # Configuration from .env
│   ├── security.py               # AES-256 credential encryption
│   ├── main.py                   # FastAPI app
│   ├── routes/                   # API routes (accounts, generation, orchestration)
│   ├── generation/               # Alpha generation, dedup, refinement
│   ├── orchestration/            # Multi-account queue and quota scheduler
│   ├── ml/                       # ML ranker and feature extraction
│   ├── filters/                  # Filtering pipeline
│   ├── workers/                  # Background orchestration worker
│   └── utils/
├── frontend/                     # React operational dashboard
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── docker-compose.yml            # Local dev environment
├── Dockerfile                    # Backend container
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore
├── setup.py                      # First-time setup script
└── README.md                     # This file
```

## Core Modules

### `core/brain_api.py`
- **BRAINSession**: Low-level BRAIN API client
  - Authentication (email/password)
  - Submit alpha expressions
  - Poll simulation progress
  - Fetch backtest results
  - Rate limit handling + retry logic
  - Session persistence

- **BRAINClient**: High-level convenience wrapper
  - `submit_and_wait(expression)`: Submit + wait for results

Key features:
- Adaptive backoff for 429 (rate limit)
- Automatic retry on network failures
- Detailed logging
- Session cookies stored for recovery

### `core/data_fields.py`
- **BRAINDataFields**: Schema of 125k+ available fields + operators
  - Field validation
  - Operator validation (FASTEXPR: ts_rank, rank, group_neutralize, etc.)
  - Basic expression validation (syntax, parentheses, injection detection)
  - Field autocompletion
  - Schema export (for frontend)

### `models.py`
SQLAlchemy ORM tables:
- **Account**: User BRAIN credentials (encrypted)
- **Simulation**: Alpha expression submission record
- **Result**: Backtest results from BRAIN
- **LeaderboardAlpha**: Cached public alpha data (for ML training)
- **AlertConfig**: User alert settings (Slack/email)

### `security.py`
- AES-256 credential encryption
- `encrypt_credential()`: Encrypt passwords/tokens
- `decrypt_credential()`: Decrypt for use
- `generate_aes_key()`: Generate new key (one-time setup)

### `config.py`
- Pydantic settings from .env
- Database URL, API keys, logging levels

### `main.py`
- FastAPI application
- CORS middleware
- Health check endpoint
- Lifespan context (startup/shutdown)
- Route registration

### `generation/`
- **RuleBasedAlphaGenerator**: Generates candidate FASTEXPR alphas by strategy
  - Momentum
  - Mean reversion
  - Price-volume
  - Liquidity
  - Volatility
  - Size
  - Intraday
  - Analyst, sentiment, options, model/risk
- **GeneticAlphaRefiner**: Mutates and crosses over seed expressions
- **ExpressionDeduplicator**: Normalizes expressions and removes exact structural duplicates
- **AlphaCandidate**: Shared candidate metadata object

### `routes/generation.py`
- `GET /api/generation/strategies`: List supported generation strategies
- `GET /api/generation/datasets`: List dataset-family profiles
- `GET /api/generation/fields`: Browse known fields by prefix, dataset, or category
- `GET /api/generation/field-intelligence`: List persisted fields ranked by coverage, usage, type, and dataset fit
- `POST /api/generation/generate`: Generate alpha candidates
- `POST /api/generation/refine`: Produce mutations/crossovers from seeds
- `POST /api/generation/deduplicate`: Normalize and deduplicate expressions

### `orchestration/`
- **SimulationOrchestrator**: Durable queue manager using the `simulations` table
- **BrainGateway**: Injectable live BRAIN submission/polling adapter
- **Quota helpers**: Daily quota reset and remaining-capacity checks
- **Result persistence**: Completed BRAIN payloads are stored in `results`

### `routes/orchestration.py`
- `POST /api/orchestration/queue`: Queue expressions as pending simulations
- `GET /api/orchestration/queue`: List queued simulations by status
- `GET /api/orchestration/summary`: Show queue counts and account quotas
- `POST /api/orchestration/submit-next`: Submit one pending simulation
- `POST /api/orchestration/poll`: Poll running simulations and persist results
- `POST /api/orchestration/worker/start`: Start the in-process worker
- `POST /api/orchestration/worker/stop`: Stop the in-process worker
- `GET /api/orchestration/worker/status`: Inspect worker state

### `ml/`
- **ExpressionFeatureExtractor**: Converts FASTEXPR strings and result metrics into stable numeric features
- **AlphaRanker**: Heuristic-first logistic ranker for pass probability scoring
- **MLRankingService**: Trains from `leaderboard_alphas` and `results`, scores stored results, and persists model weights

### `routes/ml.py`
- `GET /api/ml/status`: Show model metadata and feature names
- `POST /api/ml/score`: Score candidate expressions
- `POST /api/ml/train`: Train from stored labeled examples
- `POST /api/ml/score-results`: Apply ML scores to stored results
- `GET /api/ml/results`: List results ranked by final score
- `GET /api/ml/good-alphas`: Return saved live results that clear the local good-alpha bar, including copy-ready alpha code and settings

### `automation/`
- **SpecialAutopilot**: Random dataset/focus/settings batch generator that scores candidates with the learner, queues the best five, and keeps the worker near the configured running cap.

### `filters/`
- **AlphaFilterPipeline**: Applies expression validation, duplicate checks, complexity limits, ML gates, and result-quality thresholds
- **ExpressionFilterConfig**: Pre-submission filter thresholds
- **ResultFilterConfig**: Post-backtest filter thresholds

### `routes/filters.py`
- `GET /api/filters/rules`: Show default thresholds
- `POST /api/filters/expressions`: Filter generated expressions
- `POST /api/filters/results`: Filter stored result rows
- `GET /api/filters/results/accepted`: List accepted stored results

### `frontend/`
- Vite + React dashboard
- Generation, random generation, good-alpha Vault, filtering, queue, ML scoring, and worker controls
- Talks to the backend through `VITE_API_BASE` or `http://127.0.0.1:8000`

## Quick Start

### 1. Prerequisites
- Python 3.11+
- pip
- Optional: Docker + Docker Compose

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. First-Time Setup
```bash
python setup.py
```
This will:
- Create `.env` from `.env.example`
- Generate AES-256 encryption key
- Initialize SQLite database
- Print next steps

### 4. Configure .env
Edit `.env` with your settings:
```bash
# Optional: BRAIN credentials (can add via dashboard later)
BRAIN_EMAIL=your@email.com
BRAIN_PASSWORD=your-password

# LLM API keys
CLAUDE_API_KEY=sk-...
OPENAI_API_KEY=sk-...

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SENDGRID_API_KEY=SG.xxx
```

`OPENAI_API_KEY` is optional. When present, `scripts/codex_agent.py --model openai` can ask an OpenAI model for planning assistance, the dashboard `OpenAI assist` option can critique/rerank generated candidates, and `Special Auto` can use the same advisor before queueing. BRAIN polling, result persistence, and the local logistic learner still do not require it. The app sends only candidate expressions/settings metadata to OpenAI for this assist layer; BRAIN passwords and session cookies are not sent.

### 5. Run Backend (Development)
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Or with Docker:
```bash
docker-compose up
```

### 6. Verify Setup
```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs

# Dashboard
open http://localhost:5173
```

### 7. Run Tests
```bash
pytest tests/test_phase1.py -v
pytest tests/test_phase2_generation.py -v
pytest tests/test_phase3_orchestration.py -v
pytest tests/test_phase4_ml.py -v
pytest tests/test_phase5_filters.py -v
```

### 8. Generate Alpha Candidates
```bash
curl -X POST http://localhost:8000/api/generation/generate \
  -H "Content-Type: application/json" \
  -d "{\"count\": 10, \"focus\": \"momentum\", \"seed\": 42}"
```

Dataset-aware generation can target a specific BRAIN data family:
```bash
curl "http://localhost:8000/api/generation/datasets"
curl "http://localhost:8000/api/generation/fields?dataset_id=option8&prefix=pcr"
curl "http://localhost:8000/api/generation/field-intelligence?dataset_id=option8&limit=10"

curl -X POST http://localhost:8000/api/generation/generate \
  -H "Content-Type: application/json" \
  -d "{\"count\": 10, \"dataset_ids\": [\"option8\"], \"seed\": 42}"
```

Local dataset profiles include `pv1`, fundamentals, analyst, options, news/social sentiment, and model/risk families. The live BRAIN API is still the exact source of truth for field availability.
When a dataset is selected in the dashboard, its recommended universe/neutralization/truncation hints are merged into the queued simulation settings. Synced live fields are persisted in `data_fields`, scored, and used as generation allow-lists for dataset-specific runs.

### 9. Queue And Dry-Run Submit
```bash
curl -X POST http://localhost:8000/api/orchestration/queue \
  -H "Content-Type: application/json" \
  -d "{\"expressions\": [\"rank(close)\"], \"validate\": true}"

curl -X POST http://localhost:8000/api/orchestration/submit-next \
  -H "Content-Type: application/json" \
  -d "{\"dry_run\": true, \"universe\": \"default\"}"
```

### Add BRAIN Account
Use the dashboard `Accounts` tab, or create one from the terminal:
```bash
python scripts/create_account.py --email your@email.com --daily-quota 450
```

### 10. Score And Filter Expressions
```bash
curl -X POST http://localhost:8000/api/ml/score \
  -H "Content-Type: application/json" \
  -d "{\"expressions\": [\"rank(close)\", \"group_neutralize(rank(ts_corr(close, volume, 20)), sector)\"]}"

curl -X POST http://localhost:8000/api/filters/expressions \
  -H "Content-Type: application/json" \
  -d "{\"expressions\": [\"rank(close)\", \"group_neutralize(rank(ts_corr(close, volume, 20)), sector)\"]}"
```

### 11. Run Dashboard
```bash
cd frontend
npm install
npm run dev
npm run test:e2e
```

## Dashboard Workflow

### A. Account Setup
1. Open `http://127.0.0.1:5173`
2. Go to `Accounts`
3. Add your WorldQuant BRAIN email, password, and local daily quota
4. Click the shield button on the account row to test live BRAIN login
5. Click the database button to sync live data fields from BRAIN into the local schema

Adding an account only stores encrypted credentials. Live BRAIN login happens when you test the account, click `Live Submit`, or start the worker with dry-run disabled.

### B. Local Dry-Run
1. Go to `Generate`
2. Pick a `Focus` and `Settings` preset, or choose `random` / click `Random` to let the app pick dataset, focus, seed, and settings hints automatically. `Seed` is the repeatable random starting number; same seed plus same options gives the same candidates, while a different seed explores different variants.
3. Click `Generate`
4. Optionally click `Filter`
5. Click `Queue`
6. Go to `Queue`
7. Click `Dry Run`
8. Open `Results` to inspect local mock result metrics

Dry-run does not submit to WorldQuant BRAIN and does not consume the local quota counter.
Dry-run metrics are deterministic local estimates for workflow testing only. Treat a `Dry Candidate` as "worth checking live", not as a real BRAIN pass.

### C. Real BRAIN Simulation
1. Go to `Accounts` and click the shield test button
2. Confirm the message says BRAIN API connection is successful
3. Go to `Generate`, create candidates, filter, and queue them
4. Choose a `Settings` preset before queueing. `balanced` now uses `USA`, `TOP3000`, `delay 1`, `decay 10`, `SUBINDUSTRY`, `truncation 0.01`, and `P5Y`.
5. Go to `Queue`
6. Click `Live Submit` to send one pending alpha to BRAIN
7. Click `Poll` periodically until it moves to completed
8. Go to `Results` and click `Score Results` / `Filter Results`

### D. Promote A Dry-Run Candidate
1. Go to `Results`
2. Find a row marked `Dry Candidate`
3. Review the settings chips, such as `delay`, `decay`, `neutral`, `trunc`, and `universe`
4. Click the rocket button in the row
5. Confirm the prompt to submit it as a fresh live WorldQuant BRAIN simulation
6. Go to `Queue` and click `Poll` until the live row completes

The row-level live submit uses the same expression and saved settings from that result. If an older result is missing saved settings, the app falls back to the research-balanced settings: `USA`, `TOP3000`, `delay 1`, `decay 10`, `SUBINDUSTRY`, `truncation 0.01`, `P5Y`, `FASTEXPR`.

### Import Submitted Alpha PDFs
Text-based PDF exports can be imported into the training set:
```bash
python scripts/import_submitted_alphas.py --path "C:\path\to\alphas_submited_by_me.pdf" --train
```

If the PDF is an image-only export, the script will report `text_chars=0`; run OCR first, then retry the import.

WorldQuant BRAIN Learn PDFs can be extracted into local research notes and imported as positive examples:
```bash
python scripts/learn_brain_docs.py --glob "C:\Users\shree\Downloads\wd_*.pdf" --import-examples --clean-imported
```

### Seed Public Research Examples
Public, metrics-backed alpha examples can be imported as training signals:
```bash
python scripts/seed_public_alpha_research.py
```

These rows are tagged with `source=training_seed` and `copy_policy=training_signal_only_do_not_copy`. They improve the ranker from reported formulas, settings, and metrics, but are not queued for live submission.

### E. Worker Mode
1. Go to `Worker`
2. Keep `Dry-run worker` checked for local testing
3. Uncheck it only when you want the worker to submit live BRAIN simulations
4. Click `Start`; click `Stop` before changing mode

### F. Special Auto Mode
1. Go to `Worker`
2. Click `Special Auto`
3. The worker starts live mode with auto-learn enabled, generates five random learner-scored candidates, queues them, submits pending rows, polls running rows, and refills whenever running simulations drop below the target.

The defaults target five running simulations and cap at six. `Special Auto` also enables `OpenAI assist`, so candidates are locally scored by the learner and then optionally reranked by the OpenAI advisor before queueing. Completed live results are always persisted in `results`; rows that pass the good-alpha thresholds appear in `Vault` with copy-ready alpha code, simulation settings, and metrics.

Model learning is incremental. With fewer than roughly 50 live completed simulations, the ranker is mostly heuristic plus seed data. Around 100-300 live results it can start learning your account-specific pass/fail patterns. A stable 30-40% local good-alpha hit rate usually needs hundreds of real completed simulations across several dataset families; no system can guarantee that rate, but auto-learn improves when the live result set is diverse and honestly labeled.

## Phase 1 Implementation Details

### BRAIN API Session Management

**Key Features:**
1. **Session Persistence**: Cookies stored across requests
2. **Rate Limiting**: Handles 429 responses with adaptive backoff
3. **Polling**: Non-blocking simulation status checks
4. **Error Recovery**: Exponential backoff for network failures

**Example Usage:**
```python
from backend.core.brain_api import BRAINClient

# Authenticate
client = BRAINClient(email="user@example.com", password="secret")

# Submit and wait for results
results = client.submit_and_wait("rank(ts_corr(close, volume, 20))")
print(results)  # {"sharpe": 1.5, "fitness": 1.2, ...}

client.close()
```

### Data Fields Validation

**Example Usage:**
```python
from backend.core.data_fields import BRAINDataFields

schema = BRAINDataFields()

# Validate field
schema.validate_field("close")  # True
schema.validate_field("fake_field")  # False

# Validate operator
schema.validate_operator("ts_rank")  # True
schema.validate_operator("fake_op")  # False

# Validate expression
valid, msg = schema.validate_expression_basic("rank(close)")
# (True, "Valid")

valid, msg = schema.validate_expression_basic("rank(close); DROP TABLE")
# (False, "Suspicious pattern detected: ;")
```

### Credential Encryption

**Example Usage:**
```python
from backend.security import encrypt_credential, decrypt_credential

password = "my-secret-password"
encrypted = encrypt_credential(password)
# Store encrypted in database

# Retrieve and decrypt
decrypted = decrypt_credential(encrypted)
assert decrypted == password
```

## Database Schema

### Account Table
```sql
CREATE TABLE accounts (
  id INTEGER PRIMARY KEY,
  user_id VARCHAR,
  brain_email VARCHAR UNIQUE,
  brain_password_encrypted VARCHAR,
  daily_quota INTEGER DEFAULT 450,
  submissions_today INTEGER DEFAULT 0,
  last_quota_reset DATETIME,
  is_active BOOLEAN DEFAULT TRUE,
  created_at DATETIME,
  updated_at DATETIME
);
```

### Simulation Table
```sql
CREATE TABLE simulations (
  id INTEGER PRIMARY KEY,
  account_id INTEGER FOREIGN KEY,
  brain_simulation_id VARCHAR UNIQUE,
  expression TEXT,
  status VARCHAR,  -- pending, running, completed, failed
  progress FLOAT,  -- 0-100
  error_message VARCHAR,
  submitted_at DATETIME,
  completed_at DATETIME
);
```

### Result Table
```sql
CREATE TABLE results (
  id INTEGER PRIMARY KEY,
  account_id INTEGER FOREIGN KEY,
  simulation_id INTEGER FOREIGN KEY,
  brain_alpha_id VARCHAR UNIQUE,
  expression TEXT,
  sharpe FLOAT,
  fitness FLOAT,
  turnover FLOAT,
  self_correlation FLOAT,
  all_checks_passed BOOLEAN,
  raw_metrics JSON,
  ml_pass_probability FLOAT,
  final_score FLOAT,
  human_approved BOOLEAN,
  submitted_to_brain BOOLEAN,
  created_at DATETIME
);
```

## Verification Checklist

✅ **Directory structure** created  
✅ **Dependencies** defined (requirements.txt)  
✅ **Configuration** system (config.py + .env.example)  
✅ **Security** module (AES-256 encryption)  
✅ **Models** defined (SQLAlchemy ORM)  
✅ **BRAIN API wrapper** (session manager + polling)  
✅ **Data fields schema** (125k+ fields + operators + validation)  
✅ **FastAPI app** initialized  
✅ **Docker** setup (docker-compose.yml + Dockerfile)  
✅ **Tests** written (test_phase1.py)  
✅ **Setup script** for first-time initialization  
[x] **Alpha generation engine** implemented  
[x] **Genetic-style refinements** implemented  
[x] **Generation API routes** registered  
[x] **Phase 2 tests** written (test_phase2_generation.py)  
[x] **Quota-aware simulation queue** implemented  
[x] **Orchestration API routes** registered  
[x] **Background worker controller** implemented  
[x] **Phase 3 tests** written (test_phase3_orchestration.py)  
[x] **ML feature extraction and ranker** implemented  
[x] **ML API routes** registered  
[x] **Filtering pipeline** implemented  
[x] **Filtering API routes** registered  
[x] **React dashboard** implemented  
[x] **Full backend test suite** passing  

## Phase 2 Implementation Details

Phase 2 now includes a dependency-light baseline generation system that works without LLM credentials:

- [x] Rule-based alpha expression generator
- [x] Genetic-style mutation and crossover refinement
- [x] Expression normalization and deduplication
- [x] Generation API routes
- [x] Unit tests for generator, refiner, deduplication, and route handlers

LLM-assisted expression generation remains an optional future enhancement. The current path is deterministic and suitable for offline testing.

## Phase 3 Implementation Details

Phase 3 adds a quota-aware simulation queue:

- [x] Multi-account pending simulation queue
- [x] Quota-aware account selection and daily quota reset
- [x] Dry-run submission path for local testing
- [x] Live submission adapter for BRAIN credentials
- [x] Polling path that stores completed backtest results
- [x] In-process worker lifecycle controls

The worker defaults to `dry_run: true` when started through the API, so local queue testing does not submit to BRAIN unless explicitly configured.

## Phase 4 Implementation Details

- [x] ML feature extraction from expressions and results
- [x] Baseline pass-probability ranker
- [x] Leaderboard/result training data ingestion
- [x] Persisted lightweight model weights
- [x] Integrated ML score into stored result ranking

## Phase 5 Implementation Details

- [x] Expression validation filters
- [x] Duplicate-expression filters
- [x] Complexity threshold filters
- [x] ML probability filters
- [x] Result metric filters for Sharpe, fitness, turnover, self-correlation, and BRAIN checks

## Phase 6 Implementation Details

- [x] Vite + React dashboard
- [x] Generation controls and candidate selection
- [x] Queue overview and dry-run submission controls
- [x] Result settings display and row-level live simulation
- [x] Research settings presets and stored per-simulation settings
- [x] Submitted-alpha PDF importer for text/OCR exports
- [x] BRAIN Learn PDF extractor and local research-principles cache
- [x] Expanded generator with price-volume, fundamental, quality, hybrid, and outlier-controlled templates
- [x] Advanced generator motifs for analyst, sentiment/news, options, and decorrelation variants
- [x] ML scoring and expression filtering tools
- [x] Worker start/stop/status controls

## What's Next

- [ ] Add authentication for multi-user deployments
- [ ] Add database migrations
- [ ] Add full BRAIN integration test mode
- [ ] Add model drift monitoring alerts
- [x] Add frontend end-to-end tests

## Known Limitations & Future Work

1. **BRAIN API**: Community reverse-engineered; no official docs
   - May break if BRAIN changes endpoints
   - Monitor for updates to `q3yi/worldquant` SDK

2. **Data Fields**: Core set of common fields + placeholders
   - Full 125k fields loaded from BRAIN API on first startup
   - Cached in database for fast validation

3. **Credential Storage**: 
   - Uses Fernet (symmetric encryption)
   - AES key must be stored securely in .env
   - Future: Consider HSM or cloud KMS

4. **Session Persistence**:
   - Cookies stored as pickle files
   - Not currently implemented; rely on requests library to handle
   - Future: Explicit session serialization for multi-process deployments

## Troubleshooting

### "Failed to authenticate with BRAIN API"
- Check BRAIN_EMAIL and BRAIN_PASSWORD in .env
- Verify account credentials on WorldQuant BRAIN website
- Check internet connection + firewall

### "Invalid AES_KEY"
- Delete .env and re-run setup.py
- Or manually set AES_KEY to output of: `python -c "from backend.security import generate_aes_key; print(generate_aes_key())"`

### Tests fail with "Requires BRAIN credentials"
- Tests requiring live BRAIN account are skipped by default
- To run integration tests: `pytest tests/test_phase1.py -k "not skip" -v`
- Requires BRAIN_EMAIL + BRAIN_PASSWORD in .env

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| FastAPI | Modern, fast, async-ready |
| SQLAlchemy | ORM flexibility, multi-database support |
| Pydantic + pydantic-settings | Type validation, env management |
| Fernet (symmetric crypto) | Simple, fast, secure for credentials |
| pytest | Standard Python testing framework |
| Docker Compose | Local dev environment consistency |

## File Locations Summary

| Component | File |
|-----------|------|
| BRAIN API | `backend/core/brain_api.py` |
| Data Fields | `backend/core/data_fields.py` |
| Models | `backend/models.py` |
| Security | `backend/security.py` |
| Config | `backend/config.py` |
| App | `backend/main.py` |
| Tests | `tests/test_phase1.py` |
| Phase 2 Tests | `tests/test_phase2_generation.py` |
| Generation | `backend/generation/expression_generator.py` |
| Refinement | `backend/generation/genetic.py` |
| Deduplication | `backend/generation/dedup.py` |
| Generation Routes | `backend/routes/generation.py` |
| Orchestration | `backend/orchestration/service.py` |
| Quotas | `backend/orchestration/quota.py` |
| Orchestration Routes | `backend/routes/orchestration.py` |
| Worker | `backend/workers/orchestration_worker.py` |
| Phase 3 Tests | `tests/test_phase3_orchestration.py` |
| ML Features | `backend/ml/features.py` |
| ML Ranker | `backend/ml/ranker.py` |
| ML Routes | `backend/routes/ml.py` |
| Phase 4 Tests | `tests/test_phase4_ml.py` |
| Filters | `backend/filters/pipeline.py` |
| Filter Routes | `backend/routes/filters.py` |
| Phase 5 Tests | `tests/test_phase5_filters.py` |
| Frontend App | `frontend/src/App.jsx` |
| Frontend Styles | `frontend/src/styles.css` |
| Setup | `setup.py` |
| Docker | `docker-compose.yml`, `Dockerfile` |

---

**Phase 1 Status**: Complete  
**Phase 2 Status**: Baseline complete  
**Phase 3 Status**: Baseline complete  
**Phase 4 Status**: Baseline complete  
**Phase 5 Status**: Baseline complete  
**Phase 6 Status**: Dashboard complete
