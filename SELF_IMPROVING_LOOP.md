# Phase 7 — Self-Improving Research Loop

This platform already generated, simulated, ranked, and filtered alphas. Phase 7
**closes the loop** so the system *learns across runs*: every attempt is now
conditioned on the outcomes of previous attempts, near-misses are repaired for
free, and confirmed wins compound into future generation. No model is retrained —
this is in-context / agentic learning with **persistent memory** layered on top of
the existing ML ranker.

> Hindi: अब system हर batch में पिछले results से सीखता है — जो fail हुआ उसे दोहराता
> नहीं, जो "लगभग pass" था उसे सस्ते में repair करके दोबारा भेजता है, और जो win हुआ
> उसे library में डालकर आगे की generation को बेहतर बनाता है।

## What was already strong (kept as-is)

| Spec pattern | Where it lives |
|---|---|
| A — Objective gates + score | `filters/pipeline.py`, `ml/ranker.py`, `routes/ml.py::_is_good_alpha` |
| F — Whitelist + validation | `core/data_fields.py` |
| I — Dedup + quota + budget | `generation/dedup.py`, `models.AlphaRegistry`, `orchestration/quota.py` |

## What Phase 7 added (the gaps that blocked learning)

New package `backend/selfimprove/` + two tables + small hooks into the autopilot.

| Module | Pattern | Role |
|---|---|---|
| `selfimprove/evaluator.py` | A | One `evaluate(metrics) -> Verdict{is_ok, failures, score, outcome}`. Diagnoses *why* an attempt fell short (`LOW_SHARPE`, `HIGH_TURNOVER`, `HIGH_SELF_CORRELATION`, `LOW_SUB_UNIVERSE_SHARPE`, `COVERAGE_FAIL`, …), gives a continuous score (defined even on a pass), and classifies each attempt **win / near / fail**. Gates live in `GateConfig` (config, not code). |
| `selfimprove/memory.py` | B, H | `AttemptMemoryService`: append/upsert every attempt + outcome (the tried[]/failures[] log), surface recent failures, near-misses, and tried signatures, and auto-promote wins into an auto-growing **library** of good alphas. |
| `selfimprove/refiner.py` | E | `DeterministicRefiner.repair(expression, verdict)`: failure→fix table that turns a near-miss into validated repaired variants **without an LLM/regen call** (turnover→`ts_decay_linear`, coverage→`ts_backfill`, self-corr→decorrelate, sub-universe→neutralization-group swap, low-signal→`winsorize`). |
| `selfimprove/feedback.py` | C, D | Turns memory into generation context: a negative-example block for the OpenAI advisor, and a rank-time penalty that steers away from operator shapes that keep failing. |

New tables (auto-created on `init_db()` for existing databases too):
- `attempt_memory` — the tried[]/failures[] log.
- `alpha_library` — the auto-growing good[] pool.

## How the loop closes (in the autopilot)

`backend/automation/special_runner.py::SpecialAutopilot.tick()`:

1. **Poll** results from BRAIN (unchanged).
2. **Absorb** — `_absorb_results()` evaluates each new live result, appends it to
   `attempt_memory`, and promotes wins into `alpha_library`. *(append outcome)*
3. **Repair first** — `_queue_repairs()` reads recent near-misses and queues cheap
   deterministic repairs **before** any fresh generation. *(Pattern E)*
4. **Generate** — `queue_random_batch()` now:
   - seeds the genetic refiner with **library winners** for the focus (Pattern H),
   - penalises candidates built from **recently-failing shapes** (Pattern D),
   - feeds **recent failures** to the OpenAI advisor as negative examples (Pattern C).
5. **Submit** to the running cap (unchanged).

Every self-improve hook is guarded and **inert on empty memory**, so behaviour is
identical to before until real outcomes accumulate — then it compounds.

## Why this makes submissions faster + more accurate

- **Free near-miss recovery:** a candidate with real signal but slightly high
  turnover/self-corr/coverage is repaired deterministically and re-queued — no new
  generation, often converting a near-miss into a pass.
- **No wasted quota on dead-ends:** failing shapes are penalised and tried ground is
  remembered, so simulation budget goes to genuinely new candidates.
- **Compounding quality:** confirmed wins seed future generation, raising the
  baseline quality of every later batch.

## Observe it (read-only API)

```bash
curl http://localhost:8000/api/selfimprove/stats        # attempts, wins, near, win_rate, library_size
curl http://localhost:8000/api/selfimprove/memory       # recent attempts (?outcome=win|near|fail)
curl http://localhost:8000/api/selfimprove/near-misses   # what the refiner will repair next
curl http://localhost:8000/api/selfimprove/library       # the growing pool of confirmed-good alphas (?focus=...)
```

## Success criteria (is it actually learning?)

1. Across-run improvement: average/best `score` of later runs > early runs.
2. No repeats: the proposer stops re-emitting already-failed directions.
3. Failure recovery: a measurable fraction of near-misses become passes via the
   refiner (no extra generative call).
4. Growing library: `library_size` rises and lifts new-candidate quality.

## Amplification — BRAIN-grounded upgrades (Phase 8)

A multi-agent research pass (live WorldQuant BRAIN docs + automation repos + the
"101 Formulaic Alphas" paper) plus a codebase audit drove a second round of changes
that make the loop *accurate*, not just self-improving:

- **Correct BRAIN check semantics** (`evaluator.py`): failure tags now match the real
  `is.checks` names. `low_turnover` is a **distinct** defect from `high_turnover`
  (previously conflated — the refiner was lowering turnover on a too-low-turnover
  alpha, the exact wrong fix); `concentrated_weight` and `units` get their own tags;
  `self`/`prod` correlation are separated; `matches_competition/pyramid/themes` are
  treated as eligibility flags, **not** quality failures; `PENDING/WAITING` ≠ fail.
- **Real operator whitelist** (`data_fields.py`): added `vector_neut`,
  `group_vector_neut`, `signed_power`, `hump`, `quantile`, `ts_quantile`, the full
  `vec_*` reducer family, and more — so the generator/refiner can finally emit the
  decorrelation and turnover primitives BRAIN actually exposes.
- **Grounded refiner fixes** (`refiner.py`): `HIGH_TURNOVER → hump / trade_when` regime
  gate (on top of decay); `LOW_TURNOVER →` sharpen/shorten (raise turnover);
  correlation `→ vector_neut(expr, cap)` residualization (the canonical decorrelator);
  `CONCENTRATED_WEIGHT →` winsorize/rank + tighter truncation; `UNITS →` rank-wrap.
- **Thompson-sampling explorer** (`bandit.py` + autopilot): focus/dataset selection is
  now a Beta-Bernoulli bandit over real PASS/FAIL win-rates from memory instead of
  hard-coded weights — provably better passes-per-quota as data accumulates.
- **Proven motif library** (`motifs.py`): 101-Alphas/BRAIN structures (negated
  price-volume reversal, ranked-correlation, vol-of-vol, quality margin, analyst
  revisions, options skew, `vector_neut` decorrelation) are validated and injected as
  seed candidates, grounding generation in structures that actually work.
- **ML signal** (`ml/service.py`): training now pulls confirmed library winners as
  positives (weight 1.5) and near-misses as hard negatives (weight 2.0); the
  autopilot also applies a memory-derived penalty for operator shapes that keep failing.

Every upgrade is **inert until real outcomes/data exist**, so existing behaviour is
preserved on a cold start.

## Tests

`tests/test_phase7_selfimprove.py` (33 tests) covers the evaluator (incl. real
check-name mapping + eligibility exclusion), memory + library, the upgraded refiner
(LOW_TURNOVER / CONCENTRATED_WEIGHT / UNITS / vector_neut / hump / trade_when), the
Thompson bandit, proven motifs, ML training injection, the closed-loop integration
(absorb → repair, win → promote → seed), and the read-only API. Run the whole suite:

```bash
pytest tests/ -q     # 118 passed, 2 skipped
```
