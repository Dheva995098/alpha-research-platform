# Deploying for real 24/7 operation

The platform has two parts with different hosting needs:

| Part | What it does | Host |
|---|---|---|
| **UI + on-demand API** | dashboard, browse results, manual actions | **Vercel** (serverless) — already set up |
| **Autopilot worker** | the 24/7 loop: generate → submit → poll → train | **always-on host** (Render / Railway / VPS) |

Vercel is serverless: it cannot run a continuous background loop, and its
filesystem is ephemeral. So:

- **Persistence:** the Vercel project MUST use an external **Postgres**
  (`DATABASE_URL=postgresql://...`). SQLite does not persist on Vercel.
- **The self-improving loop** (Special Auto) must run on a long-lived process —
  that is what `scripts/run_worker.py` is for.

Both the Vercel app and the worker point at the **same Postgres**, so everything
the worker produces (results, attempt memory, win library, trained model) shows
up in the live dashboard.

## The golden rules
1. **Same `DATABASE_URL`** (Postgres) on Vercel *and* the worker host.
2. **Same `AES_KEY`** on both — the worker decrypts the BRAIN account password
   that was saved (encrypted) in Postgres. Mismatch → the worker cannot log in.
3. **Add your BRAIN account once via the live Vercel dashboard** (Accounts tab) so
   it lands in Postgres, is active + worker-enabled. With `SINGLE_ACCOUNT_MODE=true`
   the worker uses account id `1` (set `PRIMARY_ACCOUNT_ID` / `WORKER_ACCOUNT_ID`
   if yours differs).

## Option A — Render (Blueprint)
1. Render → New → **Blueprint** → connect this repo (`render.yaml` is included).
2. On the `alpha-autopilot-worker` service set env vars:
   - `DATABASE_URL` = your Vercel Postgres URL
   - `AES_KEY` = your Vercel `AES_KEY`
   - (optional) `WORKER_DRY_RUN=false`, `WORKER_AUTO_LEARN=true`
3. Deploy. Logs show a heartbeat every 60s (`ticks=… special_runs=… queued=…`).
4. Watch the live Vercel dashboard → Queue/Results/Vault/Learning fill up.

> An always-on Render Background Worker needs a paid plan (~starter). Free Render
> web services sleep after ~15 min idle, which stops the loop.

## Option B — Railway / Heroku-style (`Procfile`)
The included `Procfile` defines `worker:` and `web:` processes.
1. Create a project from this repo; add a **Postgres** plugin (or reuse Vercel's).
2. Set `DATABASE_URL` + `AES_KEY` (same as Vercel).
3. Enable the **worker** process (`python scripts/run_worker.py`). Keep it always-on.

## Option C — Docker on a VPS
```bash
# on the VPS, in the repo:
export DATABASE_URL='postgresql://...'   # same as Vercel
export AES_KEY='...'                     # same as Vercel
pip install -r requirements.txt
python scripts/run_worker.py             # run under systemd / pm2 / nohup for 24/7
```
`docker-compose.yml` is also present if you prefer containers (add the worker
command alongside the backend).

## Verify it's working (live)
- Worker logs: heartbeat `special_runs` increasing, `error` empty.
- Dashboard: **Queue** shows running, **Results/Vault** grow, **Learning** tab
  shows attempts/wins/library climbing, and the model retrains on the learn
  interval (needs ~50–300+ real completed sims before learning is meaningful).

## Tuning (env vars on the worker)
`WORKER_DRY_RUN` (false=live BRAIN), `WORKER_AUTO_LEARN`, `WORKER_SUBMIT_INTERVAL`,
`WORKER_POLL_INTERVAL`, `WORKER_LEARN_INTERVAL`, `WORKER_BATCH_SIZE`,
`WORKER_TARGET_RUNNING`, `WORKER_MAX_RUNNING`, `WORKER_REFILL_BELOW`,
`WORKER_MAX_PENDING`. Live submits are additionally paced by
`BRAIN_SUBMIT_INTERVAL_SECONDS` to respect BRAIN rate limits.
