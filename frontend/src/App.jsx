import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  CirclePause,
  Copy,
  Database,
  Filter,
  ListChecks,
  Loader2,
  Play,
  RefreshCcw,
  Rocket,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Shuffle,
  Trash2,
  UserPlus,
  Vault,
  XCircle,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://127.0.0.1:8001" : "");

const focusOptions = [
  "random",
  "momentum",
  "mean_reversion",
  "price_volume",
  "liquidity",
  "volatility",
  "size",
  "intraday",
  "quality",
  "fundamental",
  "hybrid",
  "analyst",
  "sentiment",
  "options",
  "model_risk",
  "decorrelation",
];

const defaultSimulationSettings = {
  region: "USA",
  universe: "TOP3000",
  delay: 1,
  decay: 10,
  neutralization: "SUBINDUSTRY",
  truncation: 0.01,
  testPeriod: "P5Y",
  language: "FASTEXPR",
};

const settingsPresets = {
  balanced: defaultSimulationSettings,
  reversion: { ...defaultSimulationSettings, decay: 6, truncation: 0.01, neutralization: "SUBINDUSTRY" },
  momentum: { ...defaultSimulationSettings, decay: 12, truncation: 0.02, neutralization: "SUBINDUSTRY" },
  price_volume: { ...defaultSimulationSettings, decay: 8, truncation: 0.01, neutralization: "INDUSTRY" },
  low_turnover: { ...defaultSimulationSettings, decay: 20, truncation: 0.01, neutralization: "SUBINDUSTRY" },
  intraday: { ...defaultSimulationSettings, decay: 4, truncation: 0.01, neutralization: "SUBINDUSTRY" },
};

const WORKER_OPTIONS_STORAGE_KEY = "alphaResearch.workerOptions.v1";

const defaultWorkerOptions = {
  dry_run: true,
  submit_interval_seconds: 60,
  poll_interval_seconds: 150,
  auto_learn: false,
  learning_interval_seconds: 300,
  special_auto: false,
  special_batch_size: 5,
  special_target_running: 4,
  special_max_running: 6,
  special_refill_pending_below: 0,
  special_max_pending: 15,
  special_stale_running_minutes: 240,
  openai_assist: false,
  account_ids: [1],
};

function numericOption(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function sanitizeWorkerOptions(options = {}) {
  const merged = { ...defaultWorkerOptions, ...options };
  const accountIds = Array.isArray(merged.account_ids)
    ? [...new Set(merged.account_ids.map((item) => Number(item)).filter((item) => Number.isInteger(item) && item > 0))]
    : [1];
  const primaryAccountId = accountIds[0] || 1;
  return {
    dry_run: Boolean(merged.dry_run),
    submit_interval_seconds: numericOption(merged.submit_interval_seconds, 60, 1, 3600),
    poll_interval_seconds: numericOption(merged.poll_interval_seconds, 150, 1, 3600),
    auto_learn: Boolean(merged.auto_learn),
    learning_interval_seconds: numericOption(merged.learning_interval_seconds, 300, 60, 86400),
    special_auto: false,
    special_batch_size: numericOption(merged.special_batch_size, 5, 1, 20),
    special_target_running: numericOption(merged.special_target_running, 6, 1, 25),
    special_max_running: numericOption(merged.special_max_running, 8, 1, 30),
    special_refill_pending_below: 0,
    special_max_pending: numericOption(merged.special_max_pending, 30, 0, 1000),
    special_stale_running_minutes: numericOption(merged.special_stale_running_minutes, 240, 15, 1440),
    openai_assist: Boolean(merged.openai_assist),
    account_ids: [primaryAccountId],
  };
}

function loadStoredWorkerOptions() {
  if (typeof window === "undefined") {
    return defaultWorkerOptions;
  }
  try {
    const stored = window.localStorage.getItem(WORKER_OPTIONS_STORAGE_KEY);
    return stored ? sanitizeWorkerOptions(JSON.parse(stored)) : defaultWorkerOptions;
  } catch (_error) {
    return defaultWorkerOptions;
  }
}

function persistWorkerOptions(options) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(WORKER_OPTIONS_STORAGE_KEY, JSON.stringify(sanitizeWorkerOptions(options)));
}

function App() {
  const [activeTab, setActiveTab] = useState("generate");
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [worker, setWorker] = useState(null);
  const [learning, setLearning] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [fieldIntel, setFieldIntel] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [generated, setGenerated] = useState([]);
  const [generationSettings, setGenerationSettings] = useState({});
  const [selected, setSelected] = useState(new Set());
  const [queue, setQueue] = useState([]);
  const [results, setResults] = useState([]);
  const [goodVault, setGoodVault] = useState({ summary: {}, alphas: [] });
  const [selfImprove, setSelfImprove] = useState({ stats: null, memory: [], nearMisses: [], library: [] });
  const [queueStatusFilter, setQueueStatusFilter] = useState("all");
  const [resultSourceFilter, setResultSourceFilter] = useState("all");
  const [predictions, setPredictions] = useState([]);
  const [filterSummary, setFilterSummary] = useState(null);
  const [resultFilterSummary, setResultFilterSummary] = useState(null);
  const [expressionInput, setExpressionInput] = useState("rank(close)\ngroup_neutralize(rank(ts_corr(close, volume, 20)), sector)");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [accountForm, setAccountForm] = useState({
    brain_email: "",
    brain_password: "",
    daily_quota: 450,
  });
  const [workerOptions, setWorkerOptions] = useState(loadStoredWorkerOptions);
  const [form, setForm] = useState({
    count: 12,
    focus: "momentum",
    dataset_id: "auto",
    seed: 42,
    settingsPreset: "balanced",
    neutralize: true,
    include_refinements: true,
    use_openai: true,
  });

  const selectedExpressions = useMemo(
    () => generated.filter((_, index) => selected.has(index)).map((item) => item.expression),
    [generated, selected]
  );

  const visibleResults = useMemo(
    () => results.filter((row) => resultMatchesSource(row, resultSourceFilter)),
    [results, resultSourceFilter]
  );

  const selectedWorkerAccountIds = useMemo(() => {
    const primary = accounts.find((account) => account.worker_enabled !== false) || accounts[0];
    return primary ? [primary.id] : [1];
  }, [accounts]);

  useEffect(() => {
    refreshAll();
    const timer = window.setInterval(refreshOperationalState, 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    refreshFieldIntel(form.dataset_id);
  }, [form.dataset_id]);

  useEffect(() => {
    if (activeTab === "selfimprove") {
      refreshSelfImprove();
    }
  }, [activeTab]);

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(errorMessageFromResponse(text, response.status));
    }
    if (response.status === 204) {
      return null;
    }
    const text = await response.text();
    if (!text) {
      return null;
    }
    return JSON.parse(text);
  }

  async function run(label, task) {
    setBusy(label);
    setNotice("");
    try {
      await task();
    } catch (error) {
      setNotice(cleanDisplayError(error.message));
      if (error.message.includes("No active accounts")) {
        setActiveTab("accounts");
      }
    } finally {
      setBusy("");
    }
  }

  async function refreshOperationalState() {
    const [summaryData, workerData, learningData, goodData] = await Promise.all([
      request("/api/orchestration/summary"),
      request("/api/orchestration/worker/status"),
      request("/api/ml/learning-status"),
      request("/api/ml/good-alphas?limit=200"),
    ]);
    setSummary(summaryData);
    setWorker(workerData);
    setLearning(learningData);
    setGoodVault(goodData || { summary: {}, alphas: [] });
  }

  async function refreshSelfImprove() {
    await run("selfimprove", async () => {
      const [stats, memory, nearMisses, library] = await Promise.all([
        request("/api/selfimprove/stats"),
        request("/api/selfimprove/memory?limit=50"),
        request("/api/selfimprove/near-misses?limit=25"),
        request("/api/selfimprove/library?limit=50"),
      ]);
      setSelfImprove({
        stats: stats || null,
        memory: memory?.attempts || [],
        nearMisses: nearMisses?.near_misses || [],
        library: library?.library || [],
      });
    });
  }

  async function refreshAll() {
    await run("refresh", async () => {
      const [healthData, accountsData, queueData, resultsData, datasetData] = await Promise.all([
        request("/health"),
        request("/api/accounts/"),
        request(queuePath(queueStatusFilter, 100)),
        request("/api/ml/results?limit=500"),
        request("/api/generation/datasets"),
        refreshOperationalState(),
      ]);
      setHealth(healthData);
      setAccounts(accountsData);
      setQueue(queueData);
      setResults(resultsData);
      setDatasets(datasetData?.datasets || []);
      await refreshFieldIntel(form.dataset_id);
    });
  }

  async function refreshFieldIntel(datasetId) {
    try {
      const params = new URLSearchParams({ limit: "8" });
      if (datasetId && datasetId !== "auto") {
        params.set("dataset_id", datasetId);
      }
      const data = await request(`/api/generation/field-intelligence?${params.toString()}`);
      setFieldIntel(data);
    } catch (_error) {
      setFieldIntel(null);
    }
  }

  async function changeQueueStatus(status) {
    setQueueStatusFilter(status);
    await run("queue-filter", async () => {
      const data = await request(queuePath(status, 100));
      setQueue(data || []);
    });
  }

  async function openQueueStatus(status) {
    setActiveTab("queue");
    await changeQueueStatus(status);
  }

  async function generateCandidates() {
    await run("generate", async () => {
      const data = await request("/api/generation/generate", {
        method: "POST",
        body: JSON.stringify(generationPayload(form)),
      });
      setGenerated(data.candidates || []);
      setGenerationSettings(data.settings_overrides || {});
      setSelected(new Set((data.candidates || []).map((_, index) => index)));
      setFilterSummary(null);
      setResultFilterSummary(null);
      await refreshFieldIntel(form.dataset_id);
    });
  }

  function randomizeGenerationForm() {
    const datasetChoices = ["random", "auto", ...datasets.map((dataset) => dataset.id)];
    const presetChoices = ["random", ...Object.keys(settingsPresets)];
    setForm({
      ...form,
      count: 5,
      focus: "random",
      dataset_id: datasetChoices[Math.floor(Math.random() * datasetChoices.length)] || "random",
      seed: Math.floor(Math.random() * 2_000_000_000) + 1,
      settingsPreset: presetChoices[Math.floor(Math.random() * presetChoices.length)] || "random",
      neutralize: Math.random() > 0.15,
      include_refinements: Math.random() > 0.35,
      use_openai: true,
    });
    setNotice("Random generation armed");
  }

  async function filterSelected() {
    const expressions = selectedExpressions.length ? selectedExpressions : generated.map((item) => item.expression);
    await run("filter", async () => {
      const data = await request("/api/filters/expressions", {
        method: "POST",
        body: JSON.stringify({ expressions, min_ml_probability: 0.45 }),
      });
      setFilterSummary(data);
    });
  }

  async function queueSelected() {
    const expressions = filterSummary?.accepted?.length
      ? filterSummary.accepted.map((item) => item.expression)
      : selectedExpressions;
    await run("queue", async () => {
      const data = await request("/api/orchestration/queue", {
        method: "POST",
        body: JSON.stringify({ expressions, validate: true, settings: settingsForGeneration(form, datasets, generationSettings) }),
      });
      const skipped = data.skipped?.length || 0;
      const skippedReason = skipped ? `; skipped ${skipped}: ${summarizeSkipped(data.skipped)}` : "";
      setNotice(`Queued ${data.queued_count} expression(s)${skippedReason}`);
      await refreshAll();
    });
  }

  async function submitNext(dryRun = true) {
    if (!dryRun) {
      const ok = window.confirm("Submit one alpha to live WorldQuant BRAIN using your stored account?");
      if (!ok) {
        return;
      }
    }
    await run("submit", async () => {
      await request("/api/orchestration/submit-next", {
        method: "POST",
        body: JSON.stringify({ dry_run: dryRun, universe: "default" }),
      });
      await refreshAll();
    });
  }

  async function liveSimulateResult(result) {
    const ok = window.confirm(
      `Live simulate result ${result.id} on WorldQuant BRAIN? This uses one quota from the stored account.`
    );
    if (!ok) {
      return;
    }
    await run(`live-result-${result.id}`, async () => {
      const data = await request(`/api/orchestration/results/${result.id}/live-submit`, {
        method: "POST",
        body: JSON.stringify({ universe: "default", settings: settingsForRow(result) }),
      });
      const simulationId = data.simulations?.[0]?.id;
      await refreshAll();
      setActiveTab("queue");
      setNotice(simulationId ? `Live simulation ${simulationId} submitted` : data.message);
    });
  }

  async function approveGoodResult(result) {
    const ok = window.confirm(`Save result ${result.id} to Good Live vault?`);
    if (!ok) {
      return;
    }
    await run(`approve-good-${result.id}`, async () => {
      await request(`/api/ml/results/${result.id}/approve-good`, { method: "POST" });
      setNotice(`Result ${result.id} saved to Good Live`);
      await refreshAll();
      setActiveTab("vault");
    });
  }

  async function pollRunning() {
    await run("poll", async () => {
      await request("/api/orchestration/poll", {
        method: "POST",
        body: JSON.stringify({ limit: 25 }),
      });
      await refreshAll();
    });
  }

  async function clearFailedRows() {
    const ok = window.confirm("Clear failed and cancelled queue rows?");
    if (!ok) {
      return;
    }
    await run("clear-terminal", async () => {
      const data = await request("/api/orchestration/queue/clear-terminal", {
        method: "POST",
        body: JSON.stringify({ statuses: ["failed", "cancelled"] }),
      });
      setNotice(data.message);
      await refreshAll();
    });
  }

  async function clearPendingRows() {
    const ok = window.confirm("Clear all local pending queue rows? Running and completed rows will stay.");
    if (!ok) {
      return;
    }
    await run("clear-pending", async () => {
      const data = await request("/api/orchestration/queue/clear-pending", {
        method: "POST",
        body: JSON.stringify({ keep_latest: 0 }),
      });
      setNotice(data.message);
      await refreshAll();
    });
  }

  async function scoreInput() {
    const expressions = splitExpressions(expressionInput);
    await run("score", async () => {
      const data = await request("/api/ml/score", {
        method: "POST",
        body: JSON.stringify({ expressions }),
      });
      setPredictions(data.predictions || []);
    });
  }

  async function filterInput() {
    const expressions = splitExpressions(expressionInput);
    await run("filter-input", async () => {
      const data = await request("/api/filters/expressions", {
        method: "POST",
        body: JSON.stringify({ expressions, min_ml_probability: 0.45 }),
      });
      setFilterSummary(data);
    });
  }

  async function scoreStoredResults() {
    await run("score-results", async () => {
      const data = await request("/api/ml/score-results", {
        method: "POST",
        body: JSON.stringify({ limit: 200, only_unscored: false }),
      });
      setResults(data || []);
      setNotice(`Scored ${data?.length || 0} stored result(s)`);
      await refreshOperationalState();
    });
  }

  async function filterStoredResults() {
    await run("filter-results", async () => {
      const data = await request("/api/filters/results", {
        method: "POST",
        body: JSON.stringify({ limit: 200 }),
      });
      setResultFilterSummary(data);
      setNotice(`Accepted ${data.accepted_count}, rejected ${data.rejected_count}`);
    });
  }

  async function runAutoLearn() {
    await run("auto-learn", async () => {
      const data = await request("/api/ml/auto-learn", {
        method: "POST",
        body: JSON.stringify({ limit: 500, min_examples: 5 }),
      });
      setLearning(data);
      setNotice(data.message || "Learner updated");
      await refreshOperationalState();
    });
  }

  async function copyGoodAlpha(row) {
    const text = row.copy_text || alphaCopyText(row);
    await copyText(text);
    setNotice(`Copied good alpha ${row.id}`);
  }

  function applyBestLearning() {
    const focus = learning?.best_focuses?.[0]?.name;
    if (!focus) {
      return;
    }
    setForm({ ...form, focus });
    setActiveTab("generate");
    setNotice(`Applied learned focus: ${focus}`);
  }

  async function startWorker() {
    if (!workerOptions.dry_run) {
      const ok = window.confirm("Start live worker? It will submit pending alphas to WorldQuant BRAIN on each interval.");
      if (!ok) {
        return;
      }
    }
    await run("worker-start", async () => {
      const payload = {
        submit_interval_seconds: workerOptions.submit_interval_seconds,
        poll_interval_seconds: workerOptions.poll_interval_seconds,
        dry_run: workerOptions.dry_run,
        auto_learn: workerOptions.auto_learn,
        learning_interval_seconds: workerOptions.learning_interval_seconds,
        special_auto: false,
        special_batch_size: 1,
        special_target_running: workerOptions.special_target_running,
        special_max_running: workerOptions.special_max_running,
        special_refill_pending_below: 0,
        special_max_pending: workerOptions.special_max_pending,
        special_stale_running_minutes: workerOptions.special_stale_running_minutes,
        openai_assist: workerOptions.openai_assist,
        account_ids: selectedWorkerAccountIds.slice(0, 1),
      };
      const data = await request("/api/orchestration/worker/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      persistWorkerOptions(payload);
      setWorkerOptions(sanitizeWorkerOptions(payload));
      setWorker(data);
    });
  }

  async function startSpecialWorker() {
    const modeLabel = workerOptions.dry_run ? "dry-run" : "live";
    const ok = window.confirm(`Start ${modeLabel} Special Auto on Account ${selectedWorkerAccountIds[0] || 1}?`);
    if (!ok) {
      return;
    }
    await run("worker-special", async () => {
      const payload = {
        submit_interval_seconds: workerOptions.submit_interval_seconds,
        poll_interval_seconds: workerOptions.poll_interval_seconds,
        dry_run: workerOptions.dry_run,
        auto_learn: workerOptions.auto_learn,
        learning_interval_seconds: workerOptions.learning_interval_seconds,
        special_auto: true,
        special_batch_size: workerOptions.special_batch_size,
        special_target_running: workerOptions.special_target_running,
        special_max_running: workerOptions.special_max_running,
        special_refill_pending_below: workerOptions.special_refill_pending_below,
        special_max_pending: workerOptions.special_max_pending,
        special_stale_running_minutes: workerOptions.special_stale_running_minutes,
        openai_assist: workerOptions.openai_assist,
        account_ids: selectedWorkerAccountIds.slice(0, 1),
      };
      const data = await request("/api/orchestration/worker/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      persistWorkerOptions(payload);
      setWorker(data);
      setWorkerOptions(sanitizeWorkerOptions(payload));
      setActiveTab("queue");
      setNotice("Special Auto started for single account");
      await refreshAll();
    });
  }

  async function stopWorker() {
    await run("worker-stop", async () => {
      const data = await request("/api/orchestration/worker/stop", { method: "POST" });
      setWorker(data);
    });
  }

  async function createAccount() {
    await run("account-create", async () => {
      const created = await request("/api/accounts/", {
        method: "POST",
        body: JSON.stringify({
          brain_email: accountForm.brain_email,
          brain_password: accountForm.brain_password,
        }),
      });
      if (Number(accountForm.daily_quota) !== created.daily_quota) {
        await request(`/api/accounts/${created.id}`, {
          method: "PUT",
          body: JSON.stringify({ daily_quota: Number(accountForm.daily_quota) }),
        });
      }
      setAccountForm({ brain_email: "", brain_password: "", daily_quota: 450 });
      setNotice(`Account ${created.brain_email} added`);
      await refreshAll();
    });
  }

  async function deactivateAccount(accountId) {
    await run(`account-delete-${accountId}`, async () => {
      await request(`/api/accounts/${accountId}`, { method: "DELETE" });
      setNotice(`Account ${accountId} deactivated`);
      await refreshAll();
    });
  }

  async function testAccount(accountId) {
    await run(`account-test-${accountId}`, async () => {
      const data = await request(`/api/accounts/${accountId}/test`, { method: "POST" });
      setNotice(data.message);
    });
  }

  async function resetAccountQuota(accountId) {
    await run(`account-reset-${accountId}`, async () => {
      await request(`/api/accounts/${accountId}/quota/reset`, { method: "POST" });
      setNotice(`Account ${accountId} local quota counter reset`);
      await refreshAll();
    });
  }

  async function updateAccountSettings(accountId, patch) {
    await run(`account-update-${accountId}`, async () => {
      await request(`/api/accounts/${accountId}`, {
        method: "PUT",
        body: JSON.stringify(patch),
      });
      await refreshAll();
    });
  }

  async function syncAccountFields(accountId) {
    await run(`account-sync-${accountId}`, async () => {
      const data = await request(`/api/accounts/${accountId}/sync-fields?limit_per_dataset=100`, { method: "POST" });
      setNotice(data.message || `Account ${accountId} fields synced`);
      await refreshAll();
      await refreshFieldIntel(form.dataset_id);
    });
  }

  function toggleSelected(index) {
    const next = new Set(selected);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    setSelected(next);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">WorldQuant BRAIN automation</div>
          <h1>Alpha Research Platform</h1>
        </div>
        <div className="status-strip">
          <StatusPill label="API" ok={health?.status === "ok"} value={health?.environment || "checking"} />
          <StatusPill label="Worker" ok={worker?.running} value={worker?.running ? "running" : "idle"} />
          <button className="icon-button" title="Refresh" onClick={refreshAll} disabled={Boolean(busy)}>
            {busy === "refresh" ? <Loader2 className="spin" /> : <RefreshCcw />}
          </button>
        </div>
      </header>

      {notice && <div className="notice">{notice}</div>}

      <section className="metric-row">
        <Metric
          label="Pending"
          value={summary?.statuses?.pending ?? 0}
          tone="amber"
          active={activeTab === "queue" && queueStatusFilter === "pending"}
          onClick={() => openQueueStatus("pending")}
        />
        <Metric
          label="Running"
          value={summary?.statuses?.running ?? 0}
          tone="blue"
          active={activeTab === "queue" && queueStatusFilter === "running"}
          onClick={() => openQueueStatus("running")}
        />
        <Metric
          label="Completed"
          value={summary?.statuses?.completed ?? 0}
          tone="green"
          active={activeTab === "queue" && queueStatusFilter === "completed"}
          onClick={() => openQueueStatus("completed")}
        />
        <Metric
          label="Good Live"
          value={goodVault?.summary?.good_count ?? 0}
          tone="teal"
          active={activeTab === "vault"}
          onClick={() => setActiveTab("vault")}
        />
        <Metric
          label="Account"
          value={accounts.length ? "ready" : "none"}
          tone="violet"
          active={activeTab === "accounts"}
          onClick={() => setActiveTab("accounts")}
        />
      </section>

      <nav className="tabs" aria-label="Dashboard sections">
        <Tab id="generate" active={activeTab} setActive={setActiveTab} icon={<Sparkles />} label="Generate" />
        <Tab id="accounts" active={activeTab} setActive={setActiveTab} icon={<UserPlus />} label="Account" />
        <Tab id="queue" active={activeTab} setActive={setActiveTab} icon={<ListChecks />} label="Queue" />
        <Tab id="results" active={activeTab} setActive={setActiveTab} icon={<CheckCircle2 />} label="Results" />
        <Tab id="vault" active={activeTab} setActive={setActiveTab} icon={<Vault />} label="Vault" />
        <Tab id="decisions" active={activeTab} setActive={setActiveTab} icon={<Filter />} label="Decisions" />
        <Tab id="rank" active={activeTab} setActive={setActiveTab} icon={<Brain />} label="Rank" />
        <Tab id="learner" active={activeTab} setActive={setActiveTab} icon={<Brain />} label="Learner" />
        <Tab id="selfimprove" active={activeTab} setActive={setActiveTab} icon={<RefreshCcw />} label="Learning" />
        <Tab id="worker" active={activeTab} setActive={setActiveTab} icon={<Activity />} label="Worker" />
      </nav>

      {activeTab === "generate" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<SlidersHorizontal />} title="Generation" />
            <div className="control-grid">
              <label>
                Count
                <input
                  type="number"
                  min="1"
                  max="200"
                  value={form.count}
                  onChange={(event) => setForm({ ...form, count: Number(event.target.value) })}
                />
              </label>
              <label>
                Focus
                <select value={form.focus} onChange={(event) => setForm({ ...form, focus: event.target.value })}>
                  {focusOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Dataset
                <select
                  value={form.dataset_id}
                  onChange={(event) => setForm({ ...form, dataset_id: event.target.value })}
                >
                  <option value="random">random</option>
                  <option value="auto">auto</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input
                  type="number"
                  value={form.seed}
                  onChange={(event) => setForm({ ...form, seed: Number(event.target.value) })}
                />
              </label>
              <label>
                Settings
                <select
                  value={form.settingsPreset}
                  onChange={(event) => setForm({ ...form, settingsPreset: event.target.value })}
                >
                  <option value="random">random</option>
                  {Object.keys(settingsPresets).map((preset) => (
                    <option key={preset} value={preset}>
                      {preset}
                    </option>
                  ))}
                </select>
              </label>
              <div className="settings-preview">
                <SettingsChips settings={settingsForGeneration(form, datasets, generationSettings)} compact />
              </div>
              <FieldIntelStrip fields={fieldIntel?.fields || []} />
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={form.neutralize}
                  onChange={(event) => setForm({ ...form, neutralize: event.target.checked })}
                />
                Neutralize
              </label>
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={form.include_refinements}
                  onChange={(event) => setForm({ ...form, include_refinements: event.target.checked })}
                />
                Refine
              </label>
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={form.use_openai}
                  onChange={(event) => setForm({ ...form, use_openai: event.target.checked })}
                />
                OpenAI assist
              </label>
              <label>
                Batch size
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={workerOptions.special_batch_size}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, special_batch_size: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
            </div>
            <div className="button-row">
              <ActionButton icon={<Sparkles />} label="Generate" busy={busy === "generate"} onClick={generateCandidates} />
              <ActionButton icon={<Shuffle />} label="Random" onClick={randomizeGenerationForm} />
              <ActionButton icon={<Filter />} label="Filter" busy={busy === "filter"} onClick={filterSelected} disabled={!generated.length} />
              <ActionButton icon={<Send />} label="Queue" busy={busy === "queue"} onClick={queueSelected} disabled={!selectedExpressions.length && !filterSummary?.accepted?.length} />
            </div>
          </div>

          <div className="panel list-panel">
            <PanelTitle icon={<Rocket />} title="Candidates" />
            <CandidateList items={generated} selected={selected} onToggle={toggleSelected} />
          </div>
        </section>
      )}

      {activeTab === "accounts" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<UserPlus />} title="Add Account" />
            <div className="control-grid account-form">
              <label>
                BRAIN Email
                <input
                  type="email"
                  autoComplete="username"
                  value={accountForm.brain_email}
                  onChange={(event) => setAccountForm({ ...accountForm, brain_email: event.target.value })}
                />
              </label>
              <label>
                Password
                <input
                  type="password"
                  autoComplete="current-password"
                  value={accountForm.brain_password}
                  onChange={(event) => setAccountForm({ ...accountForm, brain_password: event.target.value })}
                />
              </label>
              <label>
                Daily Quota
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={accountForm.daily_quota}
                  onChange={(event) => setAccountForm({ ...accountForm, daily_quota: Number(event.target.value) })}
                />
              </label>
            </div>
            <div className="button-row">
              <ActionButton
                icon={<UserPlus />}
                label="Add"
                busy={busy === "account-create"}
                onClick={createAccount}
                disabled={!accountForm.brain_email || !accountForm.brain_password}
              />
            </div>
          </div>

          <div className="panel list-panel">
            <PanelTitle icon={<ShieldCheck />} title="Accounts" />
            <AccountList
              accounts={accounts}
              quotas={summary?.accounts || []}
              busy={busy}
              onTest={testAccount}
              onSyncFields={syncAccountFields}
              onResetQuota={resetAccountQuota}
              onUpdate={updateAccountSettings}
              onDeactivate={deactivateAccount}
            />
          </div>
        </section>
      )}

      {activeTab === "queue" && (
        <section className="workspace queue-workspace">
          <div className="panel queue-panel">
            <PanelTitle icon={<ListChecks />} title="Queue" />
            <div className="toolbar-row">
              <div className="button-row">
                <ActionButton icon={<Send />} label="Dry Run" busy={busy === "submit"} onClick={() => submitNext(true)} />
                <ActionButton icon={<Rocket />} label="Live Submit" busy={busy === "submit"} onClick={() => submitNext(false)} />
                <ActionButton icon={<RefreshCcw />} label="Poll" busy={busy === "poll"} onClick={pollRunning} />
                <ActionButton icon={<Trash2 />} label="Clear Pending" busy={busy === "clear-pending"} onClick={clearPendingRows} />
                <ActionButton icon={<Trash2 />} label="Clear Failed" busy={busy === "clear-terminal"} onClick={clearFailedRows} />
              </div>
              <select
                className="compact-select"
                aria-label="Queue status"
                value={queueStatusFilter}
                onChange={(event) => changeQueueStatus(event.target.value)}
                disabled={busy === "queue-filter"}
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
              </select>
            </div>
            <QueueTable rows={queue} />
          </div>
          <div className="panel quota-panel">
            <PanelTitle icon={<Activity />} title="Quotas" />
            <QuotaList accounts={summary?.accounts || []} />
          </div>
        </section>
      )}

      {activeTab === "results" && (
        <section className="workspace results-workspace">
          <div className="panel results-panel">
            <PanelTitle icon={<CheckCircle2 />} title="Results" />
            <div className="toolbar-row">
              <div className="button-row">
                <ActionButton icon={<Brain />} label="Score Results" busy={busy === "score-results"} onClick={scoreStoredResults} />
                <ActionButton icon={<Filter />} label="Filter Results" busy={busy === "filter-results"} onClick={filterStoredResults} />
                <ActionButton icon={<RefreshCcw />} label="Refresh" busy={busy === "refresh"} onClick={refreshAll} />
              </div>
              <select
                className="compact-select"
                aria-label="Result source"
                value={resultSourceFilter}
                onChange={(event) => setResultSourceFilter(event.target.value)}
              >
                <option value="all">All</option>
                <option value="live">Live</option>
                <option value="dry">Dry Run</option>
              </select>
            </div>
            <ResultsTable
              rows={visibleResults}
              busy={busy}
              onLiveSimulate={liveSimulateResult}
              onApproveGood={approveGoodResult}
            />
          </div>
        </section>
      )}

      {activeTab === "vault" && (
        <section className="workspace results-workspace">
          <div className="panel vault-panel">
            <PanelTitle icon={<Vault />} title="Good Live Alphas" />
            <div className="worker-state vault-summary">
              <StatusPill label="Good" ok value={goodVault?.summary?.good_count ?? 0} />
              <StatusPill label="Live" ok value={goodVault?.summary?.live_result_count ?? 0} />
              <StatusPill label="Hit Rate" ok value={formatPercent(goodVault?.summary?.accept_rate)} />
              <StatusPill label="Sharpe" ok value={goodVault?.summary?.thresholds?.min_sharpe ?? 1.25} />
            </div>
            <div className="button-row">
              <ActionButton icon={<RefreshCcw />} label="Refresh" busy={busy === "refresh"} onClick={refreshAll} />
              <ActionButton icon={<Brain />} label="Run Learn" busy={busy === "auto-learn"} onClick={runAutoLearn} />
            </div>
            <GoodAlphaVault rows={goodVault?.alphas || []} onCopy={copyGoodAlpha} />
          </div>
        </section>
      )}

      {activeTab === "decisions" && (
        <section className="workspace decisions-workspace">
          <div className="panel list-panel">
            <PanelTitle icon={<Filter />} title="Filter Decisions" />
            <DecisionList filterSummary={resultFilterSummary} />
          </div>
        </section>
      )}

      {activeTab === "rank" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<Brain />} title="Expressions" />
            <textarea value={expressionInput} onChange={(event) => setExpressionInput(event.target.value)} />
            <div className="button-row">
              <ActionButton icon={<Brain />} label="Score" busy={busy === "score"} onClick={scoreInput} />
              <ActionButton icon={<Filter />} label="Filter" busy={busy === "filter-input"} onClick={filterInput} />
            </div>
          </div>
          <div className="panel list-panel">
            <PanelTitle icon={<CheckCircle2 />} title="Ranked" />
            <PredictionList predictions={predictions} filterSummary={filterSummary} />
          </div>
        </section>
      )}

      {activeTab === "learner" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<Brain />} title="Auto Learner" />
            <div className="worker-state">
              <StatusPill label="Model" ok={learning?.trained} value={learning?.trained ? "trained" : "cold"} />
              <StatusPill label="Examples" ok value={learning?.model?.trained_on_count ?? 0} />
              <StatusPill label="Live" ok value={learning?.live_result_count ?? 0} />
              <StatusPill label="Accepted" ok value={formatPercent(learning?.accept_rate)} />
              <StatusPill label="Checks" ok value={`${learning?.check_summary?.pass ?? 0}/${learning?.check_summary?.fail ?? 0}`} />
            </div>
            <div className="button-row">
              <ActionButton icon={<Brain />} label="Run Learn" busy={busy === "auto-learn"} onClick={runAutoLearn} />
              <ActionButton
                icon={<Sparkles />}
                label="Apply Best"
                onClick={applyBestLearning}
                disabled={!learning?.best_focuses?.length || !(learning?.positive_result_count > 0)}
              />
            </div>
            <pre className="json-block compact-json">
              {JSON.stringify(
                {
                  message: learning?.message,
                  accuracy: learning?.training?.accuracy ?? learning?.model?.metrics?.accuracy ?? null,
                  positives: learning?.training?.positive_count ?? learning?.model?.metrics?.positive_count ?? null,
                  negatives: learning?.training?.negative_count ?? learning?.model?.metrics?.negative_count ?? null,
                  scored: learning?.scored_count ?? 0,
                  check_pass_rate: learning?.check_summary?.pass_rate ?? null,
                  failed_checks: learning?.top_failed_checks ?? [],
                },
                null,
                2
              )}
            </pre>
          </div>
          <div className="panel list-panel">
            <PanelTitle icon={<CheckCircle2 />} title="Learned Patterns" />
            <LearnerRecommendations learning={learning} />
          </div>
        </section>
      )}

      {activeTab === "selfimprove" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<RefreshCcw />} title="Self-Improving Loop" />
            <div className="worker-state">
              <StatusPill label="Attempts" ok value={selfImprove.stats?.attempts ?? 0} />
              <StatusPill label="Wins" ok={Boolean(selfImprove.stats?.wins)} value={selfImprove.stats?.wins ?? 0} />
              <StatusPill label="Near" ok value={selfImprove.stats?.near_misses ?? 0} />
              <StatusPill label="Fails" ok value={selfImprove.stats?.failures ?? 0} />
              <StatusPill label="Win rate" ok value={formatPercent(selfImprove.stats?.win_rate)} />
              <StatusPill label="Library" ok={Boolean(selfImprove.stats?.library_size)} value={selfImprove.stats?.library_size ?? 0} />
            </div>
            <div className="button-row">
              <ActionButton icon={<RefreshCcw />} label="Refresh" busy={busy === "selfimprove"} onClick={refreshSelfImprove} />
            </div>
            <PanelTitle icon={<Sparkles />} title="Near-misses queued for repair" />
            {selfImprove.nearMisses.length === 0 ? (
              <div className="empty">No repairable near-misses yet. They appear once live results land.</div>
            ) : (
              <div className="item-list">
                {selfImprove.nearMisses.map((row) => (
                  <div className="vault-row" key={`nm-${row.id}`}>
                    <div className="vault-row-main">
                      <div className="row-meta">
                        <span>sh {formatMetric(row.sharpe)}</span>
                        <span>fit {formatMetric(row.fitness)}</span>
                        <span>turn {formatMetric(row.turnover)}</span>
                        {(row.failures || []).map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                      <code>{row.expression}</code>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="panel list-panel">
            <PanelTitle icon={<Vault />} title="Win library (auto-grows)" />
            {selfImprove.library.length === 0 ? (
              <div className="empty">No confirmed wins yet.</div>
            ) : (
              <div className="item-list">
                {selfImprove.library.map((row) => (
                  <div className="vault-row" key={`lib-${row.id}`}>
                    <div className="vault-row-main">
                      <div className="row-meta">
                        <span>score {formatMetric(row.score)}</span>
                        <span>sh {formatMetric(row.sharpe)}</span>
                        <span>fit {formatMetric(row.fitness)}</span>
                        {row.focus && <span>{row.focus}</span>}
                      </div>
                      <code>{row.expression}</code>
                    </div>
                  </div>
                ))}
              </div>
            )}
            <PanelTitle icon={<Brain />} title="Recent attempts" />
            {selfImprove.memory.length === 0 ? (
              <div className="empty">No attempts recorded yet.</div>
            ) : (
              <div className="item-list">
                {selfImprove.memory.slice(0, 20).map((row) => (
                  <div className="vault-row" key={`mem-${row.id}`}>
                    <div className="vault-row-main">
                      <div className="row-meta">
                        <span>{row.outcome}</span>
                        <span>score {formatMetric(row.score)}</span>
                        {row.focus && <span>{row.focus}</span>}
                      </div>
                      <code>{row.expression}</code>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {activeTab === "worker" && (
        <section className="workspace two-column">
          <div className="panel">
            <PanelTitle icon={<Activity />} title="Worker" />
            <div className="worker-state">
              <StatusPill label="State" ok={worker?.running} value={worker?.running ? "running" : "idle"} />
              <StatusPill label="Mode" ok value={worker?.dry_run ? "dry run" : "live"} />
              <StatusPill label="Learn" ok={worker?.auto_learn} value={worker?.auto_learn ? "auto" : "manual"} />
              <StatusPill label="OpenAI" ok={worker?.openai_assist} value={worker?.openai_assist ? "assist" : "off"} />
              <StatusPill label="Ticks" ok value={worker?.iterations ?? 0} />
              <StatusPill label="Account" ok value={selectedWorkerAccountIds[0] || 1} />
            </div>
            <div className="control-grid worker-controls">
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={workerOptions.dry_run}
                  onChange={(event) => setWorkerOptions({ ...workerOptions, dry_run: event.target.checked })}
                  disabled={worker?.running}
                />
                Dry-run worker
              </label>
              <label>
                Submit interval
                <input
                  type="number"
                  min="1"
                  max="3600"
                  value={workerOptions.submit_interval_seconds}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, submit_interval_seconds: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label>
                Poll interval
                <input
                  type="number"
                  min="1"
                  max="3600"
                  value={workerOptions.poll_interval_seconds}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, poll_interval_seconds: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={workerOptions.auto_learn}
                  onChange={(event) => setWorkerOptions({ ...workerOptions, auto_learn: event.target.checked })}
                  disabled={worker?.running}
                />
                Auto-learn
              </label>
              <label>
                Learn interval
                <input
                  type="number"
                  min="60"
                  max="86400"
                  value={workerOptions.learning_interval_seconds}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, learning_interval_seconds: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={workerOptions.openai_assist}
                  onChange={(event) => setWorkerOptions({ ...workerOptions, openai_assist: event.target.checked })}
                  disabled={worker?.running}
                />
                OpenAI assist
              </label>
              <label>
                Target running
                <input
                  type="number"
                  min="1"
                  max="25"
                  value={workerOptions.special_target_running}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, special_target_running: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label>
                Max running
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={workerOptions.special_max_running}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, special_max_running: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label>
                Max pending
                <input
                  type="number"
                  min="0"
                  max="1000"
                  value={workerOptions.special_max_pending}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, special_max_pending: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
              <label>
                Stale minutes
                <input
                  type="number"
                  min="15"
                  max="1440"
                  value={workerOptions.special_stale_running_minutes}
                  onChange={(event) =>
                    setWorkerOptions({ ...workerOptions, special_stale_running_minutes: Number(event.target.value) })
                  }
                  disabled={worker?.running}
                />
              </label>
            </div>
            <div className="button-row">
              <ActionButton icon={<Play />} label="Start" busy={busy === "worker-start"} onClick={startWorker} disabled={worker?.running} />
              <SpecialButton busy={busy === "worker-special"} disabled={worker?.running} onClick={startSpecialWorker} />
              <ActionButton icon={<CirclePause />} label="Stop" busy={busy === "worker-stop"} onClick={stopWorker} disabled={!worker?.running} />
            </div>
          </div>
          <div className="panel">
            <PanelTitle icon={<Activity />} title="State" />
            <pre className="json-block">{JSON.stringify(worker || {}, null, 2)}</pre>
          </div>
        </section>
      )}
    </main>
  );
}

function splitExpressions(value) {
  return value
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function queuePath(status, limit = 100) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status && status !== "all") {
    params.set("status", status);
  }
  return `/api/orchestration/queue?${params.toString()}`;
}

function cleanDisplayError(value) {
  if (!value) {
    return "";
  }
  return String(value)
    .replace(/<linkToCommonErrorMessages>.*?<\/linkToCommonErrorMessages>/gis, "")
    .replace(/\s+/g, " ")
    .trim();
}

function summarizeSkipped(skipped = []) {
  const reasons = [...new Set(skipped.map((item) => cleanDisplayError(item.reason || "skipped")).filter(Boolean))];
  return reasons.length ? reasons.slice(0, 2).join("; ") : "not queued";
}

function shortDateTime(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function errorMessageFromResponse(text, status) {
  if (!text) {
    return `HTTP ${status}`;
  }
  try {
    const payload = JSON.parse(text);
    const detail = payload.detail || payload.error || payload.message;
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || item.message || JSON.stringify(item)).join(", ");
    }
    if (typeof detail === "string") {
      return detail;
    }
  } catch (_error) {
    // Fall through to raw text.
  }
  return text;
}

function StatusPill({ label, ok, value }) {
  return (
    <span className={`status-pill ${ok ? "ok" : "muted"}`}>
      {ok ? <CheckCircle2 /> : <XCircle />}
      <span>{label}</span>
      <strong>{value}</strong>
    </span>
  );
}

function Metric({ label, value, tone, active, onClick }) {
  const className = `metric ${tone} ${onClick ? "clickable" : ""} ${active ? "active" : ""}`;
  if (onClick) {
    return (
      <button type="button" className={className} onClick={onClick}>
        <span>{label}</span>
        <strong>{value}</strong>
      </button>
    );
  }
  return (
    <div className={className}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Tab({ id, active, setActive, icon, label }) {
  return (
    <button className={`tab ${active === id ? "active" : ""}`} onClick={() => setActive(id)}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function PanelTitle({ icon, title }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function ActionButton({ icon, label, busy, disabled, onClick }) {
  return (
    <button className="action-button" disabled={disabled || busy} onClick={onClick}>
      {busy ? <Loader2 className="spin" /> : icon}
      <span>{label}</span>
    </button>
  );
}

function SpecialButton({ busy, disabled, onClick }) {
  return (
    <button className="special-button" disabled={disabled || busy} onClick={onClick}>
      {busy ? <Loader2 className="spin" /> : <Sparkles />}
      <span>Special Auto</span>
    </button>
  );
}

function CandidateList({ items, selected, onToggle }) {
  if (!items.length) {
    return <div className="empty">No candidates</div>;
  }
  return (
    <div className="item-list">
      {items.map((item, index) => (
        <label className="candidate-row" key={`${item.expression}-${index}`}>
          <input type="checkbox" checked={selected.has(index)} onChange={() => onToggle(index)} />
          <div>
            <code>{item.expression}</code>
            <div className="row-meta">
              <span>{item.strategy}</span>
              {!!item.dataset_ids?.length && <span>{item.dataset_ids.join(",")}</span>}
              <span>{Number(item.score).toFixed(2)}</span>
            </div>
          </div>
        </label>
      ))}
    </div>
  );
}

function GoodAlphaVault({ rows, onCopy }) {
  if (!rows.length) {
    return <div className="empty">No good live alphas yet</div>;
  }
  return (
    <div className="item-list vault-list">
      {rows.map((row) => (
        <div className="vault-row" key={row.id}>
          <div className="vault-row-main">
            <div className="row-meta">
              <span>id {row.id}</span>
              <span>{shortId(row.brain_alpha_id)}</span>
              <span>sh {formatMetric(row.sharpe)}</span>
              <span>fit {formatMetric(row.fitness)}</span>
              <span>turn {formatMetric(row.turnover)}</span>
              <span>ml {formatMetric(row.ml_pass_probability)}</span>
              {row.human_approved && <span>manual</span>}
            </div>
            <code>{row.expression}</code>
            <SettingsChips settings={row.settings || settingsForRow(row)} />
          </div>
          <button className="icon-button" title="Copy alpha with settings" onClick={() => onCopy(row)}>
            <Copy />
          </button>
        </div>
      ))}
    </div>
  );
}

function FieldIntelStrip({ fields }) {
  if (!fields.length) {
    return null;
  }
  return (
    <div className="field-intel-strip">
      {fields.map((field) => (
        <span className="field-intel-chip" key={field.name}>
          <code>{field.name}</code>
          <strong>{Number(field.field_score || 0).toFixed(2)}</strong>
        </span>
      ))}
    </div>
  );
}

function QueueTable({ rows }) {
  if (!rows.length) {
    return <div className="empty">No queue rows</div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Status</th>
            <th>%</th>
            <th>Brain</th>
            <th>Settings</th>
            <th>Expression</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>
                <span className={`queue-status ${row.status}`}>{row.status}</span>
              </td>
              <td>{Math.round(row.progress || 0)}%</td>
              <td className="queue-brain-id">{shortId(row.brain_simulation_id)}</td>
              <td>
                <SettingsChips settings={row.settings || defaultSimulationSettings} compact />
              </td>
              <td>
                <code>{row.expression}</code>
                {row.error_message && <div className="queue-error">{cleanDisplayError(row.error_message)}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QuotaList({ accounts }) {
  if (!accounts.length) {
    return <div className="empty">No accounts</div>;
  }
  return (
    <div className="item-list">
      {accounts.map((account) => (
        <div className="quota-row" key={account.account_id}>
          <div>
            <strong>Account {account.account_id}</strong>
            <span>{account.is_active ? "active" : "inactive"} - worker {account.worker_enabled ? "on" : "off"}</span>
            <span>{account.running || 0} running - {account.pending || 0} pending</span>
            {account.cooldown_until && <span>cooldown {shortDateTime(account.cooldown_until)}</span>}
            {account.last_worker_error && <small className="lane-error">{account.last_worker_error}</small>}
          </div>
          <meter min="0" max={account.daily_quota || 1} value={account.remaining || 0} />
          <span>{account.remaining} left</span>
        </div>
      ))}
    </div>
  );
}

function AccountList({ accounts, quotas, busy, onTest, onSyncFields, onResetQuota, onUpdate, onDeactivate }) {
  if (!accounts.length) {
    return <div className="empty">No accounts</div>;
  }
  const quotaById = Object.fromEntries(quotas.map((quota) => [quota.account_id, quota]));
  return (
    <div className="item-list">
      {accounts.map((account) => {
        const quota = quotaById[account.id] || {};
        return (
          <div className="account-row" key={account.id}>
            <div>
              <strong>{account.brain_email}</strong>
              <div className="row-meta">
                <span>{account.is_active ? "active" : "inactive"}</span>
                <span>worker {account.worker_enabled ? "on" : "off"}</span>
                <span>{quota.remaining ?? account.daily_quota} left</span>
                <span>{account.submissions_today}/{account.daily_quota}</span>
                <span>{quota.running || 0} running</span>
                <span>{quota.pending || 0} pending</span>
                {(quota.cooldown_until || account.cooldown_until) && (
                  <span>cooldown {shortDateTime(quota.cooldown_until || account.cooldown_until)}</span>
                )}
              </div>
              {(quota.last_worker_error || account.last_worker_error) && (
                <small className="lane-error">{quota.last_worker_error || account.last_worker_error}</small>
              )}
              <div className="account-lane-controls">
                <label>
                  Run
                  <input
                    type="number"
                    min="1"
                    max="30"
                    defaultValue={account.max_running ?? 6}
                    onBlur={(event) => onUpdate(account.id, { max_running: Number(event.target.value) })}
                    disabled={busy === `account-update-${account.id}`}
                  />
                </label>
                <label>
                  Pending
                  <input
                    type="number"
                    min="0"
                    max="1000"
                    defaultValue={account.max_pending ?? 15}
                    onBlur={(event) => onUpdate(account.id, { max_pending: Number(event.target.value) })}
                    disabled={busy === `account-update-${account.id}`}
                  />
                </label>
              </div>
            </div>
            <div className="account-actions">
              <button
                className={`icon-button ${account.worker_enabled ? "success" : ""}`}
                title={account.worker_enabled ? "Disable worker lane" : "Enable worker lane"}
                onClick={() => onUpdate(account.id, { worker_enabled: !account.worker_enabled })}
                disabled={busy === `account-update-${account.id}`}
              >
                {busy === `account-update-${account.id}` ? <Loader2 className="spin" /> : <Activity />}
              </button>
              <button
                className="icon-button"
                title="Test"
                onClick={() => onTest(account.id)}
                disabled={busy === `account-test-${account.id}`}
              >
                {busy === `account-test-${account.id}` ? <Loader2 className="spin" /> : <ShieldCheck />}
              </button>
              <button
                className="icon-button"
                title="Sync fields"
                onClick={() => onSyncFields(account.id)}
                disabled={busy === `account-sync-${account.id}`}
              >
                {busy === `account-sync-${account.id}` ? <Loader2 className="spin" /> : <Database />}
              </button>
              <button
                className="icon-button"
                title="Reset quota"
                onClick={() => onResetQuota(account.id)}
                disabled={busy === `account-reset-${account.id}`}
              >
                {busy === `account-reset-${account.id}` ? <Loader2 className="spin" /> : <RefreshCcw />}
              </button>
              <button
                className="icon-button danger"
                title="Deactivate"
                onClick={() => onDeactivate(account.id)}
                disabled={busy === `account-delete-${account.id}`}
              >
                {busy === `account-delete-${account.id}` ? <Loader2 className="spin" /> : <Trash2 />}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PredictionList({ predictions, filterSummary }) {
  const filterItems = filterSummary
    ? [...(filterSummary.accepted || []), ...(filterSummary.rejected || [])]
    : [];
  if (!predictions.length && !filterItems.length) {
    return <div className="empty">No ranked rows</div>;
  }
  const rows = predictions.length ? predictions : filterItems;
  return (
    <div className="item-list">
      {rows.map((row, index) => (
        <div className="prediction-row" key={`${row.expression}-${index}`}>
          <code>{row.expression}</code>
          <div className="row-meta">
            {"pass_probability" in row && <span>p={Number(row.pass_probability).toFixed(2)}</span>}
            {"score" in row && <span>score={Number(row.score).toFixed(2)}</span>}
            {"passed" in row && <span>{row.passed ? "accepted" : "rejected"}</span>}
          </div>
          {row.reasons?.length > 0 && <small>{row.reasons.join(", ")}</small>}
        </div>
      ))}
    </div>
  );
}

function LearnerRecommendations({ learning }) {
  const focuses = learning?.best_focuses || [];
  const settings = learning?.best_settings || [];
  if (!focuses.length && !settings.length) {
    return <div className="empty">No learned patterns</div>;
  }
  return (
    <div className="item-list">
      {focuses.map((focus) => (
        <div className="prediction-row" key={`focus-${focus.name}`}>
          <code>{focus.name}</code>
          <div className="row-meta">
            <span>{focus.passed}/{focus.count}</span>
            <span>{formatPercent(focus.pass_rate)}</span>
            <span>sh {formatMetric(focus.avg_sharpe)}</span>
            <span>fit {formatMetric(focus.avg_fitness)}</span>
          </div>
        </div>
      ))}
      {settings.map((item, index) => (
        <div className="prediction-row" key={`settings-${index}`}>
          <SettingsChips settings={item.settings} />
          <div className="row-meta">
            <span>{item.passed}/{item.count}</span>
            <span>{formatPercent(item.pass_rate)}</span>
            <span>sh {formatMetric(item.avg_sharpe)}</span>
            <span>fit {formatMetric(item.avg_fitness)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ResultsTable({ rows, busy, onLiveSimulate, onApproveGood }) {
  if (!rows.length) {
    return <div className="empty">No results</div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Alpha</th>
            <th>Src</th>
            <th>Verdict</th>
            <th>Sh</th>
            <th>Fit</th>
            <th>Turn</th>
            <th>Grade</th>
            <th>ML</th>
            <th>Score</th>
            <th>Settings</th>
            <th>Expr</th>
            <th>Run</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const verdict = alphaVerdict(row);
            const liveBusy = busy === `live-result-${row.id}`;
            const approveBusy = busy === `approve-good-${row.id}`;
            const canApprove = canApproveForGoodLive(row);
            return (
              <tr key={row.id}>
                <td>{row.id}</td>
                <td>{row.brain_alpha_id || "-"}</td>
                <td>
                  <SourceChip row={row} />
                </td>
                <td>
                  <span className={`verdict ${verdict.tone}`}>{verdict.label}</span>
                </td>
                <td>{formatMetric(row.sharpe)}</td>
                <td>{formatMetric(row.fitness)}</td>
                <td>{formatMetric(row.turnover)}</td>
                <td>{row.raw_metrics?.grade || "-"}</td>
                <td>{formatMetric(row.ml_pass_probability)}</td>
                <td>{formatMetric(row.final_score)}</td>
                <td>
                  <SettingsChips settings={settingsForRow(row)} />
                </td>
                <td>
                  <code>{row.expression}</code>
                  <CheckList checks={row.raw_metrics?.is?.checks || []} />
                </td>
                <td>
                  <div className="result-actions">
                    <button
                      className="icon-button live-action"
                      title="Live simulate on WorldQuant BRAIN"
                      onClick={() => onLiveSimulate(row)}
                      disabled={liveBusy || !onLiveSimulate}
                    >
                      {liveBusy ? <Loader2 className="spin" /> : <Rocket />}
                    </button>
                    <button
                      className="icon-button success"
                      title="Save to Good Live"
                      onClick={() => onApproveGood(row)}
                      disabled={approveBusy || !canApprove || !onApproveGood}
                    >
                      {approveBusy ? <Loader2 className="spin" /> : <Vault />}
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SourceChip({ row }) {
  const dry = isDryRun(row);
  return <span className={`source-chip ${dry ? "dry" : "live"}`}>{dry ? "Dry" : "Live"}</span>;
}

function settingsForPreset(preset) {
  if (preset === "random") {
    return defaultSimulationSettings;
  }
  return settingsPresets[preset] || defaultSimulationSettings;
}

function generationPayload(form) {
  const randomize = form.focus === "random" || form.dataset_id === "random" || form.settingsPreset === "random";
  return {
    count: form.count,
    focus: form.focus === "random" ? null : form.focus,
    seed: form.seed,
    randomize,
    use_openai: Boolean(form.use_openai),
    neutralize: form.neutralize,
    include_refinements: form.include_refinements,
    dataset_ids: form.dataset_id === "auto" || form.dataset_id === "random" ? [] : [form.dataset_id],
  };
}

function settingsForGeneration(form, datasets, generationSettings = {}) {
  return {
    ...settingsForPreset(form.settingsPreset),
    ...datasetSettingsForSelection(form.dataset_id, datasets),
    ...generationSettings,
  };
}

function datasetSettingsForSelection(datasetId, datasets) {
  if (!datasetId || datasetId === "auto" || datasetId === "random") {
    return {};
  }
  const dataset = (datasets || []).find((item) => item.id === datasetId);
  return dataset?.settings_overrides || {};
}

function settingsForRow(row) {
  return row.settings || row.raw_metrics?.settings || defaultSimulationSettings;
}

function SettingsChips({ settings, compact = false }) {
  if (!settings) {
    return <span className="muted-cell">-</span>;
  }
  const items = [
    ["region", settings.region],
    ["universe", settings.universe],
    ["delay", settings.delay],
    ["decay", settings.decay],
    ["neutral", settings.neutralization],
    ["trunc", settings.truncation],
    ["test", settings.testPeriod],
    ["maxTrade", settings.maxTrade],
    ["lang", settings.language],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");

  return (
    <div className={`settings-chip-list ${compact ? "compact" : ""}`}>
      {items.map(([label, value]) => (
        <span className="settings-chip" key={label}>
          {label}: {String(value)}
        </span>
      ))}
    </div>
  );
}

function shortId(value) {
  if (!value) {
    return "-";
  }
  const text = String(value);
  if (text.startsWith("http")) {
    const parts = text.split("/").filter(Boolean);
    const id = parts[parts.length - 1] || text;
    return id.length > 14 ? `${id.slice(0, 10)}...` : id;
  }
  return text.length > 14 ? `${text.slice(0, 10)}...` : text;
}

function CheckList({ checks }) {
  if (!checks.length) {
    return null;
  }
  return (
    <div className="checks-list">
      {checks.map((check) => (
        <span className={`check-chip ${String(check.result).toLowerCase()}`} key={check.name}>
          {check.name}: {check.result}
        </span>
      ))}
    </div>
  );
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function alphaCopyText(row) {
  const settings = row.settings || settingsForRow(row);
  return [
    "Alpha Code",
    "----------",
    row.expression,
    "",
    "Simulation Settings",
    "-------------------",
    `Instrument Type: ${settings.instrumentType || "EQUITY"}`,
    `Region: ${settings.region || ""}`,
    `Universe: ${settings.universe || ""}`,
    `Language: ${settings.language || ""}`,
    `Decay: ${settings.decay ?? ""}`,
    `Delay: ${settings.delay ?? ""}`,
    `Truncation: ${settings.truncation ?? ""}`,
    `Neutralization: ${settings.neutralization || ""}`,
    `Pasteurization: ${settings.pasteurization || ""}`,
    `NaN Handling: ${settings.nanHandling || ""}`,
    `Unit Handling: ${settings.unitHandling || ""}`,
    `Max Trade: ${settings.maxTrade || ""}`,
    `Test Period: ${settings.testPeriod || ""}`,
    "",
    "Result Metrics",
    "--------------",
    `Sharpe: ${row.sharpe ?? ""}`,
    `Fitness: ${row.fitness ?? ""}`,
    `Turnover: ${row.turnover ?? ""}`,
    `Self Correlation: ${row.self_correlation ?? ""}`,
    `All Checks Passed: ${row.all_checks_passed ?? ""}`,
  ].join("\n");
}

function isDryRun(row) {
  return Boolean(
    row.raw_metrics?.dry_run ||
      row.raw_metrics?.source === "dry_run" ||
      String(row.brain_alpha_id || "").startsWith("dry-run")
  );
}

function resultMatchesSource(row, source) {
  if (source === "dry") {
    return isDryRun(row);
  }
  if (source === "live") {
    return !isDryRun(row);
  }
  return true;
}

function alphaVerdict(row) {
  const dry = isDryRun(row);
  const checks = row.raw_metrics?.is?.checks || [];
  const failedChecks = checks.filter((check) => check.result === "FAIL");
  const pendingChecks = checks.filter((check) => check.result === "PENDING");
  if (failedChecks.length > 0 || (row.sharpe ?? 0) < 1.25 || (row.fitness ?? 0) < 1.0) {
    return { label: dry ? "Dry Rej" : "Reject", tone: "bad" };
  }
  if (pendingChecks.length > 0) {
    return { label: dry ? "Dry Rev" : "Review", tone: "warn" };
  }
  if (dry) {
    return { label: "Dry Cand", tone: "warn" };
  }
  if ((row.ml_pass_probability ?? 0) >= 0.65 && (row.final_score ?? 0) >= 1.0) {
    return { label: "Candidate", tone: "good" };
  }
  return { label: "Review", tone: "warn" };
}

function canApproveForGoodLive(row) {
  if (!row || row.human_approved || isDryRun(row) || row.raw_metrics?.source !== "live") {
    return false;
  }
  const checks = row.raw_metrics?.is?.checks || [];
  const failedChecks = checks.filter((check) => check.result === "FAIL");
  return Boolean(
    failedChecks.length === 0 &&
      (row.sharpe ?? 0) >= 1.25 &&
      (row.fitness ?? 0) >= 1.0 &&
      (row.turnover === null || row.turnover === undefined || row.turnover <= 0.70) &&
      (row.self_correlation === null || row.self_correlation === undefined || row.self_correlation <= 0.70)
  );
}

function DecisionList({ filterSummary }) {
  if (!filterSummary) {
    return <div className="empty">No filter decisions</div>;
  }
  const rows = [...(filterSummary.accepted || []), ...(filterSummary.rejected || [])];
  if (!rows.length) {
    return <div className="empty">No filter decisions</div>;
  }
  return (
    <div className="item-list">
      {rows.map((row, index) => (
        <div className="prediction-row" key={`${row.expression}-${index}`}>
          <code>{row.expression}</code>
          <div className="row-meta">
            <span>{row.passed ? "accepted" : "rejected"}</span>
            {row.item_id && <span>result {row.item_id}</span>}
          </div>
          {row.reasons?.length > 0 && <small>{row.reasons.join(", ")}</small>}
        </div>
      ))}
    </div>
  );
}

function formatMetric(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  return Number(value).toFixed(2);
}

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "0%";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

export default App;
