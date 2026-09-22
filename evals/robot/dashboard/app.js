const ASSET_BASE_PATH = window.ROBOT_EVAL_CONFIG?.assetBasePath || "/artifacts/robot-eval";
const DATA_PATH = `${ASSET_BASE_PATH}/results/dashboard-data.json`;

let dashboardData = null;
let selectedTrialId = null;
let selectedTaskKey = null;
let selectedModelKey = null;
let publicVersionId = null;

const els = {
  generatedAt: document.getElementById("generatedAt"),
  taskFilter: document.getElementById("taskFilter"),
  modelFilter: document.getElementById("modelFilter"),
  runCount: document.getElementById("runCount"),
  runList: document.getElementById("runList"),
  detailCard: document.getElementById("detailCard"),
};

function normalizeArtifactPath(path) {
  if (typeof path !== "string" || !path) return path;
  if (/^(https?:)?\/\//.test(path) || path.startsWith("/")) return path;
  if (path.startsWith("../results/")) return `${ASSET_BASE_PATH}/results/${path.slice("../results/".length)}`;
  if (path.startsWith("../tasks/")) return `${ASSET_BASE_PATH}/tasks/${path.slice("../tasks/".length)}`;
  if (path.startsWith("../golden_results/")) return `${ASSET_BASE_PATH}/golden_results/${path.slice("../golden_results/".length)}`;
  return path;
}

function rewriteArtifactPaths(value) {
  if (Array.isArray(value)) return value.map(rewriteArtifactPaths);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewriteArtifactPaths(item)]));
  }
  return normalizeArtifactPath(value);
}

function pct(value) { return `${(value * 100).toFixed(1)}%`; }
function fmtScore(value) { return value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(3); }
function fmtCalc(value) { return value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(4); }
function runStatusReason(run) { const error = run?.grader_info?.error; return typeof error === "string" && error ? error : ""; }
function versionSortKey(versionId) { return typeof versionId === "string" && /^v\d+$/.test(versionId) ? Number.parseInt(versionId.slice(1), 10) : Number.MAX_SAFE_INTEGER; }
function escapeHtml(value) { return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;"); }
function hasVideo(run) { return Boolean(run?.artifacts?.videos?.length); }
function hasNonEmptyObject(value) { return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length); }
function renderNamedLink(path, label) { return path ? `<a href="${path}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>` : "<span class='muted'>none</span>"; }
function goldenForTask(task) { return dashboardData?.golden_catalog?.[task] || null; }
function taskDisplayName(runOrMeta) {
  return runOrMeta?.task_meta?.display_name || runOrMeta?.display_name || runOrMeta?.task || "";
}

function applyUrlState() {
  const params = new URLSearchParams(window.location.search);
  selectedTaskKey = params.get("task") || null;
  selectedModelKey = params.get("model") || null;
  selectedTrialId = params.get("trial") || null;
}

function buildUrl() {
  const params = new URLSearchParams();
  if (selectedTaskKey) params.set("task", selectedTaskKey);
  if (selectedModelKey) params.set("model", selectedModelKey);
  if (selectedTrialId) params.set("trial", selectedTrialId);
  const query = params.toString();
  return query ? `${window.location.pathname}?${query}` : window.location.pathname;
}

function syncUrl(mode = "replace") {
  const url = buildUrl();
  if (mode === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(null, "", url);
}

function fillSelect(select, values, label = "All", preferred = null) {
  const current = select.value || preferred || "all";
  select.innerHTML = [`<option value="all">${label}</option>`].concat(values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)).join("");
  select.value = values.includes(current) ? current : "all";
}

function currentFilters() {
  return {
    task: els.taskFilter.value,
    model: els.modelFilter.value,
  };
}

function filteredRuns() {
  const filters = currentFilters();
  return dashboardData.runs.filter((run) => {
    if (publicVersionId && run.version !== publicVersionId) return false;
    if (filters.task !== "all" && run.task !== filters.task) return false;
    if (filters.model !== "all" && run.model !== filters.model) return false;
    return true;
  });
}

function taskKeyOf(run) { return `${run.task}`; }
function taskModelKeyOf(run) { return `${run.task}::${run.provider}::${run.model || "unknown"}`; }

function groupedTasksAndModels(runs) {
  const groups = new Map();
  for (const run of runs) {
    const taskKey = taskKeyOf(run);
    if (!groups.has(taskKey)) groups.set(taskKey, { key: taskKey, task: run.task, display_name: taskDisplayName(run), runs: [], models: new Map(), latestTrialId: run.trial_id });
    const taskGroup = groups.get(taskKey);
    taskGroup.runs.push(run);
    if (run.trial_id > taskGroup.latestTrialId) taskGroup.latestTrialId = run.trial_id;
    const modelLabel = `${run.provider} / ${run.model || "unknown"}`;
    const modelKey = taskModelKeyOf(run);
    if (!taskGroup.models.has(modelKey)) {
      taskGroup.models.set(modelKey, { key: modelKey, task: run.task, provider: run.provider, model: run.model || "unknown", label: modelLabel, runs: [], latestTrialId: run.trial_id });
    }
    const modelGroup = taskGroup.models.get(modelKey);
    modelGroup.runs.push(run);
    if (run.trial_id > modelGroup.latestTrialId) modelGroup.latestTrialId = run.trial_id;
  }
  return [...groups.values()].map((group) => ({
    ...group,
    runs: group.runs.slice().sort((a, b) => b.trial_id.localeCompare(a.trial_id)),
    models: [...group.models.values()].map((modelGroup) => ({
      ...modelGroup,
      runs: modelGroup.runs.slice().sort((a, b) => b.trial_id.localeCompare(a.trial_id)),
    })).sort((a, b) => b.latestTrialId.localeCompare(a.latestTrialId)),
  })).sort((a, b) => b.latestTrialId.localeCompare(a.latestTrialId));
}

function renderRunList() {
  const runs = filteredRuns();
  const taskGroups = groupedTasksAndModels(runs);
  els.runCount.textContent = `${taskGroups.length} tasks`;
  if (!runs.length) {
    els.runList.innerHTML = `<div class="muted">No runs match the current filters.</div>`;
    els.detailCard.innerHTML = `<div class="emptyState"><h2>No matching results</h2><p>Adjust the filters to inspect a task or model slice.</p></div>`;
    return;
  }
  if (!taskGroups.some((group) => group.key === selectedTaskKey)) {
    selectedTaskKey = taskGroups[0].key;
    selectedModelKey = null;
  }
  if (selectedModelKey && !taskGroups.some((group) => group.models.some((modelGroup) => modelGroup.key === selectedModelKey))) {
    selectedModelKey = null;
  }
  els.runList.innerHTML = taskGroups.map((group) => {
    const successes = group.runs.filter((run) => run.success).length;
    const passRate = group.runs.length ? pct(successes / group.runs.length) : "0.0%";
    const scores = group.runs.map((run) => run.grader_info?.progress_score).filter((value) => typeof value === "number");
    const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
    return `<section class="taskNavGroup"><button class="taskNavItem ${group.key === selectedTaskKey && !selectedModelKey ? "active" : ""}" data-task-key="${escapeHtml(group.key)}"><div class="taskNavTop"><strong>${escapeHtml(group.display_name || group.task)}</strong></div><div class="runGroupMeta">${successes}/${group.runs.length} pass - ${passRate} - avg score ${fmtScore(avgScore)}</div></button><div class="taskModelList">${group.models.map((modelGroup) => { const modelSuccesses = modelGroup.runs.filter((run) => run.success).length; const modelScores = modelGroup.runs.map((run) => run.grader_info?.progress_score).filter((value) => typeof value === "number"); const modelAvgScore = modelScores.length ? modelScores.reduce((a, b) => a + b, 0) / modelScores.length : null; return `<button class="modelNavItem ${modelGroup.key === selectedModelKey ? "active" : ""}" data-task-key="${escapeHtml(group.key)}" data-model-key="${escapeHtml(modelGroup.key)}"><div><strong>${escapeHtml(modelGroup.label)}</strong></div><div class="runMeta">${modelSuccesses}/${modelGroup.runs.length} pass - ${fmtScore(modelAvgScore)} avg${modelGroup.runs.some((run) => hasVideo(run)) ? " - video" : ""}</div></button>`; }).join("")}</div></section>`;
  }).join("");
  for (const node of els.runList.querySelectorAll(".taskNavItem[data-task-key]")) node.addEventListener("click", () => {
    selectedTaskKey = node.dataset.taskKey;
    selectedModelKey = null;
    renderRunList();
    renderDetail();
    syncUrl("push");
  });
  for (const node of els.runList.querySelectorAll(".modelNavItem[data-model-key]")) node.addEventListener("click", () => {
    selectedTaskKey = node.dataset.taskKey;
    selectedModelKey = node.dataset.modelKey;
    renderRunList();
    renderDetail();
    syncUrl("push");
  });
}

function renderStatusPanel(run) {
  const reason = runStatusReason(run);
  return reason ? `<div class="panel"><h3>Status</h3><div class="muted">${escapeHtml(reason)}</div></div>` : "";
}

function renderScoreFormulaPanel(taskMeta) {
  return taskMeta?.score_formula ? `<div class="panel"><h3>Score Formula</h3><pre>${escapeHtml(taskMeta.score_formula)}</pre></div>` : "";
}

function metricEntries(graderInfo) {
  return Object.entries(graderInfo || {}).filter(([, value]) => {
    if (value === null || value === undefined) return false;
    if (typeof value === "number" || typeof value === "boolean" || typeof value === "string") return true;
    if (Array.isArray(value) && value.length <= 6 && value.every((item) => typeof item === "number" || typeof item === "boolean" || typeof item === "string")) return true;
    return false;
  }).sort(([a], [b]) => a.localeCompare(b));
}

function renderMetricPanel(graderInfo) {
  const entries = metricEntries(graderInfo);
  if (!entries.length) return `<div class="panel"><h3>Metrics</h3><div class="muted">No scalar metrics available for this run.</div></div>`;
  return `<div class="panel"><h3>Metrics</h3><div class="kv">${entries.map(([key, value]) => `<div class="kvRow"><div class="kvKey">${escapeHtml(key)}</div><div class="metricCell">${escapeHtml(Array.isArray(value) ? JSON.stringify(value) : String(value))}</div></div>`).join("")}</div></div>`;
}

function getScoreBreakdown(run) {
  const info = run?.grader_info || {};
  if (!(Array.isArray(info.score_components) && info.score_components.length)) return null;
  const rows = info.score_components.map((component) => ({ label: component.name, value: Number(component.value ?? 0), weight: Number(component.weight ?? 0), contribution: Number(component.value ?? 0) * Number(component.weight ?? 0) }));
  const totalWeight = rows.reduce((sum, row) => sum + row.weight, 0);
  return { type: info.score_method === "weighted_subscores" ? "Weighted subscores" : "Score components", rows, formula: rows.map((row) => `${fmtCalc(row.weight)}*${row.label}`).join(" + "), score: totalWeight ? rows.reduce((sum, row) => sum + row.contribution, 0) / totalWeight : 0 };
}

function renderScoreBreakdownPanel(run) {
  const breakdown = getScoreBreakdown(run);
  if (!breakdown) return "";
  return `<div class="panel"><h3>Score Breakdown</h3><div class="scoreBreakdownMeta">Type: ${escapeHtml(breakdown.type)}</div><div class="scoreBreakdownMeta">Formula: <code>${escapeHtml(breakdown.formula)}</code></div><div class="scoreBreakdownMeta">Final score: <strong>${fmtCalc(breakdown.score)}</strong></div><div class="scoreBreakdownTable scoreBreakdownWeighted"><div class="scoreBreakdownHead">Component</div><div class="scoreBreakdownHead">Value</div><div class="scoreBreakdownHead">Weight</div><div class="scoreBreakdownHead">Weighted term</div>${breakdown.rows.map((row) => `<div class="scoreBreakdownCell scoreBreakdownLabel">${escapeHtml(row.label)}</div><div class="scoreBreakdownCell">${fmtCalc(row.value)}</div><div class="scoreBreakdownCell">${fmtCalc(row.weight)}</div><div class="scoreBreakdownCell">${fmtCalc(row.contribution)}</div>`).join("")}</div></div>`;
}

function renderArtifacts(artifacts) {
  const sections = [];
  if (artifacts.videos.length) sections.push(`<div class="panel"><h3>Video</h3><div class="mediaGrid">${artifacts.videos.map((item) => `<div><video controls preload="metadata" src="${item.url || item.path}"></video><div class="fileList"><a href="${item.url || item.path}" target="_blank" rel="noreferrer">${escapeHtml(item.name)}</a></div></div>`).join("")}</div></div>`);
  if (artifacts.images.length) sections.push(`<div class="panel"><h3>Images</h3><div class="mediaGrid">${artifacts.images.map((item) => `<a href="${item.url || item.path}" target="_blank" rel="noreferrer"><img src="${item.url || item.path}" alt="${escapeHtml(item.name)}" /><div class="fileList">${escapeHtml(item.name)}</div></a>`).join("")}</div></div>`);
  if (artifacts.texts.length) sections.push(`<div class="panel"><h3>Files</h3><div class="fileList">${artifacts.texts.map((item) => `<div><a href="${item.url || item.path}" target="_blank" rel="noreferrer">${escapeHtml(item.name)}</a></div>`).join("")}</div></div>`);
  return sections.join("");
}

function renderAttempts(attempts) {
  if (!attempts?.length) return "";
  return `<div class="panel"><h3>Attempts</h3><div class="attemptList">${attempts.map((attempt) => `<div class="attemptCard"><div class="attemptHeader"><strong>${escapeHtml(attempt.label || `attempt ${attempt.attempt_index}`)}</strong><span class="muted">#${attempt.attempt_index}</span></div><div class="fileList"><div><a href="${attempt.path}" target="_blank" rel="noreferrer">open folder</a></div></div></div>`).join("")}</div></div>`;
}

function renderEvents(events) {
  if (!events.length) return `<div class="muted">No parsed stream events available.</div>`;
  return `<div class="eventList">${events.map((event) => {
    if (event.kind === "gap") return `<div class="event event-gap"><div class="eventKind">gap</div><div>${escapeHtml(event.text)}</div></div>`;
    if (event.kind === "user") {
      const preview = (event.text || "").split("\n").find((line) => line.trim()) || "";
      return `<details class="event event-user"><summary><span class="eventSummaryMain"><span class="eventKind">${escapeHtml(event.label || "prompt")}</span><span class="eventPreviewLine">${escapeHtml(preview)}</span></span><span class="eventMeta">${escapeHtml(event.timestamp || "")}</span></summary><pre>${escapeHtml(event.text || "")}</pre></details>`;
    }
    if (event.kind === "message") {
      const preview = (event.text || "").split("\n").find((line) => line.trim()) || "";
      return `<details class="event event-message" open><summary><span class="eventSummaryMain"><span class="eventKind">message</span><span class="eventPreviewLine">${escapeHtml(preview)}</span></span><span class="eventMeta">${escapeHtml(event.timestamp || "")}</span></summary><pre>${escapeHtml(event.text || "")}</pre></details>`;
    }
    if (event.kind === "usage") return `<details class="event event-usage"><summary><span class="eventSummaryMain"><span class="eventKind">usage</span><span class="eventPreviewLine">token and cost summary</span></span><span class="eventMeta">${escapeHtml(event.timestamp || "")}</span></summary><pre>${escapeHtml(JSON.stringify(event.usage, null, 2))}</pre></details>`;
    const commandPreview = (event.command || "").split("\n")[0];
    const firstOutputLine = event.output_preview ? event.output_preview.split("\n")[0] : "";
    return `<details class="event event-command"><summary><span class="eventSummaryMain"><span class="eventKind">command</span><span class="eventPreviewLine">${escapeHtml(commandPreview)}</span></span><span class="eventMeta">${escapeHtml(event.timestamp || "")}${event.timestamp ? " - " : ""}exit=${event.exit_code ?? "?"} - ${escapeHtml(event.status || "unknown")}</span></summary><pre>${escapeHtml(event.command || "")}</pre>${firstOutputLine ? `<div class="eventPreview">${escapeHtml(firstOutputLine)}</div>` : ""}${event.output_preview ? `<pre>${escapeHtml(event.output_preview)}</pre>` : ""}</details>`;
  }).join("")}</div>`;
}

function averageScore(runs) {
  const values = runs.map((run) => run.grader_info?.progress_score).filter((value) => typeof value === "number");
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function computeFailureCounts(runs) {
  const counts = new Map();
  for (const run of runs.filter((item) => !item.success)) {
    const key = run.failure_category || "other_failure";
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function computeTaskPassAtK(modelGroups) {
  const maxAttempts = Math.max(...modelGroups.map((group) => group.runs.length), 0);
  const points = [];
  for (let k = 1; k <= maxAttempts; k += 1) {
    let hits = 0;
    for (const group of modelGroups) {
      const ordered = group.runs.slice().sort((a, b) => a.trial_id.localeCompare(b.trial_id));
      if (ordered.slice(0, k).some((run) => run.success)) hits += 1;
    }
    points.push({ k, value: modelGroups.length ? hits / modelGroups.length : 0 });
  }
  return points;
}

function representativeRuns(modelRuns) {
  const sorted = modelRuns.slice().sort((a, b) => (b.grader_info?.progress_score ?? -1) - (a.grader_info?.progress_score ?? -1));
  if (!sorted.length) return { best: null, median: null, worst: null };
  return {
    best: sorted[0],
    median: sorted[Math.floor((sorted.length - 1) / 2)],
    worst: sorted[sorted.length - 1],
  };
}

function renderTaskOverview(taskRuns) {
  const taskMeta = taskRuns[0]?.task_meta || {};
  const golden = goldenForTask(taskRuns[0]?.task);
  const modelGroups = groupByTaskModel(taskRuns);
  const goldenPanel = golden ? `<div class="panel"><h3>Example Valid Solution</h3><div class="kv"><div class="kvRow"><div class="kvKey">Status</div><div><span class="badge ${golden.success ? "pass" : "fail"}">${golden.success ? "PASS" : "FAIL"}</span></div></div><div class="kvRow"><div class="kvKey">Run</div><div>${escapeHtml(golden.run_id || "n/a")}</div></div><div class="kvRow"><div class="kvKey">Score</div><div>${fmtScore(golden.progress_score)}</div></div><div class="kvRow"><div class="kvKey">Script</div><div>${renderNamedLink(golden.script_path, "Open golden script")}</div></div><div class="kvRow"><div class="kvKey">Result</div><div>${renderNamedLink(golden.result_path, "Open result.json")}</div></div></div>${renderArtifacts(golden.artifacts || { images: [], videos: [], texts: [] })}</div>` : `<div class="panel"><h3>Example Valid Solution</h3><div class="muted">No canonical example solution is indexed for this task yet.</div></div>`;
  return `<div class="detailHeader"><div><h2>${escapeHtml(taskDisplayName(taskRuns[0]))}</h2><div class="detailMeta">${taskRuns.length} attempts across ${modelGroups.length} models</div></div></div><div class="detailGrid"><div class="panel"><h3>Task Overview</h3><div class="kv"><div class="kvRow"><div class="kvKey">Description</div><div>${escapeHtml(taskMeta.description || "No task summary indexed.")}</div></div><div class="kvRow"><div class="kvKey">README</div><div>${renderNamedLink(taskMeta.readme_path, "Open README")}</div></div><div class="kvRow"><div class="kvKey">Simulator</div><div>${renderNamedLink(taskMeta.sim_path, "Open sim.py")}</div></div><div class="kvRow"><div class="kvKey">Grader</div><div>${renderNamedLink(taskMeta.grader_path, "Open grader.py")}</div></div><div class="kvRow"><div class="kvKey">Avg score</div><div>${fmtScore(averageScore(taskRuns))}</div></div><div class="kvRow"><div class="kvKey">Pass rate</div><div>${pct(taskRuns.filter((run) => run.success).length / taskRuns.length)}</div></div></div></div>${goldenPanel}</div><div class="panel"><h3>Models</h3><div class="sliceList modelCardGrid">${modelGroups.map((group) => { const successes = group.runs.filter((run) => run.success).length; return `<button class="sliceCard modelSliceCard" data-model-key="${escapeHtml(group.key)}"><div class="modelSliceTop"><strong>${escapeHtml(group.model)}</strong><span class="badge ${successes ? "pass" : "fail"}">${successes}/${group.runs.length}</span></div><div class="runMeta">avg score ${fmtScore(averageScore(group.runs))}</div><div class="runMeta">${successes}/${group.runs.length} pass</div><div class="runMeta">${group.runs.filter((run) => hasVideo(run)).length}/${group.runs.length} with video</div></button>`; }).join("")}</div></div>${renderScoreFormulaPanel(taskMeta)}`;
}

function renderModelOverview(modelRuns) {
  const picks = representativeRuns(modelRuns);
  const bestRun = picks.best;
  if (!selectedTrialId || !modelRuns.some((run) => run.trial_id === selectedTrialId)) {
    selectedTrialId = (modelRuns.find((run) => hasVideo(run)) || bestRun || modelRuns[0]).trial_id;
  }
  const run = modelRuns.find((item) => item.trial_id === selectedTrialId) || modelRuns[0];
  const graderInfo = JSON.stringify(run.grader_info, null, 2);
  const costInfo = JSON.stringify(run.cost_info, null, 2);
  const rawLogLinks = Object.entries(run.raw_logs).filter(([, path]) => path).map(([label, path]) => `<div><a href="${path}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a></div>`).join("");
  const taskMeta = run.task_meta || {};
  const promptText = run.prompt_text || "";
  const promptKind = run.prompt_source === "eval_prompt" ? "prompt" : "task";
  const timelineEvents = promptText ? [{ kind: "user", text: promptText, timestamp: promptKind, label: promptKind }, ...run.events] : run.events;
  return `<div class="detailHeader"><div><h2>${escapeHtml(taskDisplayName(run))}</h2><div class="detailMeta">${escapeHtml(run.model || "unknown")}</div></div></div><div class="detailGrid"><div class="panel"><h3>Model Slice</h3><div class="kv"><div class="kvRow"><div class="kvKey">Attempts</div><div>${modelRuns.length}</div></div><div class="kvRow"><div class="kvKey">Pass rate</div><div>${pct(modelRuns.filter((item) => item.success).length / modelRuns.length)}</div></div><div class="kvRow"><div class="kvKey">Avg score</div><div>${fmtScore(averageScore(modelRuns))}</div></div><div class="kvRow"><div class="kvKey">Best run</div><div>${escapeHtml(bestRun.trial_id)} - ${fmtScore(bestRun.grader_info?.progress_score)}</div></div></div><div class="representativeStrip"><button class="repChip" data-trial-id="${picks.best?.trial_id || ""}">Best: ${picks.best ? fmtScore(picks.best.grader_info?.progress_score) : "n/a"}</button><button class="repChip" data-trial-id="${picks.median?.trial_id || ""}">Median: ${picks.median ? fmtScore(picks.median.grader_info?.progress_score) : "n/a"}</button><button class="repChip" data-trial-id="${picks.worst?.trial_id || ""}">Worst: ${picks.worst ? fmtScore(picks.worst.grader_info?.progress_score) : "n/a"}</button></div><div class="attemptStrip">${modelRuns.slice().sort((a, b) => a.trial_id.localeCompare(b.trial_id)).map((item, index) => `<button class="attemptChip ${item.trial_id === run.trial_id ? "active" : ""}" data-trial-id="${item.trial_id}">#${index + 1} ${item.success ? "PASS" : "FAIL"} ${fmtScore(item.grader_info?.progress_score)}</button>`).join("")}</div></div><div class="panel"><h3>Attempt Detail</h3><div class="kv"><div class="kvRow"><div class="kvKey">Trial</div><div>${escapeHtml(run.trial_id)}</div></div><div class="kvRow"><div class="kvKey">Budget</div><div>${run.budget_seconds}s</div></div><div class="kvRow"><div class="kvKey">Elapsed</div><div>${run.elapsed_seconds}s</div></div><div class="kvRow"><div class="kvKey">Progress score</div><div>${fmtScore(run.grader_info?.progress_score)}</div></div><div class="kvRow"><div class="kvKey">Failure bucket</div><div>${escapeHtml(run.failure_category || "none")}</div></div><div class="kvRow"><div class="kvKey">Timed out</div><div>${run.timed_out ? "yes" : "no"}</div></div><div class="kvRow"><div class="kvKey">Raw logs</div><div class="fileList">${rawLogLinks || "<span class='muted'>none</span>"}</div></div></div></div>${renderArtifacts(run.artifacts)}${renderStatusPanel(run)}${renderScoreBreakdownPanel(run)}${renderMetricPanel(run.grader_info)}${renderScoreFormulaPanel(taskMeta)}<details class="panel rawPanel"><summary><h3>Grader JSON</h3></summary><pre>${escapeHtml(graderInfo)}</pre></details>${hasNonEmptyObject(run.cost_info) ? `<details class="panel rawPanel"><summary><h3>Cost info</h3></summary><pre>${escapeHtml(costInfo)}</pre></details>` : ""}</div><div class="panel" style="margin-top: 18px;"><h3>Timeline</h3>${renderEvents(timelineEvents)}</div>`;
}

function renderDetail() {
  const runs = filteredRuns();
  if (!runs.length) return;
  const taskRuns = runs.filter((run) => taskKeyOf(run) === selectedTaskKey);
  if (!taskRuns.length) {
    els.detailCard.innerHTML = `<div class="emptyState"><h2>No task selected</h2><p>Choose a task from the sidebar.</p></div>`;
    return;
  }
  if (!selectedModelKey) {
    els.detailCard.innerHTML = renderTaskOverview(taskRuns);
    for (const node of els.detailCard.querySelectorAll(".sliceCard[data-model-key]")) node.addEventListener("click", () => {
      selectedModelKey = node.dataset.modelKey;
      renderRunList();
      renderDetail();
      syncUrl("push");
    });
    return;
  }
  const modelRuns = taskRuns.filter((run) => taskModelKeyOf(run) === selectedModelKey);
  if (!modelRuns.length) {
    selectedModelKey = null;
    renderDetail();
    return;
  }
  els.detailCard.innerHTML = renderModelOverview(modelRuns);
  for (const node of els.detailCard.querySelectorAll(".attemptChip[data-trial-id]")) node.addEventListener("click", () => {
    selectedTrialId = node.dataset.trialId;
    renderDetail();
    syncUrl("push");
  });
  for (const node of els.detailCard.querySelectorAll(".repChip[data-trial-id]")) node.addEventListener("click", () => {
    if (!node.dataset.trialId) return;
    selectedTrialId = node.dataset.trialId;
    renderDetail();
    syncUrl("push");
  });
}

function groupByTaskModel(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = taskModelKeyOf(run);
    if (!groups.has(key)) groups.set(key, { key, task: run.task, provider: run.provider, model: run.model || "unknown", taskLabel: run.task, runs: [] });
    groups.get(key).runs.push(run);
  }
  return [...groups.values()].sort((a, b) => a.taskLabel.localeCompare(b.taskLabel) || a.model.localeCompare(b.model));
}

async function loadDashboard() {
  const response = await fetch(`${DATA_PATH}?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${DATA_PATH}`);
  dashboardData = rewriteArtifactPaths(await response.json());
}

function hydrateFilters() {
  const versionIds = (dashboardData.versions || []).map((version) => version.id).sort((a, b) => versionSortKey(a) - versionSortKey(b) || a.localeCompare(b));
  publicVersionId = versionIds.length ? versionIds[versionIds.length - 1] : null;
  const publicRuns = dashboardData.runs.filter((run) => !publicVersionId || run.version === publicVersionId);
  fillSelect(els.taskFilter, [...new Set(publicRuns.map((run) => run.task))].sort());
  fillSelect(els.modelFilter, [...new Set(publicRuns.map((run) => run.model).filter(Boolean))].sort());
  for (const select of [els.taskFilter, els.modelFilter]) {
    select.addEventListener("change", () => { renderRunList(); renderDetail(); syncUrl("push"); });
  }
}

function renderGeneratedAt() { els.generatedAt.textContent = `generated ${dashboardData.generated_at}`; }

async function main() {
  try { await loadDashboard(); } catch {
    els.generatedAt.textContent = "No dashboard data found";
    els.detailCard.innerHTML = `<div class="emptyState"><h2>Dashboard data missing</h2><p>Run <code>python dashboard/build_dashboard.py</code> from the repo root, then serve the repo over HTTP and open <code>/dashboard/</code>.</p></div>`;
    return;
  }
  renderGeneratedAt();
  hydrateFilters();
  applyUrlState();
  window.addEventListener("popstate", () => {
    applyUrlState();
    renderRunList();
    renderDetail();
  });
  renderRunList();
  renderDetail();
  syncUrl("replace");
}

main();
