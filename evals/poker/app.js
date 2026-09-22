const data = await fetch("./data/site-data.json").then((res) => res.json());

const heroEyebrow = document.getElementById("heroEyebrow");
const poolMeta = document.getElementById("poolMeta");
const methodologySection = document.getElementById("methodologySection");
const summaryTable = document.getElementById("summaryTable");
const distributionChart = document.getElementById("distributionChart");
const chartTooltip = document.getElementById("chartTooltip");
const attemptTable = document.getElementById("attemptTable");
const attemptSelect = document.getElementById("attemptSelect");
const eventTypeSelect = document.getElementById("eventTypeSelect");
const attemptMiniMap = document.getElementById("attemptMiniMap");
const trajectoryExplorer = document.getElementById("trajectoryExplorer");
const bubbleStream = document.getElementById("bubbleStream");
const pairMatrix = document.getElementById("pairMatrix");
const qualitativeAnalysis = document.getElementById("qualitativeAnalysis");
const replaySelect = document.getElementById("replaySelect");
const handSelect = document.getElementById("handSelect");
const topSeat = document.getElementById("topSeat");
const bottomSeat = document.getElementById("bottomSeat");
const boardState = document.getElementById("boardState");
const replayLog = document.getElementById("replayLog");
const prevStep = document.getElementById("prevStep");
const nextStep = document.getElementById("nextStep");
const stepBadge = document.getElementById("stepBadge");

if (chartTooltip && chartTooltip.parentElement !== document.body) {
  document.body.appendChild(chartTooltip);
}

let currentAttemptId = null;
let currentReplayIndex = 0;
let currentHandIndex = 0;
let currentStepIndex = 0;
const collapsedState = new Map();

const attemptMap = new Map(data.attempt_cards.map((attempt) => [attempt.id, attempt]));

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function normalizeDisplayText(text) {
  return String(text)
    .replaceAll("â€™", "’")
    .replaceAll("â€œ", "“")
    .replaceAll("â€", "”")
    .replaceAll("â€“", "–")
    .replaceAll("â€”", "—")
    .replaceAll("â€¦", "…")
    .replaceAll("Â·", "·")
    .replaceAll("ï»¿", "")
    .replaceAll("ï¿½", "—");
}

function selectAttempt(attemptId) {
  if (!attemptId || !attemptMap.has(attemptId)) return;
  currentAttemptId = attemptId;
  attemptSelect.value = attemptId;
  renderAttempts();
  renderTranscript();
  trajectoryExplorer?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function highlightPythonCode(text) {
  let html = escapeHtml(text);
  html = html.replace(/(&quot;.*?&quot;|&#39;.*?&#39;)/g, '<span class="tok-string">$1</span>');
  html = html.replace(/(^|\s)(def|class|if|elif|else|for|while|return|import|from|try|except|with|in|and|or|not|True|False|None)(?=\s|:|\()/gm, '$1<span class="tok-keyword">$2</span>');
  html = html.replace(/(^|\s)(self)(?=[\.\s,\)\]:])/gm, '$1<span class="tok-self">$2</span>');
  html = html.replace(/(^|\s)(\d+(?:\.\d+)?)(?=\s|,|\)|\]|$)/gm, '$1<span class="tok-number">$2</span>');
  html = html.replace(/(#.*)$/gm, '<span class="tok-comment">$1</span>');
  return html;
}

function renderHero() {
  if (heroEyebrow) {
    heroEyebrow.textContent = "LLM Poker Benchmark";
  }
  const heroSubtitle = document.querySelector(".hero-subtitle");
  if (heroSubtitle) {
    heroSubtitle.textContent =
      "Models are asked to write a heads-up fixed-limit Texas hold'em bot under a fixed coding budget. Each generated bot is evaluated on a public development benchmark, and final rankings are checked separately on a hidden weighted test pool.";
  }
}

function renderSummary() {
  const meta = [];
  if (data.pool?.hands) meta.push(`${data.pool.hands} hands per opponent`);
  if (data.pool?.seed_list?.length) meta.push(`${data.pool.seed_list.length} seeds`);
  poolMeta.textContent = `Public development benchmark. Higher is better. Models are ranked by the median score across five attempts, while the best attempt shows peak performance. ${meta.join(" · ")}`;

  const rows = [...data.model_summaries].sort((a, b) => b.median_mbb_per_hand - a.median_mbb_per_hand);
  const allAttemptScores = rows.flatMap((row) => row.attempt_scores || []).filter((value) => typeof value === "number");
  const maxAbsAttemptScore = Math.max(1, ...allAttemptScores.map((value) => Math.abs(value)));
  summaryTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Model</th>
          <th>Median</th>
          <th>Mean</th>
          <th>Best</th>
          <th>Attempt Distribution</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row, index) => `
          <tr>
            <td>${index + 1}</td>
            <td>${row.model}</td>
            <td>${formatNumber(row.median_mbb_per_hand)}</td>
            <td>${formatNumber(row.mean_mbb_per_hand)}</td>
            <td>${formatNumber(row.best_mbb_per_hand)}</td>
            <td>
              <div class="attempt-score-grid">
                ${(row.attempt_scores || [])
                  .map((score) => {
                    const intensity = Math.min(1, Math.abs(score) / maxAbsAttemptScore);
                    const alpha = 0.14 + intensity * 0.44;
                    const borderAlpha = 0.16 + intensity * 0.42;
                    const textColor = score >= 0 ? "var(--good)" : "var(--bad)";
                    const background = score >= 0
                      ? `rgba(30, 126, 86, ${alpha.toFixed(3)})`
                      : `rgba(154, 47, 47, ${alpha.toFixed(3)})`;
                    const border = score >= 0
                      ? `rgba(30, 126, 86, ${borderAlpha.toFixed(3)})`
                      : `rgba(154, 47, 47, ${borderAlpha.toFixed(3)})`;
                    return `<span class="attempt-score-cell" style="background:${background};border-color:${border};color:${textColor}" title="${formatNumber(score)} mbb/hand">${formatNumber(score)}</span>`;
                  })
                  .join("")}
              </div>
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function initCollapsiblePanels() {
  for (const panel of document.querySelectorAll(".collapsible-panel")) {
    const head = panel.querySelector(".panel-head");
    const toggle = panel.querySelector(".panel-toggle");
    const collapsed = panel.dataset.collapsed === "true";
    if (collapsed) {
      panel.classList.add("is-collapsed");
    }
    if (toggle) {
      toggle.textContent = panel.classList.contains("is-collapsed") ? "Expand" : "Collapse";
      toggle.setAttribute("aria-expanded", panel.classList.contains("is-collapsed") ? "false" : "true");
    }
    const apply = () => {
      const isCollapsed = panel.classList.toggle("is-collapsed");
      panel.dataset.collapsed = isCollapsed ? "true" : "false";
      if (toggle) {
        toggle.textContent = isCollapsed ? "Expand" : "Collapse";
        toggle.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
      }
    };
    if (toggle) {
      toggle.addEventListener("click", (event) => {
        event.stopPropagation();
        apply();
      });
    }
    if (head) {
      head.addEventListener("click", (event) => {
        const target = event.target;
        if (target instanceof HTMLElement && target.closest("select, button, input, textarea, a, label")) {
          return;
        }
        apply();
      });
    }
  }
}

function renderDistributionChart() {
  const rows = [...data.model_summaries].sort((a, b) => b.median_mbb_per_hand - a.median_mbb_per_hand);
  const values = rows.flatMap((row) => row.attempt_scores || []).filter((value) => typeof value === "number");
  if (!distributionChart) return;
  if (!values.length) {
    distributionChart.innerHTML = `<p class="muted">No attempt distribution data available.</p>`;
    return;
  }

  const rawMinScore = Math.min(...values);
  const rawMaxScore = Math.max(...values);
  const rawSpan = Math.max(1, rawMaxScore - rawMinScore);
  const margin = rawSpan * 0.06;
  const minScore = rawMinScore - margin;
  const maxScore = rawMaxScore + margin;
  const span = Math.max(1, maxScore - minScore);
  const widthPct = (value) => ((value - minScore) / span) * 100;
  const zeroPct = Math.min(100, Math.max(0, widthPct(0)));
  const tickCount = 5;
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const ratio = index / (tickCount - 1);
    return { value: minScore + ratio * span, ratio };
  });

  distributionChart.innerHTML = `
      <div class="distribution-chart">
        <div class="distribution-scale">
          <span class="distribution-axis-label" aria-hidden="true"></span>
          <span class="distribution-axis-title">Benchmark score (mbb/hand)</span>
          <span class="distribution-axis-label" aria-hidden="true"></span>
        </div>
        <div class="distribution-tick-row">
          <div class="distribution-tick-spacer" aria-hidden="true"></div>
          <div class="distribution-ticks">
            ${ticks.map((tick, index) => `
              <span
                class="distribution-tick ${index === 0 ? "is-start" : index === tickCount - 1 ? "is-end" : ""}"
                style="left:${tick.ratio * 100}%"
              >${formatNumber(tick.value)}</span>
            `).join("")}
          </div>
        </div>
        ${rows
        .map((row) => {
          const attemptScores = [...(row.attempt_scores || [])].sort((a, b) => a - b);
          const median = row.median_mbb_per_hand;
          const low = Math.min(...attemptScores);
          const high = Math.max(...attemptScores);
          return `
            <div class="distribution-row">
              <div class="distribution-model">
                <strong>${row.model}</strong>
                <span class="muted">median ${formatNumber(median)}</span>
              </div>
              <div class="distribution-track">
                <div class="distribution-zero" style="left:${zeroPct}%"></div>
                <div class="distribution-range" style="left:${widthPct(low)}%; width:${Math.max(1.2, widthPct(high) - widthPct(low))}%"></div>
                ${attemptScores
                  .map((score, scoreIndex) => {
                    const classes = ["distribution-dot"];
                    if (Math.abs(score - median) < 1e-9) classes.push("median");
                    classes.push(score >= 0 ? "positive" : "negative");
                      return `
                        <button
                          type="button"
                          class="${classes.join(" ")}"
                          style="left:${widthPct(score)}%"
                          data-model="${escapeHtml(row.model)}"
                          data-attempt-index="${scoreIndex + 1}"
                          data-score="${formatNumber(score)}"
                        ></button>
                      `;
                  })
                  .join("")}
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;

  for (const dot of distributionChart.querySelectorAll(".distribution-dot")) {
    const model = dot.getAttribute("data-model");
    const attemptIndex = dot.getAttribute("data-attempt-index");
    const score = dot.getAttribute("data-score");
    const tooltipText = `${model} attempt ${attemptIndex}: ${score} mbb/hand`;
    const positionTooltip = () => {
      if (!chartTooltip) return;
      const rect = dot.getBoundingClientRect();
      const tooltipWidth = chartTooltip.offsetWidth || 180;
      const tooltipHeight = chartTooltip.offsetHeight || 36;
      const margin = 8;
      let left = rect.left + rect.width / 2 - tooltipWidth / 2;
      let top = rect.top - tooltipHeight - 4;

      if (left + tooltipWidth > window.innerWidth - margin) {
        left = window.innerWidth - tooltipWidth - margin;
      }
      if (left < margin) {
        left = margin;
      }
      if (top < margin) {
        top = rect.bottom + 4;
      }
      if (top + tooltipHeight > window.innerHeight - margin) {
        top = window.innerHeight - tooltipHeight - margin;
      }

      chartTooltip.style.left = `${left}px`;
      chartTooltip.style.top = `${top}px`;
    };
    dot.addEventListener("mouseenter", () => {
      if (!chartTooltip) return;
      chartTooltip.textContent = tooltipText;
      chartTooltip.hidden = false;
      positionTooltip();
    });
    dot.addEventListener("mousemove", positionTooltip);
    dot.addEventListener("mouseleave", () => {
      if (!chartTooltip) return;
      chartTooltip.hidden = true;
    });
  }
}

function renderAttempts() {
  const rows = [...data.attempt_cards].sort((a, b) => {
    const scoreA = a.avg_mbb_per_hand || Number.NEGATIVE_INFINITY;
    const scoreB = b.avg_mbb_per_hand || Number.NEGATIVE_INFINITY;
    return scoreB - scoreA;
  });
  attemptTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Attempt</th>
          <th>Model</th>
          <th>Score</th>
          <th>CI95</th>
          <th>Outcome</th>
          <th>Messages</th>
          <th>Commands</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((attempt) => `
          <tr class="attempt-row ${attempt.id === currentAttemptId ? "active" : ""}" data-attempt-id="${escapeHtml(attempt.id)}">
            <td>${attempt.attempt}</td>
            <td>${attempt.model}</td>
            <td>${attempt.avg_mbb_per_hand === null ? "—" : formatNumber(attempt.avg_mbb_per_hand)}</td>
            <td>${attempt.ci95_low === null ? "—" : `${formatNumber(attempt.ci95_low)} to ${formatNumber(attempt.ci95_high)}`}</td>
            <td><span class="pill ${attempt.outcome === "valid" ? "valid" : attempt.outcome === "template" ? "template" : "error"}">${attempt.outcome}</span></td>
            <td>${attempt.transcript_counts.agent_message}</td>
            <td>${attempt.transcript_counts.command_execution}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  for (const row of attemptTable.querySelectorAll(".attempt-row")) {
    row.addEventListener("click", () => {
      const attemptId = row.getAttribute("data-attempt-id");
      selectAttempt(attemptId);
    });
  }
}

function renderAttemptOptions() {
  attemptSelect.innerHTML = data.attempt_cards
    .map((attempt) => `<option value="${attempt.id}">${attempt.id}</option>`)
    .join("");
  currentAttemptId = data.attempt_cards[0]?.id || null;
}

function prettyEvent(event) {
  if (event.kind === "prompt") return normalizeDisplayText(event.text);
  if (event.kind === "message") return normalizeDisplayText(event.text);
  if (event.kind === "command") return normalizeDisplayText(event.output_excerpt || event.command);
  if (event.kind === "file_change") {
    if (event.content_excerpt) {
      return normalizeDisplayText(`${event.summary}\n\n${event.content_excerpt}`);
    }
    return normalizeDisplayText(event.summary || JSON.stringify(event.changes, null, 2));
  }
  return normalizeDisplayText(event.message || "");
}

function renderMethodology() {
  if (!methodologySection) return;
  const methodology = data.methodology || {};
  const cards = methodology.cards || [];
  methodologySection.innerHTML = `
    <div class="methodology-shell">
      <p class="methodology-summary">${escapeHtml(methodology.summary || "")}</p>
      <div class="analysis-grid">
        ${cards.map((card) => `
          <article class="analysis-card">
            <h3>${escapeHtml(card.title)}</h3>
            <ul class="analysis-list">
              ${(card.items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
          </article>
        `).join("")}
      </div>
    </div>
  `;
}

function defaultCollapsed(kind) {
  return kind === "prompt" || kind === "command";
}

function bubbleStateKey(attemptId, index, kind) {
  return `${attemptId}::${index}::${kind}`;
}

function isCollapsed(attemptId, index, kind) {
  const key = bubbleStateKey(attemptId, index, kind);
  if (!collapsedState.has(key)) {
    collapsedState.set(key, defaultCollapsed(kind));
  }
  return collapsedState.get(key);
}

  function toggleCollapsed(attemptId, index, kind) {
    const key = bubbleStateKey(attemptId, index, kind);
    collapsedState.set(key, !isCollapsed(attemptId, index, kind));
    renderTranscript();
  }

function previewText(event) {
  const text = prettyEvent(event).replace(/\s+/g, " ").trim();
  if (!text) return "No content";
  return text.length > 140 ? `${text.slice(0, 140)}…` : text;
}

function renderTranscript() {
  const attempt = attemptMap.get(currentAttemptId);
  if (!attempt) return;
  const typeFilter = eventTypeSelect.value;
  const visibleEvents = attempt.trajectory.filter((event) => typeFilter === "all" || event.kind === typeFilter);
  const opponentLines = Object.entries(attempt.per_opponent || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, score]) => `<div class="muted">${name}: ${formatNumber(score)} mbb/h</div>`)
    .join("");

  attemptMiniMap.innerHTML = `
    <div class="mini-map-card">
      <div class="mini-label">Attempt</div>
      <strong>${attempt.id}</strong>
      <p class="muted">${attempt.outcome} · ${attempt.transcript_counts.prompt} prompt · ${attempt.transcript_counts.agent_message} messages · ${attempt.transcript_counts.command_execution} commands</p>
    </div>
    <div class="mini-map-card">
      <div class="mini-label">Benchmark score</div>
      <strong>${attempt.avg_mbb_per_hand === null ? "—" : `${formatNumber(attempt.avg_mbb_per_hand)} mbb/h`}</strong>
      <p class="muted">${attempt.ci95_low === null ? "No confidence interval" : `CI95 ${formatNumber(attempt.ci95_low)} to ${formatNumber(attempt.ci95_high)}`}</p>
      ${opponentLines || '<p class="muted">No opponent breakdown</p>'}
    </div>
  `;

  bubbleStream.innerHTML = "";
  for (const [index, event] of visibleEvents.entries()) {
    const bubble = document.createElement("article");
    const collapsed = isCollapsed(currentAttemptId, index, event.kind);
      bubble.className = `bubble ${event.kind} ${collapsed ? "collapsed" : "expanded"}`;
      bubble.innerHTML = `
        <div class="bubble-head">
          <span>${event.kind === "prompt" ? "benchmark prompt" : event.kind.replace("_", " ")}</span>
          <span>#${index + 1}</span>
        </div>
        <div class="bubble-preview">${escapeHtml(previewText(event))}</div>
        ${event.kind === "command" ? `<div class="muted">${event.command}</div>` : ""}
        ${event.kind === "file_change" && event.summary ? `<div class="muted">${escapeHtml(event.summary)}</div>` : ""}
        <div class="bubble-body">
          <pre>${escapeHtml(prettyEvent(event))}</pre>
        </div>
      `;
      bubble.addEventListener("click", () => {
        toggleCollapsed(currentAttemptId, index, event.kind);
      });
      bubbleStream.append(bubble);
    }
  }

function renderPairMatrix() {
  if (!data.pairings.length) {
    pairMatrix.innerHTML = `<p class="muted">No pairwise matrix available for this export.</p>`;
    return;
  }
  const rowBots = [...new Set(data.pairings.map((pair) => pair.a))];
  const colBots = [...new Set(data.pairings.map((pair) => pair.b))];
  const pairMap = new Map(data.pairings.map((pair) => [`${pair.a}|${pair.b}`, pair.mbb_per_hand]));
  pairMatrix.innerHTML = "";

  const scroller = document.createElement("div");
  scroller.className = "matrix-scroll";

  const matrix = document.createElement("div");
  matrix.className = "matrix";
  matrix.style.setProperty("--count", colBots.length);

  const header = document.createElement("div");
  header.className = "matrix-row";
  header.innerHTML = `<div class="matrix-header"></div>${colBots.map((bot) => `<div class="matrix-header">${bot}</div>`).join("")}`;
  matrix.append(header);

  for (const rowBot of rowBots) {
    const row = document.createElement("div");
    row.className = "matrix-row";
    row.innerHTML = `<div class="matrix-header">${rowBot}</div>`;
    for (const colBot of colBots) {
      const value = pairMap.get(`${rowBot}|${colBot}`);
      let content = "—";
      let cls = "matrix-cell";
      if (value !== undefined) {
        content = formatNumber(value);
        cls += value >= 0 ? " positive" : " negative";
      }
      row.innerHTML += `<div class="${cls}">${content}</div>`;
    }
    matrix.append(row);
  }
  scroller.append(matrix);
  const hint = document.createElement("p");
  hint.className = "matrix-hint muted";
  hint.textContent = "Scroll horizontally to see the full benchmark pool.";
  pairMatrix.append(hint);
  pairMatrix.append(scroller);
}

function renderQualitativeAnalysis() {
  if (!qualitativeAnalysis) return;
  const cards = data.qualitative_analysis?.cards || [];
  qualitativeAnalysis.innerHTML = `
    <div class="analysis-grid">
      ${cards.map((card) => `
        <article class="analysis-card">
          <h3>${escapeHtml(card.title)}</h3>
          <p>${escapeHtml(card.summary)}</p>
          ${(card.evidence || []).length ? `
            <ul>
              ${card.evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
          ` : ""}
          ${card.quote ? `
            <div class="analysis-quote">
              <div class="analysis-quote-head">
                <div class="mini-label">${card.quote.attempt_id && attemptMap.has(card.quote.attempt_id)
                  ? `<button type="button" class="attempt-link" data-attempt-link="${escapeHtml(card.quote.attempt_id)}">${escapeHtml(card.quote.label || "Code excerpt")}</button>`
                  : escapeHtml(card.quote.label || "Code excerpt")}</div>
              </div>
              <pre class="code-block">${highlightPythonCode(card.quote.text || "")}</pre>
            </div>
          ` : ""}
          ${(card.meta || []).length ? `
            <div class="analysis-meta">
              ${card.meta.map((item) => `<span class="stat-chip">${escapeHtml(item)}</span>`).join("")}
            </div>
          ` : ""}
        </article>
      `).join("")}
    </div>
  `;
  for (const button of qualitativeAnalysis.querySelectorAll("[data-attempt-link]")) {
    button.addEventListener("click", () => {
      selectAttempt(button.getAttribute("data-attempt-link"));
    });
  }
}

function renderReplayOptions() {
  if (!data.replays.length) {
    replaySelect.innerHTML = `<option value="0">No replay data</option>`;
    handSelect.innerHTML = `<option value="0">No hands</option>`;
    currentReplayIndex = 0;
    currentHandIndex = 0;
    currentStepIndex = 0;
    return;
  }
  replaySelect.innerHTML = data.replays
    .map((replay, index) => `<option value="${index}">${replay.matchup}</option>`)
    .join("");
  currentReplayIndex = 0;
  renderHandOptions();
}

function renderHandOptions() {
  const replay = data.replays[currentReplayIndex];
  if (!replay) {
    handSelect.innerHTML = `<option value="0">No hands</option>`;
    currentHandIndex = 0;
    currentStepIndex = 0;
    return;
  }
  handSelect.innerHTML = replay.hands
    .map((hand, index) => `<option value="${index}">Seed ${hand.seed}</option>`)
    .join("");
  currentHandIndex = 0;
  currentStepIndex = 0;
}

function parseCard(card) {
  const suitCode = card[0];
  const rank = card.slice(1);
  const suitMap = {
    S: { symbol: "&spades;", colorClass: "black" },
    H: { symbol: "&hearts;", colorClass: "red" },
    D: { symbol: "&diams;", colorClass: "red" },
    C: { symbol: "&clubs;", colorClass: "black" },
  };
  const suit = suitMap[suitCode] || { symbol: suitCode, colorClass: "black" };
  return { rank, suit: suit.symbol, colorClass: suit.colorClass };
}

function renderCardFace(card) {
  const parsed = parseCard(card);
  return `
    <div class="playing-card ${parsed.colorClass}">
      <span class="card-corner top">${parsed.rank}${parsed.suit}</span>
      <span class="card-center">${parsed.rank}</span>
      <span class="card-suit">${parsed.suit}</span>
      <span class="card-corner bottom">${parsed.rank}${parsed.suit}</span>
    </div>
  `;
}

function renderCardGroup(cards, label) {
  const normalized = Array.isArray(cards) ? cards : cards ? [cards] : [];
  if (!normalized.length) {
    return `
      <div class="playing-card back">
        <span class="card-corner top">${label}</span>
        <span class="card-back-mark">?</span>
        <span class="card-corner bottom">${label}</span>
      </div>
    `;
  }
  return normalized.map((card) => renderCardFace(card)).join("");
}

function rankValue(card) {
  const rank = card.slice(1);
  const map = { "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, T: 10, J: 11, Q: 12, K: 13, A: 14 };
  return map[rank] || 0;
}

function rankName(value) {
  const map = { 14: "Ace", 13: "King", 12: "Queen", 11: "Jack", 10: "Ten", 9: "Nine", 8: "Eight", 7: "Seven", 6: "Six", 5: "Five", 4: "Four", 3: "Three", 2: "Two", 1: "Ace" };
  return map[value] || String(value);
}

function pluralRank(value) {
  const map = { 14: "Aces", 13: "Kings", 12: "Queens", 11: "Jacks", 10: "Tens", 9: "Nines", 8: "Eights", 7: "Sevens", 6: "Sixes", 5: "Fives", 4: "Fours", 3: "Threes", 2: "Twos" };
  return map[value] || `${rankName(value)}s`;
}

function countMap(values) {
  const counts = new Map();
  for (const value of values) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  return counts;
}

function highestRankWithCount(rankCounts, targetCount, exclude = null) {
  return [...rankCounts.entries()]
    .filter(([rank, count]) => count >= targetCount && rank !== exclude)
    .map(([rank]) => rank)
    .sort((a, b) => b - a)[0];
}

function findStraightHigh(sortedUniqueRanksAsc) {
  if (!sortedUniqueRanksAsc.length) return null;
  const ranks = [...sortedUniqueRanksAsc];
  if (ranks.includes(14)) ranks.unshift(1);
  let run = 1;
  let best = null;
  for (let i = 1; i < ranks.length; i += 1) {
    if (ranks[i] === ranks[i - 1] + 1) {
      run += 1;
      if (run >= 5) best = ranks[i];
    } else if (ranks[i] !== ranks[i - 1]) {
      run = 1;
    }
  }
  return best;
}

function describeHand(holeCards, boardCards) {
  const cards = [...(holeCards || []), ...(boardCards || [])];
  if (cards.length < 2) return "No made hand yet";

  const ranks = cards.map(rankValue).sort((a, b) => b - a);
  const rankCounts = countMap(ranks);
  const countValues = [...rankCounts.values()].sort((a, b) => b - a);
  const suits = cards.map((card) => card[0]);
  const suitCounts = countMap(suits);
  const uniqueRanks = [...new Set(ranks)].sort((a, b) => a - b);
  const straightHigh = findStraightHigh(uniqueRanks);
  const flushSuit = [...suitCounts.entries()].find(([, count]) => count >= 5)?.[0] || null;
  const flushRanks = flushSuit
    ? cards.filter((card) => card[0] === flushSuit).map(rankValue).sort((a, b) => a - b)
    : [];
  const straightFlushHigh = flushSuit ? findStraightHigh([...new Set(flushRanks)]) : null;

  if (straightFlushHigh) return `${rankName(straightFlushHigh)}-high straight flush`;
  if (countValues[0] === 4) return `Four of a kind, ${pluralRank(highestRankWithCount(rankCounts, 4))}`;
  if (countValues[0] === 3 && countValues[1] >= 2) {
    return `Full house, ${pluralRank(highestRankWithCount(rankCounts, 3))} over ${pluralRank(highestRankWithCount(rankCounts, 2, highestRankWithCount(rankCounts, 3)))}`;
  }
  if (flushSuit) return `${rankName(Math.max(...flushRanks))}-high flush`;
  if (straightHigh) return `${rankName(straightHigh)}-high straight`;
  if (countValues[0] === 3) return `Three of a kind, ${pluralRank(highestRankWithCount(rankCounts, 3))}`;
  if (countValues[0] === 2 && countValues[1] === 2) {
    const topPair = highestRankWithCount(rankCounts, 2);
    const secondPair = highestRankWithCount(rankCounts, 2, topPair);
    return `Two pair, ${pluralRank(topPair)} and ${pluralRank(secondPair)}`;
  }
  if (countValues[0] === 2) return `One pair, ${pluralRank(highestRankWithCount(rankCounts, 2))}`;
  return `${rankName(ranks[0])}-high`;
}

function buildSeatStates(hand, step) {
  const actingPlayer = step.player;
  const cards = hand.hand_cards_final || [];
  const boardCards = step.public_cards || [];
  const chipStates = step.all_chips || [];
  const states = [
    {
      hand: cards[0] || [],
      handLabel: describeHand(cards[0] || [], boardCards),
      chips: chipStates[0] || null,
      isActing: actingPlayer === 0,
      chosenAction: actingPlayer === 0 ? step.chosen_action : null,
    },
    {
      hand: cards[1] || [],
      handLabel: describeHand(cards[1] || [], boardCards),
      chips: chipStates[1] || null,
      isActing: actingPlayer === 1,
      chosenAction: actingPlayer === 1 ? step.chosen_action : null,
    },
  ];

  if (step.hand) states[actingPlayer].hand = step.hand;
  if (step.my_chips !== undefined && step.my_chips !== null) states[actingPlayer].chips = step.my_chips;
  states[actingPlayer].handLabel = describeHand(states[actingPlayer].hand, boardCards);
  return states;
}

function seatTemplate(name, seat) {
  const statusText = seat.isActing ? `Action: ${seat.chosenAction}` : "Waiting";
  return `
    <div class="seat-head">
      <div>
        <div class="mini-label">Seat</div>
        <div class="seat-name">${name}</div>
      </div>
      <span class="stat-chip">${seat.chips === null ? "chips ?" : `chips ${seat.chips}`}</span>
    </div>
    <div class="seat-details">
      <div>
        <div class="mini-label">Hand</div>
        <div class="card-row">
          ${renderCardGroup(seat.hand, "Hand")}
        </div>
      </div>
      <div class="seat-meta muted">
        <span>${seat.handLabel}</span>
        <span class="seat-meta-sep">&middot;</span>
        <span>${statusText}</span>
      </div>
    </div>
  `;
}

function renderReplay() {
  if (!data.replays.length) {
    topSeat.innerHTML = `<div class="muted">No replay export available.</div>`;
    bottomSeat.innerHTML = "";
    boardState.innerHTML = `
      <div class="board-head">
        <div>
          <div class="mini-label">Replay</div>
          <div class="seat-name">Not exported</div>
        </div>
      </div>
      <p class="muted">No hand-by-hand replay payload is available in this export.</p>
    `;
    replayLog.innerHTML = `<div class="muted">No replay log available.</div>`;
    stepBadge.textContent = "No replay";
    prevStep.disabled = true;
    nextStep.disabled = true;
    return;
  }

  const replay = data.replays[currentReplayIndex];
  const hand = replay.hands[currentHandIndex];
  const step = hand.steps[currentStepIndex] || hand.steps[0];
  if (!step) return;

  const seatStates = buildSeatStates(hand, step);
  topSeat.innerHTML = seatTemplate(replay.bot_a, seatStates[0]);
  bottomSeat.innerHTML = seatTemplate(replay.bot_b, seatStates[1]);
  topSeat.classList.toggle("is-acting", step.player === 0);
  bottomSeat.classList.toggle("is-acting", step.player === 1);

  const pot = (step.all_chips || []).reduce((total, chips) => total + chips, 0);
  boardState.innerHTML = `
    <div class="board-head">
      <div>
        <div class="mini-label">Board</div>
        <div class="seat-name">Hand seed ${hand.seed}</div>
      </div>
      <span class="stat-chip">pot ${pot}</span>
    </div>
    <div class="card-row">
      ${renderCardGroup(step.public_cards, "Board")}
      <span class="stat-chip">payoff ${hand.final_payoffs.join(" / ")}</span>
    </div>
  `;

    replayLog.innerHTML = hand.steps.map((entry, index) => `
      <div class="replay-step" style="opacity:${index === currentStepIndex ? 1 : 0.58}">
        <strong>${index + 1}. ${entry.bot}</strong> ${entry.chosen_action}
        <div class="muted">Pot ${entry.all_chips.reduce((total, chips) => total + chips, 0)} &middot; Chips ${entry.all_chips.join(" / ")}</div>
      </div>
    `).join("");
  stepBadge.textContent = `Step ${currentStepIndex + 1} / ${hand.steps.length}`;
  prevStep.disabled = currentStepIndex === 0;
  nextStep.disabled = currentStepIndex >= hand.steps.length - 1;
}

attemptSelect.addEventListener("change", () => {
  currentAttemptId = attemptSelect.value;
  renderTranscript();
});

eventTypeSelect.addEventListener("change", renderTranscript);

replaySelect.addEventListener("change", () => {
  currentReplayIndex = Number(replaySelect.value);
  renderHandOptions();
  renderReplay();
});

handSelect.addEventListener("change", () => {
  currentHandIndex = Number(handSelect.value);
  currentStepIndex = 0;
  renderReplay();
});

prevStep.addEventListener("click", () => {
  currentStepIndex = Math.max(0, currentStepIndex - 1);
  renderReplay();
});

nextStep.addEventListener("click", () => {
  const maxIndex = data.replays[currentReplayIndex].hands[currentHandIndex].steps.length - 1;
  currentStepIndex = Math.min(maxIndex, currentStepIndex + 1);
  renderReplay();
});

renderHero();
initCollapsiblePanels();
renderMethodology();
renderSummary();
renderDistributionChart();
renderAttempts();
renderAttemptOptions();
renderTranscript();
renderPairMatrix();
renderQualitativeAnalysis();
renderReplayOptions();
renderReplay();

