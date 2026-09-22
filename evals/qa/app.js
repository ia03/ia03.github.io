// ===== CONSTANTS =====
const BENCH_TAB_MAP = {
  'tb2': 'tb2',
  'swebench-verified': 'swe',
  'swebench-pro': 'swepro',
  'rebench': 're',
  'corebench': 'core',
  'osworld': 'osworld',
  'cybench': 'cybench',
  'mlebench': 'mle'
};

const verdictLabel = {
  WEAK_FP: 'Accepts invalid', WEAK_FN: 'Penalizes valid',
  BOTH: 'Both', ROBUST: 'OK'
};
const verdictBadge = {
  WEAK_FP: 'badge-fp', WEAK_FN: 'badge-fn',
  BOTH: 'badge-both', ROBUST: 'badge-robust'
};
const sevLabel = { high: 'High', medium: 'Medium', low: 'Low' };
const sevBadge = { high: 'badge-high', medium: 'badge-med', low: 'badge-low' };
const sevClass = { high: 'high', medium: 'med', low: 'low' };
const sevOrder = { high: 0, medium: 1, low: 2, none: 3 };
const verdictOrder = { BOTH: 0, WEAK_FN: 1, WEAK_FP: 2, ROBUST: 3 };

// ===== TABS =====
function renderTabs() {
  const container = document.getElementById('main-tabs');
  const tabs = [
    { key: 'overview', label: 'Overview' },
    { key: 'tb2', label: 'TB2' },
    { key: 'swe', label: 'SWE-bench Verified' },
    { key: 'swepro', label: 'SWE-bench Pro' },
    { key: 're', label: 'RE-Bench' },
    { key: 'core', label: 'CORE-Bench' },
    { key: 'osworld', label: 'OSWorld' },
    { key: 'cybench', label: 'Cybench' },
    { key: 'mle', label: 'MLE-bench' }
  ];
  container.innerHTML = tabs.map((t, i) =>
    `<button class="tab${i === 0 ? ' active' : ''}" data-tab="${t.key}">${t.label}</button>`
  ).join('');
}

// Tab switching
document.addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  tab.classList.add('active');
  const target = document.getElementById('tab-' + tab.dataset.tab);
  if (target) target.classList.add('active');
});

// ===== VERIFIED BUG CARDS =====
function renderBugCard(bug, isHighlight) {
  const sev = bug.severity || 'medium';
  const cardClass = sevClass[sev] || '';
  const benchLabel = isHighlight && BENCHMARKS[bug.benchmark]
    ? `<span class="badge ${sevBadge[sev] || 'badge-med'}">${BENCHMARKS[bug.benchmark].name}</span>`
    : `<span class="badge ${sevBadge[sev] || 'badge-med'}">${sevLabel[sev] || sev}</span>`;

  const evidenceBlock = bug.evidence
    ? `<div class="evidence">${bug.evidence}</div>` : '';

  return `<div class="card ${cardClass} bug-card" data-bug-id="${bug.id}">
    <div class="card-header">
      ${benchLabel}
      <span class="task-name">${bug.title}</span>
    </div>
    <div class="card-body">
      <p class="description">${bug.description}</p>
      ${evidenceBlock}
    </div>
  </div>`;
}

function renderVerifiedBugs(containerId, benchKey) {
  const container = document.getElementById(containerId);
  if (!container) return;
  // Overview highlights use the first 8 (the "highlight reel")
  const bugs = VERIFIED_BUGS.filter(b => b.benchmark === benchKey);
  container.innerHTML = bugs.map(b => renderBugCard(b, false)).join('');
}

// ===== OVERVIEW =====
function renderOverview() {
  // Highlight grid: first 8 verified bugs are the cross-benchmark highlights
  const highlights = VERIFIED_BUGS.slice(0, 8);
  const grid = document.getElementById('overview-highlights');
  grid.innerHTML = highlights.map(b => renderBugCard(b, true)).join('');

  // Summary table
  const tableEl = document.getElementById('overview-table');
  const byBench = {};
  FINDINGS_DATA.forEach(f => {
    if (!byBench[f.benchmark]) byBench[f.benchmark] = [];
    byBench[f.benchmark].push(f);
  });

  // Worst verified bug per benchmark
  const worstBug = {};
  VERIFIED_BUGS.forEach(b => {
    if (!worstBug[b.benchmark] || (sevOrder[b.severity] || 3) < (sevOrder[worstBug[b.benchmark].severity] || 3)) {
      worstBug[b.benchmark] = b;
    }
  });

  const benchOrder = ['tb2', 'swebench-verified', 'swebench-pro', 'rebench', 'corebench', 'osworld', 'cybench', 'mlebench'];
  let rows = '';
  benchOrder.forEach(bk => {
    const info = BENCHMARKS[bk];
    if (!info) return;
    const items = byBench[bk] || [];
    const issues = items.filter(f => f.verdict !== 'ROBUST').length;
    const worst = worstBug[bk];
    rows += `<tr>
      <td>${info.name}</td>
      <td class="num">${info.tasks}</td>
      <td class="num">${issues}</td>
      <td>${worst ? worst.title : '-'}</td>
    </tr>`;
  });

  tableEl.innerHTML = `<div class="card">
    <div class="card-body" style="padding-top:1.25rem;">
      <table class="findings-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th style="text-align:center">Tasks Audited</th>
            <th style="text-align:center">Problematic</th>
            <th>Worst Bug Found</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

// ===== INSTANCE LIST =====
function renderBenchmark(containerId, findings) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Build the search/filter UI (no stats — benchmark-level stats handle that)
  container.innerHTML = `
    <div class="search-bar">
      <input type="text" class="search-input" placeholder="Search by instance ID or issue...">
      <button class="filter-btn" data-filter="all">All</button>
      <button class="filter-btn" data-filter="issues">Issues Only</button>
      <button class="filter-btn" data-filter="high">High Only</button>
      <button class="filter-btn" data-filter="WEAK_FN">Penalizes Valid</button>
      <button class="filter-btn" data-filter="WEAK_FP">Accepts Invalid</button>
      <button class="filter-btn" data-filter="ROBUST">No Issues</button>
      <span class="result-count" style="color:var(--muted);font-size:0.8rem"></span>
    </div>
    <div class="instance-list"></div>
  `;

  const listEl = container.querySelector('.instance-list');

  // Sort: high severity first, then medium, then low, then robust
  findings.sort((a, b) => {
    const sa = sevOrder[a.severity] ?? 3;
    const sb = sevOrder[b.severity] ?? 3;
    if (sa !== sb) return sa - sb;
    return (verdictOrder[a.verdict] ?? 3) - (verdictOrder[b.verdict] ?? 3);
  });

  // Build a lookup for evidence (from both VERIFIED_BUGS and EVIDENCE)
  const bugLookup = {};
  VERIFIED_BUGS.forEach(b => {
    bugLookup[b.benchmark + '/' + b.id] = b;
  });
  // EVIDENCE is a dict of benchmark/id -> HTML evidence string
  if (typeof EVIDENCE !== 'undefined') {
    for (const [key, html] of Object.entries(EVIDENCE)) {
      if (!bugLookup[key]) {
        bugLookup[key] = { evidence: html };
      } else if (!bugLookup[key].evidence) {
        bugLookup[key].evidence = html;
      }
    }
  }

  // Lazy rendering: render in batches for performance
  const BATCH_SIZE = 100;
  let rendered = 0;

  function renderBatch() {
    const end = Math.min(rendered + BATCH_SIZE, findings.length);
    for (let i = rendered; i < end; i++) {
      renderRow(findings[i]);
    }
    rendered = end;
    if (rendered < findings.length) {
      const loadMore = document.createElement('button');
      loadMore.className = 'filter-btn';
      loadMore.style.width = '100%';
      loadMore.style.marginTop = '0.5rem';
      loadMore.textContent = `Show more (${findings.length - rendered} remaining)`;
      loadMore.addEventListener('click', () => {
        loadMore.remove();
        renderBatch();
      });
      listEl.appendChild(loadMore);
    }
  }

  function renderRow(f) {
    const row = document.createElement('div');
    const cardClass = f.verdict !== 'ROBUST' && sevClass[f.severity] ? ' ' + sevClass[f.severity] : '';
    row.className = 'instance-row' + cardClass;
    row.dataset.id = f.id;
    row.dataset.issue = f.issue || '';
    row.dataset.verdict = f.verdict;
    row.dataset.severity = f.severity || 'none';
    row.dataset.benchmark = f.benchmark;

    let badges = '';
    if (f.verdict !== 'ROBUST' && f.severity && f.severity !== 'none') {
      badges += `<span class="badge ${sevBadge[f.severity] || ''}">${sevLabel[f.severity] || ''}</span> `;
    }
    badges += `<span class="badge ${verdictBadge[f.verdict] || ''}">${verdictLabel[f.verdict] || f.verdict}</span>`;

    // Check if this finding has a verified bug with evidence
    const vbug = bugLookup[f.benchmark + '/' + f.id];
    const hasDetail = f.issue || (vbug && vbug.evidence);
    const expandIcon = hasDetail && f.verdict !== 'ROBUST'
      ? '<span class="expand-icon">&#9654;</span>' : '';

    // Clean issue text: strip C1:/C2:/C3:/C4: prefixes for display
    let displayIssue = f.issue || '';
    displayIssue = displayIssue.replace(/C[1-4]:\s*/g, '').replace(/;\s*$/,'').trim();

    row.innerHTML = `
      <div class="iid">${expandIcon}${f.id}</div>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">${badges}</div>
      <div class="issue-text">${displayIssue || '<span style="color:var(--muted)">No issues found</span>'}</div>
    `;

    // Click to expand detail panel
    if (hasDetail && f.verdict !== 'ROBUST') {
      row.style.cursor = 'pointer';
      row.addEventListener('click', () => toggleDetail(row, f, vbug));
    }

    listEl.appendChild(row);
  }

  renderBatch();

  // Stats are rendered by renderBenchStats, not duplicated here

  // Wire search + filters
  const input = container.querySelector('.search-input');
  const filterBtns = container.querySelectorAll('.filter-btn');
  const countEl = container.querySelector('.result-count');

  function applyFilters() {
    const query = (input ? input.value : '').toLowerCase();
    const activeBtn = container.querySelector('.filter-btn.active');
    const activeFilter = activeBtn ? activeBtn.dataset.filter : 'all';

    let visible = 0;
    container.querySelectorAll('.instance-row').forEach(row => {
      const id = (row.dataset.id || '').toLowerCase();
      const issue = (row.dataset.issue || '').toLowerCase();
      const verdict = row.dataset.verdict || '';
      const severity = row.dataset.severity || 'none';

      const matchSearch = !query || id.includes(query) || issue.includes(query);
      let matchFilter;
      if (activeFilter === 'all') matchFilter = true;
      else if (activeFilter === 'issues') matchFilter = verdict !== 'ROBUST';
      else if (activeFilter === 'high') matchFilter = severity === 'high';
      else matchFilter = verdict === activeFilter;
      const show = matchSearch && matchFilter;

      row.classList.toggle('hidden', !show);
      if (show) visible++;
    });

    if (countEl) countEl.textContent = visible + ' results';
  }

  if (input) input.addEventListener('input', applyFilters);

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const wasActive = btn.classList.contains('active');
      filterBtns.forEach(b => b.classList.remove('active'));
      if (!wasActive && btn.dataset.filter !== 'all') {
        btn.classList.add('active');
      }
      applyFilters();
    });
  });

  if (countEl) countEl.textContent = findings.length + ' results';
}


// ===== DETAIL PANEL =====
function toggleDetail(row, finding, vbug) {
  // If already expanded, collapse
  const existing = row.nextElementSibling;
  if (existing && existing.classList.contains('detail-panel')) {
    existing.classList.remove('open');
    setTimeout(() => existing.remove(), 200);
    row.classList.remove('expanded');
    row.querySelector('.expand-icon')?.classList.remove('open');
    return;
  }

  // Close any other open detail panel in this list
  const list = row.parentElement;
  list.querySelectorAll('.detail-panel').forEach(p => {
    p.previousElementSibling?.classList.remove('expanded');
    p.previousElementSibling?.querySelector('.expand-icon')?.classList.remove('open');
    p.remove();
  });

  row.classList.add('expanded');
  row.querySelector('.expand-icon')?.classList.add('open');

  const panel = document.createElement('div');
  panel.className = 'detail-panel';

  // Severity + verdict badges
  let badgesHtml = '';
  if (finding.severity && finding.severity !== 'none') {
    badgesHtml += `<span class="badge ${sevBadge[finding.severity]}">${sevLabel[finding.severity]}</span> `;
  }
  badgesHtml += `<span class="badge ${verdictBadge[finding.verdict]}">${verdictLabel[finding.verdict]}</span>`;

  // Criteria
  const criteriaFlags = [];
  if (finding.issue) {
    for (let c = 1; c <= 4; c++) {
      if (finding.issue.includes('C' + c + ':') || finding.issue.includes('C' + c + ' ')) {
        criteriaFlags.push(c);
      }
    }
  }
  let criteriaHtml = '';
  if (criteriaFlags.length) {
    const labels = {
      1: 'Penalizes valid solutions',
      2: 'Does not penalize invalid solutions',
      3: 'Inconsistent or non-deterministic grading',
      4: 'Unjustified weighing of criteria'
    };
    criteriaHtml = '<div class="detail-criteria">' +
      criteriaFlags.map(c => `<span class="criteria-tag flagged">${labels[c]}</span>`).join(' ') +
      '</div>';
  }

  // Evidence
  let evidenceHtml = '';
  if (vbug && vbug.evidence) {
    evidenceHtml = `<div class="detail-section-label">Evidence</div><div class="evidence">${vbug.evidence}</div>`;
  }

  // Full description — format inline code references as code blocks
  let descHtml = '';
  let issueText = '';
  if (vbug && vbug.description) {
    issueText = vbug.description;
  } else if (finding.issue) {
    issueText = finding.issue.replace(/C[1-4]:\s*/g, '').replace(/;\s*$/,'').trim();
  }

  if (issueText) {
    // Convert backtick-wrapped text to <code> tags
    let formatted = issueText
      .replace(/`([^`]+)`/g, '<code>$1</code>');

    // Extract code-like patterns and show as evidence blocks
    // Look for patterns like: function_name(), assert X == Y, line numbers
    const codePatterns = issueText.match(/`[^`]+`/g);
    let codeBlockHtml = '';
    if (codePatterns && codePatterns.length >= 2 && !evidenceHtml) {
      // Build a mini evidence block from the code references
      const codeRefs = codePatterns.map(p => p.replace(/`/g, '')).join('\n');
      codeBlockHtml = `<div class="detail-section-label">Referenced code</div><div class="evidence">${codeRefs}</div>`;
    }

    descHtml = `<p class="description">${formatted}</p>${codeBlockHtml}`;
  }

  panel.innerHTML = `
    <div class="detail-inner">
      <div class="detail-badges">${badgesHtml}</div>
      ${criteriaHtml}
      ${descHtml}
      ${evidenceHtml}
    </div>
  `;

  row.after(panel);
  // Trigger animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => panel.classList.add('open'));
  });
}


// ===== BENCHMARK DESCRIPTIONS + STATS =====
function renderBenchmarkExtras(benchKey) {
  const info = BENCHMARKS[benchKey];
  if (!info) return;

  // Description
  const descEl = document.getElementById(benchKey + '-desc');
  if (descEl) descEl.textContent = info.description;

  // Top stats for small benchmarks
  const statsEl = document.getElementById(benchKey + '-top-stats');
  if (!statsEl) return;

  const items = FINDINGS_DATA.filter(f => f.benchmark === benchKey);
  const issues = items.filter(f => f.verdict !== 'ROBUST').length;
  const highCount = items.filter(f => f.severity === 'high').length;
  const robustPct = items.length ? ((items.length - issues) / items.length * 100).toFixed(0) : 0;
  const issuePct = items.length ? (issues / items.length * 100).toFixed(0) : 0;

  // Same stats template for all benchmarks
  statsEl.innerHTML = `
    <div class="stat"><span class="val">${items.length}</span><span class="lbl">Tasks</span></div>
    <div class="stat"><span class="val" style="color:var(--red)">${issues}</span><span class="lbl">Problematic</span></div>
    ${highCount ? `<div class="stat"><span class="val" style="color:var(--red)">${highCount}</span><span class="lbl">High</span></div>` : ''}
    <div class="stat"><span class="val" style="color:var(--green)">${items.length - issues}</span><span class="lbl">No Issues</span></div>
  `;
}

// ===== CARD CLICK TO EXPAND =====
document.addEventListener('click', e => {
  const card = e.target.closest('.bug-card');
  if (!card) return;
  card.classList.toggle('card-expanded');
});


// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
  if (typeof FINDINGS_DATA === 'undefined') {
    console.error('FINDINGS_DATA not found. Make sure data.js is loaded.');
    return;
  }

  renderTabs();
  renderOverview();

  // Group findings by benchmark
  const byBench = {};
  FINDINGS_DATA.forEach(f => {
    if (!byBench[f.benchmark]) byBench[f.benchmark] = [];
    byBench[f.benchmark].push(f);
  });

  // Render each benchmark tab
  for (const [bench, findings] of Object.entries(byBench)) {
    renderBenchmark(bench + '-instances', findings);
    renderVerifiedBugs(bench + '-verified-bugs', bench);
    renderBenchmarkExtras(bench);
  }
});
