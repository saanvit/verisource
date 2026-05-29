/* Source Reliability Assessor — frontend logic. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const state = { activeTab: "url", mode: "per-claim" };

function setupTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      state.activeTab = name;
      $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === name));
    });
  });
}

function setupModeToggle() {
  $$(".mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      $$(".mode").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-checked", on ? "true" : "false");
      });
    });
  });
}

async function loadHealth() {
  const el = $("#health");
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const llm = data.llm_configured
      ? `<span class="ok">LLM ●</span>`
      : `<span class="warn">LLM ○ heuristic</span>`;
    const search = data.search_configured
      ? `<span class="ok">Search ●</span>`
      : `<span class="warn">Search ○</span>`;
    el.innerHTML = `${llm} &nbsp;·&nbsp; ${search}`;
  } catch {
    el.innerHTML = `<span class="err">API unreachable</span>`;
  }
}

function verdictBadgeClass(verdict) {
  // 4-tone palette (lime collapsed into green to reduce visual noise).
  return {
    "highly-reliable": "green",
    "generally-reliable": "green",
    mixed: "yellow",
    questionable: "orange",
    unreliable: "red",
  }[verdict] || "outline";
}

function scoreColor(score) {
  // 4-bucket palette. Emerald threshold shifted 85 → 80 to compensate
  // for absorbing the lime band — 70–79 now reads as the mixed-amber
  // tone to clearly distinguish "mostly reliable" from "fully reliable".
  if (score >= 80) return "#34d399";
  if (score >= 50) return "#fbbf24";
  if (score >= 30) return "#fb923c";
  return "#f87171";
}

function scoreGlow(score) {
  // Soft RGBA halo behind the score ring.
  if (score >= 80) return "rgba(52, 211, 153, 0.35)";
  if (score >= 50) return "rgba(251, 191, 36, 0.30)";
  if (score >= 30) return "rgba(251, 146, 60, 0.30)";
  return "rgba(248, 113, 113, 0.32)";
}

function claimEdgeColor(status) {
  return {
    supported: "#34d399",
    contradicted: "#f87171",
    mixed: "#fbbf24",
    unverified: "#94a3b8",
  }[status] || "#94a3b8";
}

function claimIconBg(status) {
  return {
    supported: "rgba(52, 211, 153, 0.14)",
    contradicted: "rgba(248, 113, 113, 0.14)",
    mixed: "rgba(251, 191, 36, 0.14)",
    unverified: "rgba(148, 163, 184, 0.12)",
  }[status] || "rgba(148, 163, 184, 0.12)";
}

const STATUS_ICON = {
  supported: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="m5 12 5 5L20 7" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  contradicted: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M6 6 18 18M18 6 6 18" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
  mixed: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M6 9h12M6 15h12" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
  unverified: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M9 9a3 3 0 1 1 4 2.83V14M12 17.5v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
};

const CHEVRON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="m9 6 6 6-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function bar(name, value) {
  const v = Math.round(value);
  return `
    <div class="bar-row">
      <div class="name">${name}</div>
      <div class="bar"><span style="width:${v}%"></span></div>
      <div class="v">${v}</div>
    </div>`;
}

function statusBadgeClass(status) {
  return {
    supported: "green",
    contradicted: "red",
    mixed: "yellow",
    unverified: "outline",
  }[status] || "outline";
}

function renderEvidence(evidence) {
  if (!evidence?.length) {
    return `<div class="claim-evidence-empty">No independent evidence retrieved for this claim.</div>`;
  }
  return `
    <div class="claim-evidence">
      ${evidence
        .map((e) => {
          const tone =
            e.agreement === "supports"
              ? "green"
              : e.agreement === "contradicts"
              ? "red"
              : "outline";
          return `
            <div class="evidence">
              <div class="evidence-head">
                <a href="${e.url}" target="_blank" rel="noopener noreferrer">${escapeHtml(
                  e.title || e.url
                )}</a>
                <span class="badge ${tone}">${e.agreement}</span>
              </div>
              <div class="meta">${escapeHtml(e.domain)} · domain ${Math.round(
                e.domain_score
              )}/100</div>
              <div class="snippet">${escapeHtml(e.snippet || "")}</div>
            </div>`;
        })
        .join("")}
    </div>`;
}

const ACTION_LABEL = {
  relabel_hit: "Relabel hit",
  research_claim: "Re-search claim",
  split_claim: "Split into atoms",
  noop: "No-op",
};

const ACTION_TONE = {
  relabel_hit: "yellow",
  research_claim: "outline",
  split_claim: "green",
  noop: "outline",
};

function renderReflectionAction(a) {
  const tone = ACTION_TONE[a.type] || "outline";
  const label = ACTION_LABEL[a.type] || a.type;
  let detail = "";
  if (a.type === "relabel_hit") {
    detail = `→ <strong>${escapeHtml(a.new_label || "?")}</strong>
              <span class="action-target">on ${escapeHtml(a.hit_url || "")}</span>`;
  } else if (a.type === "research_claim") {
    detail = `→ new query <em>${escapeHtml(a.alternative_query || "")}</em>`;
  } else if (a.type === "split_claim" && Array.isArray(a.subclaims)) {
    detail = `→ split into:
              <ul class="subclaims">
                ${a.subclaims.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}
              </ul>`;
  }
  return `
    <li class="reflection-action">
      <div class="action-head">
        <span class="badge ${tone}">${escapeHtml(label)}</span>
        ${a.claim ? `<span class="action-claim">${escapeHtml(a.claim)}</span>` : ""}
      </div>
      <div class="action-detail">${detail}</div>
      ${a.reason ? `<div class="action-reason">${escapeHtml(a.reason)}</div>` : ""}
    </li>`;
}

function renderReflectionRound(r) {
  const actionCount = (r.actions || []).length;
  const deltaCount = Object.keys(r.score_deltas || {}).length;
  const summary = actionCount === 0
    ? "No fixes needed — converged."
    : `${actionCount} action${actionCount === 1 ? "" : "s"} applied · ${deltaCount} claim${deltaCount === 1 ? "" : "s"} changed`;

  const issuesHtml = (r.issues || []).length
    ? `<div class="reflection-issues">
         ${r.issues.map((i) => `<span class="badge outline">${escapeHtml(i)}</span>`).join("")}
       </div>`
    : "";

  const actionsHtml = (r.actions || []).length
    ? `<ul class="action-list">${r.actions.map(renderReflectionAction).join("")}</ul>`
    : `<p class="status">No actions this round.</p>`;

  const deltasHtml = deltaCount
    ? `<div class="reflection-deltas">
         ${Object.entries(r.score_deltas)
           .map(([claim, d]) => {
             const sign = d > 0 ? "+" : "";
             const tone = d > 0 ? "green" : d < 0 ? "red" : "outline";
             return `<div class="delta">
                       <span class="badge ${tone}">${sign}${d}</span>
                       <span class="delta-claim">${escapeHtml(claim)}</span>
                     </div>`;
           })
           .join("")}
       </div>`
    : "";

  return `
    <details class="reflection-round" ${r.round_index === 0 ? "open" : ""}>
      <summary>
        <span class="reflection-round-title">Round ${r.round_index + 1}</span>
        <span class="reflection-round-summary">${escapeHtml(summary)}</span>
      </summary>
      <div class="reflection-round-body">
        <p class="reflection-critique">${escapeHtml(r.critique || "")}</p>
        ${issuesHtml}
        ${actionsHtml}
        ${deltasHtml}
      </div>
    </details>`;
}

function renderReflectionTrace(trace) {
  if (!trace?.length) return "";
  const totalActions = trace.reduce(
    (n, r) => n + (r.actions || []).length, 0
  );
  return `
    <div class="reflection-trace">
      <div class="section-title">
        Agent reasoning trace
        <span class="section-meta">${trace.length} round${trace.length === 1 ? "" : "s"} · ${totalActions} action${totalActions === 1 ? "" : "s"}</span>
      </div>
      <p class="reflection-intro">
        Claude audited the initial per-claim verifications and proposed targeted
        fixes. Each round shows the critique, the structured actions executed,
        and the resulting score changes.
      </p>
      ${trace.map(renderReflectionRound).join("")}
    </div>`;
}

function renderClaimVerifications(verifications, coverage) {
  if (!verifications?.length) return "";
  const meta =
    coverage == null
      ? `${verifications.length} claims`
      : `${verifications.length} claims · ${(coverage * 100).toFixed(0)}% coverage`;
  return `
    <div>
      <div class="section-title">
        Per-claim verification
        <span class="section-meta">${meta}</span>
      </div>
      <div class="claim-list">
        ${verifications
          .map((v, i) => {
            const score = Math.round(v.score);
            const support = Math.round((v.support_ratio || 0) * 100);
            const contradict = Math.round((v.contradict_ratio || 0) * 100);
            const edge = claimEdgeColor(v.status);
            const iconBg = claimIconBg(v.status);
            const icon = STATUS_ICON[v.status] || STATUS_ICON.unverified;
            return `
              <details class="claim" ${i === 0 ? "open" : ""}
                       style="--claim-edge:${edge};--claim-icon-bg:${iconBg}">
                <summary>
                  <span class="claim-status-icon">${icon}</span>
                  <span class="claim-text">${escapeHtml(v.claim)}</span>
                  <span class="claim-meta">
                    <span class="claim-score">${score}</span>
                    <span class="claim-chevron">${CHEVRON}</span>
                  </span>
                </summary>
                <div class="claim-body">
                  <div class="claim-ratios">
                    <div class="ratio supports">
                      <div class="ratio-head">
                        <span class="ratio-label">Supports</span>
                        <span class="ratio-v">${support}%</span>
                      </div>
                      <div class="ratio-bar"><span style="width:${support}%"></span></div>
                    </div>
                    <div class="ratio contradicts">
                      <div class="ratio-head">
                        <span class="ratio-label">Contradicts</span>
                        <span class="ratio-v">${contradict}%</span>
                      </div>
                      <div class="ratio-bar"><span style="width:${contradict}%"></span></div>
                    </div>
                  </div>
                  <div class="claim-counts">
                    <strong>${v.n_evidence}</strong> evidence ·
                    <strong>${v.n_high_quality}</strong> high-quality ·
                    status <strong>${v.status}</strong>
                  </div>
                  ${renderEvidence(v.evidence)}
                </div>
              </details>`;
          })
          .join("")}
      </div>
    </div>
  `;
}

function renderSources(sources) {
  if (!sources?.length) {
    return `<p class="status">No corroborating sources retrieved.</p>`;
  }
  return `
    <div class="source-list">
      ${sources
        .map((s) => {
          const tone =
            s.agreement === "supports"
              ? "green"
              : s.agreement === "contradicts"
              ? "red"
              : "outline";
          return `
            <div class="source">
              <div>
                <a href="${s.url}" target="_blank" rel="noopener noreferrer">${escapeHtml(
                  s.title || s.url
                )}</a>
                <div class="meta">${escapeHtml(s.domain)} · domain ${Math.round(
                  s.domain_score
                )}/100</div>
              </div>
              <div><span class="badge ${tone}">${s.agreement}</span></div>
              <div class="snippet">${escapeHtml(s.snippet || "")}</div>
            </div>`;
        })
        .join("")}
    </div>`;
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}

function listOrNone(items, cls, emptyMsg = "None detected.") {
  if (!items?.length) return `<p class="status">${escapeHtml(emptyMsg)}</p>`;
  return `<ul class="${cls}">${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`;
}

// Strip the verdict + score from the explanation string so the headline
// reads as a single fluid sentence ("Strongest signal: …; Caveat: …")
// without redundantly repeating "This source is X (Y/100)" right next to
// the giant score ring.
function heroSummary(explanation) {
  if (!explanation) return "";
  return explanation.replace(/^This source is [^.]+\.\s*/i, "").trim();
}

function renderReport(report) {
  const overall = Math.round(report.overall_score);
  const color = scoreColor(overall);
  const glow = scoreGlow(overall);
  const w = report.weights || {};
  const ca = report.content_analysis || {};
  const xref = report.cross_reference || {};
  const dom = report.domain_reputation || {};
  const hasPerClaim = Array.isArray(ca.claim_verifications) && ca.claim_verifications.length > 0;
  const hasReflection = Array.isArray(ca.reflection_trace) && ca.reflection_trace.length > 0;
  const hasRedFlags = Array.isArray(ca.red_flags) && ca.red_flags.length > 0;
  const summary = heroSummary(report.explanation);

  // Dominant cross-reference panel: only render the rich panel for
  // per-claim modes (where the per-source list is meaningful).
  const crossRefPanel = `
    <div class="cross-ref-panel" data-reveal="cross-ref">
      <div class="cross-ref-head">
        <div class="cross-ref-label">
          <span class="section-meta">Cross-reference</span>
          <h3>Independent corroboration</h3>
        </div>
        <div class="cross-ref-score-block">
          <div class="cross-ref-score" style="color:${scoreColor(Math.round(xref.score || 0))}">${Math.round(xref.score || 0)}</div>
          <div class="cross-ref-score-meta">
            <span class="consensus-pill ${escapeHtml(xref.consensus || 'no-data')}">
              <span class="dot"></span>
              ${escapeHtml((xref.consensus || 'no-data').replaceAll('-', ' '))}
            </span>
            <span class="sub">weight ${(w.cross_reference * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
      <div class="cross-ref-stats">
        <div class="stat">
          <div class="stat-num">${xref.n_sources || 0}</div>
          <div class="stat-label">independent sources</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat">
          <div class="stat-num">${xref.n_high_quality || 0}</div>
          <div class="stat-label">high-quality</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat">
          <div class="stat-num">${Math.round((xref.n_high_quality || 0) / Math.max(xref.n_sources || 1, 1) * 100)}<span class="stat-num-sub">%</span></div>
          <div class="stat-label">high-quality rate</div>
        </div>
      </div>
    </div>`;

  return `
    <div class="score-hero" style="--score-glow:${glow}" data-reveal="hero">
      <div class="score-hero-top">
        <div class="score-ring" style="--pct:${overall};--ring:${color}">
          <div class="score-num">${overall}</div>
          <div class="score-denom">/ 100</div>
        </div>
        <div class="score-content">
          <div class="verdict-headline">
            <span class="badge ${verdictBadgeClass(report.verdict)} verdict-badge">
              <span class="dot"></span>
              ${report.verdict.replaceAll("-", " ")}
            </span>
            <h1 class="hero-h1">${escapeHtml(verdictHeadline(report.verdict))}</h1>
          </div>
          <p class="hero-summary">${escapeHtml(summary)}</p>
        </div>
      </div>
      <div class="score-hero-bar">
        <span class="meta-pill"><span class="meta-key">Confidence</span><span class="meta-val">${(report.confidence * 100).toFixed(0)}%</span></span>
        ${
          dom.domain && dom.domain !== "unknown"
            ? `<span class="meta-pill"><span class="meta-key">Source</span><span class="meta-val">${escapeHtml(dom.domain)}</span></span>`
            : ""
        }
        ${
          dom.bias && dom.bias !== "unknown"
            ? `<span class="meta-pill"><span class="meta-key">Bias</span><span class="meta-val">${escapeHtml(dom.bias)}</span></span>`
            : ""
        }
        ${
          xref.n_sources > 0
            ? `<span class="meta-pill"><span class="meta-key">Sources</span><span class="meta-val">${xref.n_sources} · ${xref.n_high_quality} HQ</span></span>`
            : ""
        }
      </div>
    </div>

    ${hasReflection ? renderReflectionTrace(ca.reflection_trace) : ""}

    ${crossRefPanel}

    <div class="grid-2" data-reveal="sub-cards">
      <div class="card card-sub">
        <div class="card-sub-head">
          <span class="section-meta">Content analysis</span>
          <div class="card-sub-num">${Math.round(report.content_analysis.score)}<span class="card-sub-denom">/100</span></div>
        </div>
        <div class="card-sub-sub">weight ${(w.content * 100).toFixed(0)}% · ${escapeHtml(ca.summary || "").slice(0, 60)}${(ca.summary || "").length > 60 ? "…" : ""}</div>
        <div style="margin-top:10px">
          ${bar("Factuality", report.content_analysis.factuality)}
          ${bar("Objectivity", report.content_analysis.objectivity)}
          ${bar("Transparency", report.content_analysis.transparency)}
          ${bar("Restraint", report.content_analysis.sensationalism)}
        </div>
      </div>
      <div class="card card-sub">
        <div class="card-sub-head">
          <span class="section-meta">Domain prior</span>
          <div class="card-sub-num">${Math.round(dom.score || 0)}<span class="card-sub-denom">/100</span></div>
        </div>
        <div class="card-sub-sub">
          <strong>${escapeHtml(dom.type || "unknown")}</strong> · weight ${(w.domain * 100).toFixed(0)}%
        </div>
        <div class="card-sub-sub" style="margin-top:8px">${escapeHtml(dom.rationale || "")}</div>
      </div>
    </div>

    ${
      hasPerClaim
        ? renderClaimVerifications(ca.claim_verifications, ca.coverage)
        : `<div data-reveal="main-claims">
             <div class="section-title">
               Main claims
               <span class="section-meta">${(ca.main_claims || []).length} extracted</span>
             </div>
             ${listOrNone(
               ca.main_claims,
               "claims",
               (ca.summary || "").startsWith("Heuristic-only")
                 ? "No claims extracted by the heuristic. Set MISTRAL_API_KEY for LLM-based extraction."
                 : "No factual claims identified."
             )}
           </div>`
    }

    ${
      hasRedFlags
        ? `<div class="red-flags-block" data-reveal="red-flags">
             <div class="section-title red-flags-title">
               <span class="alert-icon" aria-hidden="true">
                 <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                   <path d="M12 9v4M12 17v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                   <path d="M10.3 3.86 2.07 18a2 2 0 0 0 1.73 3h16.4a2 2 0 0 0 1.73-3L13.7 3.86a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                 </svg>
               </span>
               Red flags
               <span class="section-meta">${ca.red_flags.length} flagged</span>
             </div>
             <ul class="flags">${ca.red_flags.map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
           </div>`
        : ""
    }

    ${
      hasPerClaim
        ? ""
        : `<div data-reveal="sources">
             <div class="section-title">
               Corroborating sources
               <span class="section-meta">${(xref.sources || []).length} retrieved</span>
             </div>
             ${renderSources(xref.sources)}
           </div>`
    }
  `;
}

// One-word verdict → presentation headline. Kept distinct from the
// machine-readable verdict so the UI can phrase it naturally.
function verdictHeadline(v) {
  return {
    "highly-reliable": "Highly reliable",
    "generally-reliable": "Generally reliable",
    mixed: "Mixed signals",
    questionable: "Questionable",
    unreliable: "Unreliable",
  }[v] || v;
}

// Track API duration so we can skip the staged-reveal theater for slow
// responses — adding 5s of animation on top of a 10s wait is sadism.
const SLOW_API_MS = 6000;

async function runAssessment() {
  const url = $("#url").value.trim();
  const text = $("#text").value.trim();
  const claim = $("#claim").value.trim();

  const body = {};
  if (state.activeTab === "url") {
    if (!url) {
      setStatus("Please enter a URL.", "warn");
      return;
    }
    body.url = url;
  } else {
    if (!text) {
      setStatus("Please paste some article text.", "warn");
      return;
    }
    body.text = text;
  }
  if (claim) body.claim = claim;

  const btn = $("#run");
  btn.disabled = true;
  setStatus(
    state.mode === "per-claim-reflective"
      ? "Decomposing, verifying, then running Claude critique + auto-corrections…"
      : state.mode === "per-claim"
      ? "Decomposing claims, retrieving evidence, scoring each independently…"
      : "Fetching, analyzing with the LLM, and cross-referencing…"
  );
  $("#results").classList.add("hidden");

  const reqStart = performance.now();
  try {
    const resp = await fetch(`/api/assess?mode=${encodeURIComponent(state.mode)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const report = await resp.json();
    const apiDurationMs = performance.now() - reqStart;
    const results = $("#results");
    results.innerHTML = renderReport(report);
    results.classList.remove("hidden");
    setStatus("");
    // Staged-reveal: ring fills with gauge overshoot, sections stagger
    // in, reflection theater plays. Skip the theater if the API was
    // already slow — user has waited enough.
    if (typeof window.revealReport === "function") {
      window.revealReport(results, {
        score: Math.round(report.overall_score),
        skipTheater: apiDurationMs > SLOW_API_MS,
      });
    }
  } catch (e) {
    setStatus(`Error: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupModeToggle();
  loadHealth();
  $("#run").addEventListener("click", runAssessment);
  $("#url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runAssessment();
  });
});
