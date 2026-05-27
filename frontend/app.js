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
  return {
    "highly-reliable": "green",
    "generally-reliable": "lime",
    mixed: "yellow",
    questionable: "orange",
    unreliable: "red",
  }[verdict] || "outline";
}

function scoreColor(score) {
  // Modern dark palette: emerald → amber → rose.
  if (score >= 85) return "#34d399";
  if (score >= 70) return "#84cc16";
  if (score >= 50) return "#fbbf24";
  if (score >= 30) return "#fb923c";
  return "#f87171";
}

function scoreGlow(score) {
  // Soft RGBA halo behind the score ring.
  if (score >= 85) return "rgba(52, 211, 153, 0.35)";
  if (score >= 70) return "rgba(132, 204, 22, 0.30)";
  if (score >= 50) return "rgba(251, 191, 36, 0.28)";
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

function renderReport(report) {
  const overall = Math.round(report.overall_score);
  const color = scoreColor(overall);
  const glow = scoreGlow(overall);
  const w = report.weights || {};
  const ca = report.content_analysis || {};
  const hasPerClaim = Array.isArray(ca.claim_verifications) && ca.claim_verifications.length > 0;

  return `
    <div class="score-hero" style="--score-glow:${glow}">
      <div class="score-ring" style="--pct:${overall};--ring:${color}">
        <div class="score-num">${overall}</div>
        <div class="score-denom">/ 100</div>
      </div>
      <div class="score-content">
        <div class="verdict-row">
          <span class="badge ${verdictBadgeClass(report.verdict)}">
            <span class="dot"></span>
            ${report.verdict.replaceAll("-", " ")}
          </span>
          <span class="badge outline">Confidence ${(report.confidence * 100).toFixed(0)}%</span>
          <span class="badge outline">${escapeHtml(report.domain_reputation.domain)}</span>
          <span class="badge outline">Bias · ${escapeHtml(report.domain_reputation.bias)}</span>
        </div>
        <p class="explanation">${escapeHtml(report.explanation)}</p>
      </div>
    </div>

    <div class="grid-3">
      <div class="card">
        <h3>Domain prior</h3>
        <div class="big">${Math.round(report.domain_reputation.score)}<span class="sub">/100</span></div>
        <div class="sub"><strong>${escapeHtml(report.domain_reputation.type)}</strong> · weight ${(w.domain * 100).toFixed(0)}%</div>
        <div class="sub">${escapeHtml(report.domain_reputation.rationale)}</div>
      </div>
      <div class="card">
        <h3>Content analysis</h3>
        <div class="big">${Math.round(report.content_analysis.score)}<span class="sub">/100</span></div>
        <div class="sub">weight ${(w.content * 100).toFixed(0)}%</div>
        <div style="margin-top:6px">
          ${bar("Factuality", report.content_analysis.factuality)}
          ${bar("Objectivity", report.content_analysis.objectivity)}
          ${bar("Transparency", report.content_analysis.transparency)}
          ${bar("Restraint", report.content_analysis.sensationalism)}
        </div>
      </div>
      <div class="card">
        <h3>Cross-reference</h3>
        <div class="big">${Math.round(report.cross_reference.score)}<span class="sub">/100</span></div>
        <div class="sub">
          <strong>${report.cross_reference.n_sources}</strong> sources ·
          <strong>${report.cross_reference.n_high_quality}</strong> high-quality ·
          weight ${(w.cross_reference * 100).toFixed(0)}%
        </div>
        <div class="sub">Consensus · <strong>${escapeHtml(report.cross_reference.consensus)}</strong></div>
      </div>
    </div>

    ${
      hasPerClaim
        ? renderClaimVerifications(ca.claim_verifications, ca.coverage)
        : `<div>
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

    <div>
      <div class="section-title">
        Red flags
        <span class="section-meta">${(ca.red_flags || []).length} flagged</span>
      </div>
      ${listOrNone(ca.red_flags, "flags", "No red flags identified.")}
    </div>

    ${
      hasPerClaim
        ? ""
        : `<div>
             <div class="section-title">
               Corroborating sources
               <span class="section-meta">${(report.cross_reference.sources || []).length} retrieved</span>
             </div>
             ${renderSources(report.cross_reference.sources)}
           </div>`
    }
  `;
}

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
    state.mode === "per-claim"
      ? "Decomposing claims, retrieving evidence, scoring each independently…"
      : "Fetching, analyzing with the LLM, and cross-referencing…"
  );
  $("#results").classList.add("hidden");

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
    $("#results").innerHTML = renderReport(report);
    $("#results").classList.remove("hidden");
    setStatus("");
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
