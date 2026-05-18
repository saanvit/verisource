/* Source Reliability Assessor — frontend logic. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const state = { activeTab: "url" };

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

async function loadHealth() {
  const el = $("#health");
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const llm = data.llm_configured
      ? `<span class="ok">LLM ✓</span>`
      : `<span class="warn">LLM ✗ (heuristic mode)</span>`;
    const search = data.search_configured
      ? `<span class="ok">Search ✓</span>`
      : `<span class="warn">Search ✗</span>`;
    el.innerHTML = `${llm} · ${search}`;
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
  if (score >= 85) return "#4ade80";
  if (score >= 70) return "#a3e635";
  if (score >= 50) return "#facc15";
  if (score >= 30) return "#fb923c";
  return "#f87171";
}

function bar(name, value) {
  const v = Math.round(value);
  return `
    <div class="bar-row">
      <div class="name">${name}</div>
      <div class="bar"><span style="width:${v}%"></span></div>
      <div class="v">${v}</div>
    </div>`;
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
  const w = report.weights || {};

  return `
    <div class="score-hero">
      <div class="score-circle"
           style="--pct:${overall};--ring-color:${color}">
        <div class="num">${overall}</div>
        <div class="label">/ 100</div>
      </div>
      <div>
        <div class="verdict-row">
          <span class="badge ${verdictBadgeClass(report.verdict)}">
            ${report.verdict.replaceAll("-", " ")}
          </span>
          <span class="badge outline">confidence ${(report.confidence * 100).toFixed(0)}%</span>
          <span class="badge outline">domain ${escapeHtml(
            report.domain_reputation.domain
          )}</span>
          <span class="badge outline">bias: ${escapeHtml(report.domain_reputation.bias)}</span>
        </div>
        <p class="explanation">${escapeHtml(report.explanation)}</p>
      </div>
    </div>

    <div class="grid-3">
      <div class="card">
        <h3>Domain prior</h3>
        <div class="big">${Math.round(report.domain_reputation.score)}<span class="sub" style="font-size:14px;color:var(--text-dim)">/100</span></div>
        <div class="sub">${escapeHtml(report.domain_reputation.type)} · weight ${(w.domain * 100).toFixed(0)}%</div>
        <div class="sub" style="margin-top:8px">${escapeHtml(report.domain_reputation.rationale)}</div>
      </div>
      <div class="card">
        <h3>Content analysis</h3>
        <div class="big">${Math.round(report.content_analysis.score)}<span class="sub" style="font-size:14px;color:var(--text-dim)">/100</span></div>
        <div class="sub">weight ${(w.content * 100).toFixed(0)}%</div>
        <div style="margin-top:10px">
          ${bar("Factuality", report.content_analysis.factuality)}
          ${bar("Objectivity", report.content_analysis.objectivity)}
          ${bar("Transparency", report.content_analysis.transparency)}
          ${bar("Restraint", report.content_analysis.sensationalism)}
        </div>
      </div>
      <div class="card">
        <h3>Cross-reference</h3>
        <div class="big">${Math.round(report.cross_reference.score)}<span class="sub" style="font-size:14px;color:var(--text-dim)">/100</span></div>
        <div class="sub">
          ${report.cross_reference.n_sources} sources ·
          ${report.cross_reference.n_high_quality} high-quality ·
          weight ${(w.cross_reference * 100).toFixed(0)}%
        </div>
        <div class="sub" style="margin-top:8px">consensus: <strong style="color:var(--text)">${escapeHtml(
          report.cross_reference.consensus
        )}</strong></div>
      </div>
    </div>

    <div class="section-title">Main claims</div>
    ${listOrNone(
      report.content_analysis.main_claims,
      "claims",
      report.content_analysis.summary.startsWith("Heuristic-only")
        ? "No claims extracted by the heuristic. Set MISTRAL_API_KEY for LLM-based extraction."
        : "No factual claims identified."
    )}

    <div class="section-title">Red flags</div>
    ${listOrNone(report.content_analysis.red_flags, "flags", "No red flags identified.")}

    <div class="section-title">Corroborating sources</div>
    ${renderSources(report.cross_reference.sources)}
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
  setStatus("Fetching, analyzing with the LLM, and cross-referencing… this can take 10-30s.");
  $("#results").classList.add("hidden");

  try {
    const resp = await fetch("/api/assess", {
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
  loadHealth();
  $("#run").addEventListener("click", runAssessment);
  $("#url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runAssessment();
  });
});
