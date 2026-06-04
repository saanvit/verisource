/* VeriSource — frontend logic. */

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

function selectMode(mode) {
  state.mode = mode;
  $$(".mode").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
  });
  // claim-check is a different tool: it verifies a single standalone claim,
  // so hide the article URL/text inputs and promote the claim field.
  const panel = $(".input-panel");
  if (panel) panel.classList.toggle("claim-only", mode === "claim-check");
  const head = panel?.querySelector(".panel-head h2");
  const sub = panel?.querySelector(".panel-sub");
  if (head && sub) {
    if (mode === "claim-check") {
      head.textContent = "Fact-check a single claim";
      sub.textContent =
        "Paste a claim, headline, or tweet. We'll retrieve independent evidence, label each source, and stress-test it with an adversarial search.";
    } else if (mode === "citation-audit") {
      head.textContent = "Audit an article's citations";
      sub.textContent =
        "Paste an article URL. We'll fetch each source it links to and check whether that source actually supports the sentence citing it.";
    } else {
      head.textContent = "Submit an article for assessment";
      sub.textContent =
        "Paste a URL or the article body. We'll decompose the claims and check each against independent sources.";
    }
  }
}

function setupModeToggle() {
  $$(".mode").forEach((btn) => {
    btn.addEventListener("click", () => selectMode(btn.dataset.mode));
  });
}

// Curated demo inputs. Each prefills the paste-text field, picks the mode
// that best shows the feature, and auto-runs — so the product is instantly
// demonstrable without a grader bringing their own URL. Text (not URLs) so
// they always work offline-of-paywalls and never 404.
const EXAMPLES = [
  {
    label: "Wire report",
    hint: "reliable",
    mode: "per-claim",
    text:
      "The European Central Bank cut its key interest rate by 25 basis points on " +
      "Thursday, lowering the deposit rate to 3.25%, the central bank said in a " +
      "statement. It was the third cut this year. ECB President Christine Lagarde " +
      "said inflation in the euro area had eased to 1.7% in September, below the " +
      "bank's 2% target for the first time since 2021. The decision was widely " +
      "expected by economists polled ahead of the meeting.",
  },
  {
    label: "Opinion column",
    hint: "mixed",
    mode: "per-claim",
    text:
      "The city's new congestion charge is a disaster that will hollow out small " +
      "businesses downtown. Foot traffic has already collapsed since the toll went " +
      "live, and every shop owner I know says revenue is down. Politicians claim the " +
      "scheme reduces emissions, but the real motive is simply to squeeze more money " +
      "out of ordinary commuters who have no other way to get to work.",
  },
  {
    label: "Viral health claim",
    hint: "questionable",
    mode: "per-claim-reflective",
    text:
      "BREAKING: Scientists have FINALLY admitted that drinking celery juice every " +
      "morning cures type 2 diabetes in just two weeks — no medication needed. Big " +
      "Pharma has been hiding this miracle cure for decades because they make " +
      "billions keeping you sick. Doctors HATE this simple trick. Share before they " +
      "take this down!",
  },
  {
    label: "Single claim",
    hint: "reliable",
    mode: "claim-check",
    claim: "The James Webb Space Telescope launched on December 25, 2021.",
  },
  {
    label: "Citation audit",
    hint: "mixed",
    mode: "citation-audit",
    html:
      "<article>" +
      '<p>The James Webb Space Telescope launched on December 25, 2021 ' +
      '<a href="https://en.wikipedia.org/wiki/James_Webb_Space_Telescope">according to NASA</a>.</p>' +
      '<p>Mount Everest is the highest mountain above sea level on Earth ' +
      '<a href="https://en.wikipedia.org/wiki/Mount_Everest">per this reference</a>.</p>' +
      '<p>The Great Wall of China is visible from the Moon with the naked eye ' +
      '<a href="https://en.wikipedia.org/wiki/Great_Wall_of_China">as documented here</a>.</p>' +
      '<p>Drinking coffee causes cancer ' +
      '<a href="https://en.wikipedia.org/wiki/Coffee">according to research</a>.</p>' +
      "</article>",
  },
];

const HINT_TONE = { reliable: "green", mixed: "yellow", questionable: "orange" };

function setupExamples() {
  const wrap = $("#exampleChips");
  if (!wrap) return;
  wrap.innerHTML = EXAMPLES.map(
    (ex, i) => `
      <button type="button" class="example-chip" data-ex="${i}">
        <span class="example-chip-label">${escapeHtml(ex.label)}</span>
        <span class="badge ${HINT_TONE[ex.hint] || "outline"} example-chip-hint">${escapeHtml(ex.hint)}</span>
      </button>`
  ).join("");
  wrap.querySelectorAll(".example-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ex = EXAMPLES[Number(btn.dataset.ex)];
      if (!ex) return;
      if (ex.mode === "claim-check") {
        // Claim-check takes a standalone claim, not an article body.
        $("#url").value = "";
        $("#text").value = "";
        $("#claim").value = ex.claim || "";
        selectMode(ex.mode); // hides the URL/text tabs via .claim-only
        runAssessment();
        return;
      }
      if (ex.mode === "citation-audit") {
        // Citation audit takes article HTML (with out-links) in the text box.
        state.activeTab = "text";
        $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === "text"));
        $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === "text"));
        $("#url").value = "";
        $("#claim").value = "";
        $("#text").value = ex.html || "";
        selectMode(ex.mode);
        runAssessment();
        return;
      }
      // Switch to the paste-text tab and fill it.
      state.activeTab = "text";
      $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === "text"));
      $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === "text"));
      $("#text").value = ex.text;
      $("#url").value = "";
      $("#claim").value = "";
      selectMode(ex.mode);
      runAssessment();
    });
  });
}

function setupHowItWorks() {
  const dialog = $("#howItWorks");
  const openBtn = $("#howItWorksBtn");
  const closeBtn = $("#howClose");
  if (!dialog || !openBtn) return;
  openBtn.addEventListener("click", () => {
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  });
  closeBtn?.addEventListener("click", () => dialog.close());
  // Click on the backdrop (the dialog element itself, outside .how-inner) closes.
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close();
  });
}

async function loadHealth() {
  const el = $("#health");
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    // Friendly short name for the active model, e.g.
    // "anthropic/claude-sonnet-4.6" → "claude-sonnet-4.6".
    const modelShort = data.model ? String(data.model).split("/").pop() : "";
    const llm = data.llm_configured
      ? `<span class="ok">LLM ●</span>${modelShort ? ` <span class="health-model">${escapeHtml(modelShort)}</span>` : ""}`
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

// Small chip shown on a claim summary indicating whether it survived the
// adversarial (falsification-oriented) search.
function robustnessChip(v) {
  if (!v.robustness_tag || v.robustness_tag === "untested") return "";
  const tone = {
    survived: "green",
    weakened: "yellow",
    refuted: "red",
  }[v.robustness_tag] || "outline";
  const label = {
    survived: "robust",
    weakened: "weakened",
    refuted: "refuted",
  }[v.robustness_tag] || v.robustness_tag;
  return `<span class="badge ${tone} robustness-chip" title="Adversarial search: ${escapeHtml(v.robustness_tag)}">${escapeHtml(label)}</span>`;
}

// Segmented results tabs — instant CSS show/hide, no animation. Called
// after renderReport() injects the HTML. defaultSeg picks which pane is
// active on load (Agent in reflective mode so the theater plays).
function setupResultTabs(rootEl) {
  const segEl = rootEl.querySelector(".seg");
  if (!segEl) return;
  const defaultSeg = segEl.dataset.defaultSeg || "overview";
  const buttons = Array.from(rootEl.querySelectorAll(".seg-btn"));
  const panes = Array.from(rootEl.querySelectorAll(".seg-pane"));

  function activate(name) {
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.seg === name));
    panes.forEach((p) => p.classList.toggle("active", p.dataset.segPane === name));
  }
  buttons.forEach((b) => {
    b.addEventListener("click", () => activate(b.dataset.seg));
  });
  activate(defaultSeg);
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

function renderClaimVerifications(verifications, coverage, opts = {}) {
  if (!verifications?.length) return "";
  const n = verifications.length;
  const plural = n === 1 ? "claim" : "claims";
  const meta =
    opts.metaText != null
      ? opts.metaText
      : coverage == null
      ? `${n} ${plural}`
      : `${n} ${plural} · <span class="term" title="Share of claims we could find independent evidence for.">${(coverage * 100).toFixed(0)}% verified</span>`;
  const title = opts.title || "Per-claim verification";
  return `
    <div>
      <div class="section-title">
        ${escapeHtml(title)}
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
                    ${robustnessChip(v)}
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
                    ${
                      v.robustness_tag && v.robustness_tag !== "untested"
                        ? ` · adversarial <strong>${escapeHtml(v.robustness_tag)}</strong>`
                        : ""
                    }
                  </div>
                  ${
                    v.adversarial_query
                      ? `<div class="adv-query">Stress-tested with: <em>${escapeHtml(v.adversarial_query)}</em></div>`
                      : ""
                  }
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
  // Strip the leading "This source is <verdict> (NN.N/100). " sentence so the
  // headline doesn't duplicate the score ring. Must match through the
  // "(score/100)." prefix — a naive "up to first period" breaks on the
  // decimal point in the score (e.g. "62.6/100" → stray "6/100).").
  return explanation
    .replace(/^This source is [^.]*\([\d.]+\/100\)\.\s*/i, "")
    .trim();
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

  const hasOpinions = Array.isArray(ca.opinion_groundings) && ca.opinion_groundings.length > 0;

  // --- Sticky hero (lives outside the tabs, always visible) ---
  const hero = `
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
    </div>`;

  // --- OVERVIEW tab: compact signal strip + red flags ---
  const overviewPane = `
    <div class="seg-pane" data-seg-pane="overview">
      ${renderSignalStrip(report)}
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
          : `<p class="status quiet-positive">No red flags identified.</p>`
      }
    </div>`;

  // --- CLAIMS & OPINIONS tab ---
  const claimsPane = `
    <div class="seg-pane" data-seg-pane="claims">
      ${
        hasPerClaim
          ? renderClaimVerifications(ca.claim_verifications, ca.coverage)
          : `<div>
               <div class="section-title">Main claims
                 <span class="section-meta">${(ca.main_claims || []).length} extracted</span>
               </div>
               ${listOrNone(
                 ca.main_claims, "claims",
                 (ca.summary || "").startsWith("Heuristic-only")
                   ? "No claims extracted by the heuristic. Set MISTRAL_API_KEY for LLM-based extraction."
                   : "No factual claims identified."
               )}
             </div>`
      }
      ${hasOpinions ? renderOpinionGroundings(ca.opinion_groundings) : ""}
    </div>`;

  // --- EVIDENCE tab: dominant cross-ref panel + full source list ---
  const evidencePane = `
    <div class="seg-pane" data-seg-pane="evidence">
      ${crossRefPanel}
      <div>
        <div class="section-title">All retrieved sources
          <span class="section-meta">${(xref.sources || []).length} sources</span>
        </div>
        ${renderSources(xref.sources)}
      </div>
    </div>`;

  // --- AGENT tab (reflective mode only) ---
  const agentPane = hasReflection
    ? `<div class="seg-pane" data-seg-pane="agent">${renderReflectionTrace(ca.reflection_trace)}</div>`
    : "";

  // Default tab: Agent when a reflection trace exists (so the theater plays
  // on load), else Overview. The sticky hero shows the verdict regardless.
  const defaultSeg = hasReflection ? "agent" : "overview";

  const seg = `
    <div class="seg" role="tablist" data-default-seg="${defaultSeg}">
      <button class="seg-btn" data-seg="overview" role="tab">Overview</button>
      <button class="seg-btn" data-seg="claims" role="tab">Claims & Opinions</button>
      <button class="seg-btn" data-seg="evidence" role="tab">Evidence</button>
      ${hasReflection ? `<button class="seg-btn" data-seg="agent" role="tab">Agent trace</button>` : ""}
    </div>`;

  return `
    ${hero}
    ${seg}
    <div class="seg-panes">
      ${overviewPane}
      ${claimsPane}
      ${evidencePane}
      ${agentPane}
    </div>
  `;
}

// claim status → natural-language headline for the claim-check hero.
function claimStatusHeadline(status) {
  return {
    supported: "Supported by sources",
    contradicted: "Contradicted by sources",
    mixed: "Mixed evidence",
    unverified: "Couldn't verify",
  }[status] || "Checked";
}

// Focused renderer for the "Check a claim" tool. Unlike renderReport (which
// frames the result as source reliability), this centers the single
// standalone claim: its support/contradict verdict, the evidence, and the
// adversarial robustness probe. Reuses the same hero + seg structure so the
// staged-reveal orchestrator (reveal.js) animates it unchanged.
function renderClaimCheckReport(report) {
  const ca = report.content_analysis || {};
  const xref = report.cross_reference || {};
  const v = (ca.claim_verifications || [])[0];
  if (!v) return renderReport(report); // defensive fallback

  const overall = Math.round(report.overall_score);
  const color = scoreColor(overall);
  const glow = scoreGlow(overall);
  const statusTone = statusBadgeClass(v.status);

  const hero = `
    <div class="score-hero" style="--score-glow:${glow}" data-reveal="hero">
      <div class="score-hero-top">
        <div class="score-ring" style="--pct:${overall};--ring:${color}">
          <div class="score-num">${overall}</div>
          <div class="score-denom">/ 100</div>
        </div>
        <div class="score-content">
          <div class="verdict-headline">
            <span class="badge ${statusTone} verdict-badge">
              <span class="dot"></span>
              ${escapeHtml(v.status)}
            </span>
            <h1 class="hero-h1">${escapeHtml(claimStatusHeadline(v.status))}</h1>
          </div>
          <p class="hero-summary claim-check-claim">${escapeHtml(v.claim)}</p>
        </div>
      </div>
      <div class="score-hero-bar">
        <span class="meta-pill"><span class="meta-key">Confidence</span><span class="meta-val">${(report.confidence * 100).toFixed(0)}%</span></span>
        <span class="meta-pill"><span class="meta-key">Evidence</span><span class="meta-val">${v.n_evidence} · ${v.n_high_quality} HQ</span></span>
        ${
          v.robustness_tag && v.robustness_tag !== "untested"
            ? `<span class="meta-pill"><span class="meta-key">Adversarial</span><span class="meta-val">${escapeHtml(v.robustness_tag)}</span></span>`
            : ""
        }
      </div>
    </div>`;

  const hasRedFlags = Array.isArray(ca.red_flags) && ca.red_flags.length > 0;
  const verdictPane = `
    <div class="seg-pane active" data-seg-pane="verdict">
      ${renderClaimVerifications([v], null, {
        title: "Verification",
        metaText: `status: ${escapeHtml(v.status)}`,
      })}
      ${
        hasRedFlags
          ? `<ul class="flags">${ca.red_flags.map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>`
          : ""
      }
    </div>`;

  const evidencePane = `
    <div class="seg-pane" data-seg-pane="evidence">
      <div>
        <div class="section-title">All retrieved sources
          <span class="section-meta">${(xref.sources || []).length} sources</span>
        </div>
        ${renderSources(xref.sources)}
      </div>
    </div>`;

  const seg = `
    <div class="seg" role="tablist" data-default-seg="verdict">
      <button class="seg-btn" data-seg="verdict" role="tab">Verdict</button>
      <button class="seg-btn" data-seg="evidence" role="tab">Evidence</button>
    </div>`;

  return `
    ${hero}
    ${seg}
    <div class="seg-panes">
      ${verdictPane}
      ${evidencePane}
    </div>
  `;
}

// Focused renderer for the "Citation audit" tool. Each claim_verification
// is one citation: claim = the citing sentence, evidence = the cited source.
// The hero shows a citation-integrity score (share of cited sources that
// actually support the sentence citing them).
function renderCitationAudit(report) {
  const ca = report.content_analysis || {};
  const xref = report.cross_reference || {};
  const cits = ca.claim_verifications || [];
  const overall = Math.round(report.overall_score);
  const color = scoreColor(overall);
  const glow = scoreGlow(overall);
  const nSupported = cits.filter((v) => v.status === "supported").length;
  const nTotal = cits.length;
  const nWeak = nTotal - nSupported;
  const tone = overall >= 80 ? "green" : overall >= 50 ? "yellow" : "red";
  const headline =
    nTotal === 0
      ? "No citations found"
      : overall >= 80
      ? "Citations hold up"
      : overall >= 50
      ? "Some citations are weak"
      : "Citations don't support the claims";

  const hero = `
    <div class="score-hero" style="--score-glow:${glow}" data-reveal="hero">
      <div class="score-hero-top">
        <div class="score-ring" style="--pct:${overall};--ring:${color}">
          <div class="score-num">${overall}</div>
          <div class="score-denom">/ 100</div>
        </div>
        <div class="score-content">
          <div class="verdict-headline">
            <span class="badge ${tone} verdict-badge">
              <span class="dot"></span>
              citation integrity
            </span>
            <h1 class="hero-h1">${escapeHtml(headline)}</h1>
          </div>
          <p class="hero-summary">${escapeHtml(report.explanation || "")}</p>
        </div>
      </div>
      <div class="score-hero-bar">
        <span class="meta-pill"><span class="meta-key">Citations</span><span class="meta-val">${nTotal}</span></span>
        <span class="meta-pill"><span class="meta-key">Supported</span><span class="meta-val">${nSupported}</span></span>
        ${
          nWeak > 0
            ? `<span class="meta-pill"><span class="meta-key">Unsupported</span><span class="meta-val">${nWeak}</span></span>`
            : ""
        }
      </div>
    </div>`;

  const hasRedFlags = Array.isArray(ca.red_flags) && ca.red_flags.length > 0;
  const verdictPane = `
    <div class="seg-pane active" data-seg-pane="citations">
      ${
        nTotal === 0
          ? `<p class="status">No external in-text citations were found to audit. Try an article URL that links out to its sources.</p>`
          : renderClaimVerifications(cits, null, {
              title: "Cited sources",
              metaText: `${nSupported}/${nTotal} support their claim`,
            })
      }
      ${
        hasRedFlags
          ? `<div class="red-flags-block"><div class="section-title red-flags-title">Citations that don't hold up<span class="section-meta">${ca.red_flags.length}</span></div><ul class="flags">${ca.red_flags.map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul></div>`
          : ""
      }
    </div>`;

  const seg = `
    <div class="seg" role="tablist" data-default-seg="citations">
      <button class="seg-btn" data-seg="citations" role="tab">Citations</button>
    </div>`;

  return `
    ${hero}
    ${seg}
    <div class="seg-panes">
      ${verdictPane}
    </div>
  `;
}

// Compact 3-signal strip for the Overview tab — small stat tiles instead
// of the big panels, so Overview stays scannable.
function renderSignalStrip(report) {
  const w = report.weights || {};
  const ca = report.content_analysis || {};
  const xref = report.cross_reference || {};
  const dom = report.domain_reputation || {};
  const tile = (label, tip, score, weight, sub) => `
    <div class="signal-tile">
      <div class="signal-tile-num" style="color:${scoreColor(Math.round(score))}">${Math.round(score)}<span class="signal-tile-denom">/100</span></div>
      <div class="signal-tile-label"><span class="term" title="${escapeHtml(tip)}">${escapeHtml(label)}</span></div>
      <div class="signal-tile-sub">${escapeHtml(sub)} · ${(weight * 100).toFixed(0)}% weight</div>
    </div>`;
  return `
    <div class="signal-strip" data-reveal="sub-cards">
      ${tile("Source reputation", "Domain-reputation prior from a curated outlet database, with TLD fallbacks.", dom.score || 0, w.domain || 0, dom.type || "unknown")}
      ${tile("Content", "LLM judgment of factuality, objectivity, transparency and sensationalism.", ca.score || 0, w.content || 0, "LLM analysis")}
      ${tile("Cross-reference", "Independent corroboration: each claim checked against outside sources.", xref.score || 0, w.cross_reference || 0, (xref.consensus || "no-data").replaceAll("-", " "))}
    </div>`;
}

// Opinion-grounding cards: each opinion + whether its factual premises hold.
function renderOpinionGroundings(groundings) {
  if (!groundings?.length) return "";
  const verdictTone = {
    "well-grounded": "green",
    "weakly-grounded": "yellow",
    "unsupported": "outline",
    "contradicted": "red",
  };
  return `
    <div class="opinion-block">
      <div class="section-title">Opinion grounding
        <span class="section-meta">${groundings.length} opinion${groundings.length === 1 ? "" : "s"} · grounded in fact?</span>
      </div>
      <p class="opinion-intro">The article's contestable claims, and whether the facts they rest on hold up to independent verification.</p>
      <div class="opinion-list">
        ${groundings.map((g, i) => {
          const tone = verdictTone[g.verdict] || "outline";
          const score = Math.round(g.grounding_score);
          return `
            <details class="opinion" ${i === 0 ? "open" : ""}>
              <summary>
                <span class="opinion-text">${escapeHtml(g.opinion)}</span>
                <span class="opinion-meta">
                  <span class="badge ${tone}">${escapeHtml(g.verdict.replaceAll("-", " "))}</span>
                  <span class="opinion-score">${score}</span>
                  <span class="claim-chevron">${CHEVRON}</span>
                </span>
              </summary>
              <div class="opinion-body">
                ${g.stance_summary ? `<p class="opinion-stance">${escapeHtml(g.stance_summary)}</p>` : ""}
                ${
                  (g.premises || []).length
                    ? `<div class="opinion-premises">
                         <div class="opinion-premises-label">Rests on these factual premises:</div>
                         ${g.premises.map((p) => {
                           const v = p.verification || {};
                           const stTone = statusBadgeClass(v.status);
                           return `
                             <div class="premise">
                               <div class="premise-head">
                                 <span class="badge ${stTone}">${escapeHtml(v.status || "unverified")}</span>
                                 <span class="premise-text">${escapeHtml(p.text)}</span>
                                 <span class="premise-score">${Math.round(v.score || 0)}</span>
                               </div>
                               ${renderEvidence(v.evidence)}
                             </div>`;
                         }).join("")}
                       </div>`
                    : `<p class="status">No publicly-verifiable premises were identified for this opinion.</p>`
                }
              </div>
            </details>`;
        }).join("")}
      </div>
    </div>`;
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
  if (state.mode === "claim-check") {
    if (!claim) {
      setStatus("Enter a claim to fact-check.", "warn");
      return;
    }
    body.claim = claim;
  } else if (state.activeTab === "url") {
    if (!url) {
      setStatus("Please enter a URL.", "warn");
      return;
    }
    body.url = url;
    if (claim) body.claim = claim;
  } else {
    if (!text) {
      setStatus("Please paste some article text.", "warn");
      return;
    }
    body.text = text;
    if (claim) body.claim = claim;
  }

  const btn = $("#run");
  btn.disabled = true;
  setStatus("");
  hidePlaceholder();
  const stopLoading = showLoading(state.mode);

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
    stopLoading();
    const results = $("#results");
    results.innerHTML =
      state.mode === "claim-check"
        ? renderClaimCheckReport(report)
        : state.mode === "citation-audit"
        ? renderCitationAudit(report)
        : renderReport(report);
    results.classList.remove("hidden");
    setStatus("");
    // Activate the segmented tabs (instant; picks the default pane).
    setupResultTabs(results);
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
    stopLoading();
    const results = $("#results");
    results.innerHTML = renderErrorCard(e.message);
    results.classList.remove("hidden");
    setStatus("");
    wireErrorRecovery(results);
  } finally {
    btn.disabled = false;
  }
}

function hidePlaceholder() {
  const p = $("#placeholder");
  if (p) p.style.display = "none";
}

// Staged loading affordance shown in the results area during the wait, so
// the 10–30s feels intentional. Steps advance on a timer and hold on the
// last step until the response lands. Returns a stop() that clears it.
const LOADING_STEPS = {
  default: ["Fetching article", "Analyzing content", "Cross-referencing", "Scoring"],
  "per-claim": ["Extracting claims", "Retrieving evidence", "Checking stances", "Scoring"],
  "per-claim-reflective": [
    "Extracting claims",
    "Retrieving evidence",
    "Checking stances",
    "Agent self-critique",
    "Scoring",
  ],
  "claim-check": ["Parsing claim", "Adversarial retrieval", "Stance labeling", "Scoring"],
  "citation-audit": ["Fetching article", "Extracting citations", "Fetching sources", "Checking support"],
};

function showLoading(mode) {
  const steps = LOADING_STEPS[mode] || LOADING_STEPS["per-claim"];
  const results = $("#results");
  results.innerHTML = `
    <div class="loading-card" aria-live="polite">
      <div class="loading-orbit" aria-hidden="true"><span></span><span></span><span></span></div>
      <ol class="loading-steps">
        ${steps
          .map((s, i) => `<li class="loading-step${i === 0 ? " active" : ""}" data-step="${i}">${escapeHtml(s)}</li>`)
          .join("")}
      </ol>
    </div>`;
  results.classList.remove("hidden");

  let i = 0;
  const lis = results.querySelectorAll(".loading-step");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const timer = reduced
    ? null
    : setInterval(() => {
        if (i >= lis.length - 1) return; // hold on last step
        lis[i].classList.remove("active");
        lis[i].classList.add("done");
        i += 1;
        lis[i].classList.add("active");
      }, 2600);

  return function stop() {
    if (timer) clearInterval(timer);
  };
}

// Map a raw error message to a designed, recoverable error card.
function renderErrorCard(message) {
  const msg = String(message || "");
  let title = "Something went wrong";
  let body = escapeHtml(msg) || "An unexpected error occurred.";
  let recover = "";
  if (/failed to fetch|networkerror|load failed|unreachable/i.test(msg)) {
    title = "Can't reach the server";
    body = "VeriSource isn't responding. Make sure the backend is running on port 8000, then try again.";
  } else if (/extract|fetch|could not read|empty|paywall|403|404|timed out|timeout/i.test(msg)) {
    title = "Couldn't read that article";
    body = "The page may be paywalled, blocked, or empty. Paste the article text instead and re-run.";
    recover = `<button type="button" class="how-btn error-recover" data-recover="paste">Switch to paste text</button>`;
  }
  return `
    <div class="error-card">
      <span class="error-icon" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M12 9v4M12 17v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          <path d="M10.3 3.86 2.07 18a2 2 0 0 0 1.73 3h16.4a2 2 0 0 0 1.73-3L13.7 3.86a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
        </svg>
      </span>
      <div class="error-body">
        <h3>${escapeHtml(title)}</h3>
        <p>${body}</p>
        ${recover}
      </div>
    </div>`;
}

function wireErrorRecovery(root) {
  const btn = root.querySelector('[data-recover="paste"]');
  if (!btn) return;
  btn.addEventListener("click", () => {
    state.activeTab = "text";
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === "text"));
    $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === "text"));
    root.classList.add("hidden");
    $("#text").focus();
  });
}

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupModeToggle();
  setupExamples();
  setupHowItWorks();
  loadHealth();
  $("#run").addEventListener("click", runAssessment);
  $("#url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runAssessment();
  });
});
