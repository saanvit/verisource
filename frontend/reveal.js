/* ============================================================
 * VeriSource — staged-reveal animation orchestrator.
 *
 * Even though the API returns the full report (including the
 * reflection trace) in one shot, we choreograph the display in
 * sequence:
 *   1. Score-hero appears, ring fills with gauge overshoot.
 *   2. Score numeral counts up.
 *   3. Cross-reference panel reveals.
 *   4. Demoted sub-cards stagger in.
 *   5. (Per-claim mode) Per-claim list staggers in.
 *      (Per-claim-reflective mode) Reflection theater:
 *        for each round →
 *          show "Agent is reasoning…" skeleton,
 *          hold ~1.2s,
 *          replace with real round card.
 *   6. Red flags + sources reveal.
 *
 * Respects prefers-reduced-motion: collapses to a single 200ms
 * fade. Auto-skips the theater for slow API responses (>6s).
 * ============================================================ */

(function () {
  "use strict";

  const EASE = "cubic-bezier(0.22, 1, 0.36, 1)";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- low-level helpers ---- */

  function delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function animate(el, keyframes, options) {
    if (!el) return Promise.resolve();
    const opts = Object.assign({ easing: EASE, fill: "both" }, options || {});
    const anim = el.animate(keyframes, opts);
    return anim.finished.catch(() => {}); // swallow cancelations
  }

  function fadeRise(el, duration, delayMs) {
    return animate(
      el,
      [
        { opacity: 0, transform: "translateY(14px) scale(0.985)" },
        { opacity: 1, transform: "translateY(0) scale(1)" },
      ],
      { duration, delay: delayMs || 0 }
    );
  }

  function fadeIn(el, duration, delayMs) {
    return animate(
      el,
      [
        { opacity: 0 },
        { opacity: 1 },
      ],
      { duration, delay: delayMs || 0 }
    );
  }

  /* ---- score ring + numeral tween (with overshoot) ---- */

  function tweenScoreRing(ring, finalScore) {
    if (!ring || finalScore == null) return Promise.resolve();
    if (REDUCED) {
      ring.style.setProperty("--pct", finalScore);
      return Promise.resolve();
    }
    // Two-stage tween for a "gauge settle" feel:
    //   0 → finalScore + 4 (overshoot) over 1000ms
    //   finalScore + 4 → finalScore over 200ms (settle back)
    const overshoot = Math.min(100, finalScore + 4);
    const t0 = performance.now();
    const stage1Ms = 1000;
    const stage2Ms = 200;

    return new Promise((resolve) => {
      function frame(now) {
        const elapsed = now - t0;
        let val;
        if (elapsed < stage1Ms) {
          const t = elapsed / stage1Ms;
          // cubic-bezier(0.22, 1, 0.36, 1) approximation via 1 - (1-t)^3
          const eased = 1 - Math.pow(1 - t, 3);
          val = eased * overshoot;
        } else if (elapsed < stage1Ms + stage2Ms) {
          const t = (elapsed - stage1Ms) / stage2Ms;
          const eased = 1 - Math.pow(1 - t, 2);
          val = overshoot + (finalScore - overshoot) * eased;
        } else {
          val = finalScore;
          ring.style.setProperty("--pct", finalScore);
          resolve();
          return;
        }
        ring.style.setProperty("--pct", val);
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  function countUpNumeral(el, finalValue) {
    if (!el || finalValue == null) return Promise.resolve();
    if (REDUCED) {
      el.textContent = finalValue;
      return Promise.resolve();
    }
    const t0 = performance.now();
    const durMs = 1000;
    const overshoot = Math.min(100, finalValue + 4);
    const settleMs = 200;

    return new Promise((resolve) => {
      function frame(now) {
        const elapsed = now - t0;
        let val;
        if (elapsed < durMs) {
          const t = elapsed / durMs;
          const eased = 1 - Math.pow(1 - t, 3);
          val = Math.round(eased * overshoot);
        } else if (elapsed < durMs + settleMs) {
          const t = (elapsed - durMs) / settleMs;
          val = Math.round(overshoot + (finalValue - overshoot) * t);
        } else {
          el.textContent = finalValue;
          resolve();
          return;
        }
        el.textContent = val;
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  /* ---- delta-badge pulse on landing ---- */

  function pulseBadge(el) {
    if (!el || REDUCED) return Promise.resolve();
    return animate(
      el,
      [
        { transform: "scale(1)" },
        { transform: "scale(1.08)" },
        { transform: "scale(1)" },
      ],
      { duration: 320 }
    );
  }

  /* ---- reflection theater ---- */

  // Build a skeleton element that shows "Agent is reasoning…" with a
  // shimmer effect. Sized to match the eventual round card's height so
  // there's no layout shift on swap.
  function buildThinkingSkeleton(targetHeight) {
    const el = document.createElement("div");
    el.className = "reflection-thinking";
    el.style.minHeight = targetHeight ? `${targetHeight}px` : "120px";
    el.innerHTML = `
      <div class="thinking-shimmer" aria-hidden="true"></div>
      <div class="thinking-content">
        <span class="thinking-dots" aria-hidden="true">
          <span></span><span></span><span></span>
        </span>
        <span class="thinking-label">Agent is reasoning…</span>
      </div>`;
    return el;
  }

  async function playReflectionTheater(traceContainer) {
    if (!traceContainer) return;
    const rounds = Array.from(traceContainer.querySelectorAll(".reflection-round"));
    if (rounds.length === 0) return;

    // Reveal the section header instantly (the rounds are the theater).
    const introEls = traceContainer.querySelectorAll(".section-title, .reflection-intro");
    introEls.forEach((el) => fadeIn(el, 320));

    if (REDUCED) {
      // Just show everything at once.
      rounds.forEach((r) => (r.style.opacity = "1"));
      return;
    }

    // Measure each round's height up front so the skeleton matches
    // and there's no layout jump. Then hide the round visually until
    // its turn.
    const heights = rounds.map((r) => r.offsetHeight || 120);
    rounds.forEach((r) => {
      r.style.opacity = "0";
      r.style.transform = "translateY(14px) scale(0.985)";
      r.style.willChange = "opacity, transform";
      r.style.display = "none";
    });

    for (let i = 0; i < rounds.length; i++) {
      // 1. Insert skeleton at the round's position.
      const round = rounds[i];
      const skeleton = buildThinkingSkeleton(heights[i]);
      round.parentNode.insertBefore(skeleton, round);

      // 2. Hold for thinking time (1200ms for round 1, 900ms after).
      await delay(i === 0 ? 1200 : 900);

      // 3. Swap skeleton out, swap round in with a fade-rise.
      skeleton.style.transition = "opacity 240ms";
      skeleton.style.opacity = "0";
      setTimeout(() => skeleton.remove(), 260);
      round.style.display = "";
      // Auto-open latest round, close all prior.
      rounds.forEach((r, j) => {
        if (j < i) r.removeAttribute("open");
      });
      round.setAttribute("open", "");

      await fadeRise(round, 480, 60);

      // 4. Pulse score-delta badges in this round.
      const deltaBadges = round.querySelectorAll(".reflection-deltas .badge");
      deltaBadges.forEach((b, k) => setTimeout(() => pulseBadge(b), k * 100));
    }

    // After all rounds, reopen round 1 so the user can review it.
    await delay(400);
    rounds.forEach((r, i) => {
      if (i === 0) r.setAttribute("open", "");
      else r.removeAttribute("open");
    });
  }

  /* ---- main orchestrator ---- */

  async function revealReport(rootEl, opts) {
    if (!rootEl) return;
    const options = opts || {};
    const skipTheater = !!options.skipTheater;
    const finalScore = options.score;

    const hero = rootEl.querySelector(".score-hero");
    const ring = rootEl.querySelector(".score-ring");
    const numeral = rootEl.querySelector(".score-ring .score-num");
    const heroBadges = rootEl.querySelectorAll(".score-hero-bar .meta-pill, .verdict-badge");
    const reflectionEl = rootEl.querySelector(".reflection-trace");
    const crossRefEl = rootEl.querySelector(".cross-ref-panel");
    const subCardsEl = rootEl.querySelector(".grid-2");
    const claimList = rootEl.querySelector(".claim-list");
    const redFlagsEl = rootEl.querySelector(".red-flags-block");
    const sourcesEl = rootEl.querySelector('[data-reveal="sources"]');

    // Initial state: everything invisible.
    [hero, reflectionEl, crossRefEl, subCardsEl, claimList, redFlagsEl, sourcesEl].forEach((el) => {
      if (!el) return;
      el.style.opacity = "0";
      el.style.transform = "translateY(14px)";
      el.style.willChange = "opacity, transform";
    });
    if (numeral) numeral.textContent = "0";
    if (ring) ring.style.setProperty("--pct", "0");

    // If the API took too long, skip the theater for everyone's sanity.
    if (REDUCED || skipTheater) {
      [hero, reflectionEl, crossRefEl, subCardsEl, claimList, redFlagsEl, sourcesEl].forEach((el) => {
        if (!el) return;
        el.style.opacity = "1";
        el.style.transform = "none";
      });
      if (ring && finalScore != null) ring.style.setProperty("--pct", finalScore);
      if (numeral && finalScore != null) numeral.textContent = finalScore;
      return;
    }

    /* === 1. Score-hero reveal === */
    await fadeRise(hero, 600);

    /* === 2. Ring fill + numeral count-up in parallel === */
    await Promise.all([
      tweenScoreRing(ring, finalScore),
      countUpNumeral(numeral, finalScore),
    ]);

    /* === 3. Hero meta badges stagger === */
    heroBadges.forEach((b, i) => {
      b.style.opacity = "0";
      b.style.transform = "translateY(6px)";
      setTimeout(() => {
        animate(
          b,
          [
            { opacity: 0, transform: "translateY(6px)" },
            { opacity: 1, transform: "translateY(0)" },
          ],
          { duration: 320, delay: 0 }
        );
      }, i * 40);
    });

    /* === 4. Cross-reference panel === */
    if (crossRefEl) {
      await delay(80);
      await fadeRise(crossRefEl, 500);
    }

    /* === 5. Sub-cards stagger === */
    if (subCardsEl) {
      const cards = subCardsEl.querySelectorAll(".card");
      subCardsEl.style.opacity = "1";
      subCardsEl.style.transform = "none";
      cards.forEach((c) => {
        c.style.opacity = "0";
        c.style.transform = "translateY(10px)";
      });
      const promises = Array.from(cards).map((c, i) => fadeRise(c, 400, i * 80));
      await Promise.all(promises);
    }

    /* === 6. Reflection theater OR per-claim list === */
    if (reflectionEl) {
      reflectionEl.style.opacity = "1";
      reflectionEl.style.transform = "none";
      await playReflectionTheater(reflectionEl);
    }

    if (claimList) {
      const items = claimList.querySelectorAll(".claim");
      claimList.style.opacity = "1";
      claimList.style.transform = "none";
      items.forEach((c) => {
        c.style.opacity = "0";
        c.style.transform = "translateY(10px)";
      });
      Array.from(items).forEach((c, i) => fadeRise(c, 380, i * 60));
      // Reveal the parent (section-title etc.) immediately
      const parent = claimList.parentElement;
      if (parent && parent !== rootEl) fadeIn(parent.querySelector(".section-title"), 320);
    }

    /* === 7. Red flags + sources (parallel tail) === */
    if (redFlagsEl) fadeRise(redFlagsEl, 380, 100);
    if (sourcesEl) fadeRise(sourcesEl, 380, 200);
  }

  // Expose to global scope (no module bundler in this project).
  window.revealReport = revealReport;
})();
