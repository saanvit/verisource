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

  // An element is "revealable" only if it exists AND is actually visible
  // (not inside a display:none tab pane). Animating or measuring hidden
  // nodes either no-ops or returns offsetHeight=0, which breaks the
  // reflection-theater height pre-measurement. Guard everything with this.
  function visible(el) {
    return !!(el && el.offsetParent !== null);
  }

  function animate(el, keyframes, options) {
    if (!visible(el)) return Promise.resolve();
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
    // Only animate the ACTIVE pane's content — hidden panes (display:none)
    // must keep their natural opacity so they look right when tab-switched.
    const activePane = rootEl.querySelector(".seg-pane.active");
    const reflectionEl = activePane && activePane.querySelector(".reflection-trace");

    // Initial state: hide only the always-visible hero + the active pane.
    if (hero) {
      hero.style.opacity = "0";
      hero.style.transform = "translateY(14px)";
      hero.style.willChange = "opacity, transform";
    }
    if (activePane) {
      activePane.style.opacity = "0";
      activePane.style.willChange = "opacity";
    }
    if (numeral) numeral.textContent = "0";
    if (ring) ring.style.setProperty("--pct", "0");

    // If the API took too long or reduced-motion, skip the theater.
    if (REDUCED || skipTheater) {
      if (hero) { hero.style.opacity = "1"; hero.style.transform = "none"; }
      if (activePane) activePane.style.opacity = "1";
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

    /* === 4. Reveal the active pane, then stagger its top-level children === */
    if (activePane) {
      activePane.style.opacity = "1";
      // If this pane is the Agent trace, play the reflection theater.
      if (reflectionEl) {
        await delay(120);
        await playReflectionTheater(reflectionEl);
      } else {
        // Otherwise stagger the pane's direct content blocks in.
        const blocks = Array.from(activePane.children);
        blocks.forEach((c) => {
          c.style.opacity = "0";
          c.style.transform = "translateY(12px)";
        });
        await delay(80);
        await Promise.all(blocks.map((c, i) => fadeRise(c, 440, i * 90)));
      }
    }
  }

  // Expose to global scope (no module bundler in this project).
  window.revealReport = revealReport;
})();
