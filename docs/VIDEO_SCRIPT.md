# VeriSource — Demo Video Script (≤ 3:00)

**Track:** Application / Product
**Target length:** 2:40–2:55. Record at 1×; a live claim-check / citation-audit
takes ~15–30 s, so pre-run each once and cut to the result, or talk over the
staged-reveal animation while it loads.

**Setup before recording:**
- Use the **live deployment**: https://veritysource-ylv7z.ondigitalocean.app
  (or `python -m backend.main` → `http://localhost:8000` for full Opus quality).
- Health pill should read `LLM ●  · Search ●` (both green).
- Have these ready to paste:
  - A viral false claim, e.g. *"5G cell towers spread the COVID-19 virus."*
  - The citation-audit HTML snippet from the README (2 good + 2 mislinked
    citations) — paste into Citation audit → "Paste text".
- Hide bookmarks/other tabs; pick a clean recording size.

---

## 0:00–0:20 — Q1: Why I built this (the bottleneck + the insight)

**On screen:** the landing page; the five mode tiles visible.

> "Most 'is this reliable?' tools score the *publisher*. But two things slip
> through: a viral claim with no publisher at all, and a respectable article
> that links to a source which doesn't actually back up what it's citing.
> VeriSource checks the *claims and the citations themselves* — and the core
> idea is simple: don't just look for confirmation, actively search for evidence
> that a claim is **false**."

---

## 0:20–1:10 — Q2a: Check a claim (the headline demo)

**On screen:** Click **Check a claim**. Paste *"5G cell towers spread the
COVID-19 virus."* Hit Assess.

> "No article needed — just the claim. VeriSource retrieves independent
> evidence, labels each source supports / contradicts / unclear, and then runs
> an **adversarial search** that deliberately tries to falsify the claim."

**On screen:** Result reveals — low score, **contradicted**, the evidence list,
and the **refuted** robustness chip. Expand it to show the adversarial query.

> "Here it's refuted: the falsification search surfaced credible sources
> debunking it, and you can click straight through to every one. Compare that to
> a true claim —" *(optionally paste "NASA's Perseverance rover landed on Mars
> on February 18, 2021." → supported / survived)* "— which survives the same
> adversarial probe."

---

## 1:10–1:50 — Q2b: Citation audit (the second tool)

**On screen:** Click **Citation audit** → Paste text → paste the HTML snippet →
Assess.

> "Second tool: paste an article and VeriSource fetches every source it links to
> and checks whether that source actually supports the sentence citing it."

**On screen:** Result — a citation-integrity score, with the good citations
**supported** (green) and the mislinked ones **flagged** (red).

> "These two citations check out; these two don't — the linked page doesn't
> support the claim attached to it. That's citation laundering and
> hallucinated references made visible — something publisher-level scoring can't
> catch."

---

## 1:50–2:25 — Q2c + Evaluation: does it actually work?

**On screen:** Cut to the **LIAR results table** (from the README / a slide).

> "I evaluated the claim checker on LIAR, a real fact-checking benchmark. The
> full pipeline hits 0.65 accuracy — but the interesting part is the ablation:
> remove the adversarial search, or use a plain zero-shot LLM, and accuracy
> collapses to chance because the model just calls everything 'reliable.' The
> adversarial step is what does the work — it lifts macro-F1 from 0.33 to 0.62."

**On screen:** the error-analysis bullet / a couple of misclassified examples.

> "And I looked at where it fails: almost every error is a *subtly* false claim
> that's literally true but misleadingly framed — which is exactly where human
> fact-checkers add judgment. I'm honest about that limit in the repo."

---

## 2:25–2:45 — Q3: Use cases & impact

**On screen:** back to a clean result; score hero + evidence.

> "This is for anyone who needs a fast, *auditable* reliability read — a reader
> facing a viral claim, a student checking a citation, a journalist triaging a
> source. The point isn't a black-box score; every judgment is traceable to
> evidence you can click through and disagree with."

---

## 2:45–2:55 — Q4: What I'd add next

**On screen:** slide or talking head.

> "Next: a 'framing' check layered on the literal-fact verification to catch
> misleading-but-true claims, in-line highlighting on the article itself, and a
> browser extension. It's deployed, the code is reproducible, and the full
> evaluation, error analysis, and AI-usage disclosure are in the repo."

**End card:** `github.com/saanvit/verisource` · live at
`veritysource-ylv7z.ondigitalocean.app`

---

## Coverage checklist (rubric-aligned)

- [x] Q1 Why / bottleneck + the core insight — 0:00
- [x] Q2 How it works — two tools demoed live (claim-check, citation-audit) + the
      adversarial-retrieval mechanism — 0:20
- [x] Evaluation: LIAR ablation table + error analysis (limitations) — 1:50
- [x] Q3 Use cases / societal value — 2:25
- [x] Q4 Future work — 2:45
- [x] Functional artifact, deployed and end-to-end (Execution)
- [x] Evidence + honest limitations (Evaluation)
- [x] Repo + live URL + docs (Communication, Disclosure)
