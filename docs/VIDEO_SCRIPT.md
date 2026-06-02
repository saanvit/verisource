# VeriSource — Demo Video Script (≤ 3:00)

**Track:** Application / Product
**Target length:** 2:40–2:55. Record at 1× speed; if a real assessment takes
~20–30 s, pre-run it and cut to the result (or use a Quick-mode run live).

**Setup before recording:**
- Server running: `python -m backend.main` → open `http://localhost:8000`.
- Health pill should read `LLM ● claude-opus-4.8 · Search ●`.
- Have the 2026 Texas Senate article URL (or text) ready in your clipboard — it
  shows the adversarial-refutation and the Deep-mode self-correction well.
- Pick a clean screen recording size; hide bookmarks/other tabs.

---

## 0:00–0:25 — Q1: Why I built this (the bottleneck)

**On screen:** VeriSource landing page (the dotted-grid hero, depth toggle).

> "Anyone can publish anything, and checking whether a claim actually holds up
> means opening ten tabs and reading every source yourself. That manual
> cross-referencing is the bottleneck. VeriSource automates it: paste a URL,
> and it decomposes the article into atomic claims, searches the open web for
> each one, and tells you which claims are actually supported by independent
> evidence — with its work shown."

---

## 0:25–1:40 — Q2: How it works (architecture, live run)

**On screen:** Click **How it works** → show the three-signal diagram briefly,
then close it and paste the Texas article. Choose **Standard**, hit Assess.

> "Three independent signals get fused. One: a domain-reputation prior from
> Media Bias/Fact Check, Ad Fontes, and AllSides. Two: an LLM content analysis
> — factuality, objectivity, transparency. Three, the core: per-claim
> verification."

**On screen:** Results reveal. Open the **Claims** tab; expand a claim.

> "Each atomic claim gets its own web search, and every retrieved source is
> stance-labeled — supports, contradicts, or unclear — then weighted by that
> source's own reputation. And it doesn't just look for confirmation: an
> adversarial search actively tries to *falsify* each claim."

**On screen:** Point at a claim with a **robustness chip** (survived/refuted).

> "This claim was stress-tested with a falsification query and survived — that's
> the robustness tag. The scores are isotonic-calibrated, so a 70 actually
> means roughly 70-percent likely reliable."

---

## 1:40–2:15 — Q2 cont.: The agent (the differentiator)

**On screen:** Re-run (or switch to) **Deep** mode; open the **Agent trace** tab.

> "Stance labels are the hard part — labelers sometimes mark obviously-
> supporting evidence as contradicting. Deep mode adds a self-critique agent:
> Claude audits its own verifications, catches mislabels, re-searches weak
> queries, and splits compound claims — and you can read every correction it
> made, right here."

**On screen:** Briefly show a round card with a relabel/score-delta.

> "I caught this exact failure live — a 'won the primary' claim was wrongly
> labeled refuted — which is what drove the agent design and a switch to a
> stronger stance model."

---

## 2:15–2:40 — Q3: Use cases & impact

**On screen:** Back to a clean result; the score hero + verdict.

> "This is for anyone who needs a fast, auditable reliability read — journalists
> triaging a source, students checking a citation, or readers facing a viral
> claim. The point isn't a black-box score; it's that every judgment is
> traceable to evidence you can click through and disagree with."

---

## 2:40–2:55 — Q4: What I'd add next

**On screen:** Slide or talking head.

> "Next: in-line claim highlighting on the article text itself, a
> claim-verifiability benchmark like AVeriTeC where the per-claim pipeline
> should win, and a browser extension to assess any page you're reading. The
> evaluation, failure analysis, and AI-usage disclosure are all in the repo."

**End card:** `github.com/saanvit/verisource`

---

## Coverage checklist (rubric-aligned)

- [x] Q1 Why / bottleneck — 0:00
- [x] Q2 How it works (product architecture: 3 signals, per-claim, calibration, agent) — 0:25
- [x] Q3 Use cases / societal value — 2:15
- [x] Q4 Future work — 2:40
- [x] Shows a functional artifact end-to-end (Execution)
- [x] Shows evidence/limitations awareness (the live failure case) (Evaluation)
- [x] Points to repo + docs (Communication, Disclosure)
