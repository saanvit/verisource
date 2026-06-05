# VeriSource — Demo Video Script (~4:30, ≤ 10:00)

**Track:** Application / Product
**Target length:** 4:15–4:45 (the rubric cap is 10 min; 3–5 is ideal). Record at
1×; a live claim-check / citation-audit takes ~15–30 s, so pre-run each once and
cut to the result, or talk over the staged-reveal animation while it loads.

**Setup before recording:**
- Use the **live deployment**: https://veritysource-ylv7z.ondigitalocean.app
  (or `python -m backend.main` → `http://localhost:8000` for full Opus quality).
- Health pill should read `LLM ●  · Search ●` (both green).
- Have these ready to paste:
  - A viral false claim, e.g. *"5G cell towers spread the COVID-19 virus."*
  - A true claim, e.g. *"NASA's Perseverance rover landed on Mars on February 18, 2021."*
  - The citation-audit HTML snippet from the README (2 good + 2 mislinked
    citations) — paste into Citation audit → "Paste text".
  - One article for Deep mode (a viral health/opinion piece shows the agent best).
- Have the LIAR results table from the README open on a slide.
- Hide bookmarks/other tabs; pick a clean recording size.

---

## 0:00–0:30 — Q1: Why I built this (the hook + the insight)

**On screen:** the landing page; the five mode tiles visible.

> "Anyone can publish anything online now, and AI is writing more of it every day
> — fluent, confident, and often wrong. Fact-checking has always taken an
> organization — rooms of researchers at PolitiFact or Snopes. This is one person
> asking how much of that an AI stack can now do on its own.
>
> Most reliability tools just score the *publisher* — Times good, random blog
> bad. They miss two things: a viral claim with no publisher at all, and a
> trustworthy-looking article that links to a source which doesn't actually back
> up what it's citing. VeriSource checks the *claims and citations themselves*.
> And the core idea is dead simple: don't just look for evidence a claim is true
> — go hunting for evidence it's **false**."

---

## 0:25–1:05 — Q2: How it works (architecture)

**On screen:** Click **How it works**; show the three-signal diagram, then close it.

> "Under the hood, it blends three signals.
>
> One: how reputable the source is — using ratings from Media Bias/Fact Check and
> AllSides. Two: an LLM reads the writing itself. And three, the main one —
> cross-referencing.
>
> It breaks the article into individual claims, searches the web for each one, and
> labels every result supports, contradicts, or unclear — weighted by how
> trustworthy the source is. Those labels come from a DeBERTa model I fine-tuned
> myself.
>
> The scores are calibrated, too — so a 70 really does mean about seventy-percent
> likely reliable. And the best part: it's all traceable. Open any claim, see the
> evidence, and check it yourself."

---

## 1:05–1:50 — Q2a: Check a claim (the headline demo)

**On screen:** Click **Check a claim**. Paste **`5G networks spread the COVID-19
virus.`** Hit Assess.  *(verified live: score 3, contradicted, refuted)*

> "Start with the simplest mode — just a claim, no article. It retrieves
> independent evidence, labels each source, and then runs an **adversarial
> search**: a second query deliberately built to falsify the claim."

**On screen:** Result reveals — score 3, **contradicted**, the evidence list,
the **refuted** robustness chip. Expand it to show the adversarial query.

> "It's refuted — the falsification search surfaced credible sources debunking
> it, and I can click straight through to each one."

**On screen:** Paste **`The James Webb Space Telescope launched on December 25,
2021.`**  *(verified live: score 100, supported, survived)*

> "A true claim survives the same probe — supported, and it 'survived' the
> adversarial search instead of being refuted."

---

## 1:50–2:30 — Q2b: Citation audit (the second tool)

**On screen:** Click **Citation audit** → Paste text → paste the HTML snippet →
Assess.

> "The second tool flips it around. Paste an article, and VeriSource fetches
> every source it links to and checks whether that source actually supports the
> sentence citing it."

**On screen:** Result — citation-integrity score; good citations **supported**
(green), mislinked ones **flagged** (red).

> "These two citations check out; these two don't — the linked page just doesn't
> support the claim attached to it. As more of the web gets written by AI, that's
> a growing problem: real-looking links on claims the source never made. This
> makes it visible."

---

## 2:30–3:00 — Q2c: Deep mode — the self-critique agent

**On screen:** Switch to **Deep** mode, run an article, open the **Agent trace** tab.

> "The trickiest step is judging whether a source actually supports a claim — and
> the model sometimes gets it wrong, marking clear evidence as 'unclear.' So Deep
> mode adds a second pass: an agent that checks its own work.
>
> After the first round, Claude reviews every verification, finds the mistakes,
> and fixes them — it'll relabel a source it misjudged, re-search a weak query, or
> split a claim that was really two claims in one."

**On screen:** point at a round card with a relabel + the score change.

> "And here's the part I like: it shows you everything it changed — every
> correction, and how it moved the score. So the most error-prone step in the
> whole system is right out in the open, instead of hidden."

---

## 3:00–3:45 — Evaluation: does it actually work?

**On screen:** the **LIAR results table** on a slide.

> "I evaluated the claim checker on LIAR, a real PolitiFact fact-checking
> benchmark. The full pipeline hits 0.65 accuracy — but the key result is the
> ablation: remove the adversarial search, or use a plain zero-shot LLM, and it
> collapses to chance, because the model just trusts everything. That one step
> lifts macro-F1 from 0.33 to 0.62 — it nearly doubles."

**On screen:** GonzaloA / calibration numbers.

> "On a separate fake-news benchmark, web search lifts ranking from a coin-flip
> to 0.78 AUC, and calibration turns that into accuracy — 0.47 up to 0.80. And
> the fine-tuned NLI labeler beat the LLM labeler, 0.88 to 0.71, while running
> sixty times faster."

**On screen:** the error-analysis bullets.

> "I also looked at where it fails: almost every error is a claim that's
> *literally* true but misleadingly framed — cherry-picked stats, paraphrases —
> which is exactly where a human fact-checker still adds judgment. I'm upfront
> about that, and about the small sample sizes, in the repo."

---

## 3:45–4:15 — Q3: Use cases & impact

**On screen:** back to a clean result; score hero + evidence.

> "This is for anyone who needs a fast, *auditable* reliability read — a reader
> facing a viral claim, a student checking a citation, a journalist triaging a
> source. The point isn't a black-box score you have to trust; every judgment is
> traceable to evidence you can click through, and disagree with."

---

## 4:15–4:40 — Q4: What I'd add next + reproducibility

**On screen:** slide or talking head.

> "Next: a 'framing' check on top of the literal-fact verification to catch those
> misleading-but-true claims, in-line highlighting on the article itself, and a
> browser extension. It's deployed on DigitalOcean, it runs from a clean clone —
> even with no API keys, thanks to a keyless search fallback — and the full
> evaluation, error analysis, and AI-usage disclosure are all in the repo."

**On screen:** quick scroll over the repo / the five tiles.

> "Everything you just saw — the retrieval pipeline, a fine-tuned stance model, a
> self-critique agent, the eval suite against real benchmarks, the deployment —
> is one person with frontier tools building what used to take a whole team.
> That's the bet behind this, and that's VeriSource."

**End card:** `github.com/saanvit/verisource` · live at
`veritysource-ylv7z.ondigitalocean.app`

---

## Coverage checklist (rubric-aligned)

- [x] Q1 Why / bottleneck + the core insight — 0:00
- [x] Q2 How it works — architecture (3 signals, per-claim, fine-tuned NLI,
      calibration) + two tools demoed live + the self-critique agent — 0:25
- [x] Evaluation: LIAR ablation, calibration + NLI results, error analysis — 3:00
- [x] Q3 Use cases / societal value — 3:45
- [x] Q4 Future work + reproducibility — 4:15
- [x] Functional artifact, deployed and end-to-end (Execution)
- [x] Evidence + honest limitations (Evaluation)
- [x] Repo + live URL + docs (Communication, Disclosure)
