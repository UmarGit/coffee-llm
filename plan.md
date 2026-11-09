# Plan: from neg-v1 results to an ACL submission

Written from the reviewer's side of the table. The question at every step is
what an ACL area chair will read, what they will doubt, and what evidence
removes that doubt before they can raise it.

Inputs: `analysis/report.md` (findings), `analysis/out/claims.md` (every
number), `attacks.md` (13 reviewer attacks, 12 supervisor comments). Only
neg-v1 results are used. `runs-*` directories are not used anywhere.

## Status (2026-11-09)

- **Paper (P0 to P4): done for Gate D.** ACL template; generators read neg-v1
  through `tools/analysis`; all tables and figures new; `make -C paper check`
  reports 0 overfull, 0 undefined, 0 em dashes; body ends on page 6 of 8.
  `make -C paper review` builds the anonymous PDF. Supervisor comments B1 to B12
  are all addressed (see `attacks.md`). One TODO left in the source: total GPU
  hours for the Ethics statement.
- **Site (W): data and text done; model swap waiting.** `site/data/` rebuilt from
  the campaign, guardrail section shows corrections, build and Chromium/WebKit
  QA pass, matcher parity 100% on 507,618 predictions. The live demo still runs
  the October neg30 model until the campaign GGUF is uploaded to Hugging Face.
  Not deployed.
- **Experiments: X1 and X3 ready to run** (`scripts/followup.sh`, handover Part
  5). X2 needs WORDS and EXR grammars written first.
- Open: the number check that fails on hand-typed digits, the `bench` page,
  supplementary archive, code version `0434a53` (A13).

## 0. Gates

| Gate | What it means | Who signs off |
|---|---|---|
| **D. Supervisor draft** | Paper on the ACL template, every number from neg-v1, every supervisor comment (B1–B12) addressed or answered, site updated | Authors, then send |
| **E. Reviewer-proof** | Control (A2), constrained decoding (A3) and LR edges (A6) in the paper; supervisor feedback on D merged | Supervisors |
| **F. Submit** | Anonymised PDF, checklist, supplementary code, page limit met, internal mock review passed | All authors |

Gate D needs no GPU. Gate E needs about 1.5 GPU-days. Before planning backwards
from the deadline, confirm the next ARR cycle date and the commitment venue
(ACL main or Findings) on the ARR site. Don't rely on memory for those dates.

## 1. What the paper claims after neg-v1

**Title (working):** *Parsed, Plausible, Wrong: Output Format Controls Silent
Errors in Small-Model Semantic Parsing.*

**One-sentence claim:** On a narrow parsing task, writing the target in natural
words rather than symbols makes a sub-1B model as accurate as the native
format. It uses a quarter of the tokens, and silent errors fall by about half.
This holds across five data sizes, five seeds, two model families, and with a
refusal and self-correction guardrail in training.

**Contributions (rewritten):**
1. A six-format ladder of lossless encodings of PIZZA, with two matched-length
   pairs that separate length from surface form (WORDS/DSL, EXR/JSON).
2. Length alone is not the mechanism. WORDS beats DSL at equal length in
   10/10 model×size cells, separating in 9 after Holm correction. Shortening
   EXR to EXR-short never helps.
3. The silent error rate as the deployment objective, reported as part of a
   four-way split A + S + D-parse + D-refuse = 1. WORDS cuts S by 2.3x on Qwen
   and 1.8x on LFM2, separating at every size. The error analysis points to
   negation as the mechanism.
4. A guardrail that trains refusal of off-menu input and in-utterance
   self-correction. It holds for every format at 96–99% on held-out
   negatives. It has a measured cost: 2–10% of polite real orders are wrongly
   refused. WORDS learns corrections best.
5. A statistical protocol fixed before results were read: 5 seeds, a
   hierarchical bootstrap over seeds and utterances, and Holm correction per
   family. This replaces the single-rerun "noise floor".

**Dropped or narrowed (C1, C9):**
- Drop "every ten tokens removed costs X points".
- Narrow the data-efficiency claim to the WORDS−EXR gap, and to "pool size"
  rather than "examples seen".
- Remove the café task as evidence.

## 2. Workstream P: the paper

### P0. Template and plumbing (first, one day)

- [x] Move `paper/paper.tex` from IEEEtran to the official ACL style
      (`acl.sty`, `review` option for line numbers and anonymity). Long paper:
      8 pages of content; Limitations, Ethics and references don't count;
      appendices are allowed. Re-check the current call for any change.
- [x] Single source of numbers. `make_tables.py` and `make_figures.py` read
      `report.primary_runs()` from `runs/`, i.e. old runs. Point them at
      `neg-v1-results/campaign` (via `BENCH_RUNS`, or by reading
      `analysis/out/tables/*.csv`). Recommended: the generators import
      `tools/analysis/data.py` and `stats.py`, so the paper's intervals are the
      same bootstrap draws as the analysis. No second implementation of the
      statistics.
- [x] Macros. Delete `\noiseFloor` and `\slopeEmSmall`. Add macros for seeds,
      run counts (300 grid, plus 78 sweep and determinism runs), the S ratio
      and CI per model, the false-reject range, corrections EM, and
      negatives handled.
- [x] `make -C paper check` passes: no overfull boxes, no em dashes, page
      count printed. Add a check that fails on any number in `paper.tex`
      typed outside a macro, with a small allowlist (years, model sizes).

### P1. Structure and page budget (8 pages)

| § | Section | Pages | Changes |
|---|---|---|---|
| 1 | Introduction | 1.0 | Café shrinks to 2 sentences of motivation (B1). Metric order stated here (B12). New contributions |
| 2 | Related work | 0.6 | Add output-format and structured-output work, constrained decoding, refusal/abstention, self-correction / disfluency in task-oriented parsing |
| 3 | Task, data and formats | 1.2 | PIZZA plus the pool statement (B2, exact wording in attacks.md). Ladder, matched pairs, tokenization figure with characters (B3). The "seventh encoding" as a one-line measurement or removed (B4) |
| 4 | Guardrail data | 0.5 | Negatives (kinds, per-format refusal string), corrections (9 kinds, held-out cues), probes. Why each format gets its own refusal spelling |
| 5 | Setup and statistics | 0.8 | Models, LoRA (r=16, α=16, with citation and the reason it is held fixed: B5, A11). LR sweep and checkpoint criterion. Examples seen vs pool size (A9). The bootstrap and Holm protocol, pre-registered |
| 6 | Results | 3.0 | 6.1 Main table (both models). 6.2 Length vs form (pairs across sizes). 6.3 Silent errors and why (negation). 6.4 Guardrail. 6.5 Pool size. 6.6 Latency (short) |
| 7 | Discussion | 0.5 | When to use EM vs S. Strict refusal as a design choice. What transfers beyond PIZZA |
| 8 | Conclusion | 0.2 | |
| – | Limitations | (free) | One dataset; author-generated guardrail sets; human test set contains neither negatives nor corrections; 305-word grammar; 32 cue-phrase orders (A7); 8 LFM2 runs near step limit; LR grid edges; LFM2 GGUF agreement 93% |
| – | Ethics | (free) | PIZZA licence CC-BY-NC-4.0, no personal data, compute used |
| App. | A integrity checks, B per-size tables, C guardrail by kind, D LR choices, E prompts and formats in full, F deployment (iPad, café) as a demonstration only | | |

### P2. Tables

Every table shows A and S side by side (B12). Every number has a spread or an
interval (B6, B8).

| ID | Content | Source | Fixes |
|---|---|---|---|
| **T1** | Main: both models × six formats, full pool. A ± SD, S ± SD, D-parse, D-refuse, tokens, latency ms. Best per column bold, ties (not separating from the best) underlined | `cells.csv`, latency jsonl | B6, B8 (the replication table merges in here) |
| **T2** | Paired differences: rows WORDS−DSL, EXR−JSON, EXR−EXR-short, WORDS−EXR; columns 5 sizes × 2 models; cells Δ with a dagger when Holm-separated; 95% CIs in appendix B | `c2..c5_*.csv` | B7. The caption says why JSON is there: it is EXR's matched-length control |
| **T3** | Guardrail at the full pool: negatives handled, false reject, corrections EM, corrections applied, probes /40, reported failure fixed /5 | `cells.csv`, N5 | new |
| T-A1..A5 | Appendix: EM and S by size; negatives by kind (N1); corrections by kind (N4, N4b); LR choices with dev EM margins; integrity summary | claims.md, integrity.txt | A6, A10 |

The current Table 2 (one size, per-seed rows) and Table 4 (café compact) are
deleted (B7, B9). Current Table 3 (replication) merges into T1.

### P3. Figures

Palette: Okabe–Ito (colour-blind safe) in `paper/palette.py`. Every series also
gets a distinct marker and line style, so it survives greyscale (B10, B11).
Label lines directly instead of using a legend where possible. Show 95%
bootstrap bands or seed-SD error bars on every data point.

| ID | Figure | Change |
|---|---|---|
| F1 | Pipeline | Keep only if space allows; candidate for cutting |
| F2 | Tokenization of one order in six formats | Add a character-width track per format on the same scale (B3) |
| F3 | EM vs tokens, both models, full pool | Add CIs; dashed lines for the matched pairs |
| F4 | Error composition (current Fig. 5) | Four-way stacked bars, both models; greyscale-safe hatching (B10) |
| F5 | EM by pool size (current Fig. 7) | Both models, bands, markers; x-axis labelled "training pool size" (B11, A9) |
| F6 | Latency vs tokens | Small; can move to the appendix |
| F7 (new) | Silent errors by type (negation, number of lines, entity, other) per format | The mechanism figure for contribution 3 |

### P4. Supervisor comments: where each one lands

| # | Comment (short) | Status today | Lands in |
|---|---|---|---|
| B1 | Why the café section | fix in text | §1 two sentences; App. F |
| B2 | Pool cleaned or not | fix in text (wording ready) | §3 |
| B3 | Fig. 3: characters | fix in figure | F2 |
| B4 | Seventh encoding unsupported | fix in text | §3, one sentence or removed |
| B5 | Why r=16, α=16 | fix in text (+ optional sweep A11) | §5 |
| B6 | Spread only for EM | **answered by neg-v1** | T1 |
| B7 | Table 2 artificial | **answered by neg-v1** | T2 |
| B8 | Table 3 no intervals | **answered by neg-v1** | T1 |
| B9 | Café examples own-made | removed with B1 | App. F, as a demonstration |
| B10 | Fig. 5 colours | fix in figure | F4 |
| B11 | Fig. 7 greyscale, no CIs | fix in figure | F5 |
| B12 | EM vs silent: which matters | fix in text | §1, §5, every table |

The draft sent at Gate D carries a one-page response to the supervisors, in
this table's form, so each comment can be checked against a page and line.

## 3. Workstream X: experiments

Run in this order; each one closes a named attack.

| # | Run | Closes | Size | Decision rule |
|---|---|---|---|---|
| X1 | **Guardrail ablation.** WORDS and EXR, full pool, 5 seeds, both models, for `--negatives 0.3 --corrections 0` and for `--negatives 0 --corrections 0` | A2, A7 | 40 runs, ~15 GPU-h on the two GPUs | Reports the cost of the guardrail in A and S. If the format result is unchanged without the guardrail, it becomes a robustness sentence. If it changes, it becomes a subsection |
| X2 | **Constrained decoding.** `bench grammar` today writes a DSL grammar only. Add GBNF grammars for WORDS and EXR (refusal string included), then decode the 5 full-pool seeds of each with and without the grammar | A3 | code ~1 day, inference only | Shows where unparseable outputs go. The expected outcome is that S cannot fall below the unconstrained S; this is reported whatever happens |
| X3 | **LR edges.** Add 2e-5 and 1e-3 to the sweep for Qwen EXR-short, Qwen JSON, LFM2 DSL | A6 | 3 arms × 2 LRs | Rerun the grid cells only if a new LR wins on dev |
| X4 | Rank sweep r ∈ {8, 16, 32}, WORDS and EXR, one size, 3 seeds | A11, B5 | ~half a day | Optional; the paper stands on the citation |
| X5 | Larger model (Qwen ~2–4B), WORDS/DSL/EXR, full pool, 3–5 seeds | A5 | ~1 day | Optional; decides whether "small models" stays in the title |
| X6 | Second dataset (TOPv2 or MTOP), same ladder and pairs, one model | A1 | 1–2 weeks | The step from Findings to main track. Go/no-go at Gate D, based on the time left before the deadline |
| X7 | Independent challenge set, 2–3 annotators outside the project, ~300 inputs, labelled blind | A4 | people time | Optional; strongest answer to "your own templates" |

X1–X3 are required for Gate E. X4–X7 are decided at Gate D.

Every new run records the VERSION file (A13). Before X1 starts, recover the
code version of the original campaign (`0434a53`) from the server's source
tree. If it can't be recovered, say so in Reproducibility and pin the current
commit for all new runs.

## 4. Workstream W: the website

The site is public and carries names, so it is **not linked from the
submission** (A12). Its numbers must still agree with the paper, because
reviewers search.

- [x] `tools/export_site_data.py`: read from `neg-v1-results/campaign` instead
      of `runs/`. Regenerate `site/data/*.json` and `site/raw/*.json`.
      `guardrail.json` and `model.json` currently cite the old run
      `words-n2456443-s0-lr0.0001-neg30-fresh-20261029`.
- [ ] Model: publish the campaign's Qwen WORDS GGUF (seed 0, full pool) and
      point `model.json` at it. The LFM2 GGUF stays unpublished, or is
      published with its 93% agreement stated.
- [x] Sections whose text carries numbers:
      - `rank`: the ladder is not monotone; drop any per-token cost.
      - `silent`: `silentRatio` 2.3x Qwen.
      - `race`: `speedup` 3.7x.
      - `again`: LFM2, WORDS 12.0 vs DSL 11.4 tokens, +0.054.
      - `gate`: refusal plus the new corrections, including the fixed
        "wait no cheese" example.
- [x] New or changed content: a short "listens to corrections" beat in
      `gate` (the reported failure, now fixed in 60/60 runs). Optional: the
      false-reject cost stated honestly.
- [x] `cafe` section: keep on the site as the origin story; the paper no
      longer relies on it.
- [ ] `bench` page (new, uncommitted: `bench.astro`, `bench.ts`,
      `bench_orders.json`, `catalog.json`, `words.ts`): finish against the new
      model or hold it back. Check the PIZZA licence for redistributing test
      orders in `bench_orders.json` (CC-BY-NC allows it with attribution).
- [ ] `npm run build` (runs `check-data.mjs`), `npm run check`, `npm run parity`,
      `npm run qa`, then deploy.

## 5. Workstream R: release and submission hygiene

- [ ] Anonymity. No names, HF user, GitHub, site URL or "our deployed café"
      details that identify the client. The iPad measurements are described
      without naming the client.
- [ ] Supplementary material: anonymised code archive (with `tools/analysis/`,
      the generators, the campaign scripts). No PIZZA-derived data in it; give
      instructions to download PIZZA.
- [ ] ARR Responsible NLP checklist: compute (GPU type, hours), licences,
      hyperparameters, seeds, statistics, the human-written test set.
- [ ] The camera-ready release (code, GGUF, site) is prepared separately and
      only after acceptance.

## 6. Mock review before Gate F

Two people outside the project, or the supervisors, review the Gate E draft
on the ARR form: soundness, excitement, overall, confidence. Ask them for the
weakest claim specifically. Target before submitting: no "Soundness ≤ 2"
comment that the paper doesn't already answer.

What a strict area chair will say about the Gate D draft, and the answer the
paper must already contain:

| Likely objection | Answer in the paper |
|---|---|
| "One synthetic-grammar dataset" | X6 if done; otherwise a Limitations paragraph plus the human-written test set as the strongest evidence that it isn't just the grammar |
| "Would a grammar solve this?" | X2 |
| "Is the effect produced by your guardrail data?" | X1 |
| "Differences of 1–2 points" | Every claim stated with CI and Holm; C5 worded as a tie in accuracy, not a win |
| "Refusing 6–10% of real orders" | T3 states it, Discussion gives the cause, Limitations names the strict rule |
| "Why should NLP care beyond pizza?" | Discussion: format is a free design choice in every structured-output system; S is measurable without new labels |

## 7. Order of work

1. **P0**: template, plumbing, macros.
2. **P2, P3**: generators emit T1–T3 and F2–F7 from neg-v1.
3. **P1, P4**: text rewrite section by section. Supervisor comments closed as
   each section is rewritten.
4. **W**: site data export, model, section text, QA, deploy.
5. **Gate D**: PDF plus the supervisor response table plus the site link →
   supervisors.
6. **X1–X3** start in parallel with step 3 (GPU time is the long pole).
7. Merge supervisor feedback and X1–X3 results → **Gate E**.
8. **R** and the mock review → **Gate F**, submit.

## 8. Open decisions (authors)

1. Framing A confirmed (recommended), or B (safety first: format, refusal,
   correction as three levers).
2. X6, second dataset: yes or no, set by the deadline.
3. Strict refusal rule kept, with false rejection as a stated cost, or a
   softened rule studied.
4. ACL main track vs Findings as the commitment target.
5. Whether the `bench` page ships with this update.
