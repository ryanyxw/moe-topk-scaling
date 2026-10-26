"""Render the downstream-eval section of the report from results.json (run collect.py first).
Usage: report_evals.py <report.html>  -- replaces the block between <!--EVALS--> and <!--/EVALS-->."""
import json, re, sys, os, datetime as dt
E = os.environ.get("EVAL_ROOT", "/mnt/nvme/scaling/eval")  # exports, configs, NEL outputs
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = json.load(open(f"{RES}/evals/results.json"))
VAL = {"c1_k1": 2.7574, "c1_k2": 2.8271, "c1_k4": 2.7428, "c1_k8": 2.7421, "c1_k16": 2.8100,
       "c2_k4": 2.5726, "c2_k8": 2.5707, "c2_k16": 2.6016, "c2_k32": 2.6903,
       "c3_k4": 2.4602, "c3_k8": 2.4385, "c3_k16": 2.4565, "c3_k32": 2.5197,
       "c4_k4": 2.3898, "c4_k8": 2.3619, "c4_k16": 2.3750, "final_k12": 2.2405,
       "final_k12_26k": 2.458, "final_k12_52k": 2.414, "final_k12_78k": 2.394, "final_k12_104k": 2.375}
TOK = {"c1": "8e18", "c2": "2e19", "c3": "5e19", "c4": "1e20", "final": "4.5e20"}
COLS = ["HellaSwag", "ARC-C", "ARC-E", "PIQA", "Winogrande 5s", "OBQA", "SciQ", "LAMBADA", "MMLU 5s", "ARC-C 25s"]
HEAD = ["HellaSwag", "ARC-C", "ARC-E", "PIQA", "Wino&shy;grande", "OBQA", "SciQ", "LAMBADA", "MMLU", "ARC-C 25-shot"]
AVG = COLS[:8]  # tasks above chance; MMLU and the letter-answer ARC-C stay at chance for every model
CHANCE = {"HellaSwag": 25, "ARC-C": 25, "ARC-E": 25, "PIQA": 50, "Winogrande 5s": 50, "OBQA": 25, "SciQ": 25,
          "LAMBADA": 0, "MMLU 5s": 25, "ARC-C 25s": 25}
statuses = {}  # worker status files; without them (fresh clone), any model with results counts as done
if not os.path.isdir(f"{E}/status"):
    statuses = {m: "DONE" for m in R}
for f in os.listdir(f"{E}/status") if os.path.isdir(f"{E}/status") else []:
    if not f.endswith(".lock"):
        statuses[f] = open(f"{E}/status/{f}").read().split(" ", 2)[-1].strip()
queue = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_queue.txt")) if l.strip()]
done = [m for m in queue if statuses.get(m) == "DONE"]

def label(m):
    if m.startswith("final_k12_"): return f"final run @ {m.split('_')[-1]} steps (pre-cooldown)"
    if m == "final_k12": return "final run, top-k 12"
    b, k = m.split("_"); return f"{b.upper()} top-k {k[1:]}"

def rows(models, best_within=False):
    ms = [m for m in models if m in R]
    if not ms: return '<tr><td colspan="13" class="cap">not finished yet</td></tr>'
    best = {c: max((R[m][c][0] for m in ms if c in R[m]), default=None) for c in COLS} if best_within else {}
    out = []
    for m in ms:
        r = R[m]; cells = []
        for c in COLS:
            if c in r:
                v = 100 * r[c][0]; cls = "r best" if best_within and len(ms) > 1 and r[c][0] == best[c] else "r"
                cells.append(f'<td class="{cls}">{v:.1f}</td>')
            else:
                cells.append('<td class="r">–</td>')
        have = [100 * r[c][0] for c in AVG if c in r]
        avg = f"{sum(have)/len(have):.1f}" if len(have) == len(AVG) else "–"
        st = "" if statuses.get(m) == "DONE" else ' <span class="chip run">running</span>'
        out.append(f'<tr><td>{label(m)}{st}</td><td class="r">{VAL[m]:.4f}</td><td class="r"><b>{avg}</b></td>{"".join(cells)}</tr>')
    return "\n".join(out)

def table(title, models, cap, best_within=False):
    th = "".join(f'<th class="r">{h}</th>' for h in HEAD)
    return f'''  <h3>{title}</h3>
  <div class="tbl"><table>
    <thead><tr><th>Model</th><th class="r">Val loss</th><th class="r">Avg (8)</th>{th}</tr></thead>
    <tbody>
{rows(models, best_within)}
    </tbody>
  </table></div>
  <p class="cap">{cap}</p>'''

now = dt.datetime.utcnow().strftime("%a %d %b, %H:%M UTC")
html = f'''<!--EVALS-->
<section>
  <h2>Downstream evaluations</h2>
  <p><b>{len(done)} of {len(queue)} models finished</b> (updated {now}). Each checkpoint was converted to Hugging Face format with Megatron-Bridge, served with vLLM 0.12 on one GPU, and scored with the NeMo Evaluator launcher (0.2.6) using NVIDIA's lm-evaluation-harness container (26.03). All tasks are multiple-choice or completion tasks scored by answer log-likelihood, which works for base models of this size.</p>
  <p class="cap">Conversion check: on the same validation text, vLLM's per-token losses match the Megatron checkpoint to within 0.003 at every position, and every served model's validation loss sits a consistent 0.01–0.02 below the Megatron number (different windows). The newer Hugging Face <code>transformers</code> implementation does not match (≈0.14 per-token gap), so it is not used anywhere.</p>
  <p class="cap">Scores are accuracy in %: length-normalized (acc_norm) for HellaSwag, ARC, PIQA, OBQA and SciQ, plain accuracy for Winogrande, LAMBADA and MMLU. Shots: HellaSwag 10, Winogrande 5, MMLU 5, ARC-C 25-shot variant 25, others 0. Chance: 25% (50% for PIQA and Winogrande). <b>Avg (8)</b> averages the eight tasks that are above chance. MMLU and the 25-shot ARC-C (which asks for an answer letter) stay at chance for every model, as expected at this scale. Error bars (1 s.e.): about ±0.5 for HellaSwag, ±1.0–1.4 for ARC, PIQA and Winogrande, ±2.2 for OBQA.</p>
  <div class="note">
    <h3>Result: downstream accuracy follows validation loss</h3>
    <ul>
      <li><b>Loss predicts accuracy.</b> Across the 17 cooled-down models, the 8-task average correlates with validation loss at r = −0.99 (HellaSwag −0.996, PIQA −0.993, LAMBADA −0.991; Winogrande is the noisiest at −0.84). Top-k changes neither the slope nor the offset: models with different top-k but the same loss score the same.</li>
      <li><b>The loss-optimal top-k is also the accuracy-optimal top-k.</b> At all four budgets, top-k 8 has both the lowest loss and the highest 8-task average (C1 45.5, C2 49.4, C3 53.6, C4 56.1). The margins over the runner-up are 0.2–0.9 points, about the size of the noise, so this agrees with the loss ranking but doesn't independently confirm it.</li>
      <li><b>The final run</b> (top-k 12, 4.5e20) averages 60.0, up 3.9 points on the best C4 model: HellaSwag 61.1, ARC-E 68.7, SciQ 86.9, LAMBADA 46.4.</li>
      <li><b>The cooldown buys about 3.6 points.</b> The final run's checkpoint at step 104k (cooldown start, loss 2.375) averages 56.4; after the cooldown it averages 60.0. Pre-cooldown checkpoints fall on the same loss–accuracy line as cooled-down models (step 26k: loss 2.458, avg 53.6; C3 top-k 8: 2.439, 53.6), so what matters is the loss reached, not how it was reached.</li>
      <li><b>Not informative at this scale:</b> MMLU (24–26% for every model) and the letter-answer 25-shot ARC-C (24–28%) stay at chance; the 0-shot ARC-C shows the trend instead (25 → 40).</li>
    </ul>
  </div>
  <figure class="plot">{open(f"{RES}/figures/evals.svg").read().strip()}
  <figcaption class="cap">Each point is one checkpoint. Circles: isoFLOP runs, colored by budget. Hollow triangles: the final run's pre-cooldown checkpoints (26k, 52k, 78k, 104k steps). Star: the finished final run. The line is a least-squares fit to the 17 cooled-down models.</figcaption></figure>
{table("A. Compute ladder (best top-k ≈ 8 at each budget, final run at 12)", ["c1_k8", "c2_k8", "c3_k8", "c4_k8", "final_k12"], "Do downstream scores improve smoothly as validation loss falls?")}
{table("B. Top-k at fixed compute: C4 (1e20)", ["c4_k4", "c4_k8", "c4_k16"], "Best score in each column is highlighted. Loss gaps here are only 0.013–0.028, near the resolution of most tasks.", True)}
{table("B. Top-k at fixed compute: C3 (5e19)", ["c3_k4", "c3_k8", "c3_k16", "c3_k32"], "Best score in each column is highlighted.", True)}
{table("C. Top-k at fixed compute: C2 (2e19)", ["c2_k4", "c2_k8", "c2_k16", "c2_k32"], "Best score in each column is highlighted.", True)}
{table("C. Top-k at fixed compute: C1 (8e18)", ["c1_k1", "c1_k2", "c1_k4", "c1_k8", "c1_k16"], "Best score in each column is highlighted. Top-k 1 uses unnormalized router weights (see above).", True)}
{table("C. Final run checkpoints", ["final_k12_26k", "final_k12_52k", "final_k12_78k", "final_k12_104k", "final_k12"], "The intermediate checkpoints were saved at full learning rate, before the cooldown, so they are not comparable to cooled-down models of the same loss; they show the trajectory of a single run. Their val loss is the 20-batch in-run evaluation.")}
</section>
<!--/EVALS-->'''
p = sys.argv[1]; s = open(p).read()
if "<!--EVALS-->" in s:
    s = re.sub(r"<!--EVALS-->.*?<!--/EVALS-->", lambda _: html, s, flags=re.S)
else:
    anchor = "<section>\n  <h2>LR sweep results"
    assert anchor in s; s = s.replace(anchor, html + "\n\n" + anchor, 1)
open(p, "w").write(s)
print(f"{len(done)}/{len(queue)} done; {len(R)} models with any results")
