---
layout: post
title: "GPTQ and AWQ Explained: The Hessian, the Activations, and the Data That Decides Everything"
date: 2026-05-30 09:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, ptq, gptq, awq, calibration, int4, w4a16, llm]
math: true
description: "GPTQ and AWQ for W4A16 INT4 — why RTN's independent rounding is suboptimal, how Hessian-guided error redistribution (GPTQ) and activation-magnitude scaling (AWQ) each improve on it, and why calibration domain typically matters more than algorithm choice."
---


*The weight you round wrong only matters as much as the activation it multiplies.*


<!-- Opening block: questions + prerequisites — stacked vertically -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Why is RTN's per-weight independent rounding suboptimal?</li>
      <li>How does GPTQ use the inverse Hessian to reduce reconstruction error vs. RTN?</li>
      <li>Why does AWQ look at activations to fix weight quantization?</li>
      <li>What do <code>desc_act</code> and <code>damp_percent</code> actually change?</li>
      <li>Why can calibration data alone swing GSM8K accuracy by 10 points?</li>
      <li>When should you pick GPTQ over AWQ, and vice versa?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #c4b5fd;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>INT4/INT8 quantization: scale, zero-point, per-tensor vs. per-channel</li>
      <li>PyTorch <code>nn.Linear</code> and forward hooks</li>
      <li>The matmul identity $Y = XW^T$</li>
      <li><a href="/posts/quantization-1/">Phase 1: quantization basics (RTN, PTQ flow, scale &amp; zero-point)</a></li>
    </ul>
  </div>
</div>


RTN rounds each weight to the nearest grid point on its own, blind to how much that weight actually moves the layer's output. GPTQ and AWQ both fix this for the **W4A16** regime — weights at INT4, activations left at BF16 — but from opposite directions. Let's chase down what RTN misses, watch each algorithm repair it, and then find the variable that quietly outweighs the choice between them.


---


## 1. Why isn't round-to-nearest good enough?


*RTN rounds every weight independently — what does treating all weights as equal actually cost us?*


The output of a linear layer is $Y = XW^T$. Quantizing $W$ to $\hat{W}$ introduces a per-weight rounding error $\epsilon_i = \hat{w}_i - w_i$, and that error reaches the output *multiplied by its input activation* $x_i$. So the same rounding slip costs wildly different amounts depending on which channel it lands on:


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Same rounding error, four different channels</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">fixed $\epsilon_i = 0.05$ · output error $= \epsilon_i \cdot |x_i|$ — bar width is the resulting output error</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:120px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">ch 0 · |x| = 0.3</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:0.4%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:120px;font-size:12px;font-weight:600;color:#14532d;flex-shrink:0;">0.015 — silent</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:120px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">ch 1 · |x| = 0.7</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:0.9%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:120px;font-size:12px;font-weight:600;color:#14532d;flex-shrink:0;">0.035 — silent</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:120px;text-align:right;font-size:12px;color:#78350f;flex-shrink:0;font-weight:600;">ch 2 · |x| = 4.2</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:5%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:120px;font-size:12px;font-weight:600;color:#78350f;flex-shrink:0;">0.21 — notable</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:120px;text-align:right;font-size:12px;color:#7f1d1d;flex-shrink:0;font-weight:700;">ch 3 · |x| = 85</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:120px;font-size:12px;font-weight:700;color:#7f1d1d;flex-shrink:0;">4.25 — dominant ×283</span>
    </div>
  </div>
</div>


Read the bars: the rounding error is *identical* on all four rows, yet the output damage spans a factor of 283. The quiet channels (mint) barely register; the outlier channel at $\lvert x \rvert = 85$ (rose) swallows the entire error budget by itself. RTN can't see this — it only looks at the weight distribution, which is roughly the same across all four columns.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · Per-column output error</div>


$$
\begin{aligned}
\delta y_i &= (w_i - \hat{w}_i)\, x_i = -\epsilon_i \, x_i \\[4pt]
\|\delta Y\|_F &= \Big\| \textstyle\sum_i \epsilon_i \, x_i \Big\|_F
\end{aligned}
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $\epsilon_i$ = rounding error for weight entry $i$ · $x_i$ = input activation on that channel.<br>
    Output error scales with <strong>activation magnitude</strong>, not weight magnitude — so you cannot predict sensitivity from the weights alone.
  </div>
</div>


> **The core failure.** RTN minimizes per-weight error; what we actually care about is per-output error. Those are the same objective only if every activation has the same magnitude — which is never true for LLMs.
{: .prompt-warning }


So the fix has to be *output-aware*. GPTQ and AWQ are two answers to the same question — which weights matter, and how do we protect them? Let's take GPTQ first, where the answer is written in a matrix.


---


## 2. How does GPTQ redistribute the error?


*If some columns matter more, how does the Hessian tell us which — and how does GPTQ repair the damage as it goes?*


GPTQ reframes quantization as a reconstruction problem: find the grid-snapped $\hat{W}$ that least disturbs the layer output on calibration data.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · GPTQ reconstruction objective</div>


$$
\min_{\hat{W}} \; \| W X - \hat{W} X \|_F^2,
\qquad
H = 2 X^T X
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $X$ = calibration activations $[N \times \text{in}]$ · $H$ = Hessian $[\text{in} \times \text{in}]$.<br>
    A large diagonal $H_{ii}$ means channel $i$ carries high activation energy — its weights swing the output most, so round them most carefully.
  </div>
</div>


The Hessian is exactly the formalization of what we saw in §1: $H_{ii}$ is large precisely for the loud channels. But GPTQ does more than rank columns — it *repairs* as it goes. It quantizes columns left to right; after locking column $i$, the rounding error $e_i$ is pushed forward into the still-unquantized columns, weighted by the inverse Hessian, so they can pre-compensate before their own turn:


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · GPTQ error propagation</div>


$$
\delta W_j = -\,e_i \cdot \frac{H^{-1}_{ij}}{H^{-1}_{ii}}, \qquad j > i
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $e_i$ = rounding error for column $i$ · $H^{-1}_{ij}/H^{-1}_{ii}$ = OLS regression coefficient of column $j$ on column $i$ in the calibration activations.<br>
    Columns that co-activate with $i$ absorb a larger correction — the update is a closed-form solution to the residual least-squares problem.
  </div>
</div>


Let's watch this run on a 2×4 weight matrix. The state of each column is the whole story — so we color by state: lavender = untouched, mint = locked forever, yellow = pre-adjusted by a prior column's error.


<!-- V6-style: column-by-column quantize + propagate -->
<div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#4c1d95;font-size:13px;margin-bottom:16px;">GPTQ: quantize one column, push its error right, repeat</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;margin-bottom:14px;">


    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">① Initial W</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c0</div>
          <div style="font-size:10px;line-height:1.6;color:#4c1d95;">0.40<br>0.76</div>
        </div>
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c1</div>
          <div style="font-size:10px;line-height:1.6;color:#4c1d95;">0.62<br>0.18</div>
        </div>
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c2</div>
          <div style="font-size:10px;line-height:1.6;color:#4c1d95;">0.91<br>0.55</div>
        </div>
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c3</div>
          <div style="font-size:10px;line-height:1.6;color:#4c1d95;">0.27<br>0.43</div>
        </div>
      </div>
      <div style="font-size:9px;color:#64748b;margin-top:5px;">all unquantized</div>
    </div>


    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>


    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">② Quantize c0 → propagate</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.50<br>1.00</div>
        </div>
        <div style="background:#fdffb6;border:1px solid #fde047;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#713f12;font-weight:600;margin-bottom:3px;">c1 ✦</div>
          <div style="font-size:10px;line-height:1.6;color:#713f12;">0.60<br>0.16</div>
        </div>
        <div style="background:#fdffb6;border:1px solid #fde047;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#713f12;font-weight:600;margin-bottom:3px;">c2 ✦</div>
          <div style="font-size:10px;line-height:1.6;color:#713f12;">0.89<br>0.53</div>
        </div>
        <div style="background:#fdffb6;border:1px solid #fde047;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#713f12;font-weight:600;margin-bottom:3px;">c3 ✦</div>
          <div style="font-size:10px;line-height:1.6;color:#713f12;">0.25<br>0.41</div>
        </div>
      </div>
      <div style="font-size:9px;color:#a16207;margin-top:5px;">✦ = adjusted by c0 error via H⁻¹</div>
    </div>


    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>


    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">③ Quantize c1 → propagate</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.50<br>1.00</div>
        </div>
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c1 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.50<br>0.00</div>
        </div>
        <div style="background:#fdffb6;border:1px solid #fde047;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#713f12;font-weight:600;margin-bottom:3px;">c2 ✦</div>
          <div style="font-size:10px;line-height:1.6;color:#713f12;">0.91<br>0.55</div>
        </div>
        <div style="background:#fdffb6;border:1px solid #fde047;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#713f12;font-weight:600;margin-bottom:3px;">c3 ✦</div>
          <div style="font-size:10px;line-height:1.6;color:#713f12;">0.27<br>0.43</div>
        </div>
      </div>
      <div style="font-size:9px;color:#a16207;margin-top:5px;">error flows right, never back</div>
    </div>


    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>


    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">④ Final Ŵ — all locked</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.50<br>1.00</div>
        </div>
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c1 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.50<br>0.00</div>
        </div>
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c2 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">1.00<br>0.50</div>
        </div>
        <div style="background:#caffbf;border:2px solid #86efac;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c3 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;color:#14532d;">0.00<br>0.50</div>
        </div>
      </div>
      <div style="font-size:9px;color:#14532d;margin-top:5px;">all quantized, errors absorbed</div>
    </div>
  </div>


  <div style="background:#f5f3ff;border:1px solid #c4b5fd;border-radius:6px;padding:10px 14px;font-size:11px;color:#4c1d95;line-height:1.7;">
    <strong>Error always flows left → right.</strong> Once a column turns mint (✓) it never changes again. The yellow (✦) columns have already been steered toward an output-preserving target, so when their turn comes the rounded value is pre-corrected. The $H^{-1}_{ij}/H^{-1}_{ii}$ term is large when columns $i$ and $j$ co-activate in calibration and near-zero when they're independent.
  </div>
</div>


The algorithm in full — note line 8, where all the work happens:


<!-- GPTQ Pseudocode -->
<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:14px;">GPTQ — algorithm pseudocode</div>
  <pre style="background:#fffffc;color:#1e293b;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#1e3a8a;font-weight:600;">Input:</span>  W [out × in]  — weight matrix for one linear layer
        X [N × in]    — calibration activations (N samples)
<span style="color:#1e3a8a;font-weight:600;">Output:</span> Ŵ [out × in]  — quantized weight matrix


<span style="color:#a16207;">1.</span> H     = 2 × Xᵀ × X                         <span style="color:#64748b;"># Hessian [in × in] — sensitivity matrix</span>
<span style="color:#a16207;">2.</span> H    += λ · mean(diag H) · I                <span style="color:#64748b;"># damping: λ = damp_percent (default 0.01)</span>
<span style="color:#a16207;">3.</span> H_inv = Cholesky_inverse(H)                 <span style="color:#64748b;"># stable inverse, computed once per layer</span>


<span style="color:#a16207;">4.</span> <span style="color:#7f1d1d;font-weight:600;">for</span> i = 0 → in_features − 1:                <span style="color:#64748b;"># process each column left → right</span>
<span style="color:#a16207;">5.</span>     Ŵ[:, i] = Q( W[:, i] )                  <span style="color:#64748b;"># round-to-nearest; lock this column forever</span>
<span style="color:#a16207;">6.</span>     e = W[:, i] − Ŵ[:, i]                   <span style="color:#64748b;"># quantization error vector [out_features]</span>
<span style="color:#a16207;">7.</span>     <span style="color:#7f1d1d;font-weight:600;">for</span> j = i+1 → in_features − 1:
<span style="color:#a16207;">8.</span>         W[:, j] -= e · H_inv[i, j] / H_inv[i, i]  <span style="color:#64748b;"># propagate error to remaining columns</span>


<span style="color:#a16207;">9.</span> <span style="color:#7f1d1d;font-weight:600;">return</span> Ŵ</pre>
  <div style="background:#9bf6ff;border:1px solid #67e8f9;border-radius:6px;padding:10px 14px;font-size:11px;color:#164e63;margin-top:12px;line-height:1.7;">
    <strong>Why Cholesky, not a plain inverse?</strong> $H = 2X^TX$ is symmetric positive semi-definite. Cholesky exploits that for a 2× cheaper factorization and stays numerically stable near singularity — standard LU inversion suffers catastrophic cancellation when a diagonal entry of $H$ approaches zero. Working with the square-root factor $L$ (where $H = LL^T$) keeps the near-zero case well-conditioned after damping.
  </div>
</div>


### What do the GPTQ config knobs actually change?


*You call `quantize()` with a handful of parameters — which ones move accuracy, and what breaks if you leave the defaults?*


> **Illustrative, not copy-paste.** The calibration-data format `GPTQModel.quantize()` expects varies by version — check it against your installed `gptqmodel`. For runnable code see [Lab 2 — GPTQ &amp; AWQ under the hood](/posts/gptq-awq-underthehood/).
{: .prompt-info }


```python
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer


model_id = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# ── Calibration dataset ────────────────────────────────────────────
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# Over-select before filtering so we reliably end up with 128 samples.
samples = [
    tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512)
    for row in dataset.select(range(256))         # over-select; trim after filter
    if len(row["text"].strip()) > 50
][:128]


# ── QuantizeConfig — the knobs that matter ─────────────────────────
quant_config = QuantizeConfig(
    bits=4,              # 4 = INT4 (W4A16), 8 = INT8
    group_size=128,      # weights split into groups of 128; each gets its own
                         #   scale+zero. Smaller → better accuracy, more overhead.
                         #   GPTQ uses ONE global Hessian for the whole column loop;
                         #   group_size only sets grid granularity, not a per-group H.
    desc_act=False,      # reorder columns by descending activation magnitude before
                         #   quantizing, so high-sensitivity columns go first and their
                         #   errors propagate INTO low-sensitivity ones. True buys
                         #   ~0.1–0.2 PPL; False is faster and simpler.
    damp_percent=0.01,   # Levenberg-Marquardt damping: H += λ·mean(diag H)·I.
                         #   Floors near-zero diagonals from "dead" channels that are
                         #   ~0 across all samples (else H⁻¹ blows up). Raise to
                         #   0.05–0.1 if you hit NaN/Inf during quant.
    sym=False,           # asymmetric (per-group zero-point) vs symmetric (zp=0).
                         #   False is the W4A16 default; better for ReLU-heavy FFNs.
)


model = GPTQModel.from_pretrained(model_id, quant_config)
model.quantize(samples)
model.save_quantized("./qwen3-4b-gptq-w4a16")
```


> **`damp_percent` and NaN.** If early layers produce NaN weights, raise `damp_percent` from 0.01 to 0.05 or 0.1. The damping prevents division by a near-zero diagonal of $H^{-1}$.
{: .prompt-danger }


> **`desc_act` memory myth.** The full Hessian is materialized per layer *regardless* of `desc_act` — it's needed to compute $H^{-1}$ at all, and that's the real memory driver (a 28672-wide FFN gives a ~3 GB Hessian at 70B). `desc_act` itself only adds a column permutation; its true cost at scale is act-order bookkeeping and slower kernels, not extra memory.
{: .prompt-warning }


GPTQ pays for its precision with that per-layer Hessian inversion. Which raises the obvious question: can we protect the same outlier channels *without* ever building $H$?


---


## 3. Why does AWQ look at activations instead?


*GPTQ inverts a Hessian per layer — expensive at scale. Can we protect the same weights with nothing but a forward pass?*


AWQ goes back to the identity we derived in §1: output error is $\epsilon_i \cdot x_i$, so the dangerous columns are simply the ones paired with large-magnitude activations. We already know which those are from a single calibration pass — no second-order matrix required. The trick is what AWQ does about them: **scale the salient weight columns up before quantizing, then divide the matching activations down at runtime** so the math is unchanged but the grid lands finer where it counts.


<div style="background:#fffffc;border:1px solid #86efac;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#14532d;font-size:13px;margin-bottom:16px;">AWQ: rank channels by activation, then scale the loud ones up</div>


  <!-- Step 1: rank by activation -->
  <div style="margin-bottom:18px;">
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">① Rank channels by activation magnitude |x_i| — louder ⇒ bigger scale</div>
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#1e293b;text-align:right;">ch 0</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:0.4%;height:100%;background:#caffbf;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#14532d;width:90px;">|x|=0.3 — safe</span>
        <span style="font-size:10px;color:#64748b;">s = 1.0 (no scaling)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#1e293b;text-align:right;">ch 1</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:1%;height:100%;background:#caffbf;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#14532d;width:90px;">|x|=0.8 — safe</span>
        <span style="font-size:10px;color:#64748b;">s = 1.0 (no scaling)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#78350f;font-weight:600;text-align:right;">ch 2</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:5%;height:100%;background:#ffd6a5;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#78350f;font-weight:600;width:90px;">|x|=4.2 — notable</span>
        <span style="font-size:10px;color:#78350f;">s = 2.0 (moderate)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#7f1d1d;font-weight:700;text-align:right;">ch 3</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:100%;height:100%;background:#ffadad;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#7f1d1d;font-weight:700;width:90px;">|x|=85 — outlier</span>
        <span style="font-size:10px;color:#7f1d1d;font-weight:700;">s = 9.2 (max)</span>
      </div>
    </div>
  </div>


  <!-- Step 2: before/after grid -->
  <div style="margin-bottom:18px;">
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">② Scaling the salient column up ⇒ finer effective grid for it</div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #fca5a5;border-radius:7px;padding:12px 14px;">
        <div style="font-size:10px;font-weight:700;color:#7f1d1d;margin-bottom:8px;">ch 3 BEFORE scaling (s = 1)</div>
        <div style="position:relative;height:16px;background:#ffadad;border-radius:3px;margin-bottom:4px;">
          <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:25%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:75%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:calc(100% - 2px);top:0;width:2px;height:100%;background:#7f1d1d;opacity:.9;"></div>
          <div style="position:absolute;left:73%;top:0;width:3px;height:100%;background:#7f1d1d;"></div>
        </div>
        <div style="font-size:9px;color:#7f1d1d;display:flex;justify-content:space-between;"><span>0</span><span style="font-weight:700;">w=0.73 ↑</span><span>1.0</span></div>
        <div style="font-size:10px;color:#7f1d1d;margin-top:5px;">step = 0.25 · w rounds to <strong>0.75</strong> · error = +0.02</div>
        <div style="font-size:10px;color:#7f1d1d;">output error = 0.02 × 85 = <strong>+1.70</strong></div>
      </div>
      <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #86efac;border-radius:7px;padding:12px 14px;">
        <div style="font-size:10px;font-weight:700;color:#14532d;margin-bottom:8px;">ch 3 AFTER scaling (s = 9.2)</div>
        <div style="position:relative;height:16px;background:#caffbf;border-radius:3px;margin-bottom:4px;">
          <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:12.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:25%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:37.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:62.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:75%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:87.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:calc(100% - 2px);top:0;width:2px;height:100%;background:#14532d;opacity:.9;"></div>
          <div style="position:absolute;left:73%;top:0;width:3px;height:100%;background:#14532d;"></div>
        </div>
        <div style="font-size:9px;color:#14532d;display:flex;justify-content:space-between;"><span>0</span><span style="font-weight:700;">w·s=6.72↑</span><span>9.2</span></div>
        <div style="font-size:10px;color:#14532d;margin-top:5px;">step ≈ 0.61 (scaled space) · rounds to ≈<strong>6.74</strong> → /9.2 ≈ 0.733</div>
        <div style="font-size:10px;color:#14532d;">output error = 0.003 × 85 = <strong>+0.26</strong> (&gt;6× smaller)</div>
      </div>
    </div>
    <div style="background:#fdffb6;border:1px solid #fde047;border-radius:6px;padding:10px 14px;font-size:11px;color:#713f12;margin-top:12px;line-height:1.7;">
      <strong>INT4 always has 16 levels.</strong> Scaling adds no levels — it changes <em>which channel sets the group scale</em>. INT4 step = group_max / 15. Before AWQ a non-salient channel can dominate the group range, forcing the salient column to share the 16-level budget. Scaling the salient column up makes it the new group_max, so all 16 levels cover its range — finer precision for the column that matters, coarser for the ones that don't.
    </div>
  </div>


  <!-- Step 3: runtime identity -->
  <div>
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">③ Runtime: ÷s folded into the preceding LayerNorm — zero extra ops</div>
    <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:7px;padding:12px 14px;font-size:11px;color:#1e293b;line-height:1.8;">
      <div>At export: <code style="background:#a0c4ff;color:#1e3a8a;padding:1px 4px;border-radius:3px;">W_fused[:, i] = W[:, i] × s[i]</code> baked into the model file.</div>
      <div>At export: the ÷s correction folds into the preceding LayerNorm gain (<code style="background:#a0c4ff;color:#1e3a8a;padding:1px 4px;border-radius:3px;">gamma /= s</code>), so activations emerge pre-divided.</div>
      <div style="margin-top:6px;padding-top:6px;border-top:1px solid #e2e8f0;">Y = (X / s) × (W × s)ᵀ = X × Wᵀ &nbsp;✓&nbsp; <span style="color:#64748b;">— the rescaling lives inside the LayerNorm that was already running.</span></div>
    </div>
  </div>
</div>


The before/after grids tell the whole story: scaling ch 3 up by 9.2 shrinks its output error from +1.70 to +0.26 — a 6× cut — purely by spending more of the 16-level budget on the channel that dominates the output. The optimal scale per channel comes from a tiny grid search over an exponent $\alpha$:


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · AWQ optimal scale search</div>


$$
\begin{aligned}
s_i^* &= \arg\min_{s_i} \big\| Q(W_i s_i)/s_i \cdot X_i - W_i X_i \big\| \\[4pt]
s_i &\propto \overline{|x_i|}^{\,\alpha}, \quad \alpha \in [0,1]
\end{aligned}
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $\alpha$ grid-searched on the calibration set (0.0 → 1.0 in 0.1 steps); $\alpha \approx 0.5$ works for most LLMs.<br>
    <span style="color:#7f1d1d;">Caveat: the ÷s fold needs a linear host (LayerNorm or a preceding matmul). It works for <code>q/k/v/gate/up</code> fed by a LayerNorm, but <code>down_proj</code> sits right after the nonlinearity with no host — AWQ scales those via the preceding <code>up_proj</code> output or skips them.</span>
  </div>
</div>


<!-- AWQ Pseudocode -->
<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:14px;">AWQ — algorithm pseudocode</div>
  <pre style="background:#fffffc;color:#1e293b;border:1px solid #e2e8f0;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#1e3a8a;font-weight:600;">Input:</span>  W [out × in]   — weight matrix for one linear layer
        X [N × in]     — calibration activations (N samples)
        α_grid         — candidate exponents, e.g. [0.0, 0.1, …, 1.0]
<span style="color:#1e3a8a;font-weight:600;">Output:</span> W_fused [out × in] — weight matrix with scale absorbed


<span style="color:#a16207;">1.</span> x_mean[i] = mean over calib of |X[:, i]|   <span style="color:#64748b;"># per-channel mean activation magnitude</span>
<span style="color:#a16207;">2.</span> w_max[i]  = max over rows of |W[:, i]|     <span style="color:#64748b;"># per-channel weight magnitude</span>


<span style="color:#a16207;">3.</span> best_s = ones(in);  best_err = ∞


<span style="color:#a16207;">4.</span> <span style="color:#7f1d1d;font-weight:600;">for</span> α in α_grid:                            <span style="color:#64748b;"># grid search over exponents</span>
<span style="color:#a16207;">5.</span>     s[i] = x_mean[i]^α / w_max[i]^(1−α)     <span style="color:#64748b;"># migration scale, per channel</span>
<span style="color:#a16207;">6.</span>     s    = clamp(s, 1e-4, 1e4)              <span style="color:#64748b;"># avoid overflow / near-zero division</span>
<span style="color:#a16207;">7.</span>     Ŵ    = Q(W × s) / s                     <span style="color:#64748b;"># scale up, quantize, undo for comparison</span>
<span style="color:#a16207;">8.</span>     err  = ‖ W·X − Ŵ·X ‖₂                   <span style="color:#64748b;"># reconstruction error on representative input</span>
<span style="color:#a16207;">9.</span>     <span style="color:#7f1d1d;font-weight:600;">if</span> err &lt; best_err: best_s = s; best_err = err


<span style="color:#a16207;">10.</span> W_fused[:, i] = W[:, i] × best_s[i]        <span style="color:#64748b;"># absorb scale permanently</span>
<span style="color:#a16207;">11.</span> <span style="color:#7f1d1d;font-weight:600;">return</span> W_fused                               <span style="color:#64748b;"># runtime: x / best_s before the GEMM</span></pre>
  <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:10px 14px;font-size:11px;color:#14532d;margin-top:12px;line-height:1.7;">
    <strong>No matrix inversion.</strong> AWQ never builds or inverts a Hessian — per α candidate it's just a per-channel multiply/divide and a forward pass. That's why it runs 3–5× faster than GPTQ, and the scale search is embarrassingly parallel across layers.
  </div>
</div>


### What does each AWQ config knob control?


*Same question we asked of GPTQ — which AutoAWQ settings actually move accuracy?*


> **Illustrative, not copy-paste.** The `quantize()` signature and `calib_data` handling differ across AutoAWQ versions — verify against your installed `autoawq`. For runnable code see [Lab 2](/posts/gptq-awq-underthehood/).
{: .prompt-info }


```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


model_id = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id)


quant_config = {
    "zero_point": True,    # asymmetric: per-group zero-point centres INT4 on the
                           #   real weight distribution. Almost always beats False.
    "q_group_size": 128,   # per-group scale+zero granularity. 64 gives ~0.1 better
                           #   PPL at 2× metadata; 128 is the standard.
    "w_bit": 4,            # 4 = INT4 (W4A16); 8 = W8A16 when latency budget allows.
    "version": "GEMM",     # packing/kernel format. GEMM = throughput on RTX/A-series
                           #   at large batch; GEMV = batch=1 decode. Marlin is NOT a
                           #   version — it's a serve-time kernel vLLM picks for
                           #   GEMM-packed weights on A100/H100.
}


model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="auto")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="wikitext-2-raw-v1",  # ← replace with domain-matched data (see §5)
    max_calib_samples=128,
    max_calib_seq_len=512,           # ← set to your production P90, not the default
)
model.save_quantized("./qwen3-4b-w4a16", safetensors=True)
```


> **W4A16 ceiling.** Both GPTQ and AWQ are weight-only. For INT8 GEMM throughput (W8A8, 2× FLOP rate on Turing+), you need a different technique — SmoothQuant, in [Part 3](/posts/quantization-3/).
{: .prompt-warning }


Two algorithms, same W4A16 target, very different costs. Which one do you actually reach for?


---


## 4. GPTQ or AWQ — how do you actually pick?


*Both hit W4A16 at near-identical accuracy. So the decision lives entirely in the costs around the edges — let's lay them side by side.*


<!-- V9 comparison table -->
<style>
#gptq-awq-table td, #gptq-awq-table th {
  white-space: normal !important;
  word-break: break-word;
  overflow-wrap: anywhere;
  vertical-align: top;
}
#gptq-awq-table code { white-space: normal; word-break: break-word; }
</style>
<div style="margin:16px 0;">
<table id="gptq-awq-table" style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:22%;">
    <col style="width:39%;">
    <col style="width:39%;">
  </colgroup>
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Dimension</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#4c1d95;font-weight:600;">GPTQ</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#14532d;font-weight:600;">AWQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#1e293b;">Core mechanism</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Hessian-guided — propagate each column's rounding error forward via $H^{-1}$</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Activation scaling — scale salient columns up so their effective grid is finer</td>
    </tr>
    <tr style="background:#caffbf;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">Quantization speed</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">Slow — Cholesky + O(n²) column loop; ~30–60 min for 7B on A100</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;font-weight:600;">Fast — no inversion; α search parallel; ~5–10 min for 7B</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#1e293b;">W4A16 accuracy</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Marginally better — ~0.05–0.15 lower PPL on matched calibration</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Comparable — within 0.1–0.2 PPL; smaller than a calib domain mismatch</td>
    </tr>
    <tr style="background:#caffbf;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">Peak memory at 70B</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">Higher — full Hessian per layer (~3 GB for a 28672-wide FFN)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;font-weight:600;">Lower — per-channel stats only; fits 70B+ on one 80 GB GPU</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#1e293b;">Calibration sensitivity</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Higher — Hessian encodes the calib distribution; mismatch corrupts it</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Lower — magnitude stats degrade more gracefully under shift</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#1e293b;">Inference kernels</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;"><code>exllama_v2</code> (decode), <code>gptq-marlin</code> (A100+)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;"><code>GEMV</code> (batch=1), <code>GEMM</code> (batch≥4), <code>Marlin</code> (H100)</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#1e293b;">Pre-quantized models</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Large — most 7B–70B have GPTQ variants on HF Hub</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Growing — official Qwen/Llama/Mistral; default in vLLM &amp; TGI</td>
    </tr>
  </tbody>
</table>
</div>


The mint rows are where the gap is real: speed, 70B memory, and calibration robustness all tip toward AWQ. Accuracy is a near-tie. So the decision tree collapses to a few yes/no questions:


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:20px 22px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:18px;">Decision workflow — GPTQ or AWQ?</div>


  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">1</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is quantization time a hard constraint? (nightly re-quant, CI)</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:6px 12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ</strong>. 3–5× faster; fine to iterate.</div>
        <div style="background:#fffffc;border:1px solid #cbd5e1;border-radius:6px;padding:6px 12px;color:#475569;">No → continue ↓</div>
      </div>
    </div>
  </div>


  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">2</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is the model ≥ 70B?</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:6px 12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ</strong>. GPTQ's ~3 GB/layer Hessian + weights strains one 80 GB GPU.</div>
        <div style="background:#fffffc;border:1px solid #cbd5e1;border-radius:6px;padding:6px 12px;color:#475569;">No → continue ↓</div>
      </div>
    </div>
  </div>


  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">3</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is your calibration data well-matched to the deployment domain?</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:6px;padding:6px 12px;color:#4c1d95;font-weight:600;">Yes → <strong>GPTQ</strong>. Domain-accurate Hessian pays off; <code>desc_act=True</code> squeezes the last 0.1–0.2 PPL.</div>
        <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:6px 12px;color:#14532d;font-weight:600;">No / mixed → <strong>AWQ</strong>. Degrades more gracefully under shift.</div>
      </div>
    </div>
  </div>


  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">4</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is inference throughput the primary goal? (multi-GPU serving, H100)</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:6px 12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ + Marlin</strong>. Best tokens/s on A100/H100 at batch ≥ 8.</div>
        <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:6px;padding:6px 12px;color:#4c1d95;font-weight:600;">Latency / batch=1 → <strong>AWQ GEMV</strong> or <strong>GPTQ exllama_v2</strong>.</div>
      </div>
    </div>
  </div>


  <div style="background:#9bf6ff;border:1px solid #67e8f9;border-radius:7px;padding:10px 14px;color:#164e63;margin-top:4px;line-height:1.7;">
    <strong>Short version:</strong> default to AWQ — faster, scales to any size, less calibration-sensitive. Reach for GPTQ with domain-matched data on a sub-70B model, or when you need the last fraction of a perplexity point.
  </div>
</div>


> **The algorithm is rarely your biggest lever.** Notice every branch above mentions calibration data. That's the real variable — and it deserves its own section.
{: .prompt-tip }


---


## 5. Why does calibration data swing accuracy more than the algorithm?


*The calibration set never runs at inference — so why does swapping it move GSM8K by 10 points while WikiText-2 PPL barely twitches?*


Both algorithms estimate sensitivity — GPTQ's Hessian, AWQ's per-channel stats — *from the calibration distribution*. Estimate it on the wrong domain and you protect the wrong channels. Here's the same Qwen2-7B W4 AWQ model, identical algorithm, only the calibration source changed:


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">GSM8K 5-shot accuracy — Qwen2-7B W4 AWQ by calibration source</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">illustrative; your numbers will differ by model and base accuracy</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">FP16 baseline</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#a0c4ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">63.2%</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">WikiText-2 (default)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:83%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">52.4%</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;flex-shrink:0;">−10.8pp</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">GSM8K train (128)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:95%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">60.0%</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;flex-shrink:0;">−3.2pp</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">Mixed (64+64+64)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:97%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">61.4%</span>
      <span style="background:#9bf6ff;color:#164e63;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #67e8f9;flex-shrink:0;">−1.8pp</span>
    </div>
  </div>
</div>


The default WikiText-2 calibration (rose) drops 10.8 points on GSM8K — a bigger hit than you'd ever recover by switching algorithms. Matching the domain (mint) nearly closes it, and a mixed set (cyan) is the safest bet when you don't know the deployment distribution. Before you pick a dataset, answer four questions:


<div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e3a8a;margin-bottom:16px;font-size:14px;">Calibration dataset checklist</div>


  <div style="margin-bottom:14px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">1. What is your deployment environment?</div>
    <div style="color:#1e293b;line-height:1.7;">Code, math, document QA, and chat each activate different outlier channels. Calibrating on WikiText-2 prose for a coding assistant means the Hessian (GPTQ) or activation stats (AWQ) are wrong for the channels that drive your benchmark. <span style="font-weight:600;color:#1e3a8a;">Rule:</span> cover every task type your users will actually run.</div>
  </div>


  <div style="margin-bottom:14px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">2. What are your typical input lengths? <span style="color:#7f1d1d;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">crucial</span></div>
    <div style="color:#1e293b;line-height:1.7;">The most underestimated variable. Calibrating at <code>max_length=512</code> while serving 2k–4k prompts under-samples long-range attention and leaves deep-layer outliers invisible. <span style="font-weight:600;color:#1e3a8a;">Rule:</span> set <code>max_calib_seq_len</code> to the P90 of your production prompt length, not the library default.</div>
  </div>


  <div style="margin-bottom:14px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">3. What languages or domain terminology will appear?</div>
    <div style="color:#1e293b;line-height:1.7;">Non-English text and technical vocabularies (medical, legal, code) diverge sharply from English Wikipedia token distributions. Even 16–32 domain-matched samples out of 128 recover most of the gap. <span style="font-weight:600;color:#1e3a8a;">Rule:</span> include at least one representative sample per significant language or terminology cluster.</div>
  </div>


  <div>
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">4. Single-turn or multi-turn?</div>
    <div style="color:#1e293b;line-height:1.7;">Multi-turn chat prepends history, shifting the context-position distribution and mis-calibrating early-layer KV states — worst for RoPE models. <span style="font-weight:600;color:#1e3a8a;">Rule:</span> if you serve multi-turn, wrap calibration samples in the same system-prompt + prior-turn template you use in production.</div>
  </div>
</div>


And the trap that makes this so easy to miss — PPL barely moves while the downstream metric collapses:


<div style="margin:16px 0;">
<style>
#calib-source-table td, #calib-source-table th {
  white-space: normal !important;
  word-break: break-word;
  overflow-wrap: anywhere;
  vertical-align: top;
}
</style>
<table id="calib-source-table" style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:26%;">
    <col style="width:15%;">
    <col style="width:14%;">
    <col style="width:17%;">
    <col style="width:28%;">
  </colgroup>
  <thead>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">WikiText-2 PPL</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">GSM8K 5-shot</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">HumanEval pass@1</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#caffbf;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">FP16 baseline</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">5.62</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">63.2%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">48.8%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Reference</td>
    </tr>
    <tr style="background:#ffadad;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">WikiText-2 (default, 128)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">5.81</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">52.4%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">38.1%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;font-weight:600;">Domain mismatch</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">GSM8K train (128)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">5.74</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">60.0%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">39.5%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Math matched, code suffers</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Python code (128)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">5.78</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">54.1%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">46.2%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Code matched, math suffers</td>
    </tr>
    <tr style="background:#9bf6ff;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Mixed (each dataset 64)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">5.69</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">61.4%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">45.7%</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#164e63;font-weight:600;">Best overall — use when domain unknown</td>
    </tr>
  </tbody>
</table>
</div>


Read across the WikiText-2 row: PPL rises a trivial 5.62 → 5.81, yet GSM8K craters 10.8 points and HumanEval 10.7. PPL is computed on the same prose you calibrated on, so it's blind to exactly the damage you care about. The mixed row (cyan) is within 2 points everywhere — the right default when the deployment domain is unknown.


> **PPL is not your benchmark.** WikiText-2 PPL hides domain-specific degradation. Always pair it with at least one downstream metric matching your deployment — and when they disagree, trust the downstream metric.
{: .prompt-warning }


---


RTN ignores which weights matter; GPTQ answers with a Hessian that redistributes error forward; AWQ answers with an activation-magnitude scale that needs no inversion at all. Each is a direct response to the failure of the one before it — which is what makes the choice between them non-arbitrary. But the bars in §5 are the part to carry out the door: once you've picked either algorithm, domain-matched calibration data buys you more accuracy than switching between them ever will. Next, [Part 3 — W8A8: Activation Outliers and SmoothQuant](/posts/quantization-3/) takes on the regime both algorithms leave broken: quantizing the activations themselves.


---


## References


1. Frantar, E., Ashkboos, S., Hoefler, T., &amp; Alistarh, D. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** *arXiv:2210.17323*. [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
2. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., &amp; Han, S. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** *arXiv:2306.00978*. [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
3. **GPTQModel** (library). [https://github.com/modelcloud/gptqmodel](https://github.com/modelcloud/gptqmodel)
4. **AutoAWQ** (library). [https://github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)



