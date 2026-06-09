---
layout: post
title: "GPTQ and AWQ Explained: The Hessian, the Activations, and the Data That Decides Everything"
date: 2026-05-30 09:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, ptq, gptq, awq, calibration, int4, w4a16, llm]
math: true
description: "GPTQ and AWQ for W4A16 INT4 — why RTN's independent rounding is suboptimal, how Hessian-guided error redistribution (GPTQ) and activation-magnitude scaling (AWQ) each improve on it, and why calibration domain typically matters more than algorithm choice."
---

<!-- Opening block: questions + prerequisites -->
<div style="margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:rgba(96,165,250,0.08);border:1px solid #60a5fa;border-radius:8px;padding:16px;margin-bottom:12px;">
    <div style="font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;font-size:12px;border-left:4px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;">
      <li>Why is RTN's per-weight independent rounding suboptimal?</li>
      <li>How does GPTQ use the inverse Hessian to reduce reconstruction error vs. RTN?</li>
      <li>Why does AWQ look at activations to fix weight quantization?</li>
      <li>How does GPTQ's <code>desc_act</code> flag and <code>damp_percent</code> affect accuracy?</li>
      <li>Why can calibration data alone swing GSM8K accuracy by 10 percentage points?</li>
      <li>When should you pick GPTQ over AWQ, and vice versa?</li>
    </ul>
  </div>
  <div style="background:rgba(167,139,250,0.08);border:1px solid #a78bfa;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;font-size:12px;border-left:4px solid #a78bfa;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;">
      <li>INT4/INT8 quantization: scale, zero-point, per-tensor vs. per-channel</li>
      <li>PyTorch <code>nn.Linear</code> and <code>forward hooks</code></li>
      <li>Matrix multiplication identity: $Y = XW^T$</li>
      <li><a href="/posts/quantization-1/">Phase 1: quantization basics (RTN, PTQ flow, scale & zero-point)</a></li>
    </ul>
  </div>
</div>

RTN (round-to-nearest) quantizes each weight independently, treating all weights as equally important and ignoring how much each one actually affects layer output. GPTQ fixes that with Hessian-guided error redistribution; AWQ takes a different angle, using activation magnitude to identify which weight columns are most sensitive and scaling them before quantizing. Both target the W4A16 regime — weights at INT4, activations at BF16. This post covers both algorithms, their tradeoffs, and why domain-matched calibration data is often worth more than switching between them.

---

## 1. Weight-Only PTQ: GPTQ and AWQ {#gptq-awq}

*Why do we need dedicated algorithms beyond round-to-nearest, and why does each approach the problem from a completely different direction?*

Both algorithms target the **W4A16** regime: weights compressed to INT4, activations kept at BF16. Neither touches activation quantization. They differ in *how* they decide which weights to protect.

### GPTQ — Hessian-Guided Error Redistribution {#gptq}

*Why is independent per-weight rounding suboptimal, and why does the Hessian give us a better strategy?*

Rounding-nearest-to-grid (RTN) quantizes each weight independently, ignoring how much each weight actually affects layer output. GPTQ frames weight quantization as a reconstruction problem — find $\hat{W}$ in the quantization grid that minimizes the change in layer output on calibration data:

$$
\min_{\hat{W}} \| W X - \hat{W} X \|_F^2
$$

Think of two input channels: one carries activations with magnitude ≈ 50 (a dominant outlier), the other ≈ 0.3 (nearly silent). A 0.05 rounding error on the weight for the loud channel corrupts the output by 2.5; on the quiet channel it barely registers. The **Hessian** — the matrix of second-order partial derivatives of the reconstruction loss — formalizes this intuition: a large diagonal entry $H_{ii}$ means the loss curves sharply when weight column $i$ is perturbed, marking that column as high-sensitivity.

The Hessian of this objective with respect to the weights is $H = 2X^TX$ (an $[\text{in} \times \text{in}]$ matrix), where $X$ is the calibration activation matrix of shape $[N \times \text{in}]$ — rows are samples. A large $H_{ii}$ means input channel $i$ carries high activation energy across calibration samples — weights in that column receive large inputs and therefore contribute more to output error when rounded. GPTQ quantizes columns left-to-right: after quantizing column $i$, the error $e_i$ is compensated forward into remaining unquantized columns via $H^{-1}$:

<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · GPTQ Error Propagation</div>

$$
\delta W_j = -e_i \cdot \frac{H^{-1}_{ij}}{H^{-1}_{ii}}, \quad j > i
$$

  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $e_i$ = quantization error for column $i$ · $H^{-1}_{ij}$ = inverse Hessian cross-term<br>
    Remaining columns absorb the error as continuous adjustments before their own quantization — total output error is minimized.
  </div>
</div>

<!-- GPTQ Pseudocode -->
<div style="background:#f8f9fa;border:1px solid #d0d7de;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:14px;">GPTQ — Algorithm Pseudocode</div>
  <pre style="background:#161b22;color:#c9d1d9;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#79c0ff;font-weight:600;">Input:</span>  W [out × in]  — weight matrix for one linear layer
        X [N × in]    — calibration activations (N samples)
<span style="color:#79c0ff;font-weight:600;">Output:</span> Ŵ [out × in]  — quantized weight matrix

<span style="color:#f0883e;">1.</span> H     = 2 × Xᵀ × X                         <span style="color:#8b949e;"># Hessian [in × in] — sensitivity matrix</span>
<span style="color:#f0883e;">2.</span> H    += λ · mean(diag H) · I                <span style="color:#8b949e;"># damping: λ = damp_percent (default 0.01)</span>
<span style="color:#f0883e;">3.</span> H_inv = Cholesky_inverse(H)                 <span style="color:#8b949e;"># numerically stable inverse, computed once per layer</span>

<span style="color:#f0883e;">4.</span> <span style="color:#ff7b72;font-weight:600;">for</span> i = 0 → in_features − 1:                <span style="color:#8b949e;"># process each column left → right</span>
<span style="color:#f0883e;">5.</span>     Ŵ[:, i] = Q( W[:, i] )                  <span style="color:#8b949e;"># round-to-nearest; lock this column forever</span>
<span style="color:#f0883e;">6.</span>     e = W[:, i] − Ŵ[:, i]                   <span style="color:#8b949e;"># quantization error vector [out_features]</span>
<span style="color:#f0883e;">7.</span>     <span style="color:#ff7b72;font-weight:600;">for</span> j = i+1 → in_features − 1:
<span style="color:#f0883e;">8.</span>         W[:, j] -= e · H_inv[i, j] / H_inv[i, i]  <span style="color:#8b949e;"># propagate error to remaining columns</span>

<span style="color:#f0883e;">9.</span> <span style="color:#ff7b72;font-weight:600;">return</span> Ŵ</pre>
  <div style="background:#f0f6ff;border:1px solid #d0d7de;border-radius:6px;padding:10px 14px;font-size:11px;color:#1f2328;margin-top:12px;line-height:1.7;">
    <strong>Line 8 is the core GPTQ update.</strong> After locking column <em>i</em>, its rounding error is redistributed to every remaining column weighted by <code>H_inv[i,j] / H_inv[i,i]</code>. Columns that share activation patterns with column <em>i</em> (high off-diagonal H_inv) absorb a larger correction — the algorithm minimises total <em>output</em> error, not individual weight error.
  </div>
  <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:6px;padding:10px 14px;font-size:11px;color:#92400e;margin-top:10px;line-height:1.7;">
    <strong>Why Cholesky, not a standard matrix inverse?</strong> H = 2XᵀX is symmetric positive semi-definite — Cholesky decomposition exploits that structure for a 2× cheaper factorisation and, more importantly, is numerically stable near singularity. Standard inversion (LU decomposition) suffers catastrophic cancellation when any diagonal entry of H is close to zero, amplifying floating-point errors into the result. Cholesky avoids this by working with the square root factor L (where H = LLᵀ), making the near-zero case well-conditioned after damping.
  </div>
</div>

<!-- GPTQ Column-by-Column Visualization -->
<div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#4c1d95;font-size:13px;margin-bottom:16px;">GPTQ: column-by-column quantization with error propagation</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;margin-bottom:14px;">

    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">① Initial W</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#ddd6fe;border:1px solid #8b5cf6;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c0</div>
          <div style="font-size:10px;line-height:1.6;">0.40<br>0.76</div>
        </div>
        <div style="background:#ddd6fe;border:1px solid #8b5cf6;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c1</div>
          <div style="font-size:10px;line-height:1.6;">0.62<br>0.18</div>
        </div>
        <div style="background:#ddd6fe;border:1px solid #8b5cf6;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c2</div>
          <div style="font-size:10px;line-height:1.6;">0.91<br>0.55</div>
        </div>
        <div style="background:#ddd6fe;border:1px solid #8b5cf6;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#4c1d95;font-weight:600;margin-bottom:3px;">c3</div>
          <div style="font-size:10px;line-height:1.6;">0.27<br>0.43</div>
        </div>
      </div>
      <div style="font-size:9px;color:#64748b;margin-top:5px;">all unquantized</div>
    </div>

    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>

    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">② Quantize c0 → propagate</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.50<br>1.00</div>
        </div>
        <div style="background:#fef9c3;border:1px solid #eab308;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#92400e;font-weight:600;margin-bottom:3px;">c1 ✦</div>
          <div style="font-size:10px;line-height:1.6;">0.60<br>0.16</div>
        </div>
        <div style="background:#fef9c3;border:1px solid #eab308;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#92400e;font-weight:600;margin-bottom:3px;">c2 ✦</div>
          <div style="font-size:10px;line-height:1.6;">0.89<br>0.53</div>
        </div>
        <div style="background:#fef9c3;border:1px solid #eab308;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#92400e;font-weight:600;margin-bottom:3px;">c3 ✦</div>
          <div style="font-size:10px;line-height:1.6;">0.25<br>0.41</div>
        </div>
      </div>
      <div style="font-size:9px;color:#d97706;margin-top:5px;">✦ = adjusted by c0 error via H⁻¹</div>
    </div>

    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>

    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">③ Quantize c1 → propagate</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.50<br>1.00</div>
        </div>
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c1 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.50<br>0.00</div>
        </div>
        <div style="background:#fef9c3;border:1px solid #eab308;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#92400e;font-weight:600;margin-bottom:3px;">c2 ✦</div>
          <div style="font-size:10px;line-height:1.6;">0.91<br>0.55</div>
        </div>
        <div style="background:#fef9c3;border:1px solid #eab308;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#92400e;font-weight:600;margin-bottom:3px;">c3 ✦</div>
          <div style="font-size:10px;line-height:1.6;">0.27<br>0.43</div>
        </div>
      </div>
      <div style="font-size:9px;color:#d97706;margin-top:5px;">error flows right, never back</div>
    </div>

    <div style="display:flex;align-items:center;font-size:16px;color:#7c3aed;padding-top:24px;">→</div>

    <div style="flex:1;min-width:130px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#1e293b;margin-bottom:6px;">④ Final Ŵ — all done</div>
      <div style="display:inline-flex;gap:3px;">
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c0 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.50<br>1.00</div>
        </div>
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c1 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.50<br>0.00</div>
        </div>
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c2 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">1.00<br>0.50</div>
        </div>
        <div style="background:#bbf7d0;border:2px solid #16a34a;border-radius:4px;padding:6px 8px;min-width:36px;">
          <div style="font-size:9px;color:#14532d;font-weight:700;margin-bottom:3px;">c3 ✓</div>
          <div style="font-size:10px;font-weight:600;line-height:1.6;">0.00<br>0.50</div>
        </div>
      </div>
      <div style="font-size:9px;color:#14532d;margin-top:5px;">all quantized, errors absorbed</div>
    </div>
  </div>

  <div style="background:#f5f3ff;border-radius:6px;padding:10px 14px;font-size:11px;color:#4c1d95;line-height:1.7;">
    <strong>Error propagation is always left → right.</strong> Once a column is locked (green ✓), it is never modified again. Yellow (✦) columns have been pre-adjusted by prior errors — so when they are quantized next, the rounded value is already steered toward the output-preserving target. The H⁻¹[i,j] / H⁻¹[i,i] term scales the correction: large when columns <em>i</em> and <em>j</em> are co-activated in calibration data, near-zero when they are independent. More precisely, H⁻¹<sub>ij</sub>/H⁻¹<sub>ii</sub> is the OLS regression coefficient of column <em>j</em> on column <em>i</em> in the calibration activations — it measures how much adjusting column <em>j</em> compensates for the quantization error at column <em>i</em>, making the update a closed-form solution to the residual least-squares problem.
  </div>
</div>


#### GPTQ Python code {#gptq-code}

*What parameters actually matter at call time, and what happens if you leave them at their defaults?*

> **Illustrative, not copy-paste.** This snippet is annotated to show what each knob does — the calibration-data format `GPTQModel.quantize()` expects varies by version, so check it against your installed `gptqmodel`. For runnable, end-to-end code see [Lab 2 — GPTQ & AWQ under the hood](/posts/gptq-awq-underthehood/).
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
# Selecting exactly 128 then filtering can silently leave fewer samples.
samples = [
    tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512)
    for row in dataset.select(range(256))         # over-select; trim after filter
    if len(row["text"].strip()) > 50
][:128]

# ── QuantizeConfig — key parameters explained ──────────────────────
quant_config = QuantizeConfig(
    bits=4,              # target weight bit-width: 4 = INT4, 8 = INT8
    group_size=128,      # weights are split into groups of 128; each group gets
                         #   its own scale+zero. Smaller group → better accuracy
                         #   but more overhead. 128 is the production default.
                         #   Note: GPTQ uses one global Hessian for the full
                         #   column loop — there is NO per-group Hessian
                         #   sub-block. The Hessian captures full-layer output
                         #   sensitivity; group_size only governs the
                         #   quantization grid granularity, applied column-wise
                         #   within that global error-propagation loop.
    desc_act=False,      # "descending activations": reorder columns by
                         #   decreasing activation magnitude before quantizing.
                         #   High-sensitivity columns are processed first so
                         #   their rounding errors propagate into the remaining
                         #   low-sensitivity columns — not the reverse. Without
                         #   reordering, accumulated low-sensitivity errors land
                         #   on the most sensitive columns last.
                         #   True improves accuracy ~0.1–0.2 PPL; False is
                         #   faster and simpler. Use False for speed, True for
                         #   best quality when you have time.
    damp_percent=0.01,   # Levenberg-Marquardt damping added to H diagonal:
                         #   H_damped = H + λ·I, λ = damp_percent × mean(diag(H))
                         #   Prevents near-singularity in H. The main cause:
                         #   "dead" input channels that are zero (or near-zero)
                         #   across all calibration samples contribute nothing
                         #   to XX^T, leaving H_{ii} ≈ 0 and H^{-1}_{ii} → ∞.
                         #   Damping adds a small floor to every diagonal entry,
                         #   making the inverse numerically stable.
                         #   Values: 0.005–0.1. Increase if NaN/Inf during quant.
    sym=False,           # Symmetric quantization (zero_point=0) vs asymmetric.
                         #   sym=True: INT4 range [-8,7]. sym=False: INT4 range
                         #   [0,15] with a per-group zero-point.
                         #   Asymmetric (False) is better for ReLU-heavy FFN
                         #   layers; symmetric (True) is the convention for INT8
                         #   GEMM tensor-core kernels (no zero-point term in the
                         #   accumulate), though not strictly required.
                         #   Default False for W4A16.
)

model = GPTQModel.from_pretrained(model_id, quant_config)
model.quantize(samples)
model.save_quantized("./qwen3-4b-gptq-w4a16")
```

> **`damp_percent` and NaN.** If quantization produces NaN weights in early layers, raise `damp_percent` from 0.01 to 0.05 or 0.1. The damping prevents division by near-zero diagonal elements of $H^{-1}$.
{: .prompt-danger }

> **`desc_act=True` at scale.** The full Hessian $H$ is materialized per layer regardless of `desc_act` — it is required to compute $H^{-1}$ at all — and *that* is the real memory driver at 70B (a 28672-wide FFN gives a ~3 GB Hessian even with `desc_act=False`). `desc_act` itself only adds a column permutation (argsort the diagonal, reorder columns), which is negligible memory. Its actual cost at scale is the act-order group bookkeeping and slower inference kernels, not extra Hessian memory. Set `desc_act=False` when you want the simpler, faster path; the accuracy gap is ~0.1–0.2 PPL.
{: .prompt-warning }

### AWQ — Activation-Magnitude Channel Scaling {#awq}

*Why does looking at activations — not weights — tell you which weights are most dangerous to quantize?*

**AWQ identifies salient weights using activations:** a weight column paired with a large-magnitude input channel contributes far more to the output error budget than its own distribution suggests.

GPTQ's weakness is cost: computing and inverting $H = 2X^TX$ per layer is expensive. AWQ takes a different angle — instead of Hessian-guided reconstruction, it asks: *which weight columns matter most to the output?*

The output of a linear layer is $Y = XW^T$. When we quantize $W$ to $\hat{W}$, the rounding error for element $i$ is $\epsilon_i = \hat{w}_i - w_i$ (where $\hat{w}_i$ is the quantized value), so the output error for one column $i$ is:

$$
\delta y_i = (w_i - \hat{w}_i) \cdot x_i = -\epsilon_i \cdot x_i
$$

where $\epsilon_i$ is the rounding error for that weight entry and $x_i$ is the corresponding input activation. The total layer output error expands to:

$$
\|\delta Y\|_F = \left\|\sum_i \epsilon_i \cdot x_i\right\|_F
$$

**This means if $x_i$ is large, even a small rounding error $\epsilon_i$ produces a huge output error.**  example: suppose a typical rounding error is $\epsilon_i = 0.05$ (one quantization step at INT4 group-size 128). For a normal channel with $x_i = 0.3$, the output error contribution is $0.05 \times 0.3 = 0.015$ — negligible. For the dominant outlier channel with $x_i = 85$, the same rounding error contributes $0.05 \times 85 = \mathbf{4.25}$ — 283× larger. This is why you cannot predict weight sensitivity from the weight distribution alone.

<div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:10px;padding:16px 18px;margin:16px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#1e3a8a;margin-bottom:6px;font-size:13px;">Quantization error amplification: $\lvert\delta y_i\rvert = \lvert\epsilon_i\rvert \cdot \lvert x_i\rvert$</div>
  <div style="font-size:11px;color:#1e3a8a;margin-bottom:14px;">Rounding error $\epsilon_i = 0.05$ (same for every column) — output error scales with activation magnitude</div>

  <div style="margin-bottom:10px;font-size:11px;font-weight:600;color:#1e3a8a;">Activation magnitude $|x_i|$ → output error contribution</div>
  <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:90px;font-size:11px;color:#1e293b;flex-shrink:0;text-align:right;">ch 0 · x=0.3</span>
      <div style="flex:1;background:#fdffb6;border-radius:3px;height:16px;overflow:hidden;">
        <div style="width:0.4%;height:100%;background:#a0c4ff;border-radius:3px;"></div>
      </div>
      <span style="width:140px;font-size:11px;color:#1e293b;flex-shrink:0;">err = 0.015 — tiny</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:90px;font-size:11px;color:#1e293b;flex-shrink:0;text-align:right;">ch 1 · x=0.7</span>
      <div style="flex:1;background:#fdffb6;border-radius:3px;height:16px;overflow:hidden;">
        <div style="width:0.9%;height:100%;background:#a0c4ff;border-radius:3px;"></div>
      </div>
      <span style="width:140px;font-size:11px;color:#1e293b;flex-shrink:0;">err = 0.035 — tiny</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:90px;font-size:11px;color:#78350f;flex-shrink:0;font-weight:600;text-align:right;">ch 2 · x=4.2</span>
      <div style="flex:1;background:#fdffb6;border-radius:3px;height:16px;overflow:hidden;">
        <div style="width:5%;height:100%;background:#ffd6a5;border-radius:3px;"></div>
      </div>
      <span style="width:140px;font-size:11px;color:#78350f;font-weight:600;flex-shrink:0;">err = 0.21 — notable</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:90px;font-size:11px;color:#7f1d1d;flex-shrink:0;font-weight:700;text-align:right;">ch 3 · x=85</span>
      <div style="flex:1;background:#fdffb6;border-radius:3px;height:16px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:3px;"></div>
      </div>
      <span style="width:140px;font-size:11px;color:#7f1d1d;font-weight:700;flex-shrink:0;">err = 4.25 — dominant ×283</span>
    </div>
  </div>

  <div style="background:#ffd6a5;border:1px solid #f59e0b;border-radius:6px;padding:10px 14px;font-size:11px;color:#78350f;line-height:1.7;">
    <strong>AWQ fix:</strong> scale up salient weight columns by $s_i \propto |x_i|^\alpha$ before quantizing. The quantization grid becomes proportionally finer for those columns, reducing $\epsilon_i$ for the weights that matter most. Divide activations by $s_i$ at runtime to preserve output identity — zero accuracy loss from the scaling itself.
  </div>
</div>

AWQ's mechanism: scale the salient weight columns up by $s_i$ before quantizing (grid is proportionally finer relative to their values), then divide the corresponding activations by $s_i$ at runtime to preserve mathematical identity. The optimal scale is found by grid-searching $\alpha$:

<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · AWQ Optimal Scale Search</div>

$$
s_i^* = \arg\min_{s_i} \left\| Q\!\left(W_i \cdot s_i\right) / s_i \cdot X_i - W_i \cdot X_i \right\|, \quad s_i \propto \overline{|x_i|}^\alpha, \quad \alpha \in [0,1]
$$

  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $\alpha$ grid-searched on calibration set (typically 0.0 → 1.0 in 0.1 steps) · $\alpha \approx 0.5$ works for most LLMs<br>
    Scale folded into the preceding LayerNorm's gain vector at export — the ÷s correction is pre-baked into LayerNorm's output, adding zero extra runtime operations<br>
    <span style="color:#7f1d1d;">Caveat: this folding needs a linear op (LayerNorm or a preceding matmul) to absorb the ÷s. It works for <code>q/k/v/gate/up</code> projections fed by a LayerNorm, but <code>down_proj</code> sits right after the non-linear activation with no such host — AWQ handles those columns separately (e.g. scaling fused into the preceding <code>up_proj</code> output) or skips them.</span>
  </div>
</div>

<!-- AWQ Pseudocode -->
<div style="background:#f8f9fa;border:1px solid #d0d7de;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:14px;">AWQ — Algorithm Pseudocode</div>
  <pre style="background:#161b22;color:#c9d1d9;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#79c0ff;font-weight:600;">Input:</span>  W [out × in]   — weight matrix for one linear layer
        X [N × in]     — calibration activations (N samples)
        α_grid         — candidate scales, e.g. [0.0, 0.1, …, 1.0]
<span style="color:#79c0ff;font-weight:600;">Output:</span> W_fused [out × in] — weight matrix with scale absorbed

<span style="color:#f0883e;">1.</span> x_mean[i] = mean over calibration of |X[:, i]|  <span style="color:#8b949e;"># per input-channel mean activation magnitude (AWQ paper)</span>
<span style="color:#f0883e;">2.</span> w_max[i] = max over output rows of |W[:, i]|    <span style="color:#8b949e;"># per input-channel weight magnitude</span>

<span style="color:#f0883e;">3.</span> best_s = ones(in_features)
<span style="color:#f0883e;">4.</span> best_err = ∞

<span style="color:#f0883e;">5.</span> <span style="color:#ff7b72;font-weight:600;">for</span> α in α_grid:                               <span style="color:#8b949e;"># grid search over candidate exponents</span>
<span style="color:#f0883e;">6.</span>     s[i] = x_mean[i]^α / w_max[i]^(1−α)        <span style="color:#8b949e;"># migration scale, per channel</span>
<span style="color:#f0883e;">7.</span>     s    = clamp(s, 1e-4, 1e4)                  <span style="color:#8b949e;"># avoid FP overflow / near-zero division</span>

<span style="color:#f0883e;">8.</span>     W_scaled[:, i] = W[:, i] × s[i]             <span style="color:#8b949e;"># scale weight columns UP</span>
<span style="color:#f0883e;">9.</span>     Ŵ_scaled       = Q(W_scaled)                 <span style="color:#8b949e;"># quantize the scaled weights</span>
<span style="color:#f0883e;">10.</span>    W_hat[:, i]    = Ŵ_scaled[:, i] / s[i]      <span style="color:#8b949e;"># undo scale for fair error comparison</span>

<span style="color:#f0883e;">11.</span>    err = ‖ W × X_rep − W_hat × X_rep ‖₂        <span style="color:#8b949e;"># reconstruction error on representative input</span>
<span style="color:#f0883e;">12.</span>    <span style="color:#ff7b72;font-weight:600;">if</span> err < best_err: best_s = s, best_err = err

<span style="color:#f0883e;">13.</span> W_fused[:, i] = W[:, i] × best_s[i]            <span style="color:#8b949e;"># absorb scale into model weights permanently</span>
<span style="color:#f0883e;">14.</span> <span style="color:#ff7b72;font-weight:600;">return</span> W_fused                                   <span style="color:#8b949e;"># at runtime: x / best_s before the GEMM</span></pre>
  <div style="background:#f0f6ff;border:1px solid #d0d7de;border-radius:6px;padding:10px 14px;font-size:11px;color:#1f2328;margin-top:12px;line-height:1.7;">
    <strong>No matrix inversion.</strong> AWQ never computes or inverts a Hessian. The only math per α candidate is a per-channel divide/multiply and a forward pass — this is why AWQ is 3–5× faster than GPTQ. The scale search is also embarrassingly parallel across layers.
  </div>
</div>

<!-- AWQ Step-by-Step Visualization -->
<div style="background:#fffffc;border:1px solid #86efac;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#14532d;font-size:13px;margin-bottom:16px;">AWQ: how activation magnitude guides weight scaling</div>

  <!-- Step 1: identify salient channels -->
  <div style="margin-bottom:18px;">
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">① Identify salient channels — rank by activation magnitude |x_i|</div>
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#1e293b;text-align:right;">ch 0</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:0.4%;height:100%;background:#a0c4ff;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#64748b;width:80px;">|x|=0.3 — safe</span>
        <span style="font-size:10px;color:#64748b;">s = 1.0 (no scaling)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#1e293b;text-align:right;">ch 1</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:1%;height:100%;background:#a0c4ff;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#64748b;width:80px;">|x|=0.8 — safe</span>
        <span style="font-size:10px;color:#64748b;">s = 1.0 (no scaling)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#d97706;font-weight:600;text-align:right;">ch 2</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:5%;height:100%;background:#fbbf24;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#d97706;font-weight:600;width:80px;">|x|=4.2 — notable</span>
        <span style="font-size:10px;color:#d97706;">s = 2.0 (moderate scale)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:60px;font-size:11px;color:#dc2626;font-weight:700;text-align:right;">ch 3</span>
        <div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:100%;height:100%;background:#fca5a5;border-radius:3px;"></div>
        </div>
        <span style="font-size:10px;color:#dc2626;font-weight:700;width:80px;">|x|=85 — outlier</span>
        <span style="font-size:10px;color:#dc2626;font-weight:700;">s = 9.2 (max scaling)</span>
      </div>
    </div>
  </div>

  <!-- Step 2: what scaling does to the grid -->
  <div style="margin-bottom:18px;">
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">② Scale up salient weight column → finer effective quantization grid</div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;background:#fff1f2;border:1px solid #fca5a5;border-radius:7px;padding:12px 14px;">
        <div style="font-size:10px;font-weight:700;color:#7f1d1d;margin-bottom:8px;">ch 3 BEFORE scaling (s = 1)</div>
        <div style="position:relative;height:16px;background:#ffe4e6;border-radius:3px;margin-bottom:4px;">
          <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:25%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:75%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.5;"></div>
          <div style="position:absolute;left:calc(100%-2px);top:0;width:2px;height:100%;background:#7f1d1d;opacity:.9;"></div>
          <!-- w = 0.73 marker -->
          <div style="position:absolute;left:73%;top:0;width:3px;height:100%;background:#dc2626;"></div>
        </div>
        <div style="font-size:9px;color:#7f1d1d;display:flex;justify-content:space-between;"><span>0</span><span style="font-weight:700;color:#dc2626;">w=0.73 ↑</span><span>1.0</span></div>
        <div style="font-size:10px;color:#7f1d1d;margin-top:5px;">step = 0.25 · w rounds to <strong>0.75</strong> · error = +0.02</div>
        <div style="font-size:10px;color:#7f1d1d;">output error = 0.02 × 85 = <strong style="color:#dc2626;">+1.70</strong></div>
      </div>
      <div style="flex:1;min-width:200px;background:#f0fdf4;border:1px solid #86efac;border-radius:7px;padding:12px 14px;">
        <div style="font-size:10px;font-weight:700;color:#14532d;margin-bottom:8px;">ch 3 AFTER scaling (s = 9.2)</div>
        <div style="position:relative;height:16px;background:#dcfce7;border-radius:3px;margin-bottom:4px;">
          <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:12.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:25%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:37.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:62.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:75%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:87.5%;top:0;width:2px;height:100%;background:#14532d;opacity:.5;"></div>
          <div style="position:absolute;left:calc(100%-2px);top:0;width:2px;height:100%;background:#14532d;opacity:.9;"></div>
          <!-- scaled w = 0.73 × 9.2 / 9.2 → in scaled space 0.73 marker -->
          <div style="position:absolute;left:73%;top:0;width:3px;height:100%;background:#16a34a;"></div>
        </div>
        <div style="font-size:9px;color:#14532d;display:flex;justify-content:space-between;"><span>0</span><span style="font-weight:700;color:#16a34a;">w·s=6.72↑</span><span>9.2</span></div>
        <div style="font-size:10px;color:#14532d;margin-top:5px;">step = 9.2/15 ≈ 0.61 (INT4, scaled space) · rounds to ≈<strong>6.74</strong> → /9.2 ≈ 0.733</div>
        <div style="font-size:10px;color:#14532d;">output error = 0.003 × 85 = <strong style="color:#16a34a;">+0.26</strong> (&gt;6× smaller)</div>
      </div>
    </div>
    <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:6px;padding:10px 14px;font-size:11px;color:#92400e;margin-top:12px;line-height:1.7;">
      <strong>INT4 always has 16 levels.</strong> The grids above are simplified for visual clarity — scaling does not add quantization levels. What AWQ actually changes is <em>which channel sets the group scale</em>: INT4 step = <em>group_max</em> / 15. Before AWQ, a non-salient channel may dominate the group range, forcing the salient column to share the 16-level budget with the whole group. Scaling the salient column up by s<sub>i</sub> makes it the new group_max, so all 16 levels are allocated over its own range — giving it finer effective precision at the cost of coarser precision for the low-activation channels that don't matter.
    </div>
  </div>

  <!-- Step 3: runtime identity preservation -->
  <div style="margin-bottom:10px;">
    <div style="font-size:11px;font-weight:700;color:#1e293b;margin-bottom:8px;">③ Runtime: activations arrive pre-divided — s absorbed into the preceding LayerNorm</div>
    <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:7px;padding:12px 14px;font-size:11px;color:#1e293b;line-height:1.8;">
      <div>At export: <code style="background:#e2e8f0;padding:1px 4px;border-radius:3px;">W_fused[:, i] = W[:, i] × s[i]</code> (absorbed into model file)</div>
      <div>At export: the ÷s correction is folded into the preceding LayerNorm's gain vector (<code style="background:#e2e8f0;padding:1px 4px;border-radius:3px;">gamma /= s</code>) — so the adjusted activations emerge from LayerNorm directly, with no separate division op at runtime.</div>
      <div>At runtime: <code style="background:#e2e8f0;padding:1px 4px;border-radius:3px;">x_adj = LayerNorm_output</code> (already ÷s) → GEMM → output matches FP16 exactly</div>
      <div style="margin-top:6px;padding-top:6px;border-top:1px solid #e2e8f0;">Y = (X / s) × (W × s)ᵀ = X × Wᵀ &nbsp;✓&nbsp; <span style="color:#64748b;">— zero extra runtime operations: the rescaling lives inside the LayerNorm that was already running</span></div>
    </div>
  </div>
</div>

The scale $s_i^*$ is fused into the weight matrix at export. No Hessian inversion required — AWQ typically runs 3–5× faster than GPTQ.

#### AWQ Python code {#awq-code}

*What does each config knob actually control, and which ones move accuracy most?*

> **Illustrative, not copy-paste.** The `quantize()` signature and `calib_data` handling differ across AutoAWQ versions — verify against your installed `autoawq`. For runnable, end-to-end code see [Lab 2 — GPTQ & AWQ under the hood](/posts/gptq-awq-underthehood/).
{: .prompt-info }

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ── Quantization config — key parameters explained ─────────────────
quant_config = {
    "zero_point": True,      # Asymmetric quantization: use a per-group zero-point
                             #   offset so the INT4 range is centred on the actual
                             #   weight distribution rather than forced symmetric.
                             #   True (asymmetric) almost always beats False for
                             #   W4A16. Set False only if downstream kernel
                             #   requires symmetric weights (rare).
    "q_group_size": 128,     # Group size for per-group scale+zero computation.
                             #   128 is the standard; 64 gives ~0.1 better PPL at
                             #   the cost of 2× more metadata overhead.
                             #   Values: 32, 64, 128, or -1 (per-channel = col).
    "w_bit": 4,              # Target weight bit-width. 4 = INT4 (W4A16).
                             #   Use 8 for W8A16 when latency budget allows.
    "version": "GEMM",       # Packing/kernel format written into the checkpoint.
                             #   "GEMM": fastest on RTX/A-series GPUs with
                             #     large batch; uses CUDA GEMM fused dequant.
                             #   "GEMV": optimised for batch=1 (single-token
                             #     decode); reduces overhead for small batch.
                             #   "GEMVFast": faster GEMV variant.
                             #   Note: Marlin is NOT an AutoAWQ version — it is an
                             #   inference-time kernel the serving runtime (e.g.
                             #   vLLM) selects at load for GEMM-packed weights on
                             #   A100/H100. Pick GEMV for latency-critical
                             #   single-user; GEMM (+ Marlin at serve time) for
                             #   high-throughput serving on H100.
}

model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="auto")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="wikitext-2-raw-v1",    # HuggingFace dataset name or list of strings.
                              #   Replace with domain-matched data for best results.
    max_calib_samples=128,    # Number of calibration sequences. 128 is sufficient
    max_calib_seq_len=512,    # Token length per calibration sample. Longer
                              #   sequences produce more accurate per-channel
                              #   statistics but increase memory and time linearly.
)
model.save_quantized("./qwen3-4b-w4a16", safetensors=True)
```

> **W4A16 ceiling.** Both GPTQ and AWQ are weight-only. If you need INT8 GEMM throughput (W8A8 for 2× FLOP rate on Turing+ GPUs), you need a separate technique — SmoothQuant, covered in [Part 3](/posts/quantization-3/).
{: .prompt-warning }

### GPTQ vs AWQ — Comparison and Decision Guide {#gptq-vs-awq}

*Same target (W4A16), different tradeoffs. Here's how to pick.*

<!-- Comparison Grid -->
<div style="margin:20px 0;border:1px solid #d0d7de;border-radius:10px;overflow:hidden;font-family:system-ui,sans-serif;font-size:12px;">

  <!-- header -->
  <div style="display:grid;grid-template-columns:1fr 1fr;">
    <div style="padding:10px 14px;background:#f0f0ff;border-right:1px solid #a0c4ff;font-weight:700;color:#4c1d95;font-size:13px;text-align:center;">GPTQ</div>
    <div style="padding:10px 14px;background:#f0fff4;font-weight:700;color:#14532d;font-size:13px;text-align:center;">AWQ</div>
  </div>

  <!-- Core mechanism -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Core mechanism</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">Hessian-guided — propagates each column's rounding error forward via inverse Hessian to minimise layer output error</div>
    <div style="padding:8px 14px;color:#1e293b;">Activation-magnitude scaling — scales up salient weight columns before quantizing so their effective grid is finer</div>
  </div>

  <!-- Quantization speed -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Quantization speed</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#7f1d1d;font-weight:600;">Slow — Cholesky inversion + O(n²) column loop; ~30–60 min for 7B on A100</div>
    <div style="padding:8px 14px;color:#14532d;font-weight:600;">Fast — no matrix inversion; α grid search is parallel; ~5–10 min for 7B on A100</div>
  </div>

  <!-- W4A16 accuracy -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">W4A16 accuracy</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">Marginally better — ~0.05–0.15 lower PPL on matched calibration; gap shrinks with good data</div>
    <div style="padding:8px 14px;color:#1e293b;">Comparable — within 0.1–0.2 PPL; smaller than the effect of a calibration domain mismatch</div>
  </div>

  <!-- Peak GPU memory -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Peak GPU memory during quantization</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">Higher — must materialise the full Hessian $H$ per layer (~3 GB for a 28672-wide FFN at 70B), independent of <code>desc_act</code></div>
    <div style="padding:8px 14px;color:#1e293b;">Lower — per-channel stats only; fits 70B+ on a single 80 GB GPU</div>
  </div>

  <!-- Calibration sensitivity -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Calibration sensitivity</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">Higher — Hessian directly encodes the calibration distribution; domain mismatch corrupts sensitivity estimates</div>
    <div style="padding:8px 14px;color:#1e293b;">Lower — per-channel magnitude stats are more stable; forgiving when calibration data is imperfect</div>
  </div>

  <!-- Runtime overhead -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Runtime overhead</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">None — dequantize-on-load GEMM</div>
    <div style="padding:8px 14px;color:#1e293b;">None — scale fused into weights at export</div>
  </div>

  <!-- Inference kernels -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Inference kernels</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;"><code>exllama_v2</code> (decode), <code>gptq-marlin</code> (A100+ throughput)</div>
    <div style="padding:8px 14px;color:#1e293b;"><code>GEMV</code> (batch=1), <code>GEMM</code> (batch≥4), <code>Marlin</code> (H100); widest coverage</div>
  </div>

  <!-- Pre-quantized models -->
  <div style="padding:5px 14px;background:#f8fafc;border-top:1px solid #d0d7de;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;">Pre-quantized models</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e2e8f0;">
    <div style="padding:8px 14px;border-right:1px solid #d0d7de;color:#1e293b;">Large — most 7B–70B models have GPTQ variants on HF Hub</div>
    <div style="padding:8px 14px;color:#1e293b;">Growing — official Qwen/Llama/Mistral releases; default in vLLM and TGI</div>
  </div>

</div>

<!-- Decision Flowchart -->
<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:20px 22px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:18px;">Decision workflow — GPTQ or AWQ?</div>

  <!-- Q1 -->
  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#6366f1;color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">1</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is quantization time a hard constraint? (e.g. nightly re-quantization, CI pipeline)</div>
      <div style="display:flex;gap:20px;">
        <div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;padding:6px 12px;font-size:12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ</strong>. 3–5× faster; fine to iterate.</div>
        <div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:6px;padding:6px 12px;font-size:12px;color:#475569;">No → continue ↓</div>
      </div>
    </div>
  </div>

  <!-- Q2 -->
  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#6366f1;color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">2</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is the model ≥ 70B?</div>
      <div style="display:flex;gap:20px;">
        <div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;padding:6px 12px;font-size:12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ</strong>. GPTQ's per-layer Hessian (~3 GB/layer at 70B) plus model weights strains a single 80 GB GPU; AWQ's per-channel stats stay comfortably within it.</div>
        <div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:6px;padding:6px 12px;font-size:12px;color:#475569;">No → continue ↓</div>
      </div>
    </div>
  </div>

  <!-- Q3 -->
  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#6366f1;color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">3</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is your calibration data well-matched to the deployment domain?</div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;">
        <div style="background:#ede9fe;border:1px solid #c4b5fd;border-radius:6px;padding:6px 12px;font-size:12px;color:#4c1d95;font-weight:600;">Yes → <strong>GPTQ</strong>. Domain-accurate Hessian pays off; <code>desc_act=True</code> extracts the last 0.1–0.2 PPL.</div>
        <div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;padding:6px 12px;font-size:12px;color:#14532d;font-weight:600;">No / mixed → <strong>AWQ</strong>. Per-channel magnitude stats degrade more gracefully under distribution shift.</div>
      </div>
    </div>
  </div>

  <!-- Q4 -->
  <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px;">
    <div style="background:#6366f1;color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:1px;">4</div>
    <div>
      <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Is inference throughput the primary goal? (multi-GPU serving, H100)</div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;">
        <div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;padding:6px 12px;font-size:12px;color:#14532d;font-weight:600;">Yes → <strong>AWQ + Marlin kernel</strong>. Best tokens/s on A100/H100 at batch ≥ 8.</div>
        <div style="background:#ede9fe;border:1px solid #c4b5fd;border-radius:6px;padding:6px 12px;font-size:12px;color:#4c1d95;font-weight:600;">Latency / batch=1 → <strong>AWQ GEMV</strong> or <strong>GPTQ exllama_v2</strong>. Similar latency; pick GPTQ if a pre-quantized checkpoint already exists.</div>
      </div>
    </div>
  </div>

  <!-- Summary rule -->
  <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:7px;padding:10px 14px;font-size:12px;color:#0c4a6e;margin-top:4px;line-height:1.7;">
    <strong>Short version:</strong> default to AWQ — it is faster to quantize, scales to any model size, and is less sensitive to calibration quality. Reach for GPTQ when you have domain-matched calibration data and a sub-70B model where the Hessian cost is acceptable, or when you need to squeeze the last fraction of a perplexity point.
  </div>
</div>

---

## 2. Calibration Data: The Hidden Accuracy Lever {#calibration}

*Why does the calibration dataset — which never runs at inference — determine production accuracy as much as the algorithm itself?*

Calibration data is the variable practitioners underestimate most. A correctly chosen calibration set is often worth more than switching from AWQ to GPTQ.

### Questions to answer before choosing your calibration dataset {#calibration-checklist}

*Four questions that determine both which dataset to use and how to mix it — answer them before you touch any dataset selector.*

<div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e3a8a;margin-bottom:16px;font-size:14px;">Calibration dataset checklist</div>

  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">1. What is your deployment environment?</div>
    <div style="color:#1e293b;line-height:1.7;">Code generation, math reasoning, document QA, and general chat each activate different outlier channels — the channels that carry large activation magnitudes in one domain are not the same ones that matter in another. Calibrating on WikiText-2 prose for a coding assistant means the Hessian (GPTQ) or activation statistics (AWQ) will be wrong for the channels that drive your benchmark.<br><span style="font-weight:600;color:#1e3a8a;">Rule:</span> your calibration distribution must cover every task type your users will actually run.</div>
  </div>

  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">2. What are your typical input lengths? <span style="color:#7f1d1d;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">crucial</span></div>
    <div style="color:#1e293b;line-height:1.7;">The most commonly underestimated variable. Calibration sequences much shorter than production inputs under-sample long-range attention patterns and leave the outlier channels in deep layers invisible. If your deployment serves 2 k–4 k token prompts but you calibrate at <code>max_length=512</code>, per-channel statistics for layers 20–30 will be wrong.<br><span style="font-weight:600;color:#1e3a8a;">Rule:</span> set <code>max_calib_seq_len</code> to the P90 of your production prompt length, not the library default of 512.</div>
  </div>

  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">3. What languages or domain-specific terminology will appear?</div>
    <div style="color:#1e293b;line-height:1.7;">Non-English text and technical vocabularies (medical, legal, financial, code) produce token distributions that diverge sharply from English Wikipedia. A model calibrated on English prose has an accurate Hessian for English tokens and a poor one for CJK characters or domain jargon. Even 16–32 domain-matched samples out of 128 recovers most of the gap.<br><span style="font-weight:600;color:#1e3a8a;">Rule:</span> if your deployment is multilingual or domain-heavy, include at least one representative sample per significant language or terminology cluster.</div>
  </div>

  <div>
    <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">4. Single-turn or multi-turn?</div>
    <div style="color:#1e293b;line-height:1.7;">Multi-turn chat prepends conversation history to every forward pass, shifting the context-position distribution compared to single-turn samples. Key/value states in early attention layers are calibrated against the wrong token-position structure. This matters most for RoPE-based models where position embeddings influence outlier magnitude across layers.<br><span style="font-weight:600;color:#1e3a8a;">Rule:</span> if your deployment is multi-turn, wrap calibration samples in the same system-prompt + prior-turn template you use in production.</div>
  </div>
</div>

<div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e3a8a;margin-bottom:4px;">GSM8K 5-shot accuracy — Qwen2-7B W4 AWQ by calibration source</div>
  <div style="font-size:11px;color:#1e3a8a;margin-bottom:14px;">illustrative; your numbers will differ by model and base accuracy</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#1e293b;flex-shrink:0;">FP16 baseline</span>
      <div style="flex:1;background:#fdffb6;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#a0c4ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">63.2%</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#1e293b;flex-shrink:0;">WikiText-2 (default)</span>
      <div style="flex:1;background:#fdffb6;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:83%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">52.4%</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;flex-shrink:0;">−10.8pp</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#1e293b;flex-shrink:0;">GSM8K train 128 samples</span>
      <div style="flex:1;background:#fdffb6;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:95%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">60.0%</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;flex-shrink:0;">−3.2pp</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:200px;text-align:right;font-size:12px;color:#1e293b;flex-shrink:0;">Mixed (64+64+64 samples)</span>
      <div style="flex:1;background:#fdffb6;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:97%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">61.4%</span>
      <span style="background:#9bf6ff;color:#164e63;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #67e8f9;flex-shrink:0;">−1.8pp</span>
    </div>
  </div>
</div>

The default WikiText-2 calibration can cost 10+ points on a math benchmark while looking fine on WikiText-2 PPL. The Hessian (and AWQ's activation statistics) is estimated from the calibration distribution, so it accurately reflects weight sensitivity only for that distribution. The companion question — how to find the optimal quantization *range* from calibration data (min/max vs. MSE vs. percentile) — is covered in [Phase 1 § 6 Calibration Range Methods](/posts/quantization-1/#6-calibration-range-methods); range method and calibration domain are two independent levers that stack.

<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:12px;font-family:system-ui,sans-serif;table-layout:fixed;">
  <colgroup>
    <col style="width:26%">
    <col style="width:13%">
    <col style="width:12%">
    <col style="width:14%">
    <col style="width:35%">
  </colgroup>
  <thead>
    <tr style="background:#fdffb6;">
      <th style="padding:5px 8px;border:1px solid #a0c4ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#1e3a8a;font-weight:600;">Calibration source</th>
      <th style="padding:5px 8px;border:1px solid #a0c4ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#1e3a8a;font-weight:600;">WikiText-2 PPL</th>
      <th style="padding:5px 8px;border:1px solid #a0c4ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#1e3a8a;font-weight:600;">GSM8K 5-shot</th>
      <th style="padding:5px 8px;border:1px solid #a0c4ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#1e3a8a;font-weight:600;">HumanEval pass@1</th>
      <th style="padding:5px 8px;border:1px solid #a0c4ff;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#1e3a8a;font-weight:600;">Verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#caffbf;">
      <td style="padding:5px 8px;border:1px solid #a0c4ff;font-weight:600;">FP16 baseline</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">5.62</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">63.2%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">48.8%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">Reference</td>
    </tr>
    <tr style="background:#ffadad;">
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">WikiText-2 (default, 128)</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">5.81</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">52.4%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">38.1%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;color:#7f1d1d;font-weight:600;">Domain mismatch</td>
    </tr>
    <tr>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">GSM8K train (128)</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">5.74</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">60.0%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">39.5%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">Math matched, code suffers</td>
    </tr>
    <tr>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">Python code (128)</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">5.78</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">54.1%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">46.2%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">Code matched, math suffers</td>
    </tr>
    <tr style="background:#9bf6ff;">
      <td style="padding:5px 8px;border:1px solid #a0c4ff;font-weight:600;">Mixed (each dataset 64)</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">5.69</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">61.4%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;">45.7%</td>
      <td style="padding:5px 8px;border:1px solid #a0c4ff;color:#164e63;font-weight:600;">Best overall — use when domain unknown</td>
    </tr>
  </tbody>
</table>
</div>

The mixed row almost always beats any single-domain set when the deployment distribution is unknown.

> **PPL is not your benchmark.** WikiText-2 PPL hides domain-specific degradation. Always pair PPL with at least one downstream metric matching your deployment use case. If they disagree, trust the downstream metric.
{: .prompt-warning }

---

## Putting It All Together

<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="display:flex;flex-direction:column;gap:12px;">

    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">1</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">GPTQ and AWQ solve W4A16 from opposite directions</div>
        <div style="color:#64748b;font-size:12px;">GPTQ frames quantization as a reconstruction problem: minimize layer output error by propagating each column's rounding error forward via the inverse Hessian. AWQ takes the activation view: weight columns paired with large-magnitude inputs contribute the most output error, so scale those columns up before quantizing to reduce their rounding error. GPTQ gives finer error control but is 3–5× slower; AWQ is faster and good enough when calibration data is unclear. Neither touches activations — both are weight-only (W4A16).</div>
      </div>
    </div>

    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">2</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Calibration domain is often worth more than algorithm choice</div>
        <div style="color:#64748b;font-size:12px;">The Hessian (GPTQ) and activation statistics (AWQ) are estimated from calibration data — they accurately reflect weight sensitivity only for that domain. WikiText-2 prose calibration for a math or coding assistant can cost 10+ points on GSM8K while leaving WikiText-2 PPL nearly unchanged. Switching from WikiText-2 to domain-matched data typically recovers more accuracy than switching from AWQ to GPTQ. Always pair PPL with at least one downstream metric matching your deployment.</div>
      </div>
    </div>

    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">3</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Each algorithm closed the gap left by the previous one</div>
        <div style="color:#64748b;font-size:12px;">RTN quantizes each weight independently, ignoring output error — GPTQ fixed that. GPTQ is slow and ignores which weight columns actually matter — AWQ fixed that. Both leave W8A8 broken because activations cannot be per-channel quantized — SmoothQuant (Part 3) fixes that. Understanding this lineage makes the algorithm choices non-arbitrary: each technique is a direct response to a specific failure mode of the prior approach.</div>
      </div>
    </div>

  </div>
</div>

> **Next:** [Part 3 — W8A8: Activation Outliers and SmoothQuant](/posts/quantization-3/) covers why per-tensor INT8 collapses for LLMs and how SmoothQuant migrates outlier difficulty to the static weight side.
{: .prompt-info }

---

## References

1. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** *arXiv:2210.17323*. [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)

2. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** *arXiv:2306.00978*. [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)

3. **GPTQModel** (library). [https://github.com/modelcloud/gptqmodel](https://github.com/modelcloud/gptqmodel)

4. **AutoAWQ** (library). [https://github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

