---
layout: post
title: "Below 8 bit on LLMs: Weight and Activation Outliers"
date: 2026-06-01 10:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, ptq, smoothquant, spinquant, calibration, int4, int8, w8a8, w4a4, llm]
math: true
description: "Why per-tensor INT8 collapses for LLMs, how SmoothQuant's migration factor α shifts the difficulty to weights, and why rotation matrices (SpinQuant) are needed to reach W4A4."
---


<!-- Opening block: questions + prerequisites -->
<div style="margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:rgba(96,165,250,0.08);border:1px solid #60a5fa;border-radius:8px;padding:16px;margin-bottom:12px;">
    <div style="font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;font-size:12px;border-left:4px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;">
      <li>Why does per-tensor INT8 collapse for LLMs but work for vision models?</li>
      <li>Why can't per-channel or per-group quantization fix the activation outlier problem?</li>
      <li>How does SmoothQuant's migration factor α control where quantization difficulty lands?</li>
      <li>How do you find the optimal per-layer α without exhaustive evaluation?</li>
      <li>Why does SmoothQuant break at INT4, and what does SpinQuant do differently?</li>
      <li>How do the four SpinQuant rotation matrices (R₁–R₄) map onto the Transformer architecture?</li>
    </ul>
  </div>
  <div style="background:rgba(167,139,250,0.08);border:1px solid #a78bfa;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;font-size:12px;border-left:4px solid #a78bfa;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;">
      <li>INT4/INT8 quantization: scale, zero-point, per-tensor vs. per-channel</li>
      <li>PyTorch <code>nn.Linear</code> and <code>forward hooks</code></li>
      <li>Matrix multiplication identity: $Y = XW^T$</li>
      <li><a href="/posts/quantization-1/">Phase 1: quantization basics (RTN, PTQ flow, scale & zero-point)</a></li>
      <li><a href="/posts/quantization-2/">Phase 2: GPTQ, AWQ, and calibration data</a></li>
    </ul>
  </div>
</div>


GPTQ and AWQ (covered in [Part 2](/posts/quantization-2/)) both sidestep the activation problem entirely — they are weight-only (W4A16). W8A8 INT8 GEMM is the target for 2× hardware throughput on Turing+ GPUs, but it requires quantizing activations too. This post covers why that is hard and how SmoothQuant makes it tractable.


---


## 1. The Activation Outlier Problem {#outlier-problem}


*Why does per-tensor INT8 — which works for CNNs — destroy accuracy for LLMs, and why can't per-channel or per-group quantization rescue activations?*


<div style="background:#fffffc;border:1px solid #fca5a5;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#7f1d1d;font-size:13px;margin-bottom:18px;">INT8 quantization grid: ideal vs. outlier-distorted</div>


  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:600;color:#14532d;margin-bottom:6px;">① Ideal channel — max ≈ 1.5, no outlier · 256 levels across full value range</div>
    <div style="position:relative;height:22px;background:#caffbf;border-radius:4px;overflow:hidden;">
      <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:6.67%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:13.33%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:20%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:26.67%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:33.33%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:40%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:46.67%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:53.33%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:60%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:66.67%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:73.33%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:80%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:86.67%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:93.33%;top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
      <div style="position:absolute;left:calc(100% - 2px);top:0;width:2px;height:100%;background:#14532d;opacity:.7;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#14532d;margin-top:3px;padding:0 2px;">
      <span>−1.5</span>
      <span style="font-weight:600;">step = 0.012 · fine resolution everywhere</span>
      <span>+1.5</span>
    </div>
  </div>


  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:600;color:#7f1d1d;margin-bottom:6px;">② With one outlier channel (max = 150) · scale = 1.18 · same 256 levels now span 100× wider range</div>
    <div style="position:relative;height:22px;background:#ffadad;border-radius:4px;overflow:hidden;">
      <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:6.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:13.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:20%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:26.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:33.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:40%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:46.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:49.5%;top:0;width:1%;height:100%;background:rgba(100,63,193,0.4);"></div>
      <div style="position:absolute;left:53.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:60%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:66.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:73.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:80%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:86.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:93.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:calc(100% - 2px);top:0;width:2px;height:100%;background:#7f1d1d;opacity:.9;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#7f1d1d;margin-top:3px;padding:0 2px;">
      <span>−150</span>
      <span style="font-weight:600;">◀ 99% of values crammed in this 1% band ▶</span>
      <span style="font-weight:700;">+150 (outlier)</span>
    </div>
  </div>


  <div>
    <div style="font-size:11px;font-weight:600;color:#7f1d1d;margin-bottom:6px;">③ Zoom: non-outlier region [−1.5, +1.5] — only 3 quantization levels land here</div>
    <div style="position:relative;height:22px;background:#ffd6a5;border-radius:4px;overflow:hidden;">
      <div style="position:absolute;left:10.7%;top:0;width:2px;height:100%;background:#78350f;"></div>
      <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#78350f;"></div>
      <div style="position:absolute;left:89.3%;top:0;width:2px;height:100%;background:#78350f;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#78350f;margin-top:3px;padding:0 2px;">
      <span>−1.5</span>
      <span style="font-weight:700;">k_eff = 3 — INT8 format, 2-bit effective quality for 99% of channels</span>
      <span>+1.5</span>
    </div>
  </div>
</div>


Per-tensor INT8 sets one scale for the entire tensor: scale = tensor_max / 127. When one outlier channel reaches 150 while 99% of channels stay below 1.5, the scale becomes 1.18. The non-outlier channels then map to integer values in {−1, 0, 1} — three levels, equivalent to a 2-bit quantizer stuffed into an 8-bit format.


**Scale distortion** is what happens when a single extreme value forces the quantization scale so large that the remaining values lose almost all their resolution. The scale is set globally by the tensor maximum, so a handful of outlier channels hold the entire 256-level range hostage — normal channels are compressed into a tiny fraction of those levels and become effectively unquantized noise.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · Per-Tensor Scale Distortion</div>


$$
k_\text{eff} = \left\lfloor \frac{v}{\text{scale}} \right\rfloor \times 2 + 1, \quad \text{scale} = \frac{\max(|x|)}{2^{b-1}-1}
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $v$ = non-outlier channel max (e.g. 1.5) · tensor_max = outlier channel max (e.g. 150) · $b=8$ for INT8<br>
    With $v=1.5$, max=150: $k_\text{eff} = \lfloor 1.5/1.18 \rfloor \times 2 + 1 = \mathbf{3}$ levels
  </div>
</div>


The asymmetry between weights and activations is the key structural fact. **Weights** are static — you can apply per-channel quantization at export time and each output channel gets its own scale. **Activations** are dynamic — they change with every input token, so per-channel scales cannot be fixed at calibration time. This is why W4A16 (weight-only) is easy and W8A8 requires dedicated techniques.


> **Root cause.** Activation outliers are persistent spatial features: the same 1% of channels are outliers across all tokens and all calibration samples. That stability is exactly what SmoothQuant exploits — but it must first be measured on your specific deployment domain, not on WikiText-2.
{: .prompt-info }


### Why per-channel and per-group cannot fix activation outliers {#per-channel-limit}


*Why isn't per-channel quantization — which already fixes outlier weights — enough to also fix outlier activations?*


Per-channel quantization assigns one scale per **output channel** of a weight tensor, and it works because weights are static — their channel maxima are known once at export and never change. Activations are fundamentally different: their values are produced at runtime from a specific input token, so their per-channel maxima are not known at export time and change with every request.


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Weights — per-channel works ✓</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.9;">
      <li>Values are <strong>static</strong> — fixed at export</li>
      <li>Channel maxima computed once on disk</li>
      <li>Scale tensor fused into the model file</li>
      <li>Zero runtime cost, zero input dependency</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffadad;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#7f1d1d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Activations — per-channel cannot work ✗</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.9;">
      <li>Values are <strong>dynamic</strong> — depend on input token</li>
      <li>Channel max differs per request — cannot be pre-computed</li>
      <li>Would need per-token, per-channel scale computed online</li>
      <li>Online scaling requires a reduce kernel before every GEMM → latency bottleneck, negates INT8 gain</li>
    </ul>
  </div>
</div>


Per-group is the same story at finer granularity: you still need a static group max, which activations cannot provide. This is exactly why **W4A16** (weight-only) is easy and **W8A8** requires a dedicated technique — SmoothQuant — that migrates the difficulty from dynamic activations to the static weight side before export.


> **The asymmetry.** Weights are quantized once; activations are quantized at every forward pass. Any scheme that requires a dynamic per-channel scale for activations adds a fused-reduce kernel per GEMM call — this kills the throughput gain that INT8 was supposed to deliver.
{: .prompt-warning }


---


## 2. Activation Outlier Migration: SmoothQuant {#smoothquant}


### Why SmoothQuant Exists {#smoothquant-motivation}


*Why can't you just apply per-channel INT8 to activations the way you do to weights, and what mathematical trick lets SmoothQuant sidestep that constraint entirely?*


W8A8 is the target for high-throughput serving: INT8 tensor cores on Turing+ GPUs deliver 2× FLOP throughput vs. BF16. But per-tensor INT8 of activations is destroyed by outlier channels (Section 1). SmoothQuant's insight: quantization difficulty is a property of the *channel*, not of weights or activations in isolation. You can *migrate* the difficulty from activations (hard) to weights (easy) via a per-channel scaling that is mathematically invertible.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · SmoothQuant Migration</div>


$$
Y = X W^T = \underbrace{\bigl(X \cdot \operatorname{diag}(s)^{-1}\bigr)}_{\text{smoothed activations}} \cdot \underbrace{\bigl(\operatorname{diag}(s) \cdot W^T\bigr)}_{\text{scaled weights}}
$$


$$
s_i = \frac{\max|x_i|^\alpha}{\max|w_i|^{1-\alpha}}, \quad \alpha \in [0, 1]
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $s_i$ = per-channel migration scale · $x_i$ = activation channel $i$ · $w_i$ = weight column $i$<br>
    $\alpha=0$: no migration (pure per-tensor activation quant) · $\alpha=0.5$: balanced split · $\alpha=1$: all difficulty to weights
  </div>
</div>


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffadad;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#7f1d1d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Before SmoothQuant</div>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Activations X</div>
    <ul style="margin:0 0 10px 0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier channels: max ≈ 150</li>
      <li>Normal channels: max ≈ 1.5</li>
      <li>Per-tensor INT8 scale = 1.18 → 3 effective levels for 99% of channels</li>
    </ul>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Weights W</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Well-behaved, near-normal distribution</li>
      <li>Per-channel INT8 works fine</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">After SmoothQuant — quantizable</div>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Smoothed Activations X·diag(s)⁻¹</div>
    <ul style="margin:0 0 10px 0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier channels divided by $s_i \approx 10$</li>
      <li>Max reduced from 150 → ~15</li>
      <li>Per-tensor INT8 scale ~0.12 → many effective levels</li>
    </ul>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Scaled Weights diag(s)·W^T</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier columns multiplied by $s_i$</li>
      <li>Harder to quantize, but static — per-channel scales still handle it</li>
    </ul>
  </div>
</div>


The scales $s_i$ are computed once from calibration data and fused into the weight matrix at export. At inference, the division by $s_i$ is folded into the preceding LayerNorm — zero runtime overhead.


### Alpha Grid Search: Finding the Best α Per Layer {#alpha-grid-search}


*Why is one global α value not enough, and how do you find the optimal per-layer split without exhaustive evaluation?*


The migration factor $\alpha$ is not one global value — different layers have different activation-vs-weight outlier ratios. A global $\alpha = 0.5$ is a reasonable starting point but leaves 0.1–0.3 PPL on the table. The standard practice is a **per-layer grid search** that tries each $\alpha$ candidate, measures the W8A8 quantization error on calibration inputs, and retains the $\alpha$ that minimises it.


<!-- SmoothQuant Alpha Grid Search Pseudocode -->
<div style="background:#f8f9fa;border:1px solid #d0d7de;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:14px;">SmoothQuant Alpha Grid Search — Pseudocode</div>
  <pre style="background:#161b22;color:#c9d1d9;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#79c0ff;font-weight:600;">Input:</span>  model       — pretrained LLM (nn.Linear layers)
        calib_data  — N calibration sequences
        α_grid      — [0.0, 0.1, …, 1.0]
<span style="color:#79c0ff;font-weight:600;">Output:</span> model with SmoothQuant scales absorbed into weights


<span style="color:#8b949e;"># Step 1: collect per-channel stats via forward hooks</span>
<span style="color:#f0883e;">1.</span> <span style="color:#ff7b72;font-weight:600;">for</span> each nn.Linear layer in model:
<span style="color:#f0883e;">2.</span>     hook: accumulate running max|x_i| across all calib sequences
<span style="color:#f0883e;">3.</span>     also record max|w_i| from weight matrix (static, no pass needed)


<span style="color:#f0883e;">4.</span> run forward pass over calib_data to fire all hooks → get act_max[layer]


<span style="color:#8b949e;"># Step 2: per-layer alpha search</span>
<span style="color:#f0883e;">5.</span> <span style="color:#ff7b72;font-weight:600;">for</span> each layer:
<span style="color:#f0883e;">6.</span>     best_α, best_err = 0.5, ∞
<span style="color:#f0883e;">7.</span>     <span style="color:#ff7b72;font-weight:600;">for</span> α in α_grid:
<span style="color:#f0883e;">8.</span>         s[i] = act_max[i]^α / w_max[i]^(1−α)   clamped to [1e-4, 1e4]
<span style="color:#f0883e;">9.</span>         X_smooth = X_rep / s                    <span style="color:#8b949e;"># tame activation outliers</span>
<span style="color:#f0883e;">10.</span>        W_scaled = W * s                        <span style="color:#8b949e;"># push difficulty to weights</span>
<span style="color:#f0883e;">11.</span>        X_q = fake_quant_int8_per_tensor(X_smooth)
<span style="color:#f0883e;">12.</span>        W_q = fake_quant_int8_per_channel(W_scaled)
<span style="color:#f0883e;">13.</span>        err = ‖ X_rep @ W.T − (X_q*s) @ (W_q/s).T ‖²
<span style="color:#f0883e;">14.</span>        <span style="color:#ff7b72;font-weight:600;">if</span> err < best_err: best_α = α, best_err = err


<span style="color:#8b949e;"># Step 3: absorb best scale into model weights permanently</span>
<span style="color:#f0883e;">15.</span> <span style="color:#ff7b72;font-weight:600;">for</span> each layer:
<span style="color:#f0883e;">16.</span>     s = compute scale with best_α
<span style="color:#f0883e;">17.</span>     W_new = W * diag(s)                         <span style="color:#8b949e;"># fused into weight matrix</span>
<span style="color:#f0883e;">18.</span>     store s as buffer → at inference: x / s before GEMM</pre>
  <div style="background:#f0f6ff;border:1px solid #d0d7de;border-radius:6px;padding:10px 14px;font-size:11px;color:#1f2328;margin-top:12px;line-height:1.7;">
    Line 8 is the only expensive step: 11 α candidates × number of layers, each requiring one fake-quant pass and an MSE. The rest is bookkeeping. The scale is absorbed at export (line 17) — inference overhead is just a vector divide before each GEMM.
  </div>
</div>


The grid search output tells you which layers need aggressive migration (α close to 1.0) and which are already well-behaved (α near 0.0). In most LLMs, `o_proj` and `up_proj` layers in the middle blocks push α toward 0.6–0.8, while embedding-adjacent layers prefer α ≈ 0.3.


> **α tuning range.** If W8A8 with uniform α=0.5 still shows activation NaN in early layers, check which layers return α > 0.8 from the grid search — those are the outlier-dominated layers. A per-layer α map typically recovers 0.1–0.2 PPL vs. a fixed global α.
{: .prompt-tip }


> **Scale overflow.** When a weight column has near-zero max (dead neuron), the migration scale explodes. Always clamp `s` to a finite range — the linked implementation uses `[1e-4, 1e4]`, which is mandatory to prevent FP16 `inf` in the weight matrix.
{: .prompt-danger }


---


## 3. Beyond W8A8: Why SmoothQuant Breaks at INT4 and How SpinQuant Fixes It {#spinquant}


SmoothQuant makes W8A8 tractable. Pushing further to **W4A8** (4-bit weights, 8-bit activations) or **W4A4** (both 4-bit) exposes three compounding failures in the scaling approach. SpinQuant (Meta, ICLR 2025) fixes all three with a fundamentally different primitive: **orthogonal rotation** instead of per-channel scaling.


### Why SmoothQuant Fails at INT4 {#int4-wall}


<div style="background:#fffffc;border:1px solid #fca5a5;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#7f1d1d;font-size:13px;margin-bottom:16px;">Three reasons the scaling trade-off collapses at 15 levels (INT4)</div>
  <div style="display:flex;flex-direction:column;gap:12px;">


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#fca5a5;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">1</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">The scale costs too much</div>
        <div style="color:#64748b;font-size:12px;line-height:1.7;">Scaling by $s_i = 10$ migrates outliers by expanding the weight range 10×. At INT8 (255 levels) you lose ~3.3 bits — painful but survivable. At INT4 you only have 4 bits total; losing 3.3 bits leaves fewer than 1 effective bit for that column: $\text{effective bits} = b - \log_2(s_i) \approx 4 - 3.3 = 0.7$.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#fca5a5;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">2</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">Scaling cannot change distribution shape</div>
        <div style="color:#64748b;font-size:12px;line-height:1.7;">Per-channel scaling stretches individual axes independently — no dimension ever interacts with another. Kurtosis, which measures how heavy-tailed a distribution is, is scale-invariant: $\kappa(cX) = \kappa(X)$. The outlier channel is still outlier-shaped at any scale. The ratio between the outlier and normal values is fully preserved after scaling.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#fca5a5;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">3</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">Residual outliers are fatal at 15 levels</div>
        <div style="color:#64748b;font-size:12px;line-height:1.7;">SmoothQuant's scale is calibrated on seen data; unseen tokens at inference can exceed the calibrated range, producing residual outliers that go unsmoothed. At INT8, a 2× residual outlier still leaves ~100 usable levels. At INT4, step = range/15 — any residual outlier forces normal values to collapse to 1–2 levels, destroying the information they carried.</div>
      </div>
    </div>


  </div>
</div>


### SpinQuant: Rotate Instead of Scale {#rotation-primitive}


SpinQuant's key insight: an **orthogonal rotation matrix** $R$ can be inserted between any two adjacent linear layers with zero change to the full-precision output — because $R R^T = I$.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · Rotation Identity Trick</div>


$$
Y = W_2 \cdot W_1 \cdot x
  = \underbrace{(W_2 \cdot R)}_{W_2'} \cdot \underbrace{(R^T \cdot W_1)}_{W_1'} \cdot x
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    Full-precision output is identical before and after inserting $R$. At quantization, activations flowing between $W_1$ and $W_2$ are rotation-transformed — their distribution has fundamentally changed. $W_2' = W_2 R$ and $W_1' = R^T W_1$ are pre-computed once at export time; zero runtime overhead.
  </div>
</div>


### What Rotation Actually Does: A Change of Basis {#rotation-basis}


Yes — rotation is exactly a **change of basis**. The vector of activations is the same point in space; you are just choosing a new coordinate system to describe it.


Think of each activation vector $x \in \mathbb{R}^d$ as a point in $d$-dimensional space. The standard basis has one axis per model dimension. In the original basis, one axis (say dimension 37) always points toward large values — that is the outlier channel. Applying $R$ rotates the entire coordinate frame so that no single axis dominates any more. The **point hasn't moved**; only the ruler we use to measure it has rotated.


$$
\tilde{x} = R \cdot x \qquad \text{(same point, new basis)}
$$


The downstream weight matrix also changes ruler accordingly:


$$
\tilde{W} = W \cdot R^T \qquad \text{(absorb rotation at export)}
$$


So $\tilde{W} \cdot \tilde{x} = W R^T \cdot R x = W x$ — the output is identical. The only thing that changed is what the activations *look like* in their new coordinate frame.


#### Rotation matrix: form and properties


A rotation matrix $R \in \mathbb{R}^{d \times d}$ is an **orthogonal matrix** — its columns (and rows) form an orthonormal basis. Formally it satisfies:


$$
R^T R = R R^T = I \qquad \det(R) = +1
$$


Three consequences matter for quantization:


<div style="border:1px solid #e2e8f0;border-radius:8px;font-size:0.9em;margin:16px 0;overflow:hidden;">
  <div style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">
    <div style="font-weight:600;margin-bottom:3px;">Length-preserving &nbsp;— &nbsp;$\lVert Rx \rVert_2 = \lVert x \rVert_2$</div>
    <div style="color:#475569;">Total energy in the activation vector is unchanged; rotation only redistributes it across dimensions.</div>
  </div>
  <div style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">
    <div style="font-weight:600;margin-bottom:3px;">Norm-preserving on weights &nbsp;— &nbsp;$\lVert W R^T \rVert_F = \lVert W \rVert_F$</div>
    <div style="color:#475569;">Absorbing $R$ into the weight matrix does not amplify any weight — the quantization problem does not get harder for $W$.</div>
  </div>
  <div style="padding:10px 14px;">
    <div style="font-weight:600;margin-bottom:3px;">Exactly invertible &nbsp;— &nbsp;$R^{-1} = R^T$</div>
    <div style="color:#475569;">The rotation can always be undone without numerical error; the transformation is lossless.</div>
  </div>
</div>


In practice SpinQuant uses **randomized Hadamard matrices** as $R$. A Hadamard matrix has entries $\pm 1/\sqrt{d}$ arranged so that $H H^T = I$; multiplying by it is equivalent to a fast Walsh–Hadamard transform in $O(d \log d)$ rather than the $O(d^2)$ of a general matrix multiply. The random signs ensure no direction is systematically favored after rotation.


#### Worked example: 8 dimensions, 1 outlier


Take a token's activation vector with 7 normal channels and one outlier:


$$
x = [\,\underbrace{0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2}_{\times 7},\; 12.0\,]
$$


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:16px;">Rotation removes the outlier — step by step</div>


  <!-- Step 1: original -->
  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:700;color:#7f1d1d;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">① Original activations — one axis dominates</div>
    <div>
      <div style="display:flex;gap:6px;margin-bottom:5px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#dc2626;">12.0 ★</div>
      </div>
      <div style="display:flex;gap:6px;align-items:flex-end;height:60px;border-bottom:1px solid #e2e8f0;">
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#60a5fa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:60px;background:#ef4444;border-radius:2px 2px 0 0;"></div>
      </div>
      <div style="display:flex;gap:6px;margin-top:5px;margin-bottom:6px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d0</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d1</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d3</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d4</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d5</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#64748b;">d6</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#dc2626;">d7</div>
      </div>
    </div>
    <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;padding:8px 12px;font-size:11px;color:#7f1d1d;line-height:1.6;">
      INT4 range: [0.2, 12.0] → span = 11.8 → step = 11.8 / 15 = <strong>0.79 per bucket</strong><br>
      All 7 normal values map to <strong>bin 0</strong>; outlier maps to <strong>bin 15</strong>. Only <strong>2 of 16 bins used</strong> — 87% of quantization capacity wasted.
    </div>
  </div>


  <!-- Step 2: what rotation does -->
  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:700;color:#4c1d95;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">② Apply 8×8 Hadamard rotation H₈ — outlier energy divides by √8</div>
    <div style="background:#f5f3ff;border:1px solid #c4b5fd;border-radius:8px;padding:12px 14px;font-size:11px;color:#1e293b;margin-bottom:10px;line-height:1.8;">
      <strong>H₈</strong> = (1/√8) ×
      <span style="font-family:monospace;">
        ⎡ +1 +1 +1 +1 +1 +1 +1 +1 ⎤<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 +1 −1 +1 −1 +1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 −1 −1 +1 +1 −1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 −1 +1 +1 −1 −1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 +1 +1 −1 −1 −1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 +1 −1 −1 +1 −1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 −1 −1 −1 −1 +1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎣ +1 −1 −1 +1 −1 +1 +1 −1 ⎦
      </span>
      &nbsp;&nbsp;← each row mixes all 8 dimensions equally; the last column's sign determines the outlier's contribution
    </div>
    <div style="background:#f8f9fa;border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;font-size:11px;color:#1e293b;margin-bottom:10px;line-height:2.0;font-family:monospace;">
      x̃[0] = (+1.4 + 12.0) / 2.83 = 13.4 / 2.83 = <strong style="color:#7c3aed;">+4.74</strong>  ← normals all +1, outlier +<br>
      x̃[1] = (+0.2 − 12.0) / 2.83 = −11.8 / 2.83 = <strong style="color:#7c3aed;">−4.17</strong>  ← normals ≈ cancel, outlier −<br>
      x̃[2] = (+0.2 − 12.0) / 2.83 = −11.8 / 2.83 = <strong style="color:#7c3aed;">−4.17</strong><br>
      x̃[3] = (−0.2 + 12.0) / 2.83 = +11.8 / 2.83 = <strong style="color:#7c3aed;">+4.17</strong><br>
      x̃[4] = (+0.2 − 12.0) / 2.83 = −11.8 / 2.83 = <strong style="color:#7c3aed;">−4.17</strong><br>
      x̃[5] = (−0.2 + 12.0) / 2.83 = +11.8 / 2.83 = <strong style="color:#7c3aed;">+4.17</strong><br>
      x̃[6] = (−0.2 + 12.0) / 2.83 = +11.8 / 2.83 = <strong style="color:#7c3aed;">+4.17</strong><br>
      x̃[7] = (+0.2 − 12.0) / 2.83 = −11.8 / 2.83 = <strong style="color:#7c3aed;">−4.17</strong>
    </div>
    <div>
      <div style="display:flex;gap:6px;margin-bottom:5px;">
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">+4.74</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7c3aed;">−4.17</div>
      </div>
      <div style="display:flex;gap:6px;align-items:flex-end;height:60px;border-bottom:1px solid #e2e8f0;">
        <div style="width:36px;height:60px;background:#a78bfa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#a78bfa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#a78bfa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#a78bfa;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
      </div>
      <div style="display:flex;gap:6px;margin-top:5px;margin-bottom:4px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d0</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d1</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d3</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d4</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d5</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d6</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#7c3aed;">d7</div>
      </div>
      <div style="font-size:10px;color:#7c3aed;margin-top:2px;">bar height = |value| &nbsp;·&nbsp; dark = positive, light = negative</div>
    </div>
    <div style="background:#f5f3ff;border:1px solid #c4b5fd;border-radius:6px;padding:8px 12px;font-size:11px;color:#4c1d95;line-height:1.6;">
      Each output mixes all 8 inputs. The 7 equal normals largely cancel via the alternating ±1 signs (contributing only ±0.2 per row), while the outlier is scaled by ±1 then divided by √8 ≈ 2.83. New range: [−4.17, 4.74] → step = 8.91 / 15 = <strong>0.59 per bucket</strong> — a <strong>25% reduction</strong> from 0.79.
    </div>
  </div>


  <!-- Step 3: quantization comparison -->
  <div>
    <div style="font-size:11px;font-weight:700;color:#14532d;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">③ INT4 quantization quality — before vs. after rotation</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:12px;">
        <div style="font-weight:700;color:#7f1d1d;font-size:11px;margin-bottom:8px;">Without rotation ✗</div>
        <div style="font-size:11px;color:#1e293b;line-height:1.9;font-family:monospace;">
          range=11.8 &nbsp;step=0.79<br>
          0.2 ×7 → bin &nbsp;0 → 0.20 (err≈0)<br>
          12.0 &nbsp; → bin 15 → 12.0 &nbsp;(err≈0)
        </div>
        <div style="margin-top:8px;font-size:11px;color:#7f1d1d;font-weight:600;">Only 2 of 16 bins occupied.<br>Any activation in [0.2, 0.99) is indistinguishable — fine-grained variation invisible to the quantizer.</div>
      </div>
      <div style="flex:1;min-width:200px;background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:12px;">
        <div style="font-weight:700;color:#14532d;font-size:11px;margin-bottom:8px;">After rotation ✓</div>
        <div style="font-size:11px;color:#1e293b;line-height:1.9;font-family:monospace;">
          range=8.91 &nbsp;step=0.59<br>
          +4.74 → bin 15 → +4.74 (err=0.00)<br>
          +4.17 → bin 14 → +4.15 (err=0.02)<br>
          −4.17 → bin &nbsp;0 → −4.17 (err=0.00)
        </div>
        <div style="margin-top:8px;font-size:11px;color:#14532d;font-weight:600;">Step 25% smaller. All 8 rotated values are distinguishable. No bins wasted.</div>
      </div>
    </div>
    <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:6px;padding:10px 12px;font-size:11px;color:#14532d;line-height:1.7;margin-top:10px;">
      <strong>Why this works.</strong> The outlier's energy (12.0) is divided by √8 ≈ 2.83 — no output exceeds 4.74. The 7 equal normals mostly cancel across Hadamard rows, so they don't widen the range. The dominant spike at 12.0 is replaced by 8 balanced values near ±4.17, and step size falls 25%.
    </div>
  </div>


</div>


> **The benefit scales with dimension.** At $d = 8$ the outlier (12.0) is divided by $\sqrt{8} \approx 2.83$, giving a 25% range reduction. At model scale with $d = 4096$ ($\sqrt{d} = 64$), the same outlier contributes only $12.0 / 64 \approx 0.19$ to each output — matching the normal channels. That is a **30× range reduction**: step size shrinks proportionally, and post-rotation kurtosis drops from $\kappa > 200$ to $\kappa \approx 3$ across all layers.
{: .prompt-info }


Note the key difference from scaling. Scaling the outlier dimension (dim 7) by $s = 10$ would give $x_7 = 1.2$ — the range becomes $[0.2, 1.2]$, step $= 0.067$. The normal values are now resolved, but the weight column for dimension 7 is $10\times$ larger — the difficulty moved, it did not disappear. Rotation spreads the energy without amplifying any weight column: $\|W R^T\|_F = \|W\|_F$ always.


> **One hard constraint.** Rotation can only be absorbed through **linear** layers. A nonlinearity (softmax, SiLU) breaks the cancellation — and produces a completely new activation distribution from scratch. This is why SpinQuant uses four separate matrices: one per quantization danger zone, each separated by a nonlinearity boundary.
{: .prompt-info }


### The Four Rotation Matrices {#four-rotations}


Each of R₁–R₄ is placed at a specific "outlier birthplace" in the Transformer — right before a quantization step, positioned so it can be absorbed into adjacent linear weights. Nonlinearities act as hard boundaries: a rotation applied before softmax cannot be carried through it.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:14px;">Transformer block — where each rotation matrix lives</div>
  <pre style="background:#0d1117;color:#c9d1d9;border-radius:8px;padding:18px 20px;font-size:12px;line-height:1.9;overflow-x:auto;margin:0;">  Residual stream (x)
          │
          │  ◀── <span style="color:#60a5fa;font-weight:700;">R₁</span>  absorbed into W_Q, W_K, W_V, W_Up, W_Gate
          │      <span style="color:#60a5fa;">zero runtime cost · one matrix for the whole model</span>
          ▼
  ┌──────────────────────── Attention ─────────────────────┐
  │                                                        │
  │  Q·<span style="color:#60a5fa;">R₁ᵀ</span>·W_Q    K·<span style="color:#60a5fa;">R₁ᵀ</span>·W_K    V·<span style="color:#60a5fa;">R₁ᵀ</span>·W_V·<span style="color:#a78bfa;">R₂</span>                │
  │                   │                  │                 │
  │                   └──── KV Cache ────┘                 │
  │                              │  ◀── <span style="color:#4ade80;font-weight:700;">R₃</span> Hadamard online │
  │      softmax(Q·Kᵀ/√d)  ◀─── <span style="color:#ef4444;font-weight:700;">NONLINEARITY</span>               │
  │               │                                        │
  │          × V aggregation                               │
  │               │  ◀── <span style="color:#a78bfa;font-weight:700;">R₂</span>  absorbed into W_V and W_O     │
  │               ▼                                        │
  │          W_O (output proj)                             │
  └────────────────────────────────────────────────────────┘
          │
      + residual
          │
  ┌─────────────────────────── FFN ────────────────────────┐
  │                                                        │
  │    W_Up·<span style="color:#60a5fa;">R₁ᵀ</span> (x)              W_Gate·<span style="color:#60a5fa;">R₁ᵀ</span> (x)          │
  │          │                          │                  │
  │     SiLU(·) × Gate  ◀──── <span style="color:#ef4444;font-weight:700;">NONLINEARITY</span>                │
  │                 │                                      │
  │                 │  ◀── <span style="color:#fb923c;font-weight:700;">R₄</span>  Hadamard online             │
  │                 │      <span style="color:#fb923c;">R₄ᵀ absorbed into W_Down</span>        │
  │                 ▼                                      │
  │            W_Down (down proj)                          │
  └────────────────────────────────────────────────────────┘
          │
      + residual → next block</pre>
</div>


<div style="display:flex;flex-wrap:wrap;gap:12px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #93c5fd;border-top:3px solid #60a5fa;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#1e3a8a;margin-bottom:6px;font-size:13px;">R₁ — Residual stream</div>
    <div style="color:#64748b;line-height:1.7;"><strong>Target:</strong> Persistent outlier channels that accumulate in <code>x</code> across many layers via residual addition.<br><strong>Absorbed into:</strong> W_Q, W_K, W_V, W_Up, W_Gate at export time.<br><strong>Runtime cost:</strong> None.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #c4b5fd;border-top:3px solid #a78bfa;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#4c1d95;margin-bottom:6px;font-size:13px;">R₂ — V / Output projection</div>
    <div style="color:#64748b;line-height:1.7;"><strong>Target:</strong> New outliers created by softmax attention weighting — not present in the residual stream and not addressable by R₁.<br><strong>Absorbed into:</strong> W_V and W_O per attention layer.<br><strong>Runtime cost:</strong> None.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #86efac;border-top:3px solid #4ade80;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#14532d;margin-bottom:6px;font-size:13px;">R₃ — KV cache (online)</div>
    <div style="color:#64748b;line-height:1.7;"><strong>Target:</strong> Outliers in K and V tensors when they are quantized and stored in the cache.<br><strong>Why online:</strong> K and V are dynamic — they depend on the current token and cannot be precomputed. Q absorbs Hᵀ offline at no cost.<br><strong>Runtime cost:</strong> O(d log d) Hadamard per token.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #fdba74;border-top:3px solid #fb923c;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#7c2d12;margin-bottom:6px;font-size:13px;">R₄ — MLP intermediate (online)</div>
    <div style="color:#64748b;line-height:1.7;"><strong>Target:</strong> Outliers produced by the SiLU nonlinearity inside the FFN — not present at Up/Gate input, so R₁ cannot help.<br><strong>Why online:</strong> Applied to SiLU output which is dynamic. R₄ᵀ is absorbed into W_Down offline.<br><strong>Runtime cost:</strong> O(d log d) Hadamard per token.</div>
  </div>


</div>


> **SpinQuant_no_had** uses only R₁ and R₂ — both absorbed offline, zero runtime overhead — and is well-suited for W4A8. **SpinQuant_had** adds the online R₃ and R₄ Hadamards at ~8% latency overhead and is required for the harder W4A4 setting.
{: .prompt-tip }


### Training the Rotations {#training-rotations}


Random rotations already help — but vary wildly. Testing 100 random seeds on LLaMA-2 7B W4A4 shows a **13-point accuracy spread** between the best and worst rotation. SpinQuant learns the optimal rotation instead of guessing.


The training constraint is that $R$ must stay **exactly orthogonal** ($R^T R = I$) at every update step. Ordinary gradient descent would drift off this constraint, breaking the identity trick and changing the full-precision model output. SpinQuant solves this with the **Cayley transform** — a special update rule that mathematically guarantees every step lands back on the manifold of orthogonal matrices.


<div style="background:#f8f9fa;border:1px solid #d0d7de;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:14px;">SpinQuant Rotation Training — Pseudocode</div>
  <pre style="background:#161b22;color:#c9d1d9;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#79c0ff;font-weight:600;">Input:</span>  model       — pretrained LLM, weights <strong>frozen</strong> at 16-bit
        calib_data  — 800 samples from WikiText-2
<span style="color:#79c0ff;font-weight:600;">Output:</span> R₁, R₂     — learned rotations (absorbed into weights after training)


<span style="color:#8b949e;"># Init with random Hadamard matrices (better starting point than pure random)</span>
<span style="color:#f0883e;"> 1.</span>  R₁, R₂ ← random_hadamard()


<span style="color:#f0883e;"> 2.</span>  <span style="color:#ff7b72;font-weight:600;">for</span> step in 1..100:
<span style="color:#f0883e;"> 3.</span>      x = sample(calib_data)
<span style="color:#f0883e;"> 4.</span>      ŷ = forward(model, x, R₁, R₂)   <span style="color:#8b949e;"># activations quantized; weights stay at 16-bit</span>
<span style="color:#f0883e;"> 5.</span>      loss = cross_entropy(ŷ, x)       <span style="color:#8b949e;"># STE allows gradient to pass through rounding</span>
<span style="color:#f0883e;"> 6.</span>      G = ∂loss / ∂R


<span style="color:#8b949e;">      # Cayley step: R is guaranteed to stay orthogonal after every update</span>
<span style="color:#f0883e;"> 7.</span>      Ĝ = G·Rᵀ − ½·R·(G·Rᵀ)ᵀ          <span style="color:#8b949e;"># project gradient onto tangent space of R</span>
<span style="color:#f0883e;"> 8.</span>      Y = Ĝ − Ĝᵀ                       <span style="color:#8b949e;"># skew-symmetrize → input for Cayley map</span>
<span style="color:#f0883e;"> 9.</span>      R ← (I − α/2·Y)⁻¹ · (I + α/2·Y) · R   <span style="color:#8b949e;"># RᵀR = I still holds ✓</span>


<span style="color:#8b949e;"># Absorb rotations into weight matrices (one-time cost at export)</span>
<span style="color:#f0883e;">10.</span>  W_Q  ← W_Q · R₁ᵀ  ;  W_K ← W_K · R₁ᵀ  ;  W_V ← W_V · R₁ᵀ · R₂
<span style="color:#f0883e;">11.</span>  W_Up ← W_Up · R₁ᵀ ;  W_Gate ← W_Gate · R₁ᵀ  ;  W_O ← R₂ᵀ · W_O
<span style="color:#f0883e;">12.</span>  <span style="color:#8b949e;"># R₄ᵀ absorbed into W_Down; R₃ stays as online Hadamard at inference</span>


<span style="color:#8b949e;"># Then run GPTQ on the rotation-transformed weights to handle weight quantization</span></pre>
  <div style="background:#f0f6ff;border:1px solid #d0d7de;border-radius:6px;padding:10px 14px;font-size:11px;color:#1f2328;margin-top:12px;line-height:1.7;">
    R₁ and R₂ together represent only ~0.26% of total model parameters. Training runs for 100 steps on 800 unlabeled samples: ~13 minutes for a 1B model, ~3.5 hours for 70B. Weights are completely frozen throughout — only the rotation matrices update. After absorption, inference is identical to normal; no rotation occurs at runtime for R₁ and R₂.
  </div>
</div>


---


## Putting It All Together


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="display:flex;flex-direction:column;gap:12px;">


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">1</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Activation outliers are the root cause — everything else is a consequence</div>
        <div style="color:#64748b;font-size:12px;">A persistent 1% of activation channels carry values 100× larger than the rest. This forces the per-tensor scale so coarse that the remaining 99% of channels collapse to 2–3 effective levels. Per-channel quantization cannot fix this: activation channel maxima are input-dependent and cannot be pre-computed, so any per-channel scale would require an expensive online reduce kernel before every GEMM — defeating the INT8 throughput gain entirely.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">2</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">SmoothQuant unlocks W8A8 by migrating difficulty from activations to weights</div>
        <div style="color:#64748b;font-size:12px;">The key identity is $Y = X W^T = (X \cdot s^{-1})(s \cdot W^T)$. Dividing the outlier activation channels by $s_i$ tames them for per-tensor INT8; multiplying the corresponding weight columns by $s_i$ absorbs that difficulty — but weights are static, so per-channel scales still handle it. The migration factor $\alpha$ controls how much difficulty moves to which side.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">3</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Per-layer α search recovers 0.1–0.2 PPL vs. a global α=0.5</div>
        <div style="color:#64748b;font-size:12px;">Different layers have different activation-vs-weight outlier ratios. A global α=0.5 is a reasonable starting point, but a per-layer grid search that minimises W8A8 reconstruction error on calibration inputs reliably closes the remaining gap. In most LLMs, middle-block projection layers push toward α≈0.7 while embedding-adjacent layers prefer α≈0.3.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#c4b5fd;color:#4c1d95;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">4</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">SpinQuant replaces scaling with rotation to reach W4A4</div>
        <div style="color:#64748b;font-size:12px;">SmoothQuant's diagonal scaling migrates difficulty without removing it — at INT4 this trade-off costs more bits than the format has. Orthogonal rotation mixes all dimensions, spreading outlier energy across the full embedding space and genuinely reshaping the distribution. The four rotation matrices (R₁–R₄) are each placed at a separate nonlinearity boundary where a new outlier zone begins; R₁ and R₂ are absorbed offline at zero runtime cost, R₃ and R₄ run as fast Hadamard transforms online. Optimizing rotations via Cayley SGD on 800 calibration samples takes under 30 minutes and closes the W4A4 accuracy gap from 27 points (SmoothQuant) to under 3 points.</div>
      </div>
    </div>


  </div>
</div>


> **Previous:** [Part 2 — GPTQ, AWQ, and the Calibration Trap](/posts/quantization-2/) covers weight-only W4A16 quantization and why domain-matched calibration data often matters more than algorithm choice.
{: .prompt-info }


---


## References


<div style="font-family:system-ui,sans-serif;font-size:13px;line-height:1.8;">


<p><strong>[1]</strong> Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han.<br>
<em>SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.</em><br>
ICML 2023. <a href="https://arxiv.org/abs/2211.10438">arXiv:2211.10438</a></p>


<p><strong>[2]</strong> Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort.<br>
<em>SpinQuant: LLM Quantization with Learned Rotations.</em><br>
arXiv 2024. <a href="https://arxiv.org/abs/2405.16406">arXiv:2405.16406</a></p>


<p><strong>[3]</strong> Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer.<br>
<em>LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.</em><br>
NeurIPS 2022. <a href="https://arxiv.org/abs/2208.07339">arXiv:2208.07339</a></p>


<p><strong>[4]</strong> Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman.<br>
<em>QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.</em><br>
arXiv 2024. <a href="https://arxiv.org/abs/2404.00456">arXiv:2404.00456</a></p>


</div>

