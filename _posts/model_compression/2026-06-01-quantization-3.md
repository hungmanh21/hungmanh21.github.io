---
layout: post
title: "Below 8 bit on LLMs: Weight and Activation Outliers"
date: 2026-06-01 10:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, ptq, smoothquant, spinquant, calibration, int4, int8, w8a8, w4a4, llm]
math: true
description: "Why per-tensor INT8 collapses on LLM activations, how SmoothQuant migrates the difficulty to weights via α, and why W4A4 needs rotations instead of scales."
---


*A scale moves the problem; a rotation dissolves it.*


<!-- Opening block: questions + prerequisites — stacked vertically -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Why does per-tensor INT8 collapse on LLM activations when it works fine for vision models?</li>
      <li>Why can't per-channel quantization — which fixes outlier weights — also fix outlier activations?</li>
      <li>How does SmoothQuant's α migrate difficulty from activations to weights, and how do we pick it per layer?</li>
      <li>Why does the same scaling trick fall apart at INT4, and what does an orthogonal rotation do differently?</li>
      <li>Where in a Transformer block does each of SpinQuant's four rotation matrices belong, and which run online vs. offline?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #c4b5fd;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Quantization basics: scale, zero-point, per-tensor vs. per-channel (covered in <a href="/posts/quantization-1/">Part 1</a>)</li>
      <li>Weight-only PTQ: GPTQ, AWQ, calibration data (covered in <a href="/posts/quantization-2/">Part 2</a>)</li>
      <li>Linear algebra: matrix multiplication identity $Y = XW^T$, orthogonal matrices</li>
      <li>Transformer architecture (attention projections, FFN, residual stream)</li>
    </ul>
  </div>
</div>


GPTQ and AWQ both sidestep the activation problem entirely — they are weight-only (W4A16). But INT8 GEMM on Turing+ tensor cores is 2× the throughput of BF16, and we only get it if we quantize activations too. That single requirement — touching the dynamic side of the matmul — is what makes W8A8 hard, and what eventually pushes us past scaling altogether. Let's chase the failure modes one by one.


---


## 1. Why does INT8 collapse on activations?


*Per-tensor INT8 works fine for CNNs and small encoders — so what changes when we point it at an LLM?*


<div style="background:#fffffc;border:1px solid #fca5a5;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#7f1d1d;font-size:13px;margin-bottom:18px;">INT8 grid: ideal channel vs. one outlier channel</div>


  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:600;color:#14532d;margin-bottom:6px;">① Ideal channel — max ≈ 1.5, no outlier · 256 levels span the full range</div>
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
      <span style="font-weight:600;">step ≈ 0.012 · fine resolution everywhere</span>
      <span>+1.5</span>
    </div>
  </div>


  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:600;color:#7f1d1d;margin-bottom:6px;">② One outlier channel hits 150 · scale jumps to 1.18 · same 256 levels now span 100× wider</div>
    <div style="position:relative;height:22px;background:#ffadad;border-radius:4px;overflow:hidden;">
      <div style="position:absolute;left:0%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:6.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:13.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:20%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:26.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:33.33%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:40%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:46.67%;top:0;width:2px;height:100%;background:#7f1d1d;opacity:.4;"></div>
      <div style="position:absolute;left:49.5%;top:0;width:1%;height:100%;background:#bdb2ff;"></div>
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
      <span style="font-weight:600;">◀ 99% of values crammed into this 1% band ▶</span>
      <span style="font-weight:700;">+150 (outlier)</span>
    </div>
  </div>


  <div>
    <div style="font-size:11px;font-weight:600;color:#78350f;margin-bottom:6px;">③ Zoom on the non-outlier band [−1.5, +1.5] — only 3 INT8 levels land here</div>
    <div style="position:relative;height:22px;background:#ffd6a5;border-radius:4px;overflow:hidden;">
      <div style="position:absolute;left:10.7%;top:0;width:2px;height:100%;background:#78350f;"></div>
      <div style="position:absolute;left:50%;top:0;width:2px;height:100%;background:#78350f;"></div>
      <div style="position:absolute;left:89.3%;top:0;width:2px;height:100%;background:#78350f;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#78350f;margin-top:3px;padding:0 2px;">
      <span>−1.5</span>
      <span style="font-weight:700;">k_eff = 3 — INT8 storage, 2-bit effective resolution for 99% of channels</span>
      <span>+1.5</span>
    </div>
  </div>
</div>


The three rows tell one story. In the green track everything is fine — fifteen tick marks across $[-1.5, 1.5]$, step size $\approx 0.012$, every value resolved. Drop one outlier channel at 150 into the same tensor and the scale becomes $150/127 \approx 1.18$. The exact same 256 INT8 levels now have to cover $\pm 150$, so when we zoom into where 99% of the channels actually live (the orange track), only **three** ticks remain. The format on disk is INT8; the resolution the model sees is closer to two bits. We are paying for 8-bit storage and getting the accuracy of a 2-bit quantizer.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · effective levels under per-tensor scale</div>


$$
k_\text{eff} = \left\lfloor \frac{v}{\text{scale}} \right\rfloor \times 2 + 1, \quad \text{scale} = \frac{\max(|x|)}{2^{b-1}-1}
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $v$ — non-outlier channel max (e.g. 1.5) · $\max(|x|)$ — outlier channel max (e.g. 150) · $b=8$ for INT8.<br>
    With $v=1.5$, $\max=150$: $k_\text{eff} = \lfloor 1.5/1.18 \rfloor \times 2 + 1 = \mathbf{3}$ levels.
  </div>
</div>


So the natural reflex is: if per-channel scales rescued weight quantization in [Part 1](/posts/quantization-1/), why don't they rescue activations the same way? The answer is *when* the values come into existence.


### Why per-channel cannot rescue activations


*Weights are static — we measure their per-channel max once, on disk. What goes wrong when we try the same recipe on activations?*


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Weights — per-channel works ✓</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.9;">
      <li>Values are <strong>static</strong> — fixed at export</li>
      <li>Channel maxima computed once on disk</li>
      <li>Scale tensor fused into the model file</li>
      <li>Zero runtime cost, no input dependency</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffadad;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#7f1d1d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Activations — per-channel cannot work ✗</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.9;">
      <li>Values are <strong>dynamic</strong> — produced from input tokens</li>
      <li>Channel max changes every request — cannot be pre-computed</li>
      <li>A "live" per-channel scale would need a reduce kernel before every GEMM</li>
      <li>That kernel kills the INT8 throughput gain we were after</li>
    </ul>
  </div>
</div>


The two cards split along one axis: when do we know the values? Weights live on disk, so we have unlimited compute budget at export — per-channel, per-group, anything we like. Activations are born inside the forward pass, so any per-channel scale would have to be measured *online*, on the same hot path the GEMM runs on. That extra reduction kernel before every matmul is exactly the latency we were trying to recover by going to INT8 in the first place. Per-group is the same story at finer granularity — it still needs a static group max, which activations cannot provide.


> **Root cause.** Activation outliers are persistent: the same ~1% of channels are outliers across all tokens and all calibration samples in a given model. That stability is what makes the next section's trick possible — we just have to pay attention to which side of the matmul we ask to absorb the difficulty.
{: .prompt-info }


That gives us a foothold. The outlier channels are stable across inputs, so we *do* know which channels are dangerous offline — we just can't rescale activations directly without paying a runtime kernel. Can we rearrange the GEMM so the scale ends up on the static side?


---


## 2. How does SmoothQuant migrate the difficulty?


*If activations are the dynamic side and weights are the static side, can we trade some activation difficulty for some weight difficulty before either is quantized?*


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · SmoothQuant migration identity</div>


$$
Y = X W^T = \underbrace{\bigl(X \cdot \operatorname{diag}(s)^{-1}\bigr)}_{\text{smoothed activations}} \cdot \underbrace{\bigl(\operatorname{diag}(s) \cdot W^T\bigr)}_{\text{scaled weights}}
$$


$$
s_i = \frac{\max|x_i|^\alpha}{\max|w_i|^{1-\alpha}}, \quad \alpha \in [0, 1]
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $s_i$ — per-channel migration scale · $x_i$ — activation channel $i$ · $w_i$ — weight column $i$.<br>
    $\alpha=0$: no migration (per-tensor activations stay broken) · $\alpha=0.5$: balanced split · $\alpha=1$: all difficulty pushed to weights.
  </div>
</div>


The identity is the whole game. We multiply by $\operatorname{diag}(s)^{-1} \cdot \operatorname{diag}(s) = I$ and rebrace — full-precision output unchanged. But now the activation outlier channel has been divided by some $s_i \approx 10$, and the corresponding weight column has been multiplied by the same $s_i$. The activation tensor is suddenly quantizable per-tensor; the weight tensor is harder, but it's static, so per-channel scales handle it fine.


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffadad;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#7f1d1d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Before — broken at INT8</div>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Activations $X$</div>
    <ul style="margin:0 0 10px 0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier channels: max ≈ 150</li>
      <li>Normal channels: max ≈ 1.5</li>
      <li>Per-tensor INT8 → 3 effective levels</li>
    </ul>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Weights $W$</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Well-behaved, near-normal</li>
      <li>Per-channel INT8 already works</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">After — both sides quantizable</div>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Smoothed $X \cdot \operatorname{diag}(s)^{-1}$</div>
    <ul style="margin:0 0 10px 0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier channels divided by $s_i \approx 10$</li>
      <li>Max drops 150 → ~15</li>
      <li>Per-tensor INT8 now resolves the bulk</li>
    </ul>
    <div style="margin-bottom:8px;font-weight:600;font-size:12px;color:#1e293b;">Scaled $\operatorname{diag}(s) \cdot W^T$</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Outlier columns multiplied by $s_i$</li>
      <li>Static — per-channel scales absorb it</li>
    </ul>
  </div>
</div>


The cost is paid where we can afford it. The smoothed activation max drops by an order of magnitude, recovering most of the lost INT8 resolution; the weight outlier columns grow, but each weight column has its own scale anyway, so a 10× column just means a 10× scale for that column and nothing else. And the division by $s$ at runtime fuses into the preceding LayerNorm — no extra kernel, no latency tax.


> **Where the scale lives at inference.** $s$ is computed once at export from calibration data. The activation-side division is folded into the preceding LayerNorm's $\gamma$ and $\beta$; the weight-side multiplication is fused into $W$ on disk. The runtime forward pass looks identical to a plain INT8 GEMM.
{: .prompt-tip }


That handles a single layer. The next subtlety is that one global $\alpha$ is wrong — different layers have wildly different outlier ratios.


### Picking α per layer


*Why does $\alpha = 0.5$ leave accuracy on the table, and how do we find the right value without exhaustive evaluation?*


The migration factor $\alpha$ controls *how much* difficulty moves. Different layers have different activation-vs-weight outlier ratios, so a single global $\alpha$ is a compromise that fits no layer perfectly. The standard fix is a per-layer grid search: try eleven $\alpha$ candidates, measure the W8A8 reconstruction error on calibration inputs, keep the one that wins.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:14px;">SmoothQuant α grid search — pseudocode</div>
  <pre style="background:#1e293b;color:#caffbf;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#a0c4ff;font-weight:600;">Input:</span>  model       — pretrained LLM (nn.Linear layers)
        calib_data  — N calibration sequences
        α_grid      — [0.0, 0.1, …, 1.0]
<span style="color:#a0c4ff;font-weight:600;">Output:</span> model with SmoothQuant scales absorbed into weights


<span style="color:#9bf6ff;"># Step 1: collect per-channel stats via forward hooks</span>
<span style="color:#ffd6a5;">1.</span> <span style="color:#ffadad;font-weight:600;">for</span> each nn.Linear in model:
<span style="color:#ffd6a5;">2.</span>     hook: accumulate running max|x_i| across all calib sequences
<span style="color:#ffd6a5;">3.</span>     also record max|w_i| from weight matrix (static, no pass needed)


<span style="color:#ffd6a5;">4.</span> run forward pass over calib_data → fill act_max[layer]


<span style="color:#9bf6ff;"># Step 2: per-layer α search</span>
<span style="color:#ffd6a5;">5.</span> <span style="color:#ffadad;font-weight:600;">for</span> each layer:
<span style="color:#ffd6a5;">6.</span>     best_α, best_err = 0.5, ∞
<span style="color:#ffd6a5;">7.</span>     <span style="color:#ffadad;font-weight:600;">for</span> α in α_grid:
<span style="color:#ffd6a5;">8.</span>         s[i] = act_max[i]^α / w_max[i]^(1−α)   clamp to [1e-4, 1e4]
<span style="color:#ffd6a5;">9.</span>         X_smooth = X_rep / s
<span style="color:#ffd6a5;">10.</span>        W_scaled = W * s
<span style="color:#ffd6a5;">11.</span>        X_q = fake_quant_int8_per_tensor(X_smooth)
<span style="color:#ffd6a5;">12.</span>        W_q = fake_quant_int8_per_channel(W_scaled)
<span style="color:#ffd6a5;">13.</span>        err = ‖ X_rep @ W.T − (X_q*s) @ (W_q/s).T ‖²
<span style="color:#ffd6a5;">14.</span>        <span style="color:#ffadad;font-weight:600;">if</span> err < best_err: best_α, best_err = α, err


<span style="color:#9bf6ff;"># Step 3: absorb best scale into weights permanently</span>
<span style="color:#ffd6a5;">15.</span> <span style="color:#ffadad;font-weight:600;">for</span> each layer:
<span style="color:#ffd6a5;">16.</span>     s = compute scale with best_α
<span style="color:#ffd6a5;">17.</span>     W_new = W * diag(s)
<span style="color:#ffd6a5;">18.</span>     register s as buffer → at inference: x / s before GEMM</pre>
  <div style="background:#fdffb6;border:1px solid #f59e0b;border-radius:6px;padding:10px 14px;font-size:11px;color:#713f12;margin-top:12px;line-height:1.7;">
    Lines 8–13 are the only expensive step: 11 α candidates × number of layers, each a fake-quant pass plus an MSE. Everything else is bookkeeping. The scale is absorbed at export (line 17) — at inference the only overhead is one vector divide before each GEMM.
  </div>
</div>


The grid search also gives us a free diagnostic. Layers that prefer $\alpha$ near 0.7–0.8 — typically `o_proj` and `up_proj` in the middle blocks — are the outlier-dominated ones. Layers near $\alpha \approx 0.3$ are weight-dominated and barely need migration at all. Reading the per-layer $\alpha$ map is the fastest way to spot which parts of the network are quantization-fragile, and it usually buys 0.1–0.2 PPL over a fixed global $\alpha = 0.5$.


> **Watch the clamp.** When a weight column happens to have near-zero max (a dead neuron), the migration scale explodes — $\max|w_i|^{1-\alpha} \to 0$ pushes $s_i \to \infty$. Always clamp $s$ to a finite range like $[10^{-4}, 10^4]$, otherwise FP16 weight values overflow to `inf` after fusion.
{: .prompt-danger }


That is W8A8 in one section. The migration trick scales until it doesn't — and that wall is exactly where INT4 lives.


---


## 3. Why does scaling fail at INT4, and what does rotation do differently?


*SmoothQuant gives us 256 levels and ten-fold migrations to play with. What happens when we try to repeat the trick at 15 levels?*


<div style="background:#fffffc;border:1px solid #fca5a5;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#7f1d1d;font-size:13px;margin-bottom:16px;">Three reasons the scaling trade-off collapses at 15 levels (INT4)</div>
  <div style="display:flex;flex-direction:column;gap:12px;">


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#ffadad;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">1</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">The scale itself costs too many bits</div>
        <div style="color:#1e293b;font-size:12px;line-height:1.7;">A migration scale of $s_i = 10$ stretches the weight range by $10\times$. At INT8 that costs about $\log_2 10 \approx 3.3$ bits — painful but survivable inside 8 bits. At INT4 the format only has 4 bits to begin with: $4 - 3.3 = 0.7$ effective bits left for that column.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#ffadad;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">2</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">Scaling cannot change distribution shape</div>
        <div style="color:#1e293b;font-size:12px;line-height:1.7;">Per-channel scaling stretches each axis independently — no two dimensions ever interact. Kurtosis is scale-invariant: $\kappa(cX) = \kappa(X)$. The outlier channel is still outlier-shaped at any scale. The ratio between the spike and the bulk is exactly preserved.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#ffadad;color:#7f1d1d;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex-shrink:0;margin-top:2px;">3</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:3px;">Residual outliers are fatal at 15 levels</div>
        <div style="color:#1e293b;font-size:12px;line-height:1.7;">$s$ is calibrated on seen data; a fresh input at inference can produce a 2× residual outlier. INT8 absorbs this — there are still ~100 usable levels left. INT4 has $\text{step} = \text{range}/15$, so any residual outlier crushes the bulk back into 1–2 buckets.</div>
      </div>
    </div>


  </div>
</div>


Reading down the three rows, the trade-off curve flips sign between INT8 and INT4. At eight bits, every scaling mistake is forgivable — there is plenty of grid headroom. At four bits, the scale itself competes with the data for the bits that exist. The honest conclusion is structural: **scaling moves the problem along an axis, and we need something that genuinely reshapes the distribution instead.**


### The rotation identity


*If diagonal scaling can't change distribution shape, what kind of matrix can?*


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · rotation identity</div>


$$
Y = W_2 \cdot W_1 \cdot x
  = \underbrace{(W_2 \cdot R)}_{W_2'} \cdot \underbrace{(R^T \cdot W_1)}_{W_1'} \cdot x
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $R$ is orthogonal: $R^T R = I$. We slip $R R^T = I$ between $W_2$ and $W_1$ and rebrace — full-precision output is unchanged. At quantization, the activations flowing between the two layers are now in a rotated frame; their distribution has been completely reshaped. $W_2'$ and $W_1'$ are pre-computed once at export — zero runtime overhead.
  </div>
</div>


The identity is structurally similar to SmoothQuant's, with one critical difference. SmoothQuant uses $\operatorname{diag}(s) \cdot \operatorname{diag}(s)^{-1} = I$: a *diagonal* matrix, so each axis is rescaled independently and shape is preserved. SpinQuant uses $R \cdot R^T = I$ for an *orthogonal* $R$, which mixes all dimensions. The activation vector lands at the same point in space, but the axes we use to describe it have been rotated — and dimensions where every axis contributes equally have no outliers to speak of.


### What rotation does, geometrically


*Why does mixing all dimensions automatically dilute an outlier?*


Think of an activation $x \in \mathbb{R}^d$ as a point in $d$-space. In the original basis, dimension 37 always points toward huge values — that is the outlier channel. Applying $R$ rotates the entire coordinate frame. The point hasn't moved; only the ruler has. Every new axis is a linear combination of all the old ones, so the outlier's energy spreads across the whole frame instead of piling onto one component.


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;font-size:13px;margin:16px 0;overflow:hidden;">
  <div style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">
    <div style="font-weight:600;color:#1e293b;margin-bottom:3px;">Length-preserving — $\lVert Rx \rVert_2 = \lVert x \rVert_2$</div>
    <div style="color:#1e293b;font-size:12px;">Total energy in the activation vector is unchanged; rotation only redistributes it across dimensions.</div>
  </div>
  <div style="padding:10px 14px;border-bottom:1px solid #e2e8f0;">
    <div style="font-weight:600;color:#1e293b;margin-bottom:3px;">Norm-preserving on weights — $\lVert W R^T \rVert_F = \lVert W \rVert_F$</div>
    <div style="color:#1e293b;font-size:12px;">Absorbing $R$ into the weight matrix does not amplify any weight column. Unlike SmoothQuant scaling, the weight side gets no harder.</div>
  </div>
  <div style="padding:10px 14px;">
    <div style="font-weight:600;color:#1e293b;margin-bottom:3px;">Exactly invertible — $R^{-1} = R^T$</div>
    <div style="color:#1e293b;font-size:12px;">The transformation is lossless at full precision; the identity trick is mathematically exact, not approximate.</div>
  </div>
</div>


The middle row is the one that pays the rent. SmoothQuant's $W' = \operatorname{diag}(s) \cdot W$ has columns $s_i \cdot w_i$ — outlier columns grow by $s_i$. SpinQuant's $W' = W \cdot R^T$ keeps the Frobenius norm constant. The difficulty doesn't move sideways — it actually drops, because the energy concentrated on one axis is spread over $d$ axes.


In practice we use **randomized Hadamard matrices** for $R$: entries $\pm 1/\sqrt{d}$ arranged so $H H^T = I$. Multiplication is an $O(d \log d)$ Walsh–Hadamard transform instead of $O(d^2)$ matmul, and the random signs guarantee no axis is systematically favored after rotation.


### Worked example: 8 dimensions, one outlier


*Numbers make the energy-spreading concrete. Let's quantize one activation vector before and after rotation.*


Take a token activation with seven well-behaved channels and one fat outlier:


$$
x = [\,\underbrace{0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2,\; 0.2}_{\times 7},\; 12.0\,]
$$


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:16px;">From spike to balanced — three steps</div>


  <!-- Step 1 -->
  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:700;color:#7f1d1d;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">① Original activations — one axis dominates</div>
    <div>
      <div style="display:flex;gap:6px;margin-bottom:5px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">0.2</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7f1d1d;">12.0 ★</div>
      </div>
      <div style="display:flex;gap:6px;align-items:flex-end;height:60px;border-bottom:1px solid #e2e8f0;">
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:1px;background:#a0c4ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:60px;background:#ffadad;border-radius:2px 2px 0 0;"></div>
      </div>
      <div style="display:flex;gap:6px;margin-top:5px;margin-bottom:6px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d0</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d1</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d3</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d4</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d5</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#1e293b;">d6</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#7f1d1d;">d7</div>
      </div>
    </div>
    <div style="background:#ffadad;border:1px solid #fca5a5;border-radius:6px;padding:8px 12px;font-size:11px;color:#7f1d1d;line-height:1.6;">
      INT4 range $[0.2, 12.0]$ → step $= 11.8/15 = $ <strong>0.79 per bucket</strong>. All 7 normal values land in <strong>bin 0</strong>; the outlier in <strong>bin 15</strong>. Only <strong>2 of 16 bins</strong> used — 87% of the quantization grid wasted.
    </div>
  </div>


  <!-- Step 2 -->
  <div style="margin-bottom:20px;">
    <div style="font-size:11px;font-weight:700;color:#4c1d95;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">② Apply $H_8$ — outlier energy divides by $\sqrt{8}$</div>
    <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:8px;padding:12px 14px;font-size:11px;color:#4c1d95;margin-bottom:10px;line-height:1.8;">
      <strong>$H_8$</strong> = $(1/\sqrt{8})\,\times$
      <span style="font-family:monospace;color:#1e293b;">
        ⎡ +1 +1 +1 +1 +1 +1 +1 +1 ⎤<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 +1 −1 +1 −1 +1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 −1 −1 +1 +1 −1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 −1 +1 +1 −1 −1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 +1 +1 −1 −1 −1 −1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 −1 +1 −1 −1 +1 −1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎢ +1 +1 −1 −1 −1 −1 +1 +1 ⎥<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⎣ +1 −1 −1 +1 −1 +1 +1 −1 ⎦
      </span>
      &nbsp; — every row mixes all 8 dimensions; the last column's sign decides how the outlier contributes.
    </div>
    <div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;font-size:11px;color:#1e293b;margin-bottom:10px;line-height:2.0;font-family:monospace;">
      x̃[0] = (+1.4 + 12.0) / 2.83 = <strong style="color:#4c1d95;">+4.74</strong>  ← normals all +1, outlier +<br>
      x̃[1] = (+0.2 − 12.0) / 2.83 = <strong style="color:#4c1d95;">−4.17</strong>  ← normals ≈ cancel, outlier −<br>
      x̃[2] = (+0.2 − 12.0) / 2.83 = <strong style="color:#4c1d95;">−4.17</strong><br>
      x̃[3] = (−0.2 + 12.0) / 2.83 = <strong style="color:#4c1d95;">+4.17</strong><br>
      x̃[4] = (+0.2 − 12.0) / 2.83 = <strong style="color:#4c1d95;">−4.17</strong><br>
      x̃[5] = (−0.2 + 12.0) / 2.83 = <strong style="color:#4c1d95;">+4.17</strong><br>
      x̃[6] = (−0.2 + 12.0) / 2.83 = <strong style="color:#4c1d95;">+4.17</strong><br>
      x̃[7] = (+0.2 − 12.0) / 2.83 = <strong style="color:#4c1d95;">−4.17</strong>
    </div>
    <div>
      <div style="display:flex;gap:6px;margin-bottom:5px;">
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">+4.74</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">−4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">+4.17</div>
        <div style="width:36px;text-align:center;font-size:10px;font-weight:700;color:#4c1d95;">−4.17</div>
      </div>
      <div style="display:flex;gap:6px;align-items:flex-end;height:60px;border-bottom:1px solid #e2e8f0;">
        <div style="width:36px;height:60px;background:#bdb2ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#bdb2ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#bdb2ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#bdb2ff;border-radius:2px 2px 0 0;"></div>
        <div style="width:36px;height:53px;background:#c4b5fd;border-radius:2px 2px 0 0;"></div>
      </div>
      <div style="display:flex;gap:6px;margin-top:5px;margin-bottom:4px;">
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d0</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d1</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d2</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d3</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d4</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d5</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d6</div>
        <div style="width:36px;text-align:center;font-size:10px;color:#4c1d95;">d7</div>
      </div>
      <div style="font-size:10px;color:#4c1d95;margin-top:2px;">bar height = |value| · dark = positive, light = negative</div>
    </div>
    <div style="background:#bdb2ff;border:1px solid #c4b5fd;border-radius:6px;padding:8px 12px;font-size:11px;color:#4c1d95;line-height:1.6;margin-top:8px;">
      Each output mixes all 8 inputs. The 7 equal normals largely cancel via the alternating ± signs (only $\pm 0.2$ per row), while the outlier is signed by $\pm 1$ then divided by $\sqrt{8} \approx 2.83$. New range $[-4.17, 4.74]$ → step $= 8.91/15 = $ <strong>0.59</strong>, a <strong>25% drop</strong> from 0.79.
    </div>
  </div>


  <!-- Step 3 -->
  <div>
    <div style="font-size:11px;font-weight:700;color:#14532d;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">③ INT4 quantization quality — before vs. after rotation</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;background:#ffadad;border:1px solid #fca5a5;border-radius:8px;padding:12px;">
        <div style="font-weight:700;color:#7f1d1d;font-size:11px;margin-bottom:8px;">Without rotation ✗</div>
        <div style="font-size:11px;color:#1e293b;line-height:1.9;font-family:monospace;">
          range = 11.8 &nbsp;step = 0.79<br>
          0.2 ×7 → bin &nbsp;0 → 0.20 (err≈0)<br>
          12.0 &nbsp; → bin 15 → 12.0 &nbsp;(err≈0)
        </div>
        <div style="margin-top:8px;font-size:11px;color:#7f1d1d;font-weight:600;">Only 2 of 16 bins occupied — anything in $[0.2, 0.99)$ is indistinguishable to the quantizer.</div>
      </div>
      <div style="flex:1;min-width:200px;background:#caffbf;border:1px solid #86efac;border-radius:8px;padding:12px;">
        <div style="font-weight:700;color:#14532d;font-size:11px;margin-bottom:8px;">After rotation ✓</div>
        <div style="font-size:11px;color:#1e293b;line-height:1.9;font-family:monospace;">
          range = 8.91 &nbsp;step = 0.59<br>
          +4.74 → bin 15 → +4.74 (err = 0.00)<br>
          +4.17 → bin 14 → +4.15 (err = 0.02)<br>
          −4.17 → bin &nbsp;0 → −4.17 (err = 0.00)
        </div>
        <div style="margin-top:8px;font-size:11px;color:#14532d;font-weight:600;">Step 25% smaller. All 8 rotated values are distinguishable. No bins wasted.</div>
      </div>
    </div>
    <div style="background:#caffbf;border:1px solid #86efac;border-radius:6px;padding:10px 12px;font-size:11px;color:#14532d;line-height:1.7;margin-top:10px;">
      <strong>Why this works.</strong> The outlier's energy (12.0) is divided by $\sqrt{8} \approx 2.83$ — no output exceeds 4.74. The 7 equal normals mostly cancel across Hadamard rows, so they don't widen the range. The dominant spike at 12.0 is replaced by 8 balanced values near $\pm 4.17$, and step size falls 25%.
    </div>
  </div>


</div>


> **The benefit scales with dimension.** At $d = 8$ the outlier divides by $\sqrt{8} \approx 2.83$ — a 25% range reduction. At $d = 4096$ ($\sqrt{d} = 64$) the same outlier contributes only $12.0 / 64 \approx 0.19$ to each output, indistinguishable from a normal channel. That is roughly a 30× range reduction; post-rotation kurtosis drops from $\kappa > 200$ to $\kappa \approx 3$ across all layers.
{: .prompt-info }


The contrast with SmoothQuant is what to keep in your back pocket. Scaling dim 7 by $s = 10$ would have given $x_7 = 1.2$ — the activation range becomes $[0.2, 1.2]$, step $= 0.067$ — but the weight column for dimension 7 would be $10\times$ larger. The difficulty moved sideways. Rotation spreads the outlier's energy over $d$ axes without amplifying any weight column, because $\lVert W R^T \rVert_F = \lVert W \rVert_F$ always.


> **One hard constraint.** Rotation can only be absorbed through *linear* layers. A nonlinearity (softmax, SiLU) breaks the cancellation — it produces a brand-new activation distribution from scratch. Each nonlinearity boundary in the Transformer is therefore a fresh outlier zone that needs its own rotation matrix.
{: .prompt-warning }


That single constraint dictates everything about *where* to plant the rotations.


---


## 4. Where in the Transformer does each rotation matrix live?


*One $R$ would be ideal — but every nonlinearity resets the activation distribution. So how many do we actually need, and where?*


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:14px;">Transformer block — where each rotation matrix lives</div>
  <pre style="background:#1e293b;color:#fffffc;border-radius:8px;padding:18px 20px;font-size:12px;line-height:1.9;overflow-x:auto;margin:0;">  Residual stream (x)
          │
          │  ◀── <span style="color:#a0c4ff;font-weight:700;">R₁</span>  absorbed into W_Q, W_K, W_V, W_Up, W_Gate
          │      <span style="color:#a0c4ff;">zero runtime cost · one matrix for the whole model</span>
          ▼
  ┌──────────────────────── Attention ─────────────────────┐
  │                                                        │
  │  Q·<span style="color:#a0c4ff;">R₁ᵀ</span>·W_Q    K·<span style="color:#a0c4ff;">R₁ᵀ</span>·W_K    V·<span style="color:#a0c4ff;">R₁ᵀ</span>·W_V·<span style="color:#bdb2ff;">R₂</span>                │
  │                   │                  │                 │
  │                   └──── KV Cache ────┘                 │
  │                              │  ◀── <span style="color:#caffbf;font-weight:700;">R₃</span> Hadamard online │
  │      softmax(Q·Kᵀ/√d)  ◀─── <span style="color:#ffadad;font-weight:700;">NONLINEARITY</span>               │
  │               │                                        │
  │          × V aggregation                               │
  │               │  ◀── <span style="color:#bdb2ff;font-weight:700;">R₂</span>  absorbed into W_V and W_O     │
  │               ▼                                        │
  │          W_O (output proj)                             │
  └────────────────────────────────────────────────────────┘
          │
      + residual
          │
  ┌─────────────────────────── FFN ────────────────────────┐
  │                                                        │
  │    W_Up·<span style="color:#a0c4ff;">R₁ᵀ</span> (x)              W_Gate·<span style="color:#a0c4ff;">R₁ᵀ</span> (x)          │
  │          │                          │                  │
  │     SiLU(·) × Gate  ◀──── <span style="color:#ffadad;font-weight:700;">NONLINEARITY</span>                │
  │                 │                                      │
  │                 │  ◀── <span style="color:#ffd6a5;font-weight:700;">R₄</span>  Hadamard online             │
  │                 │      <span style="color:#ffd6a5;">R₄ᵀ absorbed into W_Down</span>        │
  │                 ▼                                      │
  │            W_Down (down proj)                          │
  └────────────────────────────────────────────────────────┘
          │
      + residual → next block</pre>
</div>


The diagram traces one block top to bottom and marks every place a fresh outlier zone is born. Rotations slip in just before each quantization step, sandwiched by linear layers so they can be absorbed into adjacent weights. Where the path crosses a nonlinearity (the red `NONLINEARITY` markers) — softmax in attention, SiLU in the FFN — the rotation has to stop and a new one start on the other side. That is why we need four, not one.


<div style="display:flex;flex-wrap:wrap;gap:12px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #a0c4ff;border-top:3px solid #60a5fa;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#1e3a8a;margin-bottom:6px;font-size:13px;">R₁ — residual stream</div>
    <div style="color:#1e293b;line-height:1.7;"><strong>Target:</strong> persistent outlier channels that accumulate in $x$ across many layers via residual addition.<br><strong>Absorbed into:</strong> $W_Q, W_K, W_V, W_\text{Up}, W_\text{Gate}$ at export.<br><strong>Runtime cost:</strong> none.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #bdb2ff;border-top:3px solid #c4b5fd;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#4c1d95;margin-bottom:6px;font-size:13px;">R₂ — V / output projection</div>
    <div style="color:#1e293b;line-height:1.7;"><strong>Target:</strong> new outliers created by softmax attention weighting — not present in the residual stream, so $R_1$ can't help.<br><strong>Absorbed into:</strong> $W_V$ and $W_O$ per attention layer.<br><strong>Runtime cost:</strong> none.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-top:3px solid #86efac;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#14532d;margin-bottom:6px;font-size:13px;">R₃ — KV cache (online)</div>
    <div style="color:#1e293b;line-height:1.7;"><strong>Target:</strong> outliers in $K$ and $V$ when stored in the cache.<br><strong>Why online:</strong> $K$ and $V$ are dynamic; can't be precomputed. $Q$ absorbs $H^T$ offline at no cost.<br><strong>Runtime cost:</strong> $O(d \log d)$ Hadamard per token.</div>
  </div>


  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffd6a5;border-top:3px solid #f59e0b;border-radius:8px;padding:14px;">
    <div style="font-weight:700;color:#78350f;margin-bottom:6px;font-size:13px;">R₄ — MLP intermediate (online)</div>
    <div style="color:#1e293b;line-height:1.7;"><strong>Target:</strong> outliers produced by SiLU inside the FFN — absent at Up/Gate input, so $R_1$ misses them.<br><strong>Why online:</strong> applied to dynamic SiLU output. $R_4^T$ is absorbed into $W_\text{Down}$ offline.<br><strong>Runtime cost:</strong> $O(d \log d)$ Hadamard per token.</div>
  </div>


</div>


The four cards split along one axis: *static or dynamic input?* $R_1$ and $R_2$ rotate signals that flow through pre-known weight matrices, so we absorb them into those matrices at export time — zero runtime overhead. $R_3$ and $R_4$ rotate signals that are born during inference (KV cache writes, SiLU outputs), so half of each rotation has to run online as a fast Hadamard transform.


> **Two deployment tiers.** SpinQuant_no_had uses only $R_1$ and $R_2$ — both absorbed offline, zero runtime cost — and is enough for W4A8. SpinQuant_had adds $R_3$ and $R_4$ as online Hadamards at ~8% latency overhead, and is required for the harder W4A4 setting.
{: .prompt-tip }


### Training the rotations


*Random rotations already help — but with 13 points of accuracy spread across seeds, can we just learn the right one?*


A random Hadamard improves things on average, but seed-to-seed variance is huge: 100 random seeds on LLaMA-2 7B W4A4 span a 13-point accuracy range. SpinQuant learns $R$ instead of guessing it. The catch is that gradient descent doesn't naturally preserve orthogonality — drift off the manifold and the identity trick breaks, changing the full-precision output.


The fix is the **Cayley transform**: an update rule that mathematically guarantees every step lands back on the manifold of orthogonal matrices.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1e293b;font-size:13px;margin-bottom:14px;">SpinQuant rotation training — pseudocode</div>
  <pre style="background:#1e293b;color:#caffbf;border-radius:8px;padding:16px 18px;font-size:12px;line-height:1.85;overflow-x:auto;margin:0;"><span style="color:#a0c4ff;font-weight:600;">Input:</span>  model       — pretrained LLM, weights <strong>frozen</strong> at 16-bit
        calib_data  — 800 samples from WikiText-2
<span style="color:#a0c4ff;font-weight:600;">Output:</span> R₁, R₂     — learned rotations (absorbed into weights after training)


<span style="color:#9bf6ff;"># Init with random Hadamard matrices — better than pure random</span>
<span style="color:#ffd6a5;"> 1.</span>  R₁, R₂ ← random_hadamard()


<span style="color:#ffd6a5;"> 2.</span>  <span style="color:#ffadad;font-weight:600;">for</span> step in 1..100:
<span style="color:#ffd6a5;"> 3.</span>      x = sample(calib_data)
<span style="color:#ffd6a5;"> 4.</span>      ŷ = forward(model, x, R₁, R₂)   <span style="color:#9bf6ff;"># activations quantized; weights stay 16-bit</span>
<span style="color:#ffd6a5;"> 5.</span>      loss = cross_entropy(ŷ, x)       <span style="color:#9bf6ff;"># STE lets gradient pass through rounding</span>
<span style="color:#ffd6a5;"> 6.</span>      G = ∂loss / ∂R


<span style="color:#9bf6ff;">      # Cayley step — R stays orthogonal after every update</span>
<span style="color:#ffd6a5;"> 7.</span>      Ĝ = G·Rᵀ − ½·R·(G·Rᵀ)ᵀ          <span style="color:#9bf6ff;"># project onto tangent space of R</span>
<span style="color:#ffd6a5;"> 8.</span>      Y = Ĝ − Ĝᵀ                       <span style="color:#9bf6ff;"># skew-symmetrize → input for Cayley map</span>
<span style="color:#ffd6a5;"> 9.</span>      R ← (I − α/2·Y)⁻¹ · (I + α/2·Y) · R   <span style="color:#9bf6ff;"># RᵀR = I still holds ✓</span>


<span style="color:#9bf6ff;"># Absorb rotations into weight matrices (one-time export cost)</span>
<span style="color:#ffd6a5;">10.</span>  W_Q  ← W_Q · R₁ᵀ  ;  W_K ← W_K · R₁ᵀ  ;  W_V ← W_V · R₁ᵀ · R₂
<span style="color:#ffd6a5;">11.</span>  W_Up ← W_Up · R₁ᵀ ;  W_Gate ← W_Gate · R₁ᵀ  ;  W_O ← R₂ᵀ · W_O
<span style="color:#ffd6a5;">12.</span>  <span style="color:#9bf6ff;"># R₄ᵀ absorbed into W_Down; R₃ stays as online Hadamard at inference</span>


<span style="color:#9bf6ff;"># Then GPTQ on the rotation-transformed weights handles weight quantization</span></pre>
  <div style="background:#fdffb6;border:1px solid #f59e0b;border-radius:6px;padding:10px 14px;font-size:11px;color:#713f12;margin-top:12px;line-height:1.7;">
    $R_1$ and $R_2$ together are only ~0.26% of total parameters. 100 steps on 800 unlabeled samples: ~13 minutes for a 1B model, ~3.5 hours for 70B. Weights are completely frozen — only the rotation matrices update. After absorption, inference for $R_1$ and $R_2$ is identical to a normal forward pass.
  </div>
</div>


The numbers tell the whole economics story. Rotation training touches a tiny slice of the parameter count, freezes the weights, runs on a few hundred samples, and finishes in under an hour for production-scale models. After absorption, $R_1$ and $R_2$ are gone — they only existed during training. What's left is a normal forward pass with a slightly different set of weight values, plus two cheap Hadamard transforms for $R_3$ and $R_4$ if we are pushing all the way to W4A4.


Cayley-trained rotations close the W4A4 gap from ~27 PPL points (SmoothQuant) to under 3 (SpinQuant_had). With that, the full picture from per-tensor INT8 collapse all the way to W4A4 is in place.


---


## 5. Cheatsheet — when you hit this in the wild


*Symptom-to-cause map. When something goes wrong post-quantization, where do you look first?*


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">What you see</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">What it usually means</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">What to try</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Per-tensor W8A8 PPL collapses on an LLM</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Activation outlier channels crushing the scale</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">SmoothQuant with global $\alpha = 0.5$, then per-layer grid search</td>
    </tr>
    <tr style="background:#a0c4ff40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">SmoothQuant W8A8 still produces NaN in early layers</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Outlier-dominated layers need $\alpha &gt; 0.8$</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Check per-layer $\alpha$ map; per-layer search recovers 0.1–0.2 PPL</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;"><code>inf</code> in weight matrix after absorption</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Dead-neuron weight column → $s_i \to \infty$</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Clamp $s$ to $[10^{-4}, 10^{4}]$ before fusing</td>
    </tr>
    <tr style="background:#bdb2ff40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">SmoothQuant works at INT8 but explodes at INT4</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Scale costs more bits than the format has</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Switch to SpinQuant (rotation, not scaling)</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Random Hadamard works on some seeds, not others</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">~13-pt seed variance on W4A4 — luck of the draw</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Train rotations with Cayley SGD on 800 samples</td>
    </tr>
    <tr style="background:#caffbf40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">W4A8 ships fine, W4A4 still has a 5+ PPL gap</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Missing online rotations $R_3, R_4$ for KV / SiLU</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">SpinQuant_had — 8% latency for W4A4 quality</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Rotation-quantized model output drifts from FP16</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">$R$ drifted off the orthogonal manifold during training</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Confirm Cayley update; verify $\lVert R^T R - I \rVert &lt; 10^{-5}$</td>
    </tr>
  </tbody>
</table>
</div>


---


The two techniques compose into one decision flow. For W8A8 on an LLM, reach for SmoothQuant — a per-layer $\alpha$ search migrates outliers cheaply, and the inference path stays a plain INT8 GEMM. When the bit budget drops to W4A8 or below, scaling has run out of room and we switch to SpinQuant: $R_1$ and $R_2$ at zero runtime cost cover most of W4A8, and the online $R_3, R_4$ pair completes the picture for W4A4 at modest latency overhead. Everything else — calibration data, GPTQ-style weight rounding, kernel choices — rides on top of these two primitives.


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



