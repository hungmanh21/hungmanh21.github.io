---
layout: post
title: "Sensitivity Analysis: Finding the Layers That Refuse to Be Quantized"
date: 2026-06-04 09:00:00 +0000
categories: [Model Compression, Quantization, Debug]
tags: [quantization, ptq, sensitivity-analysis, sqnr, mixed-precision, llm]
math: true
description: "Why down_proj fails first, how to read an SQNR-per-layer report, and the W4→W8 promotion rule — sensitivity analysis for mixed-precision LLM quantization."
---


<!-- Opening block: prerequisites + questions -->
<div style="margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;margin-bottom:12px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>RTN, GPTQ, AWQ at a high level — Phase 2 PTQ post</li>
      <li>Per-tensor / per-channel / per-group quantization</li>
      <li>Transformer block layout: attention projections, FFN, residual stream, RoPE</li>
      <li>PyTorch <code>nn.Linear</code> and forward hooks</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>PTQ accuracy is bad — should I tune clipping or raise some layers to higher bits?</li>
      <li>Why are <code>down_proj</code> and the LM head almost always the most sensitive?</li>
      <li>What are the input / param / output quantizers, and which one do I actually need?</li>
      <li>How do I read an SQNR-per-layer report and decide what stays in 4-bit vs 8-bit?</li>
      <li>What does mixed precision cost me at deployment time?</li>
    </ul>
  </div>
</div>


You ran AWQ at W4A16. Perplexity went from 7.1 to 9.6. Re-running with different calibration data didn't help. That's the wall. From here you have exactly two moves: search harder for clipping ranges, or accept that not every layer is willing to live at 4 bits and promote a handful back to 8. Sensitivity analysis tells you which.


---


## 1. Hitting the Quantization Wall {#wall}


*PTQ accuracy is unacceptable — what are the actual escape hatches, and which one should you try first?*


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Move 1 — better clipping ranges</div>
    <div style="color:#1e293b;line-height:1.6;">
      Default <code>autoawq</code> grid-searches clip ratios uniformly. Swap that for KL-divergence or percentile-based search per channel. Same bit-width, same kernels, no deployment changes.
    </div>
    <div style="margin-top:10px;padding:8px 10px;background:#caffbf;color:#14532d;border-radius:4px;font-size:11px;">
      Cheap. Try first. Recovers small gaps (PPL +0.1 to +0.5).
    </div>
  </div>
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Move 2 — mixed precision</div>
    <div style="color:#1e293b;line-height:1.6;">
      Identify the few layers that genuinely cannot survive 4-bit and pin them to 8-bit (or FP16). Everything else stays compressed.
    </div>
    <div style="margin-top:10px;padding:8px 10px;background:#ffd6a5;color:#78350f;border-radius:4px;font-size:11px;">
      Recovers larger gaps (PPL +0.5 to +2). Costs deployment complexity — see §6.
    </div>
  </div>
</div>


Move 1 keeps the format uniform; only the calibration math changes. Move 2 changes the model itself and the kernels you'll dispatch to. Sensitivity analysis is the prerequisite for Move 2 — it tells you the *minimum* set of layers to upgrade so you don't pay deployment cost for free.


> **Order of operations.** Always exhaust clipping search before mixed precision. A successful clip-search recovery is invisible to the runtime; a successful mixed-precision recovery isn't.
{: .prompt-tip }


---


## 2. Why Some Layers Are Sensitive {#why-sensitive}


*Quantization noise is the same operator everywhere — why does <code>down_proj</code> degrade 12 dB worse than <code>q_proj</code>?*


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Per-layer weight-quantization SQNR — Llama-style block, W4 RTN</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">lower bar = more sensitive · sample from <code>sensitivity_analyzer.md</code></div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">down_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">9.78 dB</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;">danger</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">o_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:40%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">16.09 dB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;">warn</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">up_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:41%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">16.57 dB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;">warn</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">v_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:43%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">17.06 dB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;">warn</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">gate_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:44%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">17.73 dB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;">warn</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">k_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:48%;height:100%;background:#fdffb6;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">19.13 dB</span>
      <span style="background:#fdffb6;color:#713f12;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fde047;">note</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">q_proj</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:54%;height:100%;background:#fdffb6;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">21.64 dB</span>
      <span style="background:#fdffb6;color:#713f12;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fde047;">note</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">lm_head</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:56%;height:100%;background:#fdffb6;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">22.55 dB</span>
      <span style="background:#fdffb6;color:#713f12;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fde047;">note</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:160px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">embed_tokens</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:97%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">38.67 dB</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;">safe</span>
    </div>
  </div>
</div>


<code>down_proj</code> consistently sits at the bottom because the FFN's SiLU-gated path produces a distribution with heavy tails — a few channels carry magnitudes 10–100× the median, so a uniform 4-bit grid wastes most of its 16 levels on empty space and crushes the body of the distribution. <code>o_proj</code>, <code>up_proj</code>, and <code>v_proj</code> follow the same pattern at lower intensity. The LM head shows up next because errors here are not absorbed by any downstream layer — they hit the softmax directly. <code>embed_tokens</code> stays clean because its rows are independent token vectors and the dynamic range per row is small.


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6366f1;margin-bottom:8px;">Formula · Signal-to-Quantization-Noise Ratio</div>


$$
\mathrm{SQNR}(x, \hat{x}) = 10 \log_{10} \frac{\mathbb{E}[\|x\|^2]}{\mathbb{E}[\|x - \hat{x}\|^2]} \quad \text{(dB)}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $x$ — full-precision tensor (weight, activation, or output). $\hat{x}$ — quantized then dequantized version. Higher SQNR = less quantization noise relative to signal. Rule of thumb: <b>≥ 30 dB safe</b>, <b>20–30 dB borderline</b>, <b>&lt; 20 dB likely to hurt downstream PPL</b>.
  </div>
</div>


> **Two structural reasons for sensitivity.** (1) Wide dynamic range — a few outlier channels dominate the scale and starve the rest. (2) Proximity to logits — errors near the LM head are not averaged out by deeper layers. Both apply to <code>down_proj</code>; the second alone applies to <code>lm_head</code>.
{: .prompt-info }


---


## 3. The Three Quantizers on Every Layer {#three-quantizers}


*"Quantize the layer" is ambiguous — three different tensors flow through a linear, and each one can carry its own quantizer.*


<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:660px;display:block;margin:18px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="200" fill="#fffffc" rx="10"/>


  <!-- Top row: X → input_q → matmul → output_q → Y -->
  <rect x="20" y="60" width="80" height="40" fill="#a0c4ff" stroke="#60a5fa" rx="6"/>
  <text x="60" y="85" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">X</text>


  <text x="115" y="83" font-size="22" fill="#475569">→</text>


  <rect x="135" y="60" width="100" height="40" fill="#9bf6ff" stroke="#67e8f9" rx="6"/>
  <text x="185" y="78" text-anchor="middle" font-size="11" font-weight="700" fill="#164e63">input_q</text>
  <text x="185" y="93" text-anchor="middle" font-size="10" fill="#164e63">activation Q</text>


  <text x="245" y="83" font-size="22" fill="#475569">→</text>


  <rect x="265" y="50" width="110" height="60" fill="#fdffb6" stroke="#fde047" rx="6"/>
  <text x="320" y="75" text-anchor="middle" font-size="13" font-weight="700" fill="#713f12">Y = X̂ Ŵᵀ</text>
  <text x="320" y="93" text-anchor="middle" font-size="10" fill="#713f12">matmul</text>


  <text x="385" y="83" font-size="22" fill="#475569">→</text>


  <rect x="405" y="60" width="105" height="40" fill="#9bf6ff" stroke="#67e8f9" rx="6"/>
  <text x="457" y="78" text-anchor="middle" font-size="11" font-weight="700" fill="#164e63">output_q</text>
  <text x="457" y="93" text-anchor="middle" font-size="10" fill="#164e63">activation Q</text>


  <text x="520" y="83" font-size="22" fill="#475569">→</text>


  <rect x="540" y="60" width="80" height="40" fill="#a0c4ff" stroke="#60a5fa" rx="6"/>
  <text x="580" y="85" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">Y</text>


  <!-- Bottom row: W → param_q → into matmul -->
  <rect x="135" y="145" width="80" height="38" fill="#caffbf" stroke="#86efac" rx="6"/>
  <text x="175" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#14532d">W</text>


  <text x="225" y="168" font-size="22" fill="#475569">→</text>


  <rect x="245" y="145" width="100" height="38" fill="#bdb2ff" stroke="#c4b5fd" rx="6"/>
  <text x="295" y="161" text-anchor="middle" font-size="11" font-weight="700" fill="#4c1d95">param_q</text>
  <text x="295" y="176" text-anchor="middle" font-size="10" fill="#4c1d95">weight Q</text>


  <!-- Arrow from param_q up into matmul -->
  <path d="M 320 145 L 320 110" stroke="#c4b5fd" stroke-width="2" fill="none" marker-end="url(#arr)"/>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#c4b5fd"/>
    </marker>
  </defs>
</svg>


The activation X enters, optionally gets quantized by the **input quantizer** ($\hat{X}$), multiplies the weight that has been pre-quantized by the **param quantizer** ($\hat{W}$), and the result Y can optionally be re-quantized by the **output quantizer** before exiting the layer. W4A16 means: param_q on, both activation quantizers off. W8A8 means: param_q on plus activation quantization at every tensor boundary — conceptually all three slots filled, though as §4 shows, neighboring layers share a single quantizer at their boundary rather than each owning a private input_q and output_q. W4A8 is the same idea with 4-bit weights.


> **What "isolating a quantizer" means.** To measure one quantizer's contribution, leave the other two at full precision and quantize only the one you're studying. The drop in PPL (or the SQNR of the output vs FP) is that quantizer's marginal cost. Without isolation, blame is ambiguous.
{: .prompt-tip }


---


## 4. Why Input + Output Quantizers Look Redundant {#redundancy}


*If layer A's output is layer B's input, do we really need both an output_q on A and an input_q on B? The answer is: usually no — except when the graph branches.*


<svg viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:660px;display:block;margin:18px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="360" fill="#fffffc" rx="10"/>


  <!-- Panel A: sequential dedup -->
  <text x="20" y="25" font-size="12" font-weight="700" fill="#1e293b">(a) Sequential chain — keep only one of the two</text>
  <rect x="20" y="40" width="90" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="65" y="62" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">layer A</text>
  <rect x="115" y="40" width="80" height="36" fill="#9bf6ff" stroke="#67e8f9" rx="5"/>
  <text x="155" y="62" text-anchor="middle" font-size="10" font-weight="700" fill="#164e63">output_q</text>
  <text x="200" y="62" font-size="18" fill="#475569">→</text>
  <rect x="220" y="40" width="80" height="36" fill="#9bf6ff" stroke="#67e8f9" rx="5" stroke-dasharray="4 3" opacity=".4"/>
  <text x="260" y="62" text-anchor="middle" font-size="10" font-weight="700" fill="#164e63" opacity=".5">input_q</text>
  <rect x="305" y="40" width="90" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="350" y="62" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">layer B</text>
  <text x="410" y="62" font-size="11" fill="#14532d">✓ disable input_q on B</text>


  <!-- Panel B: residual stream -->
  <text x="20" y="120" font-size="12" font-weight="700" fill="#1e293b">(b) Residual stream — the add point merges two paths, both must be quantized</text>
  <rect x="20" y="160" width="80" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="60" y="182" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">x</text>
  <path d="M 100 178 L 140 178" stroke="#475569" stroke-width="2" fill="none"/>
  <!-- branch up -->
  <path d="M 130 178 L 130 145 L 230 145" stroke="#475569" stroke-width="2" fill="none"/>
  <rect x="230" y="125" width="120" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="290" y="147" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">attn / FFN sublayer</text>
  <path d="M 350 145 L 410 145 L 410 178 L 460 178" stroke="#475569" stroke-width="2" fill="none"/>
  <!-- skip line -->
  <path d="M 140 178 L 460 178" stroke="#475569" stroke-width="2" fill="none" stroke-dasharray="4 3"/>
  <!-- residual_add -->
  <circle cx="470" cy="178" r="14" fill="#ffadad" stroke="#fca5a5" stroke-width="2"/>
  <text x="470" y="183" text-anchor="middle" font-size="14" font-weight="700" fill="#7f1d1d">+</text>
  <rect x="495" y="160" width="100" height="36" fill="#9bf6ff" stroke="#67e8f9" rx="5"/>
  <text x="545" y="182" text-anchor="middle" font-size="10" font-weight="700" fill="#164e63">output_q</text>
  <text x="545" y="195" text-anchor="middle" font-size="9" fill="#164e63">(after +)</text>
  <text x="20" y="218" font-size="11" fill="#7f1d1d">⚠ output_q on the sublayer alone is not enough — the residual lane still arrives at full precision</text>


  <!-- Panel C: RoPE / reshape gap -->
  <text x="20" y="258" font-size="12" font-weight="700" fill="#1e293b">(c) Quantization-free op (RoPE, reshape) — next op needs its own input_q</text>
  <rect x="20" y="280" width="80" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="60" y="302" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">q_proj</text>
  <text x="105" y="302" font-size="18" fill="#475569">→</text>
  <rect x="125" y="280" width="80" height="36" fill="#fdffb6" stroke="#fde047" rx="5"/>
  <text x="165" y="298" text-anchor="middle" font-size="11" font-weight="700" fill="#713f12">RoPE</text>
  <text x="165" y="312" text-anchor="middle" font-size="9" fill="#713f12">(no quant)</text>
  <text x="210" y="302" font-size="18" fill="#475569">→</text>
  <rect x="230" y="280" width="100" height="36" fill="#9bf6ff" stroke="#67e8f9" rx="5"/>
  <text x="280" y="302" text-anchor="middle" font-size="10" font-weight="700" fill="#164e63">input_q</text>
  <text x="335" y="302" font-size="18" fill="#475569">→</text>
  <rect x="355" y="280" width="110" height="36" fill="#a0c4ff" stroke="#60a5fa" rx="5"/>
  <text x="410" y="302" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">matmul_qk</text>
  <text x="20" y="340" font-size="11" fill="#7f1d1d">⚠ q_proj's output_q is invalidated by RoPE — re-quantize the input of matmul_qk</text>
</svg>


In the simple sequential case (a), <code>q_proj.output_q</code> and the next op's <code>input_q</code> describe the same boundary; pick one and disable the other. In the residual case (b), the sublayer's quantized output and the FP skip lane merge at the `+` (the add itself runs in higher precision), so the **output_q sits *after* the add** — that's what gives downstream layers a uniform-precision tensor to consume. Without it, the next sublayer's input_q would be the only thing standing between mixed-precision arithmetic and the rest of the model — and that input_q is often disabled by the dedup rule from (a). In case (c), RoPE rotates and reshape permutes without any quantization step, which silently de-quantizes the tensor; the next op (typically <code>matmul_qk</code>) must re-quantize on its input side.


This is exactly why the example report from the source material has <code>matmul_qk</code> and <code>matmul_sv</code> in the **input_quantizers** group while <code>q_proj</code>, <code>k_proj</code>, <code>v_proj</code> do not — those three sit downstream of <code>input_layernorm.output_q</code>, so their input is already discrete.


> **Audit rule.** Every tensor that crosses from quantized to FP back to quantized produces a phantom error. Walk the graph: at every op, ask "is my input quantized at the precision I expect?" If not, add an input_q.
{: .prompt-warning }


---


## 5. Reading a Real Sensitivity Report {#reading-report}


*You ran the analyzer and got a CSV with 30+ rows of SQNR numbers — what's actually actionable in there?*


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:12px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">op_class</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">quantizer</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">SQNR (dB)</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#ffadad;"><td style="padding:6px 12px;border:1px solid #e2e8f0;"><b>down_proj</b></td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">9.78</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">⚠ promote to W8</td></tr>
    <tr style="background:#ffadad;"><td style="padding:6px 12px;border:1px solid #e2e8f0;"><b>mlp.mul</b></td><td style="padding:6px 12px;border:1px solid #e2e8f0;">output_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">28.17</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">⚠ keep activation FP</td></tr>
    <tr style="background:#ffadad;"><td style="padding:6px 12px;border:1px solid #e2e8f0;"><b>residual_add_1</b></td><td style="padding:6px 12px;border:1px solid #e2e8f0;">output_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">28.30</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">⚠ keep activation FP</td></tr>
    <tr style="background:#ffadad;"><td style="padding:6px 12px;border:1px solid #e2e8f0;"><b>residual_add_2</b></td><td style="padding:6px 12px;border:1px solid #e2e8f0;">output_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">27.75</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">⚠ keep activation FP</td></tr>
    <tr style="background:#ffd6a5;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">o_proj</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">16.09</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">consider W8</td></tr>
    <tr style="background:#ffd6a5;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">v_proj</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">17.06</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">consider W8</td></tr>
    <tr style="background:#fdffb6;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">q_proj</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">21.64</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">borderline — leave W4</td></tr>
    <tr style="background:#fdffb6;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">lm_head</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">22.55</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">borderline — often W8</td></tr>
    <tr style="background:#caffbf;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">embed_tokens</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">38.67</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">safe at W4</td></tr>
    <tr style="background:#caffbf;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">input_layernorm</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">96.46</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">trivial</td></tr>
    <tr style="background:#caffbf;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">model.norm</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">param_q</td><td style="padding:6px 12px;border:1px solid #e2e8f0;text-align:right;">103.69</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">trivial</td></tr>
  </tbody>
</table>
</div>


The actionable rows are the ones below ~20 dB on the param side and below ~30 dB on the activation side. Three patterns appear in every Llama-style report: <code>down_proj</code> at the bottom of weight-quantization SQNR; the residual adds and <code>mlp.mul</code> (the SwiGLU elementwise multiply, <code>silu(gate) * up</code>) at the bottom of output-quantization SQNR; and the LM head sitting on the borderline. The norm layers are at +90 dB because their weights are nearly uniform scalars — quantization barely touches them.


Note that in the source CSV, <code>q_proj</code>, <code>k_proj</code>, <code>v_proj</code> have **no input_quantizer row**: their input is the output of <code>input_layernorm</code>, which already carries an output_q. That's the dedup rule from §4 in action. Conversely, <code>matmul_qk</code> and <code>matmul_sv</code> *do* have input_q rows because RoPE breaks the chain in front of them.


> **The recipe.** Tier the param SQNR: **< 12 dB → promote to W8** (almost always wins more PPL than it costs); **12–20 dB → consider W8** if the deployment can absorb the memory hit, measure both ways; **> 20 dB → leave at W4** (LM head is the usual exception — promote it regardless because errors there hit softmax directly). Disable the output quantizer on any activation site with SQNR < 30 dB. Re-run PPL. If still bad, escalate to QAT.
{: .prompt-info }


---


## 6. Mixed-Precision Trade-offs {#tradeoffs}


*Promoting a layer from W4 to W8 always recovers some accuracy — what does it actually cost downstream?*


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Wins</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li><b>Accuracy recovery.</b> Promoting just <code>down_proj</code> often closes 50–70% of the gap to FP16 PPL — applied to §1's wall (FP16 7.1, W4 9.6), that's roughly 7.9–8.4.</li>
      <li><b>Cheap analysis.</b> One forward pass per layer. No retraining.</li>
      <li><b>Targeted.</b> You upgrade ~5–10% of weights, not all of them.</li>
    </ul>
  </div>
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #fca5a5;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#7f1d1d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Costs</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li><b>Memory savings shrink.</b> <code>down_proj</code> alone is ~20% of LLM parameters (one of three FFN matrices, FFN being two-thirds of each block). Promoting it from W4 to W8 across every layer adds ~20% to the W4 model size; adding <code>o_proj</code> too pushes that to ~25%. Measure the end-to-end size — don't assume the cost is a rounding error.</li>
      <li><b>Kernel fragmentation.</b> The runtime now dispatches W4 and W8 GEMMs on the same forward path — fewer fused kernels apply.</li>
      <li><b>Format support.</b> GPTQ and AWQ checkpoint formats assume uniform bit-width; mixed precision usually means a custom or <code>llm-compressor</code>-style packed format, and not every inference engine reads it.</li>
      <li><b>Per-architecture tuning.</b> Sensitivity ranking shifts between Llama, Qwen, Mistral, and MoE — you can't reuse the recipe across families.</li>
    </ul>
  </div>
</div>


> **When to skip mixed precision.** If your deployment target is a vanilla GPTQ/AWQ-only runtime (e.g., older vLLM versions or a fixed inference endpoint), mixed precision is dead on arrival. Switch to QAT or QLoRA instead — same accuracy goal, uniform output format.
{: .prompt-warning }


---


## 7. Minimal Sensitivity Analyzer {#code}


*The full analyzer in ~75 lines — fake-quantizes one weight at a time, measures output SQNR, ranks the layers.*


```python
# model_compression/phase2/labs/sensitivity_analyzer.py
import torch, torch.nn as nn, csv, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def fake_quant_weight(w, n_bits=4, group_size=128):
    out_f, in_f = w.shape
    g = group_size if in_f % group_size == 0 else in_f
    w_g = w.reshape(out_f, in_f // g, g)
    qmax = 2 ** (n_bits - 1) - 1
    scale = (w_g.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
    w_q = torch.round(w_g / scale).clamp(-qmax - 1, qmax) * scale
    return w_q.reshape(out_f, in_f).to(w.dtype)


def sqnr_db(x, x_q):
    sig = x.float().pow(2).mean()
    noise = (x - x_q).float().pow(2).mean().clamp(min=1e-12)
    return (10.0 * torch.log10(sig / noise)).item()


@torch.no_grad()
def analyze(model, inputs, n_bits, group_size):
    model.eval()
    fp_out = model(**inputs).logits
    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w_orig = mod.weight.data.clone()
        mod.weight.data = fake_quant_weight(w_orig, n_bits, group_size)
        try:
            sqnr = sqnr_db(fp_out, model(**inputs).logits)
        finally:
            mod.weight.data = w_orig
        rows.append((name, sqnr))
        print(f"{name:60s}  SQNR = {sqnr:6.2f} dB")
    return sorted(rows, key=lambda r: r[1])
```


The trick is replace-then-restore: clone the FP weight, swap in the fake-quant version, run forward, swap back. No hooks, no model surgery, no graph rewiring. Each iteration costs one forward pass; on a 0.5B model with a single calibration prompt, the whole sweep finishes in under a minute on CPU.


Sample output on Qwen2.5-0.5B with W4 RTN:


```
model.layers.0.mlp.down_proj           SQNR =   9.83 dB
model.layers.5.mlp.down_proj           SQNR =  10.41 dB
model.layers.0.self_attn.o_proj        SQNR =  16.21 dB
model.layers.0.self_attn.v_proj        SQNR =  17.04 dB
...
model.embed_tokens                     SQNR =  38.55 dB


Top 5 most sensitive (candidates for higher precision):
  model.layers.0.mlp.down_proj                                  9.83 dB
  model.layers.5.mlp.down_proj                                 10.41 dB
  model.layers.11.mlp.down_proj                                10.92 dB
  model.layers.0.self_attn.o_proj                              16.21 dB
  model.layers.0.self_attn.v_proj                              17.04 dB
```


The analyzer reports per-instance (`layers.0.mlp.down_proj`, `layers.5.mlp.down_proj`, …), but in practice you promote per **op type** — all `down_proj`s together — rather than cherry-picking individual layer indices. The spread between `layers.0.down_proj` and `layers.31.down_proj` is typically <2 dB; the spread between `down_proj` and `q_proj` is ~12 dB. Pick the right granularity, and your kernel dispatch stays sane.


> **Extension hint.** This script only measures the **param quantizer** in isolation. To extend to input/output quantizers, install a forward hook on each module, fake-quantize the activation tensor inside the hook, and rerun the same sweep. Same replace-then-restore pattern, different tensor.
{: .prompt-tip }


---


The two recovery paths from §1 compose cleanly: run sensitivity analysis first to know *which* layers need help, then decide *how* — better clipping for the 12–20 dB borderline rows, W8 promotion for the truly broken ones (SQNR < 12 dB), and the LM head promoted regardless because errors there hit softmax directly. Anything that survives both is already quantization-safe and not worth more engineering.



