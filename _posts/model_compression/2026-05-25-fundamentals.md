---
layout: post
title: "LLM Memory 101: Floating-Point Formats, Memory Arithmetic, and GPU Hierarchy"
date: 2026-05-25 00:00:00 +0000
categories: [Model Compression]
tags: [quantization, floating-point, bfloat16, fp8, fp4, kv-cache, gpu-memory, arithmetic-intensity]
math: true
description: "BF16 vs FP16, KV cache sizing, and INT8 vs FP8 by batch size — the floating-point formats and GPU-memory arithmetic behind every LLM quantization decision."
---


*What you cannot measure in bytes, you cannot compress.*


<!-- Opening block: questions + prerequisites — stacked vertically -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>If FP16 and BF16 are both 16 bits, why does FP16 overflow to NaN while BF16 trains fine?</li>
      <li>A 7B model is "7 GB in INT8" — so why does it OOM at batch 8?</li>
      <li>Why does the KV cache grow linearly with batch and sequence length, and dwarf the weights?</li>
      <li>Why does INT8 buy throughput at batch=1 but do nothing at batch=64?</li>
      <li>What is microscaling, and why does it make FP4 practical on Blackwell?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #c4b5fd;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>IEEE 754 floating-point basics (sign, exponent, mantissa)</li>
      <li>Transformer architecture (layers, attention heads, KV cache)</li>
      <li>Basic GPU concepts (VRAM, HBM bandwidth)</li>
    </ul>
  </div>
</div>


Before we touch a single quantization knob, three mental models have to be solid: **how a weight is stored in bits**, **how many bytes it demands at serving time**, and **where those bytes live on the GPU**. Let's chase each one through a figure — every compression decision later in the curriculum falls out of these three.


---


## 1. How are numerical values stored in bits?


*If FP16 and BF16 are both 16 bits, why does one overflow to NaN and the other train fine?*


Every parameter is a binary number, and IEEE 754 carves those bits into three fields:


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #60a5fa;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#1e3a8a;margin-bottom:8px;">Formula · IEEE 754 value</div>


$$
\text{value} = (-1)^{S} \times 2^{\,E - \text{bias}} \times (1 + M)
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    <strong>S</strong> (1 bit) — sign.
    <strong>E</strong> (exponent bits) — sets <em>dynamic range</em>: how large or small a number can get.
    <strong>M</strong> (mantissa bits) — sets <em>precision</em>: how many distinct values fit inside that range.
  </div>
</div>


The whole story of every format is one fixed tension: **spend bits on the exponent and you buy range; spend them on the mantissa and you buy precision.** Sixteen bits, eight bits, four bits — the question is always the same split. Let's watch three 16-bit formats make three different bets.


### Why does FP16 overflow where BF16 survives?


*Same 16 bits — so the difference has to be in how they split them.*


<div style="font-family:system-ui,sans-serif; font-size:13px; line-height:1.6; margin:20px 0;">


  <!-- FP32 row -->
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
    <span style="width:44px; font-weight:700; color:#1e293b; font-size:12px; flex-shrink:0;">FP32</span>
    <div style="flex:1; max-width:560px; display:flex; height:30px; border-radius:5px; overflow:hidden; border:1px solid #d1d5db;">
      <div style="flex:1; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">S</div>
      <div style="flex:8; background:#a0c4ff; color:#1e3a8a; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:600; font-size:11px;">Exponent · 8 bits</div>
      <div style="flex:23; background:#caffbf; color:#14532d; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:11px;">Mantissa · 23 bits</div>
    </div>
    <span style="font-size:11px; color:#1e293b; white-space:nowrap;">range ±3.4×10³⁸ · ~7 decimal digits</span>
  </div>


  <!-- FP16 row -->
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
    <span style="width:44px; font-weight:700; color:#7f1d1d; font-size:12px; flex-shrink:0;">FP16</span>
    <div style="flex:1; max-width:560px; display:flex; height:30px; border-radius:5px; overflow:hidden; border:1px solid #d1d5db;">
      <div style="flex:1; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">S</div>
      <div style="flex:5; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:11px;">Exp · 5 bits ⚠</div>
      <div style="flex:10; background:#caffbf; color:#14532d; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:11px;">Mantissa · 10 bits</div>
    </div>
    <span style="font-size:11px; color:#7f1d1d; font-weight:700; white-space:nowrap;">max 65,504 → overflow!</span>
  </div>


  <!-- BF16 row -->
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="width:44px; font-weight:700; color:#14532d; font-size:12px; flex-shrink:0;">BF16</span>
    <div style="flex:1; max-width:560px; display:flex; height:30px; border-radius:5px; overflow:hidden; border:1px solid #d1d5db;">
      <div style="flex:1; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">S</div>
      <div style="flex:8; background:#a0c4ff; color:#1e3a8a; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:11px;">Exp · 8 bits ✓</div>
      <div style="flex:7; background:#caffbf; color:#14532d; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:11px;">Mantissa · 7 bits</div>
    </div>
    <span style="font-size:11px; color:#14532d; font-weight:700; white-space:nowrap;">same range as FP32 · precision SGD absorbs</span>
  </div>


</div>


Read the exponent segments top to bottom. FP32 and BF16 both carry an **8-bit exponent** — the blue bar is the same width — so both top out around $3.4\times10^{38}$. FP16 spends one of those exponent bits on extra mantissa, and that single bit is the whole drama: a 5-bit exponent caps FP16 at **65,504**. LLM gradient norms blow past that ceiling after a few thousand steps, the value saturates to `NaN`, and the run corrupts silently and never recovers.


BF16 makes the opposite trade — it keeps FP32's range and pays with a 7-bit mantissa. That sounds reckless until you remember what training actually does: stochastic gradient descent averages over millions of updates, so it never needed seven decimal digits per step. **BF16 fixes overflow because it keeps the 8-bit exponent**, and it gets away with the coarser mantissa because SGD absorbs the noise.


| Format | Bits | Exp | Man | Max value  | Primary use                              |
| ------ | ---- | --- | --- | ---------- | ---------------------------------------- |
| FP32   | 32   | 8   | 23  | ≈3.4×10³⁸  | Master weights, full-precision reference |
| FP16   | 16   | 5   | 10  | **65,504** | Legacy (avoid for LLM training)          |
| BF16   | 16   | 8   | 7   | ≈3.4×10³⁸  | Default training + inference (2026)      |
| INT8   | 8    | —   | —   | 127        | PTQ compressed weights                   |
| INT4   | 4    | —   | —   | 7          | W4A16 deployment                         |


> **2026 default.** `torch.autocast`, HuggingFace Accelerate, and vLLM all target BF16. FP16 training is legacy — if you see a codebase still passing `fp16=True`, flag it.
{: .prompt-info }


Keep that 8-bit exponent in your back pocket. The reason floating point beats integers at low bit-widths is the same reason BF16 beat FP16 — and it shows up next when we drop to 8 bits.


### Why does FP8 exist when INT8 already does?


*INT8 is one byte, hardware-supported everywhere — so what is FP8 buying that INT8 can't sell?*


<svg viewBox="0 0 620 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="620" height="210" fill="#fffffc" rx="10" stroke="#d1d5db" stroke-width="1"/>


  <!-- ── INT8 number line ── -->
  <text x="14" y="52" font-size="12" font-weight="700" fill="#1e293b">INT8</text>
  <rect x="80" y="44" width="440" height="10" rx="3" fill="#fffffc" stroke="#d1d5db" stroke-width="1"/>
  <rect x="80" y="44" width="440" height="10" rx="3" fill="#bdb2ff" opacity=".4"/>
  <g stroke="#4c1d95" stroke-width="1.5">
    <line x1="80"  y1="44" x2="80"  y2="60"/>
    <line x1="107" y1="47" x2="107" y2="57"/>
    <line x1="134" y1="47" x2="134" y2="57"/>
    <line x1="161" y1="47" x2="161" y2="57"/>
    <line x1="188" y1="47" x2="188" y2="57"/>
    <line x1="215" y1="47" x2="215" y2="57"/>
    <line x1="242" y1="47" x2="242" y2="57"/>
    <line x1="269" y1="47" x2="269" y2="57"/>
    <line x1="300" y1="44" x2="300" y2="62"/>
    <line x1="327" y1="47" x2="327" y2="57"/>
    <line x1="354" y1="47" x2="354" y2="57"/>
    <line x1="381" y1="47" x2="381" y2="57"/>
    <line x1="408" y1="47" x2="408" y2="57"/>
    <line x1="435" y1="47" x2="435" y2="57"/>
    <line x1="462" y1="47" x2="462" y2="57"/>
    <line x1="489" y1="47" x2="489" y2="57"/>
    <line x1="520" y1="44" x2="520" y2="60"/>
  </g>
  <text x="76"  y="74" font-size="10" fill="#64748b" text-anchor="middle">−127</text>
  <text x="300" y="74" font-size="10" fill="#64748b" text-anchor="middle">0</text>
  <text x="524" y="74" font-size="10" fill="#64748b" text-anchor="middle">127</text>
  <text x="300" y="88" font-size="10" fill="#4c1d95" text-anchor="middle" font-weight="600">256 uniformly-spaced integers — fixed step size</text>


  <!-- ── FP8 number line ── -->
  <text x="14" y="132" font-size="12" font-weight="700" fill="#1e293b">FP8</text>
  <rect x="80" y="124" width="440" height="10" rx="3" fill="#fffffc" stroke="#d1d5db" stroke-width="1"/>
  <rect x="80" y="124" width="440" height="10" rx="3" fill="#ffd6a5" opacity=".5"/>
  <g stroke="#f59e0b" stroke-width="1.5">
    <line x1="80"  y1="124" x2="80"  y2="140"/>
    <line x1="95"  y1="127" x2="95"  y2="137"/>
    <line x1="108" y1="127" x2="108" y2="137"/>
    <line x1="124" y1="127" x2="124" y2="137"/>
    <line x1="138" y1="127" x2="138" y2="137"/>
    <line x1="150" y1="127" x2="150" y2="137"/>
    <line x1="161" y1="127" x2="161" y2="137"/>
    <line x1="170" y1="127" x2="170" y2="137"/>
    <line x1="200" y1="127" x2="200" y2="137"/>
    <line x1="215" y1="127" x2="215" y2="137"/>
    <line x1="228" y1="127" x2="228" y2="137"/>
    <line x1="240" y1="127" x2="240" y2="137"/>
    <line x1="252" y1="127" x2="252" y2="137"/>
    <line x1="263" y1="127" x2="263" y2="137"/>
    <line x1="274" y1="127" x2="274" y2="137"/>
    <line x1="284" y1="127" x2="284" y2="137"/>
    <line x1="293" y1="127" x2="293" y2="137"/>
    <line x1="300" y1="124" x2="300" y2="142"/>
    <line x1="307" y1="127" x2="307" y2="137"/>
    <line x1="316" y1="127" x2="316" y2="137"/>
    <line x1="326" y1="127" x2="326" y2="137"/>
    <line x1="337" y1="127" x2="337" y2="137"/>
    <line x1="348" y1="127" x2="348" y2="137"/>
    <line x1="360" y1="127" x2="360" y2="137"/>
    <line x1="372" y1="127" x2="372" y2="137"/>
    <line x1="385" y1="127" x2="385" y2="137"/>
    <line x1="400" y1="127" x2="400" y2="137"/>
    <line x1="410" y1="127" x2="410" y2="137"/>
    <line x1="420" y1="127" x2="420" y2="137"/>
    <line x1="432" y1="127" x2="432" y2="137"/>
    <line x1="444" y1="127" x2="444" y2="137"/>
    <line x1="458" y1="127" x2="458" y2="137"/>
    <line x1="476" y1="127" x2="476" y2="137"/>
    <line x1="496" y1="127" x2="496" y2="137"/>
    <line x1="520" y1="124" x2="520" y2="140"/>
  </g>
  <text x="76"  y="154" font-size="10" fill="#64748b" text-anchor="middle">−448</text>
  <text x="300" y="154" font-size="10" fill="#64748b" text-anchor="middle">0</text>
  <text x="524" y="154" font-size="10" fill="#64748b" text-anchor="middle">448</text>
  <text x="300" y="168" font-size="10" fill="#78350f" text-anchor="middle" font-weight="600">dense near zero, sparse at extremes — logarithmic spacing</text>


  <!-- callout box -->
  <rect x="50" y="180" width="520" height="24" rx="5" fill="#ffd6a5" stroke="#f59e0b" stroke-width="1"/>
  <text x="310" y="196" font-size="11" fill="#78350f" text-anchor="middle" font-weight="600">FP8 e4m3 covers ±448 vs. INT8's ±127 — 3.5× wider range with the same 8 bits</text>
</svg>


Look at the two tracks side by side. INT8's ticks are evenly spaced — a fixed step from one end to the other. FP8's ticks bunch up near zero and fan out toward the edges. That uneven spacing is the entire point: neural network activations and gradients routinely span 0.0003 and 180.0 *inside the same tensor*, and floating point packs its precision exactly where most of the values are — near zero — while still reaching out to ±448.


Now we can see the trap INT8 sets. With a uniform grid you must pick one scale, and it forces a losing choice: stretch the scale to cover the spikes and the dense region near zero turns into a coarse staircase; tighten it to the dense region and the spikes overflow. FP8 dodges the choice structurally because its grid is already dense where it needs to be. Same 8 bits — but the cyan-vs-amber spacing in the figure is why FP8 exists.


### Why two flavors of FP8 instead of one?


*8 bits can't be both wide and precise at once — so which job gets which split?*


<div style="display:flex; gap:14px; margin:20px 0; flex-wrap:wrap; font-family:system-ui,sans-serif; font-size:13px;">


  <!-- e4m3 card -->
  <div style="flex:1; min-width:230px; background:#9bf6ff; border:1px solid #67e8f9; border-radius:8px; padding:16px;">
    <div style="font-weight:700; color:#164e63; text-transform:uppercase; letter-spacing:.05em; margin-bottom:10px; font-size:11px;">FP8 e4m3 — precision variant</div>
    <div style="display:flex; height:26px; border-radius:4px; overflow:hidden; margin-bottom:12px; border:1px solid #67e8f9;">
      <div style="flex:1; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">S</div>
      <div style="flex:4; background:#a0c4ff; color:#1e3a8a; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">Exp (4)</div>
      <div style="flex:3; background:#caffbf; color:#14532d; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:10px;">Man (3)</div>
    </div>
    <ul style="list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:5px;">
      <li style="font-size:12px;">Max value: <strong>448</strong></li>
      <li style="font-size:12px;">8 mantissa levels (2³) — finer quantization steps</li>
      <li style="font-size:12px; color:#164e63; font-weight:600; margin-top:4px;">→ Weights &amp; activations (forward pass)</li>
    </ul>
  </div>


  <!-- e5m2 card -->
  <div style="flex:1; min-width:230px; background:#ffc6ff; border:1px solid #f0abfc; border-radius:8px; padding:16px;">
    <div style="font-weight:700; color:#581c87; text-transform:uppercase; letter-spacing:.05em; margin-bottom:10px; font-size:11px;">FP8 e5m2 — range variant</div>
    <div style="display:flex; height:26px; border-radius:4px; overflow:hidden; margin-bottom:12px; border:1px solid #f0abfc;">
      <div style="flex:1; background:#ffadad; color:#7f1d1d; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">S</div>
      <div style="flex:5; background:#a0c4ff; color:#1e3a8a; display:flex; align-items:center; justify-content:center; border-right:2px solid #fff; font-weight:700; font-size:10px;">Exp (5)</div>
      <div style="flex:2; background:#caffbf; color:#14532d; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:10px;">Man (2)</div>
    </div>
    <ul style="list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:5px;">
      <li style="font-size:12px;">Max value: <strong>57,344</strong></li>
      <li style="font-size:12px;">4 mantissa levels (2²) — coarser, but huge range</li>
      <li style="font-size:12px; color:#581c87; font-weight:600; margin-top:4px;">→ Gradients (backward pass)</li>
    </ul>
  </div>


</div>


The two cards run the same exponent-vs-mantissa trade we saw at 16 bits, now squeezed into 8. **e4m3** keeps a 3-bit mantissa for finer steps and tops out at 448 — exactly what weights and activations want, since they sit in a moderate band and care about precision. **e5m2** spends that mantissa bit on a fifth exponent bit, reaching 57,344 — exactly what gradients want, since they spike unpredictably and care about range. The standard H100 FP8 recipe pairs **e4m3 forward, e5m2 backward** for that reason. The split is the same lesson as BF16, just applied per-direction.


### How does FP4 stay usable with only 16 values?


*Four bits is sixteen numbers total — surely that's too coarse for a weight matrix?*


<div style="background:#fffffc; border:1px solid #d1d5db; border-radius:8px; padding:16px; margin:16px 0; font-family:system-ui,sans-serif; font-size:13px;">
  <div style="font-weight:700; color:#1e293b; margin-bottom:14px; font-size:12px; text-transform:uppercase; letter-spacing:.05em;">Microscaling (MX) — how it works</div>


  <!-- without MX -->
  <div style="margin-bottom:12px;">
    <div style="font-size:11px; color:#64748b; margin-bottom:6px; font-weight:600;">WITHOUT microscaling — per-element exponent (wastes bits)</div>
    <div style="display:flex; gap:3px; flex-wrap:wrap;">
      <div style="display:flex; flex-direction:column; align-items:center; gap:2px;" title="element 1">
        <div style="background:#ffd6a5; border:1px solid #f59e0b; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#78350f;">E</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">M</div>
      </div>
      <div style="display:flex; flex-direction:column; align-items:center; gap:2px;">
        <div style="background:#ffd6a5; border:1px solid #f59e0b; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#78350f;">E</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">M</div>
      </div>
      <div style="display:flex; flex-direction:column; align-items:center; gap:2px;">
        <div style="background:#ffd6a5; border:1px solid #f59e0b; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#78350f;">E</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">M</div>
      </div>
      <div style="display:flex; flex-direction:column; align-items:center; gap:2px;">
        <div style="background:#ffd6a5; border:1px solid #f59e0b; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#78350f;">E</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:3px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">M</div>
      </div>
      <span style="color:#64748b; font-size:10px; align-self:center; padding:0 4px;">… ×32</span>
      <span style="font-size:11px; color:#7f1d1d; align-self:center; padding-left:8px;">32 exponents × 8 bits each = 256 bits overhead</span>
    </div>
  </div>


  <!-- with MX -->
  <div>
    <div style="font-size:11px; color:#64748b; margin-bottom:6px; font-weight:600;">WITH microscaling — shared exponent per block (MXFP4)</div>
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
      <div style="background:#ffd6a5; border:1px solid #f59e0b; border-radius:4px; padding:6px 10px; font-size:11px; font-weight:700; color:#78350f;">1 shared exp<br>(FP8 · 8 bits)</div>
      <span style="font-size:18px; color:#64748b;">→</span>
      <div style="display:flex; gap:2px; flex-wrap:wrap;">
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:4px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">v₁<br>4b</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:4px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">v₂<br>4b</div>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:4px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">v₃<br>4b</div>
        <span style="color:#64748b; font-size:10px; align-self:center; padding:0 2px;">…</span>
        <div style="background:#a0c4ff; border:1px solid #60a5fa; border-radius:3px; padding:4px 5px; font-size:9px; font-weight:700; color:#1e3a8a;">v₃₂<br>4b</div>
      </div>
      <span style="font-size:11px; color:#14532d; font-weight:600; margin-left:4px;">8 bits amortized over 32 values = 0.25 bits/value overhead</span>
    </div>
  </div>
</div>


Naive FP4 really is too coarse — 16 levels can't span a weight matrix on their own. The fix in the lower row is to stop storing an exponent per element and instead share **one FP8 exponent across a block of 32 values**. Each element keeps only its 4-bit mantissa; the block-level exponent slides the whole group up or down the number line. Amortized, that scale costs $8/32 = 0.25$ bits per value instead of a full per-element exponent. The 4 bits now spend themselves entirely on precision *within* a range the shared exponent already positioned.


| Format | Bits | Scaling                              | Hardware                | Status (2026)     |
| ------ | ---- | ------------------------------------ | ----------------------- | ----------------- |
| NVFP4  | 4    | Per-tensor / per-block (proprietary) | Blackwell B100/B200     | Production        |
| MXFP4  | 4    | Block shared exponent (OCP standard) | Blackwell; partial H100 | Emerging standard |
| MXFP8  | 8    | Block shared exponent                | H100+, Blackwell        | Available         |


> **Interview signal.** FP4 on Blackwell is the successor to FP8 on Hopper, and microscaling — not a clever rounding trick — is what enables it. Without a block-level shared exponent, 4-bit floating point is too coarse to use.
{: .prompt-tip }


We now know exactly how many bits a weight costs. But bits on paper aren't bytes in VRAM at serving time — so let's count what actually has to fit on the GPU.


---


## 2. Where does the VRAM actually go?


*A 7B model is "7 GB in INT8" — so why does it OOM at batch 8?*


Two quantities fight for VRAM. **Weight memory** is fixed the moment the model loads; **KV cache** grows with every token of every request in the batch. The "7 GB" number only describes the first one — and that's the trap.


### How big are the weights?


*This part is mechanical — bytes per param times param count. So why bother with a figure?*


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #60a5fa;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#1e3a8a;margin-bottom:8px;">Formula · Weight memory</div>


$$
\text{mem}_\text{weights} = N_\text{params} \times b_\text{dtype}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $b_\text{dtype}$ in bytes — FP32 = 4 · BF16/FP16 = 2 · INT8/FP8 = 1 · W4 ≈ 0.5
  </div>
</div>


| Format          | Bytes/param | Qwen2-7B (7B) | Qwen2-70B (70B) |
| --------------- | ----------- | ------------- | --------------- |
| FP32            | 4           | 28 GB         | 280 GB          |
| BF16 / FP16     | 2           | 14 GB         | 140 GB          |
| INT8 / FP8      | 1           | 7 GB          | 70 GB           |
| **W4 + scales** | **≈0.53**   | **~3.7 GB**   | **~37 GB**      |


The table is the easy half of the budget — each row just halves the one above it. The W4 row carries a small asterisk: group quantization (group_size = 128) stores an FP32 scale per group, $\tfrac{N}{128}\times 4$ bytes, about **218 MB** for a 7B model. Negligible against the weights, which is why we round W4 to ~0.5 bytes/param and move on.


> **VRAM trap.** AWQ and GPTQ need the full FP16 model resident *before* quantization starts — 14 GB plus Hessian buffers means ~16–18 GB just to quantize a 7B. HQQ skips the calibration pass and can offload to CPU, which is the only practical route for 70B on a single node.
{: .prompt-warning }


So weights are bounded and predictable. The reason the 7 GB model still OOMs is the *other* term — the one that scales with traffic.


### Why does the KV cache dwarf the weights?


*Weights are fixed at load — so what grows until it eats the GPU?*


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #60a5fa;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#1e3a8a;margin-bottom:8px;">Formula · KV cache</div>


$$
\text{KV} = 2 \times B \times S \times L \times H \times b_\text{dtype}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $B$ = batch size · $S$ = sequence length (context + generated) · $L$ = layers · $H = n_\text{kv-heads}\times d_\text{head}$ · the leading <strong>2</strong> = one K tensor + one V tensor per layer. For Qwen2-7B ($L{=}32$, $n_\text{kv}{=}32$, $d_\text{head}{=}128$): $H = 4{,}096$.
  </div>
</div>


<div style="background:#fffffc; border:1px solid #d1d5db; border-radius:8px; padding:16px; margin:20px 0; font-family:system-ui,sans-serif; font-size:13px; overflow-x:auto;">
  <div style="font-weight:700; color:#78350f; margin-bottom:12px; font-size:13px;">KV cache vs. W4 weights — Qwen2-7B (W4 ≈ 3.7 GB constant)</div>
  <table style="border-collapse:collapse; width:100%; font-size:12px; min-width:500px;">
    <thead>
      <tr style="background:#ffd6a5; color:#78350f;">
        <th style="padding:7px 10px; text-align:left; border:1px solid #f59e0b; font-weight:700;">Config</th>
        <th style="padding:7px 10px; text-align:right; border:1px solid #f59e0b; font-weight:700;">KV BF16</th>
        <th style="padding:7px 10px; text-align:right; border:1px solid #f59e0b; font-weight:700;">KV INT8</th>
        <th style="padding:7px 10px; text-align:right; border:1px solid #f59e0b; font-weight:700;">KV INT4</th>
        <th style="padding:7px 10px; text-align:right; border:1px solid #f59e0b; font-weight:700;">KV / W4 weights</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px 10px; border:1px solid #f59e0b;">B=1, S=4K</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">2.1 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">1.1 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">0.5 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; color:#64748b;">0.6×</td>
      </tr>
      <tr style="background:#fdffb6;">
        <td style="padding:6px 10px; border:1px solid #f59e0b; font-weight:700;">B=8, S=4K</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; font-weight:700; color:#78350f;">16.8 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">8.4 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">4.2 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; font-weight:700; color:#78350f;">4.5×</td>
      </tr>
      <tr style="background:#ffadad;">
        <td style="padding:6px 10px; border:1px solid #f59e0b; font-weight:700;">B=8, S=16K</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; font-weight:700; color:#7f1d1d;">67.1 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">33.6 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">16.8 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; font-weight:700; color:#7f1d1d;">18×</td>
      </tr>
      <tr>
        <td style="padding:6px 10px; border:1px solid #f59e0b;">B=4, S=8K</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">16.8 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">8.4 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right;">4.2 GB</td>
        <td style="padding:6px 10px; border:1px solid #f59e0b; text-align:right; font-weight:700; color:#78350f;">4.5×</td>
      </tr>
    </tbody>
  </table>
</div>


Walk down the rightmost column — that's the KV cache measured in units of the entire W4 weight set. At B=1 it's a rounding error (0.6×). Bump the batch to 8 and it's already **4.5× the weights**; stretch the context to 16K and the yellow row goes red at **18×**. The weights never moved from 3.7 GB; every byte of that growth is the KV cache, because the formula is linear in both $B$ and $S$ and there's nothing sublinear to save you. That's the answer to the opening question — the "7 GB" model OOMs at batch 8 because the cache, not the weights, ran away.


> **Critical rule.** Always compute the KV footprint at your *target* batch and sequence length before picking a compression method. Weight quantization does nothing to the KV cache — at B=8, S=16K the cache is 18× the W4 weights, so long-context serving needs KV quantization (INT8/INT4 KV) as a separate, deliberate step.
{: .prompt-danger }


One caveat the formula hides: it assumes Multi-Head Attention where $n_\text{kv} = n_\text{q}$. Modern models cut KV heads with GQA/MQA, or drop the cache entirely with MLA (DeepSeek) — each shrinks $H$ and pulls the whole right column down.


We can now size the bytes exactly. But two configs with identical VRAM can have wildly different throughput — because the last question isn't *how many* bytes, it's *how fast they move and where they sit*.


---


## 3. Which bottleneck are you actually hitting?


*Why does INT8 buy throughput at batch=1 but do nothing at batch=64?*


Every GPU op is a conversation between two memories, and the gap between them is the most important number in serving.


### What's the gap between HBM and SRAM?


*Two memories, one ~10× faster than the other — which one are we waiting on?*


<div style="font-family:system-ui,sans-serif; margin:20px 0; font-size:13px;">
  <div style="display:flex; gap:12px; align-items:stretch; flex-wrap:wrap;">


    <!-- HBM -->
    <div style="flex:3; min-width:260px; background:#ffd6a5; border:2px solid #f59e0b; border-radius:10px; padding:18px 20px;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#78350f; margin-bottom:8px;">HBM — High Bandwidth Memory</div>
      <div style="font-size:22px; font-weight:700; color:#78350f; margin-bottom:2px;">40–192 GB</div>
      <div style="font-size:13px; color:#78350f; margin-bottom:14px;">2–3.35 TB/s on H100 SXM</div>
      <div style="font-size:12px; color:#78350f; font-weight:600; margin-bottom:6px;">What lives here permanently:</div>
      <ul style="margin:0; padding-left:16px; display:flex; flex-direction:column; gap:4px; font-size:12px; color:#78350f;">
        <li><strong>Model weights</strong> — all ~7B parameters of a 7B model (e.g. 3.7 GB in W4)</li>
        <li><strong>KV cache</strong> — every key/value tensor for every layer of every in-flight request</li>
        <li><strong>Optimizer states</strong> — during training: 2nd-moment Adam buffers, fp32 master weights</li>
        <li><strong>Activations</strong> — intermediate tensors during the forward/backward pass</li>
      </ul>
      <div style="margin-top:12px; font-size:11px; color:#78350f; background:#fffffc; border-radius:6px; padding:8px 10px; line-height:1.5;">
        <strong>The bottleneck.</strong> At batch=1, each weight row is loaded once from HBM to compute one output. Throughput is capped by how fast bytes arrive — not by how fast the GPU can do math.
      </div>
    </div>


    <!-- arrow -->
    <div style="display:flex; align-items:center; justify-content:center; font-size:28px; color:#94a3b8; flex-shrink:0;">⇄</div>


    <!-- SRAM -->
    <div style="flex:2; min-width:220px; background:#caffbf; border:2px solid #86efac; border-radius:10px; padding:18px 20px;">
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#14532d; margin-bottom:8px;">SRAM — On-Chip Shared Memory</div>
      <div style="font-size:22px; font-weight:700; color:#14532d; margin-bottom:2px;">~50–80 MB</div>
      <div style="font-size:13px; color:#14532d; margin-bottom:14px;">~20–30 TB/s · 10× faster than HBM</div>
      <div style="font-size:12px; color:#14532d; font-weight:600; margin-bottom:6px;">What happens here:</div>
      <ul style="margin:0; padding-left:16px; display:flex; flex-direction:column; gap:4px; font-size:12px; color:#14532d;">
        <li><strong>Active tile computation</strong> — the current matrix multiply block being processed</li>
        <li><strong>FlashAttention tiles</strong> — Q/K/V tiles reused in-register, never spilled back to HBM</li>
        <li><strong>Softmax accumulators</strong> — running max and sum kept on-chip to avoid re-reads</li>
      </ul>
      <div style="margin-top:12px; font-size:11px; color:#14532d; background:#fffffc; border-radius:6px; padding:8px 10px; line-height:1.5;">
        <strong>The goal.</strong> Kernel design is about maximizing SRAM reuse — data loaded from HBM once should be used as many times as possible before the next HBM fetch.
      </div>
    </div>


  </div>


  <!-- bandwidth callout -->
  <div style="display:flex; align-items:center; gap:10px; margin-top:12px; padding:10px 14px; background:#fffffc; border:1px solid #d1d5db; border-radius:8px;">
    <span style="font-size:20px;">📐</span>
    <span style="font-size:12px; color:#1e293b; line-height:1.5;">
      <strong>The gap that defines LLM serving:</strong> SRAM is ~10× faster than HBM but ~1,000× smaller. Every optimization — FlashAttention, quantization, KV compression — is ultimately about crossing this gap fewer times or with fewer bytes.
    </span>
  </div>
</div>


The two cards are sized to scale: HBM holds 40–192 GB but feeds at only ~3 TB/s, while SRAM holds a few dozen MB and feeds at ~20–30 TB/s. Everything heavy — weights, KV cache, activations — lives in the slow, roomy left card; only the tile currently being multiplied fits in the fast, tiny right one. So the question "which bottleneck?" reduces to: *for this workload, is the GPU waiting on the orange card or busy in the green one?*


|                 | HBM                              | SRAM                |
| --------------- | -------------------------------- | ------------------- |
| Capacity        | 40–192 GB                        | 50–80 MB total      |
| Bandwidth       | 2–3.35 TB/s                      | ~20–30 TB/s         |
| Latency         | ~600 cycles                      | ~30 cycles          |
| Holds           | Weights · KV cache · activations | Active compute tile |
| Bottleneck when | bandwidth-bound (small batch)    | —                   |


One number decides which card you're stuck on. Let's name it.


### What does arithmetic intensity tell us?


*Is this workload starved for bytes, or starved for math?*


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #60a5fa;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#1e3a8a;margin-bottom:8px;">Formula · Arithmetic intensity</div>


$$
\text{arithmetic intensity} = \frac{\text{FLOPs executed}}{\text{bytes loaded from HBM}}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    Below the hardware roofline → memory-bound (waiting on bytes). Above it → compute-bound (waiting on tensor cores). H100 roofline = 1,979 TFLOPS ÷ 3.35 TB/s ≈ <strong>590 FLOPs/byte</strong>.
  </div>
</div>


<div style="background:#fffffc; border:1px solid #d1d5db; border-radius:10px; padding:18px 20px; margin:20px 0; font-family:system-ui,sans-serif; font-size:13px;">
  <div style="font-weight:700; color:#1e293b; margin-bottom:4px; font-size:13px;">Arithmetic intensity across batch sizes</div>
  <div style="font-size:11px; color:#94a3b8; margin-bottom:14px;">Qwen2-7B · hidden = 4096 · H100 SXM</div>


  <!-- roofline marker -->
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px; padding-bottom:12px; border-bottom:1px dashed #bdb2ff;">
    <span style="width:76px; text-align:right; font-size:11px; color:#4c1d95; font-weight:700; flex-shrink:0;">Roofline</span>
    <div style="flex:1; height:3px; background:linear-gradient(90deg,#bdb2ff,#4c1d95); border-radius:2px;"></div>
    <span style="font-size:11px; color:#4c1d95; font-weight:700; white-space:nowrap; flex-shrink:0;">~590 FLOPs/byte (H100 BF16)</span>
  </div>


  <!-- rows -->
  <div style="display:flex; flex-direction:column; gap:8px;">


    <!-- batch 1 -->
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="width:76px; text-align:right; font-weight:600; font-size:12px; color:#334155; flex-shrink:0;">batch = 1</span>
      <div style="flex:1; background:#f1f5f9; border-radius:4px; height:18px; overflow:hidden;">
        <div style="width:0.17%; height:100%; background:#ffadad; border-radius:4px; min-width:4px;"></div>
      </div>
      <span style="width:110px; font-size:11px; color:#64748b; flex-shrink:0;">~1 FLOPs/byte</span>
      <span style="background:#ffadad; color:#7f1d1d; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #fca5a5;">memory-bound · W4 wins</span>
    </div>


    <!-- batch 4 -->
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="width:76px; text-align:right; font-weight:600; font-size:12px; color:#334155; flex-shrink:0;">batch = 4</span>
      <div style="flex:1; background:#f1f5f9; border-radius:4px; height:18px; overflow:hidden;">
        <div style="width:0.68%; height:100%; background:#ffd6a5; border-radius:4px; min-width:7px;"></div>
      </div>
      <span style="width:110px; font-size:11px; color:#64748b; flex-shrink:0;">~4 FLOPs/byte</span>
      <span style="background:#ffd6a5; color:#78350f; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #f59e0b;">still memory-bound</span>
    </div>


    <!-- batch 16 -->
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="width:76px; text-align:right; font-weight:600; font-size:12px; color:#334155; flex-shrink:0;">batch = 16</span>
      <div style="flex:1; background:#f1f5f9; border-radius:4px; height:18px; overflow:hidden;">
        <div style="width:2.7%; height:100%; background:#fdffb6; border-radius:4px;"></div>
      </div>
      <span style="width:110px; font-size:11px; color:#64748b; flex-shrink:0;">~16 FLOPs/byte</span>
      <span style="background:#fdffb6; color:#713f12; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #fde68a;">transitioning…</span>
    </div>


    <!-- batch 64 -->
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="width:76px; text-align:right; font-weight:600; font-size:12px; color:#334155; flex-shrink:0;">batch = 64</span>
      <div style="flex:1; background:#f1f5f9; border-radius:4px; height:18px; overflow:hidden;">
        <div style="width:10.8%; height:100%; background:#caffbf; border-radius:4px;"></div>
      </div>
      <span style="width:110px; font-size:11px; color:#64748b; flex-shrink:0;">~64 FLOPs/byte</span>
      <span style="background:#caffbf; color:#14532d; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #86efac;">compute-bound · FP8 wins</span>
    </div>


    <!-- batch 256 -->
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="width:76px; text-align:right; font-weight:600; font-size:12px; color:#334155; flex-shrink:0;">batch = 256</span>
      <div style="flex:1; background:#f1f5f9; border-radius:4px; height:18px; overflow:hidden;">
        <div style="width:43.4%; height:100%; background:#caffbf; border-radius:4px;"></div>
      </div>
      <span style="width:110px; font-size:11px; color:#64748b; flex-shrink:0;">~256 FLOPs/byte</span>
      <span style="background:#caffbf; color:#14532d; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; border:1px solid #86efac;">heavily compute-bound</span>
    </div>


  </div>
</div>


Read the bars against the dashed roofline. At **batch=1** the bar is a sliver — intensity ≈ 1 FLOP/byte, which is *1/590th* of what the H100's tensor cores could chew through. Each weight matrix gets streamed from HBM exactly once to produce a single output vector, so the GPU spends nearly all its time waiting on the orange card. That's why the batch=1 badge says **W4 wins**: halving the bytes nearly doubles decode throughput, because bytes are the whole bottleneck.


Now follow the bars right. Each batch increment reuses the same weights across more rows, so intensity climbs ~linearly: by **batch=64** the bar pushes into the green and the workload flips **compute-bound**. Here INT8 does nothing — it shrinks bytes you're no longer waiting on. What helps now is **FP8**, which doubles the tensor-core TFLOPS (the right card's throughput), or a bigger batch to amortize the HBM traffic you've already paid for. Same model, opposite tool, and the only thing that changed is where the bar sits relative to the roofline.


| Regime            | When             | Intensity           | Bottleneck         | What actually helps           |
| ----------------- | ---------------- | ------------------- | ------------------ | ----------------------------- |
| **Memory-bound**  | decode batch ≤ 4 | ~1–4 FLOPs/byte     | HBM bytes/sec      | W4/W8 (fewer bytes to stream) |
| **Compute-bound** | batch ≥ 16–32+   | hundreds FLOPs/byte | Tensor core TFLOPS | FP8 (2× TFLOPS on H100)       |


That's the full chain — bits, bytes, and the bottleneck. Before we compose them into a checklist, here's the field guide for when you hit each symptom in the wild.


---


## 4. Cheatsheet


*When you hit one of these in practice, what does it mean — and what's the move?*


<div style="margin:16px 0;font-family:system-ui,sans-serif;font-size:13px;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">


  <!-- header -->
  <div style="display:grid;grid-template-columns:2fr 3fr;background:#f1f5f9;">
    <div style="padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">What you see</div>
    <div style="padding:8px 12px;border-left:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">What it usually means</div>
  </div>


  <!-- row: NaN -->
  <div style="display:grid;grid-template-columns:2fr 3fr;background:#ffadad;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;color:#7f1d1d;">Loss goes to <code>NaN</code> a few thousand steps into training</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #f59e0b;color:#7f1d1d;">FP16 overflow past 65,504. Switch to BF16 — same range as FP32, mantissa noise SGD absorbs.</div>
  </div>


  <!-- row: OOM -->
  <div style="display:grid;grid-template-columns:2fr 3fr;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;">Model is "7 GB in INT8" but OOMs at batch 8</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #e2e8f0;">KV cache, not weights. It scales with $B\times S$ — recompute it at your real batch and context first.</div>
  </div>


  <!-- row: no speedup -->
  <div style="display:grid;grid-template-columns:2fr 3fr;background:#fdffb6;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;color:#713f12;">Quantized to W4, memory dropped, but tokens/s barely moved</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #fde68a;color:#713f12;">You're compute-bound (large batch). W4 only helps the memory-bound regime; reach for FP8 instead.</div>
  </div>


  <!-- row: long context -->
  <div style="display:grid;grid-template-columns:2fr 3fr;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;">Long-context serving blows the VRAM budget despite W4 weights</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #e2e8f0;">Weight quant never touches the KV cache. Add INT8/INT4 KV quantization as a separate step.</div>
  </div>


  <!-- row: FP8 directions -->
  <div style="display:grid;grid-template-columns:2fr 3fr;background:#caffbf;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;color:#14532d;">FP8 training run, choosing a format per direction</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #86efac;color:#14532d;">e4m3 forward (precision for weights/activations), e5m2 backward (range for spiking gradients).</div>
  </div>


  <!-- row: naive FP4 -->
  <div style="display:grid;grid-template-columns:2fr 3fr;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;">Naive FP4 wrecks accuracy</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #e2e8f0;">16 levels is too coarse alone. Use microscaling (MXFP4/NVFP4) — a shared block exponent at 0.25 bits/value.</div>
  </div>


  <!-- row: AWQ/GPTQ OOM -->
  <div style="display:grid;grid-template-columns:2fr 3fr;border-top:1px solid #e2e8f0;">
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;">AWQ/GPTQ OOMs <em>during</em> quantization on a 70B</div>
    <div style="padding:7px 12px;min-width:0;overflow-wrap:break-word;border-left:1px solid #e2e8f0;">They need the full FP16 model + Hessian buffers resident. Use HQQ (no calibration pass, CPU-offloadable).</div>
  </div>


</div>


---


## Putting It All Together


These three models collapse into one decision checklist for every compression project:


<div style="background:#fffffc; border:1px solid #d1d5db; border-radius:8px; padding:16px; margin:20px 0; font-family:system-ui,sans-serif; font-size:13px;">
  <div style="display:flex; flex-direction:column; gap:12px;">


    <div style="display:flex; gap:12px; align-items:flex-start;">
      <div style="background:#a0c4ff; color:#1e3a8a; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; flex-shrink:0; margin-top:1px;">1</div>
      <div>
        <div style="font-weight:700; color:#1e293b; margin-bottom:2px;">Pick your dtype</div>
        <div style="color:#64748b; font-size:12px;">BF16 for training. FP8 e4m3 for forward pass. FP8 e5m2 for gradients. W4 + INT8 KV for memory-constrained serving.</div>
      </div>
    </div>


    <div style="display:flex; gap:12px; align-items:flex-start;">
      <div style="background:#a0c4ff; color:#1e3a8a; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; flex-shrink:0; margin-top:1px;">2</div>
      <div>
        <div style="font-weight:700; color:#1e293b; margin-bottom:2px;">Budget your VRAM</div>
        <div style="color:#64748b; font-size:12px;">Compute: weights + KV cache at (B, S) + ~1 GB framework overhead. KV grows linearly with B and S — check it before choosing your GPU tier.</div>
      </div>
    </div>


    <div style="display:flex; gap:12px; align-items:flex-start;">
      <div style="background:#a0c4ff; color:#1e3a8a; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; flex-shrink:0; margin-top:1px;">3</div>
      <div>
        <div style="font-weight:700; color:#1e293b; margin-bottom:2px;">Diagnose the bound</div>
        <div style="color:#64748b; font-size:12px;">Small batch decode → memory-bound → reduce bytes (weight quant). Large batch inference / training → compute-bound → increase FLOP efficiency (FP8, tensor parallelism).</div>
      </div>
    </div>


  </div>
</div>


Pick the dtype, budget the bytes at your real batch and context, then read the roofline to know which lever even matters — that's the whole loop, and every method later in the curriculum is just a sharper tool for one of these three steps.



