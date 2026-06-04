---
layout: post
title: "LLM Memory 101: Floating-Point Formats, Memory Arithmetic, and GPU Hierarchy"
date: 2026-05-25 00:00:00 +0000
categories: [Model Compression]
tags: [quantization, floating-point, bfloat16, fp8, fp4, kv-cache, gpu-memory, arithmetic-intensity]
math: true
description: "A ground-up guide to floating-point formats, GPU memory arithmetic, and KV cache sizing — the three mental models every LLM compression practitioner needs before touching a quantization knob."
---


**This post answers:**
- Why does BF16 avoid overflow while FP16 doesn't, even though both are 16 bits?
- How much VRAM does a 7B model actually need at serving time — weights plus KV cache?
- Why does KV cache memory grow linearly with batch size and sequence length — and can easily dwarf the weights?
- Why does INT8 help at batch=1 but FP8 is the right tool at batch=64?
- What is microscaling and why does it make FP4 practical on Blackwell?


Before you touch a single quantization knob, three mental models need to be solid: **how weights are stored in bits**, **how much memory they demand at serving time**, and **where that memory lives on the GPU**. Everything else in model compression follows from these.


---


## 1. How Numerical Values Are Represented


Every parameter in a neural network is a binary number. IEEE 754 defines the layout as three fields:


$$
\text{value} = (-1)^{S} \times 2^{\,E - \text{bias}} \times (1 + M)
$$


- **S** (1 bit) — sign
- **E** (exponent bits) — controls *dynamic range*: how large or small numbers can get
- **M** (mantissa bits) — controls *precision*: how many distinct values fit in a given range


The design tension is constant: **more exponent bits → wider range, fewer mantissa bits → coarser precision**.


### FP32 / FP16 / BF16


The three formats differ entirely in how they split their bit budget.


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


**The FP16 overflow story.** A 5-bit exponent caps FP16 at 65,504. LLM gradient norms routinely exceed this after thousands of training steps. When a value overflows to `NaN`, the run corrupts silently and never recovers. BF16 keeps the 8-bit exponent (same ceiling as FP32) but trims the mantissa to 7 bits. Stochastic gradient descent averages over millions of updates — it doesn't need 7 decimal digits of precision per step.


| Format | Bits | Exp | Man | Max value  | Primary use                              |
| ------ | ---- | --- | --- | ---------- | ---------------------------------------- |
| FP32   | 32   | 8   | 23  | ≈3.4×10³⁸  | Master weights, full-precision reference |
| FP16   | 16   | 5   | 10  | **65,504** | Legacy (avoid for LLM training)          |
| BF16   | 16   | 8   | 7   | ≈3.4×10³⁸  | Default training + inference (2026)      |
| INT8   | 8    | —   | —   | 127        | PTQ compressed weights                   |
| INT4   | 4    | —   | —   | 7          | W4A16 deployment                         |


> **2026 default.** `torch.autocast`, HuggingFace Accelerate, and vLLM all target BF16. FP16 training is legacy. If you see a codebase still using `fp16=True`, flag it.
{: .prompt-info }


### FP8 — Why Not Just Use INT8?


INT8 is already a mature format — 1 byte per value, hardware support everywhere. Why does FP8 exist?


The answer is **dynamic range**. INT8 maps 256 evenly-spaced integers between −127 and 127. That uniform grid is fine for weights whose values happen to cluster near zero, but it fails for any tensor with a wide spread of magnitudes. Neural network activations and gradients routinely span many orders of magnitude within a single layer — values like 0.0003 and 180.0 in the same tensor.


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


The spacing between representable values in floating-point is *not* uniform. Near zero the values are packed tightly (high precision for small activations). Near the extremes they spread out (acceptable coarseness where precision matters less). This logarithmic density is exactly what neural network tensors need — most values are small, occasional spikes reach larger magnitudes.


INT8 forces you to choose: either your scale factor covers the spikes (leaving precision gaps near zero) or it covers the dense region (and the spikes overflow). FP8 sidesteps that tradeoff structurally.


### FP8 — Two Variants for Two Roles


FP8 halves BF16's footprint. But 8 bits barely cover both range and precision, so two variants exist with different exponent/mantissa splits:


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


The logic is asymmetric by design: **gradients spike unpredictably and need range**, while **weights and activations stay in a moderate band and need precision**. The standard H100 FP8 training recipe pairs e4m3 forward with e5m2 backward.


### FP4 and Microscaling — The Blackwell Bet


4 bits per element means only 16 representable values. Naive FP4 is too coarse for weights. The solution is **microscaling**: share one exponent across a *block* of 16–32 values instead of one per element.


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


| Format | Bits | Scaling                              | Hardware                | Status (2026)     |
| ------ | ---- | ------------------------------------ | ----------------------- | ----------------- |
| NVFP4  | 4    | Per-tensor / per-block (proprietary) | Blackwell B100/B200     | Production        |
| MXFP4  | 4    | Block shared exponent (OCP standard) | Blackwell; partial H100 | Emerging standard |
| MXFP8  | 8    | Block shared exponent                | H100+, Blackwell        | Available         |


> **Interview signal.** FP4 on Blackwell is the successor to FP8 on Hopper. The enabling design is microscaling — not just a quantization trick. Without block-level exponent sharing, 4-bit floating point is too coarse to be practically useful.
{: .prompt-tip }


---


## 2. The Memory Arithmetic


Once you know the format, computing memory is mechanical. Two quantities dominate: **weight memory** (fixed at load time) and **KV cache** (grows with every token and every request in the batch).


### Model Weight Memory


$$
\text{mem}_\text{weights} = N_\text{params} \times b_\text{dtype}
$$


$b_\text{dtype}$ in bytes: FP32 = 4 · BF16/FP16 = 2 · INT8/FP8 = 1 · W4 ≈ 0.5


| Format          | Bytes/param | Qwen2-7B (7B) | Qwen2-70B (70B) |
| --------------- | ----------- | ------------- | --------------- |
| FP32            | 4           | 28 GB         | 280 GB          |
| BF16 / FP16     | 2           | 14 GB         | 140 GB          |
| INT8 / FP8      | 1           | 7 GB          | 70 GB           |
| **W4 + scales** | **≈0.53**   | **~3.7 GB**   | **~37 GB**      |


Group quantization (group_size = 128) adds FP32 scale tensors: $\tfrac{N}{128} \times 4$ bytes. For a 7B model that's ~218 MB — negligible.


> **VRAM trap.** AWQ and GPTQ require the full FP16 model resident in VRAM *before* quantization begins. A 7B model needs 14 GB plus Hessian buffers: ~16–18 GB just to quantize. HQQ skips the calibration pass entirely and can offload to CPU — the only practical option for 70B on a single node.
{: .prompt-warning }


### KV Cache — The Hidden Memory Drain


Weight memory is fixed once the model loads. KV cache memory is dynamic and dominates at any real serving workload:


$$
\text{KV} = 2 \times B \times S \times L \times H \times b_\text{dtype}
$$


- $B$ = batch size
- $S$ = sequence length (context + generated tokens)
- $L$ = number of transformer layers
- $H = n_\text{kv-heads} \times d_\text{head}$
- **Factor of 2**: one K tensor + one V tensor per layer


For example with the model config: $L=32$, $n_\text{kv}=32$, $d_\text{head}=128$ → $H = 4{,}096$


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


> **Critical rule.** Always compute KV footprint at your target batch and sequence length *before* choosing a compression method. At B=8, S=16K the KV cache is **18× larger than the W4 weights**. Weight quantization does not reduce KV cache at all — for long-context serving you need KV quantization (INT8 or INT4 KV) as a separate step.
{: .prompt-danger }


The formula above assumes standard **Multi-Head Attention** where $n_\text{kv-heads} = n_\text{q-heads}$. Modern LLMs often reduce KV heads via GQA, MQA, or replace the KV cache entirely with MLA (DeepSeek) — all of which shrink the cache further.


---


## 3. GPU Memory Hierarchy


Knowing how much memory you need is one thing. Understanding where it lives and how fast it moves determines which optimizations actually matter.


### The Two-Tier Memory Model


Every GPU operation is a conversation between two memories separated by a 20× bandwidth gap. That gap is the single most important number in LLM serving performance.


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


|                 | HBM                              | SRAM                |
| --------------- | -------------------------------- | ------------------- |
| Capacity        | 40–192 GB                        | 50–80 MB total      |
| Bandwidth       | 2–3.35 TB/s                      | ~20–30 TB/s         |
| Latency         | ~600 cycles                      | ~30 cycles          |
| Holds           | Weights · KV cache · activations | Active compute tile |
| Bottleneck when | bandwidth-bound (small batch)    | —                   |


### Arithmetic Intensity — The Bottleneck Diagnostic


$$
\text{arithmetic intensity} = \frac{\text{FLOPs executed}}{\text{bytes loaded from HBM}}
$$


At **batch=1 decode**, each weight matrix is streamed from HBM exactly once to compute one output vector. You execute $\approx 2N$ FLOPs and load $\approx 2N$ bytes (BF16). Intensity ≈ **1 FLOPs/byte**.


H100's roofline ceiling: 1,979 TFLOPS ÷ 3.35 TB/s ≈ **590 FLOPs/byte**.


So batch=1 is running at 1/590th of the compute limit. The GPU is completely bottlenecked on how fast HBM can stream bytes.


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


| Regime            | When             | Intensity           | What actually helps           |
| ----------------- | ---------------- | ------------------- | ----------------------------- |
| **Memory-bound**  | decode batch ≤ 4 | ~1–4 FLOPs/byte     | W4/W8 (fewer bytes to stream) |
| **Compute-bound** | batch ≥ 16–32+   | hundreds FLOPs/byte | FP8 (2× TFLOPS on H100)       |


**Why INT8 helps at batch=1 but not batch=64.** INT8 halves the bytes streamed from HBM — so you get close to 2× decode throughput at batch=1. At batch=64 you are bottlenecked on arithmetic, not bandwidth. INT8 does not give you more FLOPs. What would help at batch=64 is FP8, which gives 2× tensor core TFLOPS on H100, or larger batch to amortize the HBM traffic you already have.


---


## Putting It All Together


These three models compose into a decision checklist for every compression project:


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





