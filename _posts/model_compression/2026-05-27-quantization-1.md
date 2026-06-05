---
layout: post
title: "Quantization Fundamentals: Memory Math, Roofline, and Calibration Granularity"
date: 2026-05-27 09:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, fundamentals]
math: true
description: "Scale, zero-point, symmetric vs. asymmetric, per-group calibration, and the roofline model — the quantization fundamentals every LLM practitioner needs before touching a compression tool."
---


<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;max-width:560px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>What exactly happens when a weight is quantized — where does the error come from?</li>
      <li>How much memory does a model use, and does quantization also speed up inference?</li>
      <li>What parts of a transformer are worth quantizing — and what should never be touched?</li>
      <li>What is calibration granularity, and why does per-group outperform per-tensor for weights?</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>IEEE 754 floating-point basics (sign, exponent, mantissa)</li>
      <li>PyTorch tensor operations</li>
      <li>Transformer architecture (linear layers, attention, FFN)</li>
      <li>Basic GPU memory concepts (VRAM, HBM bandwidth)</li>
    </ul>
  </div>
</div>


LLM weights are too large for most hardware in FP32 or BF16. Quantization replaces high-precision values with low-bit integers, making 70B models fit on two GPUs instead of eight — but the mechanism, the tradeoffs, and when it actually speeds things up require careful treatment.


---


## 1. What Is Quantization?


*Why do we need it — what breaks without it when you try to serve a 7B model on a single consumer GPU?*


The bell curve below is your weight distribution. Most values cluster near zero; a handful of outliers reach the tails. INT4 gives you 16 buckets to cover that entire range.


<!-- Bell curve + INT4 bucket mapping -->
<svg viewBox="0 0 640 275" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:660px;display:block;margin:18px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="275" fill="#fffffc" rx="10"/>


  <!-- FP32 label -->
  <text x="30" y="30" font-size="16" font-weight="800" fill="#7c3aed">FP32</text>


  <!-- Bell curve path (symmetric Gaussian, peak at x=320, y=43) -->
  <path d="M 50,205
    C 70,203 90,199 110,193
    C 130,184 150,170 170,152
    C 190,132 210,110 230,91
    C 250,74 270,61 290,53
    C 300,47 310,44 320,43
    C 330,44 340,47 350,53
    C 370,61 390,74 410,91
    C 430,110 450,132 470,152
    C 490,170 510,184 530,193
    C 550,199 570,203 590,205"
    fill="#e2e8f0" fill-opacity="0.5" stroke="#1e293b" stroke-width="2.5"/>


  <!-- FP32 dots — 8 values, symmetric about x=320.
       Boxes 3 and 4 each receive two arrows (peak region is dense). -->
  <circle cx="130" cy="184" r="7" fill="#7c3aed"/>
  <circle cx="210" cy="110" r="7" fill="#7c3aed"/>
  <circle cx="270" cy="61"  r="7" fill="#7c3aed"/>
  <circle cx="310" cy="44"  r="7" fill="#7c3aed"/>
  <circle cx="330" cy="44"  r="7" fill="#7c3aed"/>
  <circle cx="370" cy="61"  r="7" fill="#7c3aed"/>
  <circle cx="430" cy="110" r="7" fill="#7c3aed"/>
  <circle cx="510" cy="184" r="7" fill="#7c3aed"/>


  <!-- Dashed arrows: every dot connects to its nearest INT4 bucket.
       Box centers (x = 52 + i*68 + 30): 82, 150, 218, 286, 354, 422, 490, 558 -->
  <line x1="130" y1="191" x2="150" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="210" y1="117" x2="218" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="270" y1="68"  x2="286" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="310" y1="51"  x2="286" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="330" y1="51"  x2="354" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="370" y1="68"  x2="354" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="430" y1="117" x2="422" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="510" y1="191" x2="490" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>


  <!-- INT4 boxes — 8 buckets (box i: x = 52 + i*68, width=60) -->


  <!-- Box 0 — far left tail: no weights → lighter (wasted level) -->
  <rect x="52"  y="242" width="60" height="26" rx="4" fill="#fce7f3" stroke="#f9a8d4"/>


  <!-- Box 1 — left shoulder -->
  <rect x="120" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="150" cy="255" r="7" fill="#be123c"/>


  <!-- Box 2 — left slope -->
  <rect x="188" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="218" cy="255" r="7" fill="#be123c"/>


  <!-- Box 3 — left near-peak: 2 arrows in = crowded -->
  <rect x="256" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="286" cy="255" r="7" fill="#be123c"/>


  <!-- Box 4 — right near-peak: 2 arrows in = crowded -->
  <rect x="324" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="354" cy="255" r="7" fill="#be123c"/>


  <!-- Box 5 — right slope -->
  <rect x="392" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="422" cy="255" r="7" fill="#be123c"/>


  <!-- Box 6 — right shoulder -->
  <rect x="460" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="490" cy="255" r="7" fill="#be123c"/>


  <!-- Box 7 — far right tail: no weights → lighter (wasted level) -->
  <rect x="528" y="242" width="60" height="26" rx="4" fill="#fce7f3" stroke="#f9a8d4"/>


  <!-- INT4 label -->
  <text x="598" y="261" font-size="15" font-weight="800" fill="#be123c">INT4</text>
</svg>


Each purple dot is a real FP32 weight value. The dashed arrows show the core operation: **round each value to its nearest INT4 bucket**. The center boxes (3 and 4) each receive two arrows — the peak region is dense, so multiple FP32 values round to the same integer. The extreme tail boxes (lighter, no arrows) are valid INT4 levels that no weight maps to — **wasted quantization levels**. The rounding distance is the **quantization error**.


### Why We Need It


*What is the concrete hardware wall that quantization breaks through?*


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Model Size vs. dtype — 7B Model</div>
  <div style="font-size:11px;color:#94a3b8;margin-bottom:14px;">bytes/param × 7 × 10⁹ · RTX 4090 VRAM = 24 GB</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:110px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">FP32 (4 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:52px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">28 GB</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;flex-shrink:0;">no fit</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:110px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">BF16 (2 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:50%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:52px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">14 GB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;flex-shrink:0;">tight</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:110px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">INT8 (1 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:52px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">7 GB</span>
      <span style="background:#9bf6ff;color:#164e63;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #67e8f9;flex-shrink:0;">fits</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:110px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">INT4 / NF4 (0.5 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:12.5%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:52px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">3.5 GB</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;flex-shrink:0;">comfortable</span>
    </div>
  </div>
</div>


Three hard reasons to quantize: (1) **VRAM wall** — a 70B BF16 model needs 140 GB; INT4 needs 35 GB, fitting on two A100s instead of eight. (2) **Memory bandwidth is the bottleneck** during decode — at batch=1 the GPU spends most of its time waiting for weights to load from HBM, not doing arithmetic; fewer bytes per weight = higher tokens/s.


> **Central tradeoff.** You trade a small, controlled accuracy loss for a large, concrete hardware win. The art of quantization is minimizing that error — which is what the rest of this post covers.
{: .prompt-tip }


---


## 2. The Memory Wall & Roofline


### Weight Memory Economics


*Why isn't model size alone the right number to budget for — what else eats VRAM at inference?*


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Weight + KV Cache VRAM — 7B / 70B Models</div>
  <div style="font-size:11px;color:#94a3b8;margin-bottom:14px;">bytes/param × N params · FP32=4B · BF16=2B · INT8=1B · INT4=0.5B · KV cache: B=32, seq=2048, FP16</div>
  <div style="display:flex;flex-direction:column;gap:8px;">


    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:4px;">7B model</div>


    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">FP32</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">28 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">BF16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:50%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">14 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">INT8</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">7 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">INT4 / NF4</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:12.5%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">3.5 GB</span>
    </div>


    <div style="font-size:10px;color:#94a3b8;margin-top:4px;margin-bottom:-2px;font-style:italic;padding-left:2px;">KV cache — B=16, seq=2048, FP16 · L=32, 32 KV heads, d=128</div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">KV FP16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:57%;height:100%;background:#ddd6fe;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">16 GB</span>
    </div>


    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:8px;">70B model</div>


    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">BF16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">140 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">INT4 / NF4</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">35 GB</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;">fits 2×A100</span>
    </div>


    <div style="font-size:10px;color:#94a3b8;margin-top:4px;margin-bottom:-2px;font-style:italic;padding-left:2px;">KV cache — B=16, seq=2048, FP16 · L=80, 8 KV heads (GQA), d=128</div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#64748b;flex-shrink:0;">KV FP16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:7.1%;height:100%;background:#ddd6fe;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">10 GB</span>
    </div>
  </div>
</div>


The formula is simple:


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6366f1;margin-bottom:8px;">Formula · Weight Memory</div>


$$
\text{mem}_\text{weights} = N_\text{params} \times b_\text{dtype}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $N_\text{params}$ — parameter count (e.g. 7 × 10⁹ for a 7B model)<br>
    $b_\text{dtype}$ — bytes per element: FP32=4, BF16=2, INT8=1, INT4/NF4=0.5
  </div>
</div>


Weight memory is necessary but not sufficient. At inference you also carry the KV cache, which grows with batch size and sequence length and can dwarf the model weights.


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6366f1;margin-bottom:8px;">Formula · KV Cache Memory</div>


$$
\text{mem}_\text{KV} = 2 \times B \times S \times L \times H \times b_\text{dtype}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    Factor of 2 — one K tensor + one V tensor per layer<br>
    $B$ — batch size &nbsp;·&nbsp; $S$ — sequence length &nbsp;·&nbsp; $L$ — number of layers<br>
    $H$ — KV hidden size per layer = $n_\text{kv\_heads} \times d_\text{head}$ &nbsp;·&nbsp; $b_\text{dtype}$ — bytes per element<br>
  </div>
</div>


> **First check.** Before optimizing throughput, ask: does the model fit in VRAM at all, including the KV cache at your target batch and sequence length?
{: .prompt-info }


---


### Practice Question


**Q — Quantization reduces memory footprint. Does it also guarantee faster compute? Why or why not?**


<details>
<summary>💡 Hints</summary>


<div style="background:#fdffb6;border:1px solid #fcd34d;border-radius:8px;padding:14px 16px;margin-top:8px;font-family:system-ui,sans-serif;font-size:13px;line-height:1.7;">
<strong>Hint 1:</strong> At inference, is the GPU waiting for data to arrive from HBM to SRAM, or waiting for arithmetic to finish? Which operation dominates at batch=1 for a 7B model?<br><br>
<strong>Hint 2:</strong> In W4A16, the weights are stored as INT4 — but trace what dtype both operands are when the GEMM actually accumulates. Does a dequantization step happen, and where?
</div>
</details>


<details>
<summary>✅ Answer</summary>


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-top:8px;font-family:system-ui,sans-serif;font-size:13px;line-height:1.7;">


<strong>Short answer: quantization speeds up memory transport, not arithmetic.</strong>


<p style="margin-top:8px;">Reducing bits (e.g. BF16 → INT4) cuts the bytes the GPU must transfer from HBM to SRAM per weight. Since LLM decode at small batch is memory-bandwidth-bound (arithmetic intensity ~1 FLOP/byte vs. the A100 ridge at ~300), loading fewer bytes directly translates to more matrix-vector products per second. W4 loads 0.5 bytes/param vs. 2 bytes/param in BF16 — theoretically 4× the bandwidth use, producing ~2–3× real speedup (overhead closes the gap).</p>


<p>However, in W4A16, the INT4 weights are <strong>dequantized to BF16 before the GEMM</strong>. The matmul executes in BF16 on BF16 tensor cores — not on INT4 tensor cores. INT4 is a storage-only format. The speedup is entirely from reduced HBM traffic, not faster arithmetic.</p>


<p>To get genuine compute speedup you need W8A8 or FP8: both operands are integers/FP8 at GEMM time, enabling INT8/FP8 tensor cores (2–4× higher FLOP throughput than BF16). That only pays off when the workload is compute-bound (batch ≥ ~32 on A100).</p>


</div>
</details>


---


## 3. Three Quantization Targets


*Why can't you just quantize everything uniformly — what makes weights, activations, and KV cache each require a different strategy?*


### Why Linear Layers Are the Focus


A transformer forward pass is dominated by matrix multiplications. Every attention projection (Q, K, V, O) and every FFN projection (gate, up, down) is a linear layer of the form:


$$
Y = WX
$$


where $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is the weight matrix and $X \in \mathbb{R}^{d_\text{in} \times B}$ is the activation batch. These GEMMs account for over 95% of FLOPs and virtually all the parameter memory in a 7B+ model. LayerNorm, SiLU, softmax, and residual additions are elementwise operations with negligible memory footprint — there is nothing to gain from quantizing them.


Quantization replaces one or both FP operands with an integer (or lower-precision FP) representation plus a floating-point scale:


$$
Y \approx \hat{W}\hat{X}
$$


Every quantization scheme discussed below is a choice about which of $W$ and $X$ to quantize, to what bit-width, and with what granularity.


---


### Weights, Activations, and KV Cache


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Weights</div>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px;color:#334155;">
      <li>✓ Static — computed once, offline</li>
      <li>✓ No input dependency</li>
      <li>✓ No calibration data needed (per-channel)</li>
      <li style="font-weight:600;color:#4c1d95;">Format: W4 (INT4/NF4)</li>
      <li style="color:#64748b;font-size:12px;">Tools: torchao, autoawq, gptqmodel, HQQ</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Activations</div>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px;color:#334155;">
      <li>⚡ Dynamic — depend on input tokens</li>
      <li>⚡ Requires static or dynamic calibration</li>
      <li>⚡ Enables INT8/FP8 GEMMs at large batch</li>
      <li style="font-weight:600;color:#1e3a8a;">Format: INT8 (W8A8), INT8 (W4A8)</li>
      <li style="color:#64748b;font-size:12px;">Tools: torchao, bitsandbytes, llm-compressor</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">KV Cache</div>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px;color:#334155;">
      <li>📈 Grows with sequence length</li>
      <li>📈 Written + read every generation step</li>
      <li>📈 Can dwarf weight memory at long seq</li>
      <li style="font-weight:600;color:#14532d;">Format: INT8, FP8 (H100), INT4</li>
      <li style="color:#64748b;font-size:12px;">Tools: vLLM, TRT-LLM, KIVI</li>
    </ul>
  </div>
</div>


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Target</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">When computed</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Input-dependent?</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Dominant format</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Independent decision?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Weights</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Once, offline</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">No</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT4 / NF4</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
    <tr style="background:#9bf6ff20;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Activations</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">At inference</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT8 (W8A8)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
    <tr style="background:#caffbf40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">KV cache</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">At generation time</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes (grows with seq)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT8, FP8, INT4</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
  </tbody>
</table>
</div>


> **Orthogonal decisions.** W4A16 quantizes only weights; activations and KV cache stay in BF16. You can add INT8 KV cache to W4A16 without touching the weight scheme. Production deployments routinely stack all three independently.
{: .prompt-tip }


> **Embeddings and LM head.** These are weight tensors but are almost always kept at BF16. They carry quantization error into every layer (embedding) or onto every output token (LM head). They represent under 1% of parameters in 7B+ models — never worth the risk.
{: .prompt-warning }


---


## 4. Quantization Math


### Asymmetric Quantization


*Why do we need two parameters (scale and zero-point) instead of just one — what breaks with a single scale on a non-symmetric distribution?*


<!-- ── Asymmetric number-line diagram ─────────────────────────────────── -->
<svg viewBox="0 0 520 195" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:560px;display:block;margin:18px auto 4px;font-family:system-ui,sans-serif;background:#fffffc;border-radius:10px;">


  <text x="26" y="95"  font-size="15" font-weight="700" fill="#1e293b" font-style="italic">r</text>
  <text x="26" y="150" font-size="15" font-weight="700" fill="#9f1239" font-style="italic">q</text>


  <line x1="42" y1="90" x2="482" y2="90" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,86 490,90 480,94" fill="#1e293b"/>


  <rect x="60" y="44" width="400" height="26" fill="#a0c4ff" fill-opacity="0.6" rx="3"/>
  <text x="260" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#1e3a8a">Floating-point range</text>


  <line x1="60"  y1="85" x2="60"  y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <line x1="460" y1="85" x2="460" y2="95" stroke="#1e293b" stroke-width="1.4"/>


  <text x="60"  y="32" text-anchor="middle" font-size="12" fill="#1e3a8a" font-style="italic">r</text>
  <text x="68"  y="36" font-size="9"        fill="#1e3a8a">min</text>


  <line x1="220" y1="85" x2="220" y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <text x="220" y="32" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">0</text>


  <text x="448" y="32" text-anchor="middle" font-size="12" fill="#1e3a8a" font-style="italic">r</text>
  <text x="456" y="36" font-size="9"        fill="#1e3a8a">max</text>


  <line x1="60"  y1="70" x2="60"  y2="141" stroke="#9f1239" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>
  <line x1="460" y1="70" x2="460" y2="141" stroke="#9f1239" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>


  <line x1="220" y1="141" x2="220" y2="100" stroke="#1e3a8a" stroke-width="2.4"/>
  <polygon points="214,100 220,91 226,100" fill="#1e3a8a"/>


  <text x="232" y="113" font-size="13" font-weight="700" fill="#1e3a8a">&#xD7;<tspan font-style="italic">S</tspan></text>
  <text x="232" y="126" font-size="10"  fill="#1e3a8a">Floating-point</text>
  <text x="232" y="139" font-size="10"  fill="#1e3a8a">Scale</text>


  <line x1="42" y1="145" x2="482" y2="145" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,141 490,145 480,149" fill="#1e293b"/>


  <circle cx="60"  cy="145" r="5" fill="#9f1239"/>
  <circle cx="87"  cy="145" r="5" fill="#9f1239"/>
  <circle cx="113" cy="145" r="5" fill="#9f1239"/>
  <circle cx="140" cy="145" r="5" fill="#9f1239"/>
  <circle cx="167" cy="145" r="5" fill="#9f1239"/>
  <circle cx="193" cy="145" r="5" fill="#9f1239"/>
  <circle cx="220" cy="145" r="7" fill="#9f1239"/>
  <circle cx="247" cy="145" r="5" fill="#9f1239"/>
  <circle cx="273" cy="145" r="5" fill="#9f1239"/>
  <circle cx="300" cy="145" r="5" fill="#9f1239"/>
  <circle cx="327" cy="145" r="5" fill="#9f1239"/>
  <circle cx="353" cy="145" r="5" fill="#9f1239"/>
  <circle cx="380" cy="145" r="5" fill="#9f1239"/>
  <circle cx="407" cy="145" r="5" fill="#9f1239"/>
  <circle cx="433" cy="145" r="5" fill="#9f1239"/>
  <circle cx="460" cy="145" r="5" fill="#9f1239"/>


  <text x="55"  y="164" font-size="12" fill="#9f1239" font-style="italic">q</text>
  <text x="63"  y="168" font-size="9"  fill="#9f1239">min</text>
  <text x="220" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#9f1239" font-style="italic">Z</text>
  <text x="220" y="180" text-anchor="middle" font-size="10" fill="#64748b">Zero point</text>
  <text x="451" y="164" font-size="12" fill="#9f1239" font-style="italic">q</text>
  <text x="459" y="168" font-size="9"  fill="#9f1239">max</text>


</svg>


Asymmetric quantization maps a floating-point range $[x_\text{min}, x_\text{max}]$ to an integer grid using two parameters: **scale** $s$ (FP16, sets grid spacing) and **zero-point** $z$ (INT8 or FP16, shifts the grid so that FP zero maps to a specific integer). Using two parameters allows full coverage of any asymmetric range with no wasted levels — essential for ReLU activations and biases.


A single scale centered at zero wastes half the integer range when all values are positive (e.g., ReLU outputs span $[0, x_\text{max}]$, leaving the entire negative half of the grid empty). The zero-point shifts the grid to align $q_\text{min}$ with $x_\text{min}$, so every integer level maps to a real value in range.


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6366f1;margin-bottom:8px;">Formula · Asymmetric Quantization</div>


$$
s = \frac{x_\text{max} - x_\text{min}}{2^b - 1}, \qquad z = \text{round}\!\left(\frac{-x_\text{min}}{s}\right)
$$


$$
x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\ 0,\ 2^b - 1\right)
$$


$$
\hat{x} = s \cdot (x_q - z)
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $b$ — bit-width (e.g. 4 or 8) &nbsp;·&nbsp; $s$ — scale, stored as <strong>FP16</strong> &nbsp;·&nbsp; $z$ — zero-point, stored as <strong>INT8</strong> (or FP16) &nbsp;·&nbsp; $\hat{x}$ — reconstructed value
  </div>
</div>


**Worked example — unsigned INT4, range 0–15:**


Tensor: $x = [-3.2,\ 0.0,\ 1.7,\ 4.5]$


$$
s = \frac{4.5 - (-3.2)}{15} = 0.5133, \qquad z = \text{round}\!\left(\frac{3.2}{0.5133}\right) = 6
$$


<div style="overflow-x:auto;margin:12px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:7px 12px;border:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">$x$</th>
      <th style="padding:7px 12px;border:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">round($x/s$) + $z$</th>
      <th style="padding:7px 12px;border:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">$x_q$</th>
      <th style="padding:7px 12px;border:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">$\hat{x}$</th>
      <th style="padding:7px 12px;border:1px solid #e2e8f0;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Error</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:6px 12px;border:1px solid #e2e8f0;">−3.2</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">−6 + 6 = 0</td><td style="padding:6px 12px;border:1px solid #e2e8f0;font-weight:600;">0</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">−3.080</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">0.120</td></tr>
    <tr style="background:#caffbf;"><td style="padding:6px 12px;border:1px solid #e2e8f0;">0.0</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">0 + 6 = 6</td><td style="padding:6px 12px;border:1px solid #e2e8f0;font-weight:600;">6</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">0.000</td><td style="padding:6px 12px;border:1px solid #e2e8f0;color:#14532d;">0.000 ✓</td></tr>
    <tr><td style="padding:6px 12px;border:1px solid #e2e8f0;">1.7</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">3 + 6 = 9</td><td style="padding:6px 12px;border:1px solid #e2e8f0;font-weight:600;">9</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">1.540</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">0.160</td></tr>
    <tr><td style="padding:6px 12px;border:1px solid #e2e8f0;">4.5</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">9 + 6 = 15</td><td style="padding:6px 12px;border:1px solid #e2e8f0;font-weight:600;">15</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">4.620</td><td style="padding:6px 12px;border:1px solid #e2e8f0;">0.120</td></tr>
  </tbody>
</table>
</div>


The zero-point ensures FP zero maps exactly to integer 6 and dequantizes back to exactly 0.0 — the defining property of the asymmetric construction.


<div style="background:#9bf6ff;border:1px solid #67e8f9;border-left:3px solid #164e63;border-radius:0 6px 6px 0;padding:12px 16px;margin:16px 0;font-family:system-ui,sans-serif;font-size:13px;line-height:1.7;">
<strong style="color:#164e63;">Max error bound.</strong> Every in-range value satisfies <span style="white-space:nowrap;">|<em>e</em>| ≤ <em>s</em>/2</span>. Here <em>s</em>/2 = 0.257; all errors in the table above are below this bound. Smaller scale = smaller error per element, but narrower representable range — the central tradeoff calibration optimizes.
</div>


```python
import torch


def quantize_asymmetric(x: torch.Tensor, bits: int = 8):
    """Per-tensor asymmetric quantization (unsigned integer grid)."""
    q_min, q_max = 0, 2**bits - 1
    x_min, x_max = x.min().item(), x.max().item()


    scale = (x_max - x_min) / (q_max - q_min)          # FP16 in production
    zero_point = round(-x_min / scale)                   # INT8 in production
    zero_point = int(max(q_min, min(q_max, zero_point))) # clamp to int grid


    x_q = torch.clamp(torch.round(x / scale + zero_point), q_min, q_max).to(torch.int8)
    x_dq = scale * (x_q.float() - zero_point)           # dequantize → FP32


    return x_q, x_dq, scale, zero_point




# --- smoke test ---
x = torch.tensor([-3.2, 0.0, 1.7, 4.5])
x_q, x_dq, s, z = quantize_asymmetric(x, bits=4)
print(f"scale={s:.4f}  zero_point={z}")
print(f"quantized : {x_q.tolist()}")          # [0, 6, 9, 15]
print(f"dequantized: {x_dq.tolist()}")        # [-3.08, 0.0, 1.54, 4.62]
print(f"max error : {(x - x_dq).abs().max():.4f}")  # ≤ s/2 = 0.257
```


---


### Symmetric Quantization


*Why is symmetric the default for weights but a poor choice for activations?*


<!-- ── Symmetric number-line diagram ──────────────────────────────────── -->
<svg viewBox="0 0 520 195" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:560px;display:block;margin:18px auto 4px;font-family:system-ui,sans-serif;background:#fffffc;border-radius:10px;">


  <text x="26" y="95"  font-size="15" font-weight="700" fill="#1e293b" font-style="italic">r</text>
  <text x="26" y="150" font-size="15" font-weight="700" fill="#9f1239" font-style="italic">q</text>


  <line x1="42" y1="90" x2="482" y2="90" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,86 490,90 480,94" fill="#1e293b"/>


  <rect x="60" y="44" width="400" height="26" fill="#a0c4ff" fill-opacity="0.6" rx="3"/>
  <text x="260" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#1e3a8a">Floating-point range</text>


  <line x1="60"  y1="85" x2="60"  y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <line x1="460" y1="85" x2="460" y2="95" stroke="#1e293b" stroke-width="1.4"/>


  <text x="60" y="30" text-anchor="middle" font-size="12" fill="#1e3a8a">&#x2212;|r|</text>
  <text x="72" y="39" font-size="9"        fill="#1e3a8a">max</text>


  <line x1="260" y1="85" x2="260" y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <text x="260" y="32" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">0</text>


  <text x="448" y="32" text-anchor="middle" font-size="12" fill="#1e3a8a">|r|</text>
  <text x="456" y="38" font-size="9"        fill="#1e3a8a">max</text>


  <line x1="60"  y1="70" x2="60"  y2="141" stroke="#9f1239" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>
  <line x1="460" y1="70" x2="460" y2="141" stroke="#9f1239" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>


  <line x1="260" y1="141" x2="260" y2="100" stroke="#1e3a8a" stroke-width="2.4"/>
  <polygon points="254,100 260,91 266,100" fill="#1e3a8a"/>


  <text x="272" y="113" font-size="13" font-weight="700" fill="#1e3a8a">&#xD7;<tspan font-style="italic">S</tspan></text>
  <text x="272" y="126" font-size="10"  fill="#1e3a8a">Floating-point</text>
  <text x="272" y="139" font-size="10"  fill="#1e3a8a">Scale</text>


  <line x1="42" y1="145" x2="482" y2="145" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,141 490,145 480,149" fill="#1e293b"/>


  <circle cx="60"  cy="145" r="5" fill="#9f1239"/>
  <circle cx="89"  cy="145" r="5" fill="#9f1239"/>
  <circle cx="117" cy="145" r="5" fill="#9f1239"/>
  <circle cx="146" cy="145" r="5" fill="#9f1239"/>
  <circle cx="174" cy="145" r="5" fill="#9f1239"/>
  <circle cx="203" cy="145" r="5" fill="#9f1239"/>
  <circle cx="231" cy="145" r="5" fill="#9f1239"/>
  <circle cx="260" cy="145" r="7" fill="#9f1239"/>
  <circle cx="289" cy="145" r="5" fill="#9f1239"/>
  <circle cx="317" cy="145" r="5" fill="#9f1239"/>
  <circle cx="346" cy="145" r="5" fill="#9f1239"/>
  <circle cx="374" cy="145" r="5" fill="#9f1239"/>
  <circle cx="403" cy="145" r="5" fill="#9f1239"/>
  <circle cx="431" cy="145" r="5" fill="#9f1239"/>
  <circle cx="460" cy="145" r="5" fill="#9f1239"/>


  <text x="55"  y="164" font-size="12" fill="#9f1239" font-style="italic">q</text>
  <text x="63"  y="168" font-size="9"  fill="#9f1239">min</text>
  <text x="260" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#9f1239" font-style="italic">Z</text>
  <text x="276" y="164" font-size="12" font-weight="600" fill="#9f1239">= 0</text>
  <text x="451" y="164" font-size="12" fill="#9f1239" font-style="italic">q</text>
  <text x="459" y="168" font-size="9"  fill="#9f1239">max</text>


</svg>


Symmetric quantization fixes $z = 0$ — the integer grid is centered at zero. Scale is determined by the maximum absolute value. This eliminates the subtract-then-multiply in dequantization (just $\hat{x} = s \cdot x_q$), simplifying kernels. It wastes some representable levels when the distribution is not zero-centered, but transformer weight matrices are approximately zero-centered after training, making symmetric the default for weight quantization.


<div style="background:#f1f5f9;border:1px solid #e2e8f0;border-left:3px solid #6366f1;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6366f1;margin-bottom:8px;">Formula · Symmetric Quantization</div>


$$
s = \frac{\max|x|}{2^{b-1} - 1}, \qquad z = 0
$$


$$
x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right),\ -2^{b-1},\ 2^{b-1}-1\right), \qquad \hat{x} = s \cdot x_q
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $s$ — stored as <strong>FP16</strong> &nbsp;·&nbsp; INT8 symmetric: range −127 to 127, $s = \max|x|/127$ &nbsp;·&nbsp; INT4 symmetric: range −7 to 7, $s = \max|x|/7$
  </div>
</div>


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Asymmetric</div>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:5px;color:#334155;font-size:12px;">
      <li>Two params: scale (FP16) + zero-point (INT8/FP16)</li>
      <li>Covers any range $[x_\text{min}, x_\text{max}]$ exactly</li>
      <li>Best for ReLU activations, biases</li>
      <li>Dequant: $\hat{x} = s(x_q - z)$ — two ops</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Symmetric — preferred for weights</div>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:5px;color:#334155;font-size:12px;">
      <li>One param: scale (FP16) only, $z=0$</li>
      <li>Wastes 1 negative level (−128 unused for INT8)</li>
      <li>Best for zero-centered weight distributions</li>
      <li>Dequant: $\hat{x} = s \cdot x_q$ — one multiply</li>
    </ul>
  </div>
</div>


> **Relative error explodes on small values.** Relative error ≈ $s / (2|x|)$. When one outlier at 5.0 sets the scale for a tensor where 99% of values are in [−0.3, 0.3], those small values get 17× inflated error. Per-tensor symmetric quantization on LLM activations almost always fails for this reason — activations have channel-wise outliers.
{: .prompt-danger }


```python
import torch


def quantize_symmetric(x: torch.Tensor, bits: int = 8):
    """Per-tensor symmetric quantization (signed integer grid, z=0)."""
    q_max = 2**(bits - 1) - 1   # INT8 → 127, INT4 → 7


    scale = x.abs().max().item() / q_max    # FP16 in production
    zero_point = 0                          # always 0 — no shift needed


    x_q = torch.clamp(torch.round(x / scale), -q_max, q_max).to(torch.int8)
    x_dq = scale * x_q.float()             # dequantize → FP32


    return x_q, x_dq, scale




# --- smoke test ---
x = torch.tensor([-3.2, 0.0, 1.7, 4.5])
x_q, x_dq, s = quantize_symmetric(x, bits=4)
print(f"scale={s:.4f}  zero_point=0")
print(f"quantized : {x_q.tolist()}")         # [-5, 0, 3, 7]
print(f"dequantized: {x_dq.tolist()}")       # [-3.21, 0.0, 1.93, 4.5]
print(f"max error : {(x - x_dq).abs().max():.4f}")
```


---


## 5. Calibration Granularity


*Why does the number of scale factors matter — what breaks when one outlier controls the scale for thousands of weights?*


How many (scale, zero-point) pairs you compute for a tensor is as important as the bit-width. A single pair for the entire tensor (per-tensor) is fast but vulnerable to outliers. One pair per output channel (per-channel) isolates outlier channels. One pair per block of $g$ contiguous values (per-group) provides the finest accuracy with manageable overhead for static weights.


### Per-Tensor, Per-Channel, and Per-Group


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0;font-family:system-ui,sans-serif;font-size:13px;">


  <!-- PER-TENSOR -->
  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">PER-TENSOR — one scale for the entire weight matrix</div>
  <div style="display:flex;gap:3px;flex-wrap:wrap;align-items:flex-start;margin-bottom:16px;">
    <div style="background:#bdb2ff;color:#4c1d95;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;white-space:nowrap;align-self:center;">s, z</div>
    <div style="display:flex;gap:2px;flex-wrap:wrap;align-items:center;">
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₁</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₂</div>
      <div style="background:#ffadad;border:1px solid #fca5a5;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0!</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₄</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₅</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₆</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₇</div>
      <div style="background:#e2e8f0;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₈</div>
    </div>
    <div style="font-size:11px;color:#7f1d1d;font-weight:600;align-self:center;margin-left:6px;">one outlier inflates s for all 8 → 17× error on small values</div>
  </div>


  <!-- PER-CHANNEL -->
  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">PER-CHANNEL — one scale per output row (channel)</div>
  <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#a0c4ff;color:#1e3a8a;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;flex-shrink:0;">s₀, z₀</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₀</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₁</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₂</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₃</div>
      </div>
      <div style="font-size:10px;color:#1e3a8a;margin-left:4px;">← normal range</div>
    </div>
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#ffadad;color:#7f1d1d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;flex-shrink:0;">s₁, z₁</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">4.8</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.1</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">4.9</div>
      </div>
      <div style="font-size:10px;color:#7f1d1d;font-weight:600;margin-left:4px;">← outlier channel gets its own large s₁ — row 0 unaffected</div>
    </div>
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#a0c4ff;color:#1e3a8a;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;flex-shrink:0;">s₂, z₂</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₀</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₁</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₂</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₃</div>
      </div>
      <div style="font-size:10px;color:#1e3a8a;margin-left:4px;">← fine scale, low error</div>
    </div>
  </div>


  <!-- PER-GROUP -->
  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">PER-GROUP (g=4 shown) — one scale per g contiguous weights within a row</div>
  <div style="display:flex;flex-direction:column;gap:5px;">
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
      <div style="background:#caffbf;color:#14532d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;flex-shrink:0;">s₀₀</div>
      <div style="display:flex;gap:2px;margin-right:4px;">
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₁</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₂</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₃</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₄</div>
      </div>
      <div style="background:#ffadad;color:#7f1d1d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;flex-shrink:0;">s₀₁</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₆</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₇</div>
        <div style="background:#fffffc;border:1px solid #caffbf;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₈</div>
      </div>
      <div style="font-size:10px;color:#14532d;font-weight:600;margin-left:4px;">outlier confined to group s₀₁ — group s₀₀ unaffected</div>
    </div>
  </div>


</div>


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Granularity</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Scales stored</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Accuracy</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Overhead</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Typical use</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#ffadad;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#7f1d1d;">Per-tensor</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">1 per layer</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">Lowest</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">None</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#7f1d1d;">Legacy INT8, fast prototyping</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Per-channel</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">1 per output row</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Good</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Minimal</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Weights in INT8, activations</td>
    </tr>
    <tr style="background:#caffbf;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">Per-group (g=128)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">N/g per row</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">Best</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">+~3% params</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">W4 weights (GPTQ, AWQ, HQQ)</td>
    </tr>
  </tbody>
</table>
</div>


> **Scale metadata overhead.** Per-group with g=128 on a 4096×4096 weight matrix stores 4096×(4096/128) = 131,072 FP16 scale values = 256 KB per layer. Over 32 layers that is ~8 MB — negligible versus the 3.5 GB W4 model. This is why g=128 is the default for almost all W4 production deployments.
{: .prompt-tip }


```python
import torch


def _sym_quant(x: torch.Tensor, bits: int):
    q_max = 2**(bits - 1) - 1
    scale = x.abs().max() / q_max
    x_q   = torch.clamp(torch.round(x / scale), -q_max, q_max).to(torch.int8)
    return x_q, scale




def quantize_per_tensor(W: torch.Tensor, bits: int = 8):
    x_q, scale = _sym_quant(W, bits)
    x_dq = scale * x_q.float()
    return x_q, x_dq, scale




def quantize_per_channel(W: torch.Tensor, bits: int = 8):
    q_max  = 2**(bits - 1) - 1
    scale  = W.abs().amax(dim=1, keepdim=True) / q_max
    x_q    = torch.clamp(torch.round(W / scale), -q_max, q_max).to(torch.int8)
    x_dq   = scale * x_q.float()
    return x_q, x_dq, scale




def quantize_per_group(W: torch.Tensor, bits: int = 4, group_size: int = 128):
    out, inp = W.shape
    assert inp % group_size == 0
    q_max  = 2**(bits - 1) - 1
    W_g    = W.view(out, inp // group_size, group_size)
    scale  = W_g.abs().amax(dim=-1, keepdim=True) / q_max
    x_q    = torch.clamp(torch.round(W_g / scale), -q_max, q_max).to(torch.int8)
    x_dq   = (scale * x_q.float()).view(out, inp)
    return x_q.view(out, inp), x_dq, scale.squeeze(-1)




W = torch.randn(4096, 4096)
_, dq_t, _ = quantize_per_tensor(W, bits=8)
_, dq_c, _ = quantize_per_channel(W, bits=8)
_, dq_g, _ = quantize_per_group(W, bits=4, group_size=128)


def rmse(a, b): return (a - b).pow(2).mean().sqrt().item()
print(f"per-tensor  RMSE={rmse(W, dq_t):.4f}")  # ~0.0252
print(f"per-channel RMSE={rmse(W, dq_c):.4f}")  # ~0.0087
print(f"per-group   RMSE={rmse(W, dq_g):.4f}")  # ~0.0031
```


---


### Practice Question


**Q — What granularity is typically used for weights vs. activations, and why do they differ?**


<details>
<summary>💡 Hints</summary>


<div style="background:#fdffb6;border:1px solid #fcd34d;border-radius:8px;padding:14px 16px;margin-top:8px;font-family:system-ui,sans-serif;font-size:13px;line-height:1.7;">
<strong>Hint 1:</strong> Think about <em>when</em> the scale is computed — offline (once, before deployment) vs. at runtime (every forward pass). Which gives you freedom to use per-group without overhead?<br><br>
<strong>Hint 2:</strong> In W8A8, if the activation scale is stored in FP16 and the activation values are INT8, what dtype must the GEMM accumulation actually use? What does that mean for the integer tensor-core benefit?
</div>
</details>


<details>
<summary>✅ Answer</summary>


<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-top:8px;font-family:system-ui,sans-serif;font-size:13px;line-height:1.7;">


<strong>Weights: per-group (g=32, 64, or 128) — the standard for all W4 deployments.</strong>


<p style="margin-top:8px;">Weights are static. You compute group scales once offline and store them alongside the quantized values. At inference, loading the scales costs a few cache-line fetches — negligible. Per-group isolates outlier channels and dramatically reduces reconstruction error vs. per-channel at INT4. g=128 is the standard default; g=32 is used when maximum accuracy is needed at the cost of ~1.5% more metadata.</p>


<strong>Activations: per-tensor (in practice for W8A8).</strong>


<p>Activations are dynamic — they change with every input token. Computing per-group or per-channel scales at runtime means extra reads/writes to store group scales for every token, and since the scale is FP16 and the activation is INT8, the GEMM must apply the scale in FP16 during accumulation — you cannot issue a pure INT8 GEMM. You get INT8-packed storage but FP16 arithmetic, losing the entire throughput benefit of INT8 tensor cores. Per-tensor uses one scale per tensor, fused as a single multiply, and is compatible with pure INT8 GEMM accumulation.</p>


</div>
</details>


---


## Putting It All Together


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="display:flex;flex-direction:column;gap:12px;">


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">1</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Quantization targets linear layers — everything else is not worth touching</div>
        <div style="color:#64748b;font-size:12px;">Over 95% of parameters and FLOPs live in the weight matrices of attention and FFN projections. LayerNorm, SiLU, softmax, and embeddings are left in FP16/BF16. Quantization is a compression scheme for matmul operands, not a general model transformation.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">2</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">W4A16 guarantees memory reduction — it does not guarantee a compute speedup</div>
        <div style="color:#64748b;font-size:12px;">In W4A16, weights are stored as INT4 but dequantized to BF16 before the GEMM. The matmul runs on BF16 tensor cores — not INT4. The only win is fewer bytes streamed from HBM. At batch=1 (arithmetic intensity ≈ 1 FLOP/byte) that translates to 2–3× decode throughput. At batch≥32 the workload is compute-bound and weight compression contributes near zero. The roofline tells you which regime you are in.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">3</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Symmetric for weights, asymmetric for activations — for different reasons</div>
        <div style="color:#64748b;font-size:12px;">Weight tensors are roughly zero-centered after training, so a single scale ($z=0$) wastes no grid levels and dequantization reduces to one multiply. Activations after ReLU span $[0, x_\text{max}]$ — a symmetric grid leaves half the integer range empty. Asymmetric adds a zero-point to shift the grid, recovering those wasted levels at the cost of one extra subtract per dequantization.</div>
      </div>
    </div>


    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div style="background:#a0c4ff;color:#1e3a8a;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;margin-top:1px;">4</div>
      <div>
        <div style="font-weight:700;color:#1e293b;margin-bottom:2px;">Granularity is the accuracy knob — per-group is the practical default for weights</div>
        <div style="color:#64748b;font-size:12px;">Per-tensor lets one outlier inflate the scale for every weight in a layer. Per-group (g=128) isolates that outlier to 128 values — dramatically tighter error bounds at the cost of a small metadata overhead. For activations, per-tensor is the only practical choice: dynamic per-group scales cannot be fused into a pure INT8 GEMM.</div>
      </div>
    </div>


  </div>
</div>


---


## References


- Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — approachable visual walkthrough of quantization concepts; good companion read for the math in Section 4.



