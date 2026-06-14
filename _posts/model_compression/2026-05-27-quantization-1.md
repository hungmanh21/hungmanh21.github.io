---
layout: post
title: "Quantization Fundamentals: Memory Math, Roofline, and Calibration Granularity"
date: 2026-05-27 09:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, fundamentals]
math: true
---

*We trade a small, controlled error for a large, concrete hardware win — and the art is keeping the error controlled.*

<!-- Opening block: questions + prerequisites — stacked vertically -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>What exactly happens when a weight is quantized — where does the error come from?</li>
      <li>How much memory does a model use, and does quantization also speed up inference?</li>
      <li>What parts of a transformer are worth quantizing — and what should never be touched?</li>
      <li>What is calibration granularity, and why does per-group outperform per-tensor for weights?</li>
      <li>Min/max, percentile, MSE, KL — which range method should we pick?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #c4b5fd;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>IEEE 754 floating-point basics (sign, exponent, mantissa)</li>
      <li>PyTorch tensor operations</li>
      <li>Transformer architecture (linear layers, attention, FFN)</li>
      <li>Basic GPU memory concepts (VRAM, HBM bandwidth)</li>
    </ul>
  </div>
</div>

A 7B model in FP32 is 28 GB — it doesn't fit on a single consumer GPU. Quantization gets us to 3.5 GB at INT4, but the mechanism, the speedup story, and the calibration knobs are all worth understanding before reaching for a tool. Let's chase each piece in turn.

---

## 1. What breaks when we don't quantize?

*Where does the hardware wall actually appear, and what does the rounding step look like up close?*

<svg viewBox="0 0 640 275" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:660px;display:block;margin:18px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="275" fill="#fffffc" rx="10"/>

  <text x="30" y="30" font-size="16" font-weight="800" fill="#4c1d95">FP32</text>

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
    fill="#bdb2ff" fill-opacity="0.35" stroke="#1e293b" stroke-width="2.5"/>

  <circle cx="130" cy="184" r="7" fill="#4c1d95"/>
  <circle cx="210" cy="110" r="7" fill="#4c1d95"/>
  <circle cx="270" cy="61"  r="7" fill="#4c1d95"/>
  <circle cx="310" cy="44"  r="7" fill="#4c1d95"/>
  <circle cx="330" cy="44"  r="7" fill="#4c1d95"/>
  <circle cx="370" cy="61"  r="7" fill="#4c1d95"/>
  <circle cx="430" cy="110" r="7" fill="#4c1d95"/>
  <circle cx="510" cy="184" r="7" fill="#4c1d95"/>

  <line x1="130" y1="191" x2="150" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="210" y1="117" x2="218" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="270" y1="68"  x2="286" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="310" y1="51"  x2="286" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="330" y1="51"  x2="354" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="370" y1="68"  x2="354" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="430" y1="117" x2="422" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="510" y1="191" x2="490" y2="242" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>

  <rect x="52"  y="242" width="60" height="26" rx="4" fill="#ffc6ff" stroke="#f0abfc"/>
  <rect x="120" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="150" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="188" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="218" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="256" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="286" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="324" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="354" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="392" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="422" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="460" y="242" width="60" height="26" rx="4" fill="#ffadad" stroke="#fca5a5"/>
  <circle cx="490" cy="255" r="7" fill="#7f1d1d"/>
  <rect x="528" y="242" width="60" height="26" rx="4" fill="#ffc6ff" stroke="#f0abfc"/>

  <text x="598" y="261" font-size="15" font-weight="800" fill="#7f1d1d">INT4</text>
</svg>

The bell curve is our weight distribution — most values cluster near zero, a few stretch into the tails. Each lavender dot is a real FP32 weight. We give ourselves only sixteen INT4 buckets to cover the whole thing, and the dashed arrows show the actual operation: round each value to its nearest bucket. Notice the two center buckets each receive two arrows — the peak is dense, so distinct FP32 values collapse into the same integer. The pink end-buckets are valid INT4 levels that no weight maps to: wasted resolution. **The rounding distance is the quantization error**, and where to spend grid resolution is the question that drives everything below.

### So why pay this cost at all?

*What is the concrete hardware wall that a 7B model hits in FP32?*

<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Model size vs. dtype — 7B model</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">bytes/param × 7 × 10⁹ · RTX 4090 VRAM = 24 GB</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:140px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">FP32 (4 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">28 GB</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;flex-shrink:0;">no fit</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:140px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">BF16 (2 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:50%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">14 GB</span>
      <span style="background:#ffd6a5;color:#78350f;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #f59e0b;flex-shrink:0;">tight</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:140px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">INT8 (1 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">7 GB</span>
      <span style="background:#9bf6ff;color:#164e63;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #67e8f9;flex-shrink:0;">fits</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:140px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">INT4 / NF4 (0.5 B)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:12.5%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">3.5 GB</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;flex-shrink:0;">comfortable</span>
    </div>
  </div>
</div>

The chart reads top-down as a sequence of "no fit / tight / fits / comfortable". FP32 overshoots a 24 GB consumer card by 4 GB before we have loaded a single token of context. BF16 lands inside the card but with no room for the KV cache we will need at runtime. INT8 halves that again, and INT4 leaves us 20 GB free for activations, KV cache, and headroom. Scale this up: 70B in BF16 needs 140 GB and eight A100s; in INT4 it is 35 GB and fits on two.

But VRAM is only half the story. At decode time with batch 1, the GPU spends most of its cycles waiting for weights to stream in from HBM, not doing arithmetic — fewer bytes per weight is fewer microseconds per matrix-vector product. Keep that "memory-bound at small batch" fact in your back pocket; we will use it twice in section 2.

> **Central tradeoff.** A small, controlled accuracy loss buys a large, concrete hardware win. Everything below is about keeping that error controlled.
{: .prompt-tip }

That handles whether we *can* fit the model. The harder question is whether we have budgeted for everything *else* that lives in VRAM.

---

## 2. Where does the memory actually go?

*Weights fit at our chosen dtype — but does the rest fit at our batch and sequence length?*

<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Weight + KV cache VRAM — 7B / 70B</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">FP32=4B · BF16=2B · INT8=1B · INT4=0.5B · KV: B=16, seq=2048, FP16</div>
  <div style="display:flex;flex-direction:column;gap:8px;">

    <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-top:4px;">7B model</div>

    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">FP32</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">28 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">BF16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:50%;height:100%;background:#ffd6a5;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">14 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">INT8</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#9bf6ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">7 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">INT4 / NF4</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:12.5%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">3.5 GB</span>
    </div>

    <div style="font-size:10px;color:#94a3b8;margin-top:4px;font-style:italic;padding-left:2px;">KV cache — L=32, 32 KV heads, d=128</div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">KV FP16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:57%;height:100%;background:#bdb2ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">16 GB</span>
    </div>

    <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-top:8px;">70B model</div>

    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">BF16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">140 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">INT4 / NF4</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:25%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">35 GB</span>
      <span style="background:#caffbf;color:#14532d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #86efac;">fits 2×A100</span>
    </div>

    <div style="font-size:10px;color:#94a3b8;margin-top:4px;font-style:italic;padding-left:2px;">KV cache — L=80, 8 KV heads (GQA), d=128</div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:100px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">KV FP16</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:7.1%;height:100%;background:#bdb2ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">10 GB</span>
    </div>
  </div>
</div>

The lavender bar is the surprise. For the 7B model at batch 16 and 2k context, the KV cache is **larger than the INT8 weights and roughly 5× the INT4 weights**. Quantizing weights to INT4 and forgetting the KV cache lands us at 19.5 GB on a 24 GB card, half of which is a tensor that grew with our serving config rather than the model. The 70B row tells the opposite story: GQA collapses 64 heads to 8, the KV cache shrinks to 10 GB, and weights dominate again. The numbers do not generalize; we read them per architecture.

<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #bdb2ff;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · weight memory</div>

$$
\text{mem}_\text{weights} = N_\text{params} \times b_\text{dtype}
$$

  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $N_\text{params}$ — parameter count (e.g. 7 × 10⁹ for a 7B model)<br>
    $b_\text{dtype}$ — bytes per element: FP32=4, BF16=2, INT8=1, INT4/NF4=0.5
  </div>
</div>

<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #bdb2ff;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · KV cache memory</div>

$$
\text{mem}_\text{KV} = 2 \times B \times S \times L \times H \times b_\text{dtype}
$$

  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    Factor 2 — one K tensor + one V tensor per layer · $B$ batch · $S$ sequence length · $L$ layers · $H = n_\text{kv\_heads} \times d_\text{head}$ · $b_\text{dtype}$ bytes per element
  </div>
</div>

The KV formula is linear in $B$ and $S$ — every token in every batch slot pays for itself, every layer. That linearity is why a 32-batch, 4k-context server explodes the KV cache to dozens of gigabytes even on a modest model, and why "INT8 KV cache" became a production knob in its own right.

> **First check.** Before optimizing throughput, confirm the model fits in VRAM at the *target batch and sequence length*, KV cache included. Weight bytes alone is not the budget.
{: .prompt-info }

So we now know what eats the memory. The next question is what we should — and absolutely should not — quantize inside that budget.

---

## 3. What inside a transformer is worth quantizing?

*Weights, activations, KV cache — three targets, three different stories. Why can't one strategy cover all of them?*

A transformer forward pass is overwhelmingly matrix multiplications. Every attention projection (Q, K, V, O) and every FFN projection (gate, up, down) is a linear of the form

$$
Y = W X
$$

with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ the weight matrix and $X \in \mathbb{R}^{d_\text{in} \times B}$ the activation batch. Those GEMMs account for over 95% of FLOPs and essentially all the parameter memory in a 7B+ model. LayerNorm, SiLU, softmax, and residual additions are elementwise — negligible memory and arithmetic compared to the matmuls. There is nothing to gain from quantizing them.

Quantization replaces one or both FP operands with a low-precision representation plus a floating-point scale, so the GEMM becomes

$$
Y \approx \hat{W}\hat{X}.
$$

Every scheme below — W4A16, W8A8, FP8, INT8 KV — is a choice about which of $W$ and $X$ to quantize, to what bit-width, and at what granularity.

<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Weights</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;">
      <li>Static — computed once, offline</li>
      <li>No input dependency</li>
      <li>No calibration needed for per-channel</li>
      <li><strong>Format:</strong> W4 (INT4 / NF4)</li>
      <li style="color:#64748b;font-size:12px;">Tools: torchao, autoawq, gptqmodel, HQQ</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Activations</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;">
      <li>Dynamic — depend on input tokens</li>
      <li>Need static or dynamic calibration</li>
      <li>Enable INT8/FP8 GEMMs at large batch</li>
      <li><strong>Format:</strong> INT8 (W8A8), W4A8</li>
      <li style="color:#64748b;font-size:12px;">Tools: torchao, bitsandbytes, llm-compressor</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">KV cache</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;">
      <li>Grows with sequence length</li>
      <li>Written and read every decode step</li>
      <li>Can dwarf weights at long seq</li>
      <li><strong>Format:</strong> INT8, FP8 (H100), INT4</li>
      <li style="color:#64748b;font-size:12px;">Tools: vLLM, TRT-LLM, KIVI</li>
    </ul>
  </div>
</div>

The three cards split cleanly along one axis: *when do we know the values?* Weights are known offline, so we get unlimited compute budget for the calibration step and can afford fancy schemes. Activations are born at inference, so any calibration overhead is paid per token. KV cache lives between those two — it is written once per token and re-read on every subsequent step, so its compression cost is amortized but its bandwidth bill is enormous.

<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Target</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">When computed</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Input-dependent?</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Dominant format</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Independent?</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#bdb2ff40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Weights</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Once, offline</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">No</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT4 / NF4</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
    <tr style="background:#a0c4ff40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">Activations</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">At inference</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT8 (W8A8)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
    <tr style="background:#caffbf40;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;">KV cache</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">At generation</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes (grows with seq)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">INT8, FP8, INT4</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Yes</td>
    </tr>
  </tbody>
</table>
</div>

> **Orthogonal decisions.** W4A16 quantizes only weights; activations and KV cache stay in BF16. We can stack INT8 KV cache on top without touching the weight scheme. Production deployments routinely combine all three independently.
{: .prompt-tip }

> **Embeddings and the LM head are off-limits.** They are weight tensors, but quantization error here propagates into every layer (embedding) or every output token (LM head). Combined they are under 1% of params in 7B+ models — never worth the risk.
{: .prompt-warning }

There is one more subtlety hiding in the W4A16 row. We compress the weights to INT4 — but does that mean the GEMM runs in INT4? Hold that question; it sets up the math in the next section.

---

## 4. How does the math actually work?

*Scale and zero-point — why two parameters, and when does one suffice?*

### Asymmetric: the general case

*What does a single scale miss when the distribution isn't centered on zero?*

<svg viewBox="0 0 520 195" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:560px;display:block;margin:18px auto 4px;font-family:system-ui,sans-serif;background:#fffffc;border-radius:10px;">

  <text x="26" y="95"  font-size="15" font-weight="700" fill="#1e293b" font-style="italic">r</text>
  <text x="26" y="150" font-size="15" font-weight="700" fill="#7f1d1d" font-style="italic">q</text>

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

  <line x1="60"  y1="70" x2="60"  y2="141" stroke="#7f1d1d" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>
  <line x1="460" y1="70" x2="460" y2="141" stroke="#7f1d1d" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>

  <line x1="220" y1="141" x2="220" y2="100" stroke="#1e3a8a" stroke-width="2.4"/>
  <polygon points="214,100 220,91 226,100" fill="#1e3a8a"/>

  <text x="232" y="113" font-size="13" font-weight="700" fill="#1e3a8a">×<tspan font-style="italic">S</tspan></text>
  <text x="232" y="126" font-size="10"  fill="#1e3a8a">Floating-point</text>
  <text x="232" y="139" font-size="10"  fill="#1e3a8a">Scale</text>

  <line x1="42" y1="145" x2="482" y2="145" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,141 490,145 480,149" fill="#1e293b"/>

  <circle cx="60"  cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="87"  cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="113" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="140" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="167" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="193" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="220" cy="145" r="7" fill="#7f1d1d"/>
  <circle cx="247" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="273" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="300" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="327" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="353" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="380" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="407" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="433" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="460" cy="145" r="5" fill="#7f1d1d"/>

  <text x="55"  y="164" font-size="12" fill="#7f1d1d" font-style="italic">q</text>
  <text x="63"  y="168" font-size="9"  fill="#7f1d1d">min</text>
  <text x="220" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#7f1d1d" font-style="italic">Z</text>
  <text x="220" y="180" text-anchor="middle" font-size="10" fill="#64748b">Zero point</text>
  <text x="451" y="164" font-size="12" fill="#7f1d1d" font-style="italic">q</text>
  <text x="459" y="168" font-size="9"  fill="#7f1d1d">max</text>

</svg>

The top axis is the floating-point range $[r_\text{min}, r_\text{max}]$, and the bottom is the integer grid. Asymmetric quantization places those two side-by-side and uses two parameters to map between them: a **scale** $s$ that sets the grid spacing and a **zero-point** $z$ that decides which integer corresponds to FP zero. Notice the pivot — FP zero lands on $Z$, not on $q_\text{min}$. That shift is what makes asymmetric work for ReLU outputs in $[0, x_\text{max}]$ or any other distribution that isn't centered: every integer level lands on a real value in range, no levels wasted.

<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #bdb2ff;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · asymmetric quantization</div>

$$
s = \frac{x_\text{max} - x_\text{min}}{2^b - 1}, \qquad z = \text{round}\!\left(\frac{-x_\text{min}}{s}\right)
$$

$$
x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\ 0,\ 2^b - 1\right), \qquad \hat{x} = s \cdot (x_q - z)
$$

  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $b$ — bit-width · $s$ stored as <strong>FP16</strong> · $z$ stored as <strong>INT8</strong> (or FP16) · $\hat{x}$ — reconstructed value
  </div>
</div>

A small worked example fixes the mechanics. Take $x = [-3.2, 0.0, 1.7, 4.5]$ at INT4 unsigned (range 0–15). The two parameters fall out:

$$
s = \frac{4.5 - (-3.2)}{15} = 0.5133, \qquad z = \text{round}\!\left(\frac{3.2}{0.5133}\right) = 6.
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

The green row is the defining property: FP zero round-trips exactly because $z$ was constructed to send it there. Every other error is bounded by half a grid step — here $s/2 = 0.257$, and the table confirms every entry stays under it. A smaller $s$ would tighten that bound but cover a narrower range, and that tradeoff is exactly what calibration in section 6 will optimize.

```python
import torch

def quantize_asymmetric(x: torch.Tensor, bits: int = 8):
    """Per-tensor asymmetric quantization (unsigned integer grid)."""
    q_min, q_max = 0, 2**bits - 1
    x_min, x_max = x.min().item(), x.max().item()

    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = round(-x_min / scale)
    zero_point = int(max(q_min, min(q_max, zero_point)))

    x_q  = torch.clamp(torch.round(x / scale + zero_point), q_min, q_max).to(torch.int8)
    x_dq = scale * (x_q.float() - zero_point)
    return x_q, x_dq, scale, zero_point


x = torch.tensor([-3.2, 0.0, 1.7, 4.5])
x_q, x_dq, s, z = quantize_asymmetric(x, bits=4)
print(f"scale={s:.4f}  zero_point={z}")
print(f"q : {x_q.tolist()}")    # [0, 6, 9, 15]
print(f"dq: {x_dq.tolist()}")   # [-3.08, 0.0, 1.54, 4.62]
```

That covers any range. So why isn't asymmetric the universal default?

### Symmetric: when zero-point is wasted

*Why is symmetric the default for weights but a poor choice for activations?*

<svg viewBox="0 0 520 195" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:560px;display:block;margin:18px auto 4px;font-family:system-ui,sans-serif;background:#fffffc;border-radius:10px;">

  <text x="26" y="95"  font-size="15" font-weight="700" fill="#1e293b" font-style="italic">r</text>
  <text x="26" y="150" font-size="15" font-weight="700" fill="#7f1d1d" font-style="italic">q</text>

  <line x1="42" y1="90" x2="482" y2="90" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,86 490,90 480,94" fill="#1e293b"/>

  <rect x="60" y="44" width="400" height="26" fill="#a0c4ff" fill-opacity="0.6" rx="3"/>
  <text x="260" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#1e3a8a">Floating-point range</text>

  <line x1="60"  y1="85" x2="60"  y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <line x1="460" y1="85" x2="460" y2="95" stroke="#1e293b" stroke-width="1.4"/>

  <text x="60" y="30" text-anchor="middle" font-size="12" fill="#1e3a8a">−|r|</text>
  <text x="72" y="39" font-size="9"        fill="#1e3a8a">max</text>

  <line x1="260" y1="85" x2="260" y2="95" stroke="#1e293b" stroke-width="1.4"/>
  <text x="260" y="32" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">0</text>

  <text x="448" y="32" text-anchor="middle" font-size="12" fill="#1e3a8a">|r|</text>
  <text x="456" y="38" font-size="9"        fill="#1e3a8a">max</text>

  <line x1="60"  y1="70" x2="60"  y2="141" stroke="#7f1d1d" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>
  <line x1="460" y1="70" x2="460" y2="141" stroke="#7f1d1d" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.55"/>

  <line x1="260" y1="141" x2="260" y2="100" stroke="#1e3a8a" stroke-width="2.4"/>
  <polygon points="254,100 260,91 266,100" fill="#1e3a8a"/>

  <text x="272" y="113" font-size="13" font-weight="700" fill="#1e3a8a">×<tspan font-style="italic">S</tspan></text>
  <text x="272" y="126" font-size="10"  fill="#1e3a8a">Floating-point</text>
  <text x="272" y="139" font-size="10"  fill="#1e3a8a">Scale</text>

  <line x1="42" y1="145" x2="482" y2="145" stroke="#1e293b" stroke-width="1.8"/>
  <polygon points="480,141 490,145 480,149" fill="#1e293b"/>

  <circle cx="60"  cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="89"  cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="117" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="146" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="174" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="203" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="231" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="260" cy="145" r="7" fill="#7f1d1d"/>
  <circle cx="289" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="317" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="346" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="374" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="403" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="431" cy="145" r="5" fill="#7f1d1d"/>
  <circle cx="460" cy="145" r="5" fill="#7f1d1d"/>

  <text x="55"  y="164" font-size="12" fill="#7f1d1d" font-style="italic">q</text>
  <text x="63"  y="168" font-size="9"  fill="#7f1d1d">min</text>
  <text x="260" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#7f1d1d" font-style="italic">Z</text>
  <text x="276" y="164" font-size="12" font-weight="600" fill="#7f1d1d">= 0</text>
  <text x="451" y="164" font-size="12" fill="#7f1d1d" font-style="italic">q</text>
  <text x="459" y="168" font-size="9"  fill="#7f1d1d">max</text>

</svg>

Symmetric pins $z = 0$. The integer grid is centered on zero and the scale is set by the maximum absolute value, $s = \max\lvert x \rvert / (2^{b-1} - 1)$. Two consequences fall out of the picture: the dequantization collapses to one multiply $\hat{x} = s \cdot x_q$ — no subtract — and one negative integer level is permanently unused (INT8 uses −127 to 127, not −128). That's a 0.4% waste on INT8, but the real reason it works for transformer weights is in the histogram: trained weights are approximately zero-centered, so we'd put $z$ near 0 anyway.

<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #bdb2ff;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · symmetric quantization</div>

$$
s = \frac{\max|x|}{2^{b-1} - 1}, \qquad z = 0
$$

$$
x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right),\ -2^{b-1},\ 2^{b-1}-1\right), \qquad \hat{x} = s \cdot x_q
$$

  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    INT8 symmetric: range −127 to 127 · INT4 symmetric: range −7 to 7 · $s$ stored as FP16
  </div>
</div>

<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Asymmetric</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>Two params: scale + zero-point</li>
      <li>Covers any range $[x_\text{min}, x_\text{max}]$ exactly</li>
      <li>Best for ReLU activations, biases</li>
      <li>Dequant: $\hat{x} = s(x_q - z)$ — two ops</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Symmetric — preferred for weights</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>One param: scale only, $z=0$</li>
      <li>Wastes one negative level</li>
      <li>Best for zero-centered weight distributions</li>
      <li>Dequant: $\hat{x} = s \cdot x_q$ — one multiply</li>
    </ul>
  </div>
</div>

> **Relative error explodes on small values.** Per-tensor symmetric on activations almost always fails for this reason — relative error scales as $s / (2 \lvert x \rvert)$, so a single outlier at 5.0 in a tensor whose 99% bulk lives in $[-0.3, 0.3]$ inflates error on those small values by roughly 17×. Activations carry channel-wise outliers; weights, mostly, do not.
{: .prompt-danger }

```python
import torch

def quantize_symmetric(x: torch.Tensor, bits: int = 8):
    """Per-tensor symmetric quantization (signed grid, z=0)."""
    q_max = 2**(bits - 1) - 1
    scale = x.abs().max().item() / q_max
    x_q   = torch.clamp(torch.round(x / scale), -q_max, q_max).to(torch.int8)
    x_dq  = scale * x_q.float()
    return x_q, x_dq, scale


x = torch.tensor([-3.2, 0.0, 1.7, 4.5])
x_q, x_dq, s = quantize_symmetric(x, bits=4)
print(f"scale={s:.4f}, z=0")
print(f"q : {x_q.tolist()}")    # [-5, 0, 3, 7]
print(f"dq: {x_dq.tolist()}")   # [-3.21, 0.0, 1.93, 4.5]
```

The danger callout flagged the failure mode: one bad value can ruin an entire tensor. That points squarely at the next knob — how *many* scales do we use per tensor?

---

## 5. Why does the number of scales matter?

*Per-tensor, per-channel, per-group — what does an outlier actually do to the rest of the values?*

The scale formula takes a range as input. The question is: a range over *what*? The whole tensor? Each output channel? Each block of 128 contiguous weights? That choice is the **granularity** knob, and it controls how far an outlier's blast radius reaches.

### Three granularities, side by side

*What does an outlier look like under each scheme?*

<div style="background:#fffffc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0;font-family:system-ui,sans-serif;font-size:13px;">

  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">Per-tensor — one scale for the entire matrix</div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:16px;">
    <div style="background:#bdb2ff;color:#4c1d95;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;white-space:nowrap;">s, z</div>
    <div style="display:flex;gap:2px;flex-wrap:wrap;align-items:center;">
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₁</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₂</div>
      <div style="background:#ffadad;border:1px solid #fca5a5;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0!</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₄</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₅</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₆</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₇</div>
      <div style="background:#f1f5f9;border:1px solid #cbd5e1;width:28px;height:22px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#64748b;">w₈</div>
    </div>
    <div style="font-size:11px;color:#7f1d1d;font-weight:600;margin-left:6px;">one outlier inflates s for all 8 → ~17× error on small values</div>
  </div>

  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">Per-channel — one scale per output row</div>
  <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#a0c4ff;color:#1e3a8a;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;">s₀, z₀</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₀</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₁</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₂</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₀₃</div>
      </div>
      <div style="font-size:10px;color:#1e3a8a;margin-left:4px;">← normal range</div>
    </div>
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#ffadad;color:#7f1d1d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;">s₁, z₁</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">4.8</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.1</div>
        <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">4.9</div>
      </div>
      <div style="font-size:10px;color:#7f1d1d;font-weight:600;margin-left:4px;">← outlier channel gets its own large s₁ — row 0 unaffected</div>
    </div>
    <div style="display:flex;align-items:center;gap:4px;">
      <div style="background:#a0c4ff;color:#1e3a8a;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;">s₂, z₂</div>
      <div style="display:flex;gap:2px;">
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₀</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₁</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₂</div>
        <div style="background:#fffffc;border:1px solid #a0c4ff;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#1e3a8a;">w₂₃</div>
      </div>
      <div style="font-size:10px;color:#1e3a8a;margin-left:4px;">← fine scale, low error</div>
    </div>
  </div>

  <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;">Per-group (g=4 shown) — one scale per g contiguous weights within a row</div>
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
    <div style="background:#caffbf;color:#14532d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;">s₀₀</div>
    <div style="display:flex;gap:2px;margin-right:4px;">
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₁</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₂</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₃</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₄</div>
    </div>
    <div style="background:#ffadad;color:#7f1d1d;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700;white-space:nowrap;">s₀₁</div>
    <div style="display:flex;gap:2px;">
      <div style="background:#ffadad;border:1px solid #fca5a5;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#7f1d1d;font-weight:700;">5.0</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₆</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₇</div>
      <div style="background:#fffffc;border:1px solid #86efac;width:26px;height:20px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#14532d;">w₈</div>
    </div>
    <div style="font-size:10px;color:#14532d;font-weight:600;margin-left:4px;">outlier confined to group s₀₁ — group s₀₀ unaffected</div>
  </div>

</div>

Read it as a sequence of containment radii. Per-tensor: one outlier in eight values poisons the scale for all eight. Per-channel: the outlier is trapped in its own row, the other rows keep a fine grid. Per-group with $g=4$: only three other values share the contaminated scale. The pattern is general — the smaller the group, the smaller the blast radius — and we are paying for the privilege only in metadata, not in compute.

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
      <td style="padding:7px 12px;border:1px solid #e2e8f0;">Weights at INT8, activations</td>
    </tr>
    <tr style="background:#caffbf;">
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">Per-group (g=128)</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">N/g per row</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">Best</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;color:#14532d;">+~3% params</td>
      <td style="padding:7px 12px;border:1px solid #e2e8f0;font-weight:600;color:#14532d;">W4 weights — GPTQ, AWQ, HQQ</td>
    </tr>
  </tbody>
</table>
</div>

> **The metadata math is forgiving.** For a 4096×4096 weight matrix at $g=128$, we store $4096 \times (4096/128) = 131{,}072$ FP16 scales — 256 KB per layer. Across 32 layers that's ~8 MB on a 3.5 GB W4 model. Negligible. This is why $g=128$ is the production default.
{: .prompt-tip }

```python
import torch

def quantize_per_tensor(W: torch.Tensor, bits: int = 8):
    q_max = 2**(bits - 1) - 1
    scale = W.abs().max() / q_max
    x_q   = torch.clamp(torch.round(W / scale), -q_max, q_max).to(torch.int8)
    return x_q, scale * x_q.float(), scale


def quantize_per_channel(W: torch.Tensor, bits: int = 8):
    q_max = 2**(bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True) / q_max
    x_q   = torch.clamp(torch.round(W / scale), -q_max, q_max).to(torch.int8)
    return x_q, scale * x_q.float(), scale


def quantize_per_group(W: torch.Tensor, bits: int = 4, group_size: int = 128):
    out, inp = W.shape
    assert inp % group_size == 0
    q_max = 2**(bits - 1) - 1
    W_g   = W.view(out, inp // group_size, group_size)
    scale = W_g.abs().amax(dim=-1, keepdim=True) / q_max
    x_q   = torch.clamp(torch.round(W_g / scale), -q_max, q_max).to(torch.int8)
    x_dq  = (scale * x_q.float()).view(out, inp)
    return x_q.view(out, inp), x_dq, scale.squeeze(-1)


W = torch.randn(4096, 4096)
def rmse(a, b): return (a - b).pow(2).mean().sqrt().item()

print(f"per-tensor  RMSE={rmse(W, quantize_per_tensor(W, 8)[1]):.4f}")   # ~0.0252
print(f"per-channel RMSE={rmse(W, quantize_per_channel(W, 8)[1]):.4f}")  # ~0.0087
print(f"per-group   RMSE={rmse(W, quantize_per_group(W, 4, 128)[1]):.4f}")  # ~0.0031
```

Per-group at INT4 beats per-channel at INT8 on RMSE — the granularity knob can recover bits. But notice every recipe so far quantizes weights, not activations. That's not a coincidence: activations are dynamic, so a per-group activation scale must be computed at runtime, which forces the GEMM into a mixed-precision (FP16 × INT8) accumulation and erases the entire INT8 tensor-core throughput benefit. Per-tensor is the only granularity that fuses cleanly into a pure INT8 matmul, which is why W8A8 production deployments use per-channel weights but per-tensor activations.

Granularity decides *how many* scales we compute. The next question is *how* we compute each one.

---

## 6. How do we pick the range?

*Min/max, percentile, MSE, KL — what does each method actually optimize, and what does it cost?*

The scale formula $s = (x_\text{max} - x_\text{min}) / (2^b - 1)$ takes a range as input, but we have not yet said where that range comes from. Different rules make very different clipping–rounding tradeoffs.

<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #d1d5db;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;font-size:11px;">Min/Max</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>Range = [min(x), max(x)]</li>
      <li>No calibration data needed</li>
      <li>One outlier blows up the scale</li>
      <li>Worst at INT4</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;font-size:11px;">Percentile</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>Range = [p%, 100−p%]</li>
      <li>Clips tails, robust to outliers</li>
      <li>Hyperparameter $\alpha$ to tune</li>
      <li>Needs calibration set</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;font-size:11px;">MSE</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>Minimize $\mathbb{E}[(x - \hat{x})^2]$</li>
      <li>Optimal grid spacing</li>
      <li>Requires calibration set</li>
      <li>Costs a 1D grid search</li>
    </ul>
  </div>
  <div style="flex:1;min-width:200px;background:#fffffc;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;font-size:11px;">KL Divergence</div>
    <ul style="margin:0;padding-left:14px;color:#1e293b;line-height:1.7;font-size:12px;">
      <li>Minimize KL(P‖Q)</li>
      <li>Preserves distribution shape</li>
      <li>Requires calibration set</li>
      <li>Histogram + 1D search</li>
    </ul>
  </div>
</div>

### Min/max: the lazy default

*Why does min/max get away with it for weights but fail on activations?*

The range is taken straight from the data: $x_\text{min} = \min(x)$, $x_\text{max} = \max(x)$. Zero clipping error, no calibration step. The catch is that one outlier — say a weight at 5.0 in a tensor whose 99% bulk lives in $[-0.3, 0.3]$ — sets a coarse grid that crushes all the small values into a handful of bins. Min/max wins when the distribution is well-behaved or when no calibration data is available; it loses when tails are heavy.

> **Works for weights, risky for activations.** Weight distributions are stable across inputs and rarely have pathological outliers. Activation ranges shift with every token — a single calibration batch with an extreme input inflates the scale permanently.
{: .prompt-warning }

### Percentile: clip the tails, keep the bulk

*If a few values are wrecking the grid, what if we just throw them out?*

Use the $\alpha$-th and $(1-\alpha)$-th percentiles as the range, and clamp anything outside to those endpoints before quantizing. Common settings: $\alpha = 0.001$ (clip 0.1% per tail) or $\alpha = 0.0001$ for smoother distributions.

<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:640px;display:block;margin:18px auto;font-family:system-ui,sans-serif;background:#fffffc;border-radius:10px;border:1px solid #e2e8f0;">

  <path d="M 40,130 C 60,128 85,122 110,112 C 135,100 155,83 175,65 C 190,51 205,41 220,35 C 235,29 248,27 260,26 C 272,27 285,29 300,35 C 315,41 330,51 345,65 C 365,83 385,100 410,112 C 435,122 460,128 480,130"
        fill="#a0c4ff" fill-opacity="0.35" stroke="none"/>

  <path d="M 40,130 C 60,128 85,122 110,112 C 120,106 128,100 135,94 L 135,130 Z"
        fill="#ffadad" fill-opacity="0.7"/>
  <path d="M 385,94 C 392,100 400,106 410,112 C 435,122 460,128 480,130 L 480,130 L 385,130 Z"
        fill="#ffadad" fill-opacity="0.7"/>

  <path d="M 40,130 C 60,128 85,122 110,112 C 135,100 155,83 175,65 C 190,51 205,41 220,35 C 235,29 248,27 260,26 C 272,27 285,29 300,35 C 315,41 330,51 345,65 C 365,83 385,100 410,112 C 435,122 460,128 480,130"
        fill="none" stroke="#1e293b" stroke-width="2"/>

  <line x1="135" y1="20" x2="135" y2="135" stroke="#7f1d1d" stroke-width="1.8" stroke-dasharray="5,3"/>
  <line x1="385" y1="20" x2="385" y2="135" stroke="#7f1d1d" stroke-width="1.8" stroke-dasharray="5,3"/>

  <text x="135" y="15" text-anchor="middle" font-size="11" font-weight="700" fill="#7f1d1d">α percentile</text>
  <text x="385" y="15" text-anchor="middle" font-size="11" font-weight="700" fill="#7f1d1d">(1−α) percentile</text>

  <text x="87"  y="148" text-anchor="middle" font-size="10" fill="#7f1d1d" font-weight="600">clipped</text>
  <text x="432" y="148" text-anchor="middle" font-size="10" fill="#7f1d1d" font-weight="600">clipped</text>
  <text x="260" y="148" text-anchor="middle" font-size="10" fill="#14532d" font-weight="600">quantized range</text>

  <line x1="30" y1="132" x2="495" y2="132" stroke="#94a3b8" stroke-width="1"/>
</svg>

The rose tails get clamped — those few values pay a clipping error — and in exchange the bulk of the distribution gets a much finer grid. It's a knob, not a method: pick $\alpha$ too low and we are back to min/max; pick it too high and we clip the body of the distribution. So the natural follow-up is — can we set $\alpha$ automatically?

### MSE: balance clipping against rounding

*What value of $\alpha$ minimizes the actual reconstruction error?*

Search for the $\alpha$ that minimizes mean squared error over a calibration set:

$$
\alpha^* = \arg\min_\alpha\ \mathbb{E}\!\left[\left(x - Q_\alpha(x)\right)^2\right]
$$

where $Q_\alpha$ is the quantize→dequantize round-trip with range clipped to $[-\alpha, \alpha]$. In practice this is a one-dimensional grid search over $\alpha$ on a small batch (~128–512 samples). The MSE decomposes into two opposing terms:

<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #bdb2ff;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">MSE · clipping vs. rounding decomposition</div>

$$
\text{MSE}(\alpha) = \underbrace{\frac{\alpha^2}{3 \cdot (2^b - 1)^2}}_{\text{rounding}} \cdot N_\text{in} \;+\; \underbrace{\sum_{|x_i| > \alpha} (|x_i| - \alpha)^2}_{\text{clipping}}
$$

  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $N_\text{in}$ — values inside the clipped range · $b$ — bit-width · $\alpha$ — clip threshold (symmetric)
  </div>
</div>

The two terms pull in opposite directions: shrinking $\alpha$ tightens the grid (less rounding) but pushes more values into the clip region (more clipping). Their crossover is the optimum, and the grid search just finds it.

```python
import torch

def mse_calibrate(x: torch.Tensor, bits: int = 8, steps: int = 200):
    """Search for the symmetric clip threshold alpha* that minimizes MSE."""
    q_max = 2**(bits - 1) - 1
    best_alpha, best_mse = x.abs().max().item(), float("inf")

    for i in range(1, steps + 1):
        alpha = x.abs().max().item() * i / steps
        scale = alpha / q_max
        x_q   = torch.clamp(torch.round(x.clamp(-alpha, alpha) / scale), -q_max, q_max)
        mse   = (x - scale * x_q.float()).pow(2).mean().item()
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return best_alpha
```

This is the default in **GPTQ**, **AWQ**, and most post-training pipelines: cheap (one forward pass plus a search), reliably better than min/max, no tuning required from the user.

### KL divergence: preserve the shape

*When the distribution itself carries the signal, what does element-wise error miss?*

Instead of element-wise reconstruction error, minimize the information loss between the original distribution $P$ and the quantized distribution $Q$:

$$
\alpha^* = \arg\min_\alpha\ D_\text{KL}(P \,\|\, Q_\alpha)
= \arg\min_\alpha \sum_i P(i) \log \frac{P(i)}{Q_\alpha(i)}.
$$

Both $P$ and $Q_\alpha$ are histograms over the calibration data. KL penalizes missing probability mass strongly — if $Q$ assigns zero where $P$ has non-zero, the divergence is infinite. That makes KL the right choice for distributions where the *shape* matters more than per-element fidelity. Softmax attention scores are the textbook example: a few large values carry most of the signal, and we want to preserve their relative weight, not minimize their absolute reconstruction error. TensorRT's legacy INT8 calibration uses KL by default for exactly this reason.

```python
import numpy as np

def kl_calibrate(x, bits=8, num_bins=2048, steps=100):
    """Find alpha* that minimizes KL(P || Q) over a calibration tensor."""
    x_np = x.float().numpy().ravel()
    abs_max = np.abs(x_np).max()
    hist, edges = np.histogram(x_np, bins=num_bins, range=(-abs_max, abs_max))
    P = hist.astype(np.float64); P /= P.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])

    best_alpha, best_kl = abs_max, float("inf")
    for i in range(steps, 0, -1):
        alpha = abs_max * i / steps
        levels = 2**bits - 1
        scale  = alpha / (levels // 2)
        mask   = np.abs(centers) <= alpha
        Pc = P.copy(); Pc[~mask] = 0
        if Pc.sum() < 1e-12: continue
        Pc /= Pc.sum()
        q_idx = np.clip(np.round(centers[mask] / scale).astype(int),
                        -(levels // 2), levels // 2)
        Q = np.zeros(num_bins)
        for level in np.unique(q_idx):
            bins = np.where(mask)[0][q_idx == level]
            if len(bins): Q[bins] = Pc[bins].sum() / len(bins)
        if Q.sum() < 1e-12: continue
        Q /= Q.sum()
        nz = (Pc > 0) & (Q > 0)
        kl = (Pc[nz] * np.log(Pc[nz] / Q[nz])).sum()
        if kl < best_kl:
            best_kl, best_alpha = kl, alpha
    return best_alpha
```

### Comparing the four

*Which method should we reach for first?*

<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;min-width:560px;font-size:13px;font-family:system-ui,sans-serif;">
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 10px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Method</th>
      <th style="padding:8px 10px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Calib data</th>
      <th style="padding:8px 10px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Robustness</th>
      <th style="padding:8px 10px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Cost</th>
      <th style="padding:8px 10px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;font-weight:600;">Typical use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;font-weight:600;">Min/Max</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">None</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;color:#7f1d1d;">Poor</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Trivial</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Static weights, fast prototyping</td>
    </tr>
    <tr style="background:#a0c4ff40;">
      <td style="padding:7px 10px;border:1px solid #e2e8f0;font-weight:600;">Percentile</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Small batch</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;color:#78350f;">Good</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Low</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Activations with known tails</td>
    </tr>
    <tr style="background:#bdb2ff40;">
      <td style="padding:7px 10px;border:1px solid #e2e8f0;font-weight:600;">MSE</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Small batch</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;color:#14532d;">Best</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Medium (grid search)</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;font-weight:600;">W4 weights, GPTQ/AWQ default</td>
    </tr>
    <tr style="background:#caffbf40;">
      <td style="padding:7px 10px;border:1px solid #e2e8f0;font-weight:600;">KL Divergence</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Small batch</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;color:#14532d;">Best</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Medium (histogram)</td>
      <td style="padding:7px 10px;border:1px solid #e2e8f0;">Activations, TensorRT INT8 default</td>
    </tr>
  </tbody>
</table>
</div>

> **Practical recommendation.** For W4 weight quantization, MSE on a 128–512 sample calibration set is the standard. For W8A8 activation quantization, KL or percentile on a representative calibration set beats min/max significantly. Min/max is competitive only for static weights at INT8 with mild outliers.
{: .prompt-tip }

To make the difference concrete: imagine an activation tensor with 99.9% of values in $[-1, 1]$ and 0.1% of outliers reaching ±8.0. Min/max sets the INT8 range to $[-8, 8]$, giving a grid spacing of $16/255 \approx 0.063$ — every value in the bulk loses two bits of effective precision so the few outliers don't clip. MSE or KL clips the tails to roughly ±1; the 0.1% pay a clipping error of at most 7, contributing $0.001 \times 49 = 0.049$ to the mean squared error, while the 99.9% inside get a grid spacing of $2/255 \approx 0.008$ — eight times finer. The bulk is what shows up in metrics, so MSE and KL win by a wide margin.

---

We started with a 28 GB FP32 model and ended with a stack of independent knobs — bit-width, target (W/A/KV), symmetric vs. asymmetric, granularity, and range method — that compose into a single deployment recipe. The decision flow is short: confirm the model fits at the target batch and sequence including KV cache; quantize weights with per-group symmetric and MSE calibration; only quantize activations or KV cache when memory or compute demands force it; and never touch embeddings or the LM head. Everything else in this curriculum — GPTQ, AWQ, SmoothQuant, FP8, QAT — is a refinement of one of those five knobs.

---

## References

- Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — approachable visual walkthrough; good companion read for section 4.
- NVIDIA TensorRT, [Post-Training Quantization Using Calibration](https://archive.docs.nvidia.com/tensorrt/tensorrt-861/developer-guide/index.html#enable_int8_c) — TensorRT 8.6 documentation on INT8 calibration, including the KL-divergence algorithm.
- Hao Wu et al., [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) — NVIDIA paper covering quantization parameter choices including calibration range methods across vision, speech, and language models.


