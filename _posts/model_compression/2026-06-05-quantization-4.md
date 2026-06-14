---
layout: post
title: "Advanced Quantization: VLMs, Mixture-of-Experts, and the KV Cache"
date: 2026-06-05 09:00:00 +0000
categories: [Model Compression, Quantization]
tags: [quantization, vlm, mixture-of-experts, moe, kv-cache, fp8, int4, calibration, llm]
math: true
description: "Three places where the standard LLM PTQ playbook breaks: vision-language models (two activation distributions, a fragile projector), mixture-of-experts (a brittle router and unevenly-exercised experts), and the KV cache (per-channel keys, per-token values). Why each needs its own recipe, and what people actually ship."
---


*The standard recipe assumes one weight matrix and one activation distribution — each of these three architectures breaks one of those.*


<!-- Opening block: questions + prerequisites — stacked vertically -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #60a5fa;padding-left:8px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Why does a single per-tensor scale break a VLM but work on a plain LLM?</li>
      <li>Why is the MoE router the one component you must never quantize?</li>
      <li>Why are Keys quantized per-channel but Values per-token?</li>
      <li>When does the KV cache stop being optional to quantize?</li>
      <li>What do you actually <em>ship</em> for each — and what stays in FP16?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;border-left:3px solid #c4b5fd;padding-left:8px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>Weight PTQ basics: scale, zero-point, per-channel vs. per-token</li>
      <li><a href="/posts/quantization-2/">GPTQ / AWQ and the idea of calibration data</a></li>
      <li>Transformer attention and the Q/K/V projections</li>
      <li>FP8 / INT4 number formats and their memory footprint</li>
    </ul>
  </div>
</div>


The W4A16 GPTQ/AWQ recipe that works on a dense text LLM quietly fails on three increasingly common architectures. A vision-language model mixes two activation distributions in one network. A mixture-of-experts model hides a discrete switch that a rounding error can flip. The KV cache isn't weights at all — it's a runtime tensor that, at long context, grows larger than the model itself. Let's take each in turn and find the one component that has to stay in high precision.


---


## 1. Why does one calibration pass corrupt a VLM?


*Why does a calibration recipe that works on a text LLM silently degrade the language quality of a VLM?*


A VLM is not one model — it is three stacked stages with very different jobs, and two of those stages don't exist in a plain LLM.


<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="300" fill="#fffffc" rx="10"/>
  <!-- image path -->
  <rect x="20" y="30" width="90" height="34" rx="6" fill="#a0c4ff" stroke="#60a5fa"/>
  <text x="65" y="52" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="700">image</text>
  <!-- vision encoder -->
  <rect x="150" y="20" width="150" height="54" rx="8" fill="#9bf6ff" stroke="#67e8f9"/>
  <text x="225" y="42" text-anchor="middle" font-size="12.5" fill="#164e63" font-weight="700">Vision Encoder (ViT)</text>
  <text x="225" y="60" text-anchor="middle" font-size="10.5" fill="#164e63">patchify → d_vision feats · W8A8</text>
  <!-- projector -->
  <rect x="150" y="100" width="150" height="54" rx="8" fill="#ffd6a5" stroke="#f59e0b"/>
  <text x="225" y="122" text-anchor="middle" font-size="12.5" fill="#78350f" font-weight="700">Projector</text>
  <text x="225" y="140" text-anchor="middle" font-size="10.5" fill="#78350f">d_vision → d_llm · keep FP16</text>
  <!-- arrows image->enc->proj -->
  <line x1="110" y1="47" x2="148" y2="47" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="225" y1="74" x2="225" y2="98" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <!-- visual tokens -->
  <rect x="150" y="180" width="150" height="34" rx="6" fill="#caffbf" stroke="#86efac"/>
  <text x="225" y="202" text-anchor="middle" font-size="11.5" fill="#14532d" font-weight="700">visual tokens</text>
  <line x1="225" y1="154" x2="225" y2="178" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <!-- text path -->
  <rect x="20" y="180" width="90" height="34" rx="6" fill="#a0c4ff" stroke="#60a5fa"/>
  <text x="65" y="202" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="700">text</text>
  <rect x="20" y="232" width="90" height="34" rx="6" fill="#caffbf" stroke="#86efac"/>
  <text x="65" y="254" text-anchor="middle" font-size="11" fill="#14532d" font-weight="700">text tokens</text>
  <line x1="65" y1="214" x2="65" y2="230" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <!-- concat -->
  <rect x="360" y="200" width="110" height="40" rx="8" fill="#fdffb6" stroke="#facc15"/>
  <text x="415" y="224" text-anchor="middle" font-size="12" fill="#713f12" font-weight="700">concat</text>
  <line x1="300" y1="197" x2="358" y2="214" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="110" y1="249" x2="358" y2="228" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <!-- LLM decoder -->
  <rect x="510" y="195" width="110" height="50" rx="8" fill="#bdb2ff" stroke="#c4b5fd"/>
  <text x="565" y="216" text-anchor="middle" font-size="12.5" fill="#4c1d95" font-weight="700">LLM Decoder</text>
  <text x="565" y="233" text-anchor="middle" font-size="10.5" fill="#4c1d95">W4A16</text>
  <line x1="470" y1="220" x2="508" y2="220" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar)"/>
  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/>
    </marker>
  </defs>
</svg>


The image is split into patches by the **vision encoder** (a ViT), each patch becoming a `d_vision`-dim token. The **projector** maps those features into the LLM's hidden space — either a cheap 2-layer MLP (LLaVA-style, one token per patch) or a cross-attention resampler (Q-Former, Qwen merger) that compresses hundreds of patches to a fixed set. The **LLM decoder** then sees the visual tokens concatenated in front of the text tokens and decodes autoregressively. The two new stages — encoder and projector — are exactly the two new failure points for quantization.


### Why do two distributions break one scale?


*Why can't we just run the whole VLM through an existing AWQ pipeline?*


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Vision activations</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;color:#1e293b;">
      <li>Strong outliers after LayerNorm / in attention</li>
      <li>Thousands of patch tokens, mostly redundant background</li>
      <li>A few salient tokens carry the answer</li>
    </ul>
  </div>
  <div style="flex:1;min-width:220px;background:#eff6ff;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Text activations</div>
    <ul style="margin:0;padding-left:16px;line-height:1.8;color:#1e293b;">
      <li>Different dynamic range and outlier structure</li>
      <li>Far fewer tokens per sample</li>
      <li>Each token is comparatively high-value</li>
    </ul>
  </div>
</div>


Read the two cards side by side: a single per-tensor scale tuned on text will clip vision, and one tuned on vision will starve text — **because** the two modalities have genuinely different dynamic ranges and outlier patterns. That forces the first VLM rule: quantize **component-wise (modality-aware)**, not globally. The projector compounds this — every visual token flows through it, so a small error there corrupts *all* visual information before it reaches the LLM, which is why it stays in high precision despite being tiny.


> **The modality-imbalance trap.** Standard PTQ weights every calibration token equally. In a VLM the visual tokens vastly outnumber text tokens, so the reconstruction loss is dominated by vision — and the decoder gets tuned for the wrong modality while text quality silently drops.
{: .prompt-danger }


The research fix (MBQ: re-weight the reconstruction loss per modality) isn't in mainstream tooling yet. The **engineering workaround** is to control your calibration mix and validate text perplexity separately — which is why VLM evaluation needs three numbers, not one. But first, how do we keep the two halves from contaminating each other during calibration?


### How do we calibrate the two parts separately?


*Why feed the decoder FP16 visual tokens during its own calibration instead of quantized ones?*


<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="250" fill="#fffffc" rx="10"/>
  <text x="20" y="28" font-size="12.5" font-weight="700" fill="#1e293b">Calibrating the decoder — vision stays frozen in FP16</text>
  <rect x="20" y="45" width="150" height="30" rx="6" fill="#a0c4ff" stroke="#60a5fa"/>
  <text x="95" y="65" text-anchor="middle" font-size="11" fill="#1e3a8a" font-weight="700">Image+Text pairs</text>
  <!-- vision frozen -->
  <rect x="210" y="40" width="160" height="40" rx="7" fill="#9bf6ff" stroke="#67e8f9" stroke-dasharray="5,3"/>
  <text x="290" y="56" text-anchor="middle" font-size="11" fill="#164e63" font-weight="700">ViT + Projector</text>
  <text x="290" y="72" text-anchor="middle" font-size="10" fill="#164e63">FP16 · frozen · NOT quantized</text>
  <line x1="170" y1="60" x2="208" y2="60" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <!-- visual tokens -->
  <rect x="210" y="105" width="160" height="28" rx="6" fill="#caffbf" stroke="#86efac"/>
  <text x="290" y="124" text-anchor="middle" font-size="10.5" fill="#14532d" font-weight="700">realistic visual tokens</text>
  <line x1="290" y1="80" x2="290" y2="103" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <!-- text emb -->
  <rect x="20" y="105" width="150" height="28" rx="6" fill="#caffbf" stroke="#86efac"/>
  <text x="95" y="124" text-anchor="middle" font-size="10.5" fill="#14532d" font-weight="700">text token embeds</text>
  <line x1="95" y1="75" x2="95" y2="103" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <!-- concat -->
  <rect x="150" y="165" width="120" height="32" rx="7" fill="#fdffb6" stroke="#facc15"/>
  <text x="210" y="185" text-anchor="middle" font-size="11" fill="#713f12" font-weight="700">concatenate</text>
  <line x1="95" y1="133" x2="190" y2="163" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <line x1="290" y1="133" x2="232" y2="163" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <!-- decoder -->
  <rect x="360" y="158" width="260" height="48" rx="8" fill="#bdb2ff" stroke="#c4b5fd"/>
  <text x="490" y="178" text-anchor="middle" font-size="12" fill="#4c1d95" font-weight="700">LLM Decoder</text>
  <text x="490" y="195" text-anchor="middle" font-size="10.5" fill="#4c1d95">◄── quant nodes calibrated HERE</text>
  <line x1="270" y1="181" x2="358" y2="181" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar2)"/>
  <defs>
    <marker id="ar2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/>
    </marker>
  </defs>
</svg>


The decoder's activation statistics depend on what the visual tokens actually look like. If we fed it *quantized* visual tokens during calibration, we'd entangle two error sources — vision noise and decoder rounding — and couldn't tell which one a bad scale came from. Freezing the vision tower in FP16 keeps the decoder's calibration clean. The vision part is calibrated independently, driven by vision-heavy images and checked with feature SQNR.


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · Signal-to-Quantization-Noise Ratio</div>


$$
\text{SQNR}_{\text{dB}} = 10 \cdot \log_{10}\!\left( \frac{\lVert x_{\text{fp}} \rVert^2}{\lVert x_{\text{fp}} - x_{\text{quant}} \rVert^2} \right)
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $x_{\text{fp}}$ — full-precision visual features · $x_{\text{quant}}$ — features after quantizing a layer. Higher = cleaner. A sharp SQNR drop on a specific layer localizes the damage to that layer.
  </div>
</div>


Because component metrics can miss interactions, we evaluate three things — and the three scores let us attribute any degradation to vision vs. language:


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:34%;">
    <col style="width:24%;">
    <col style="width:42%;">
  </colgroup>
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Stage</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Metric</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Driven by</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#9bf6ff33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Vision (encoder + projector)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Feature SQNR</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Vision-heavy images (COCO, DocVQA)</td></tr>
    <tr style="background:#bdb2ff33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Language (decoder)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Perplexity</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Text dataset (as a normal LLM)</td></tr>
    <tr style="background:#caffbf33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">End-to-end</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">VLM benchmark</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">VQAv2, DocVQA, MMMU, TextVQA</td></tr>
  </tbody>
</table>
</div>


### What do people actually ship?


*Why is "quantize the decoder only" the right first move, and which modules go on the ignore list?*


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:22%;">
    <col style="width:22%;">
    <col style="width:56%;">
  </colgroup>
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Component</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Precision</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Why</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#ffd6a533;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Projector</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">FP16 (at most W8)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Tiny, but every visual token flows through it — huge sensitivity, negligible memory win</td></tr>
    <tr style="background:#9bf6ff33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Vision encoder</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">W8A8 safe; W4 risky</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">ViT activations have strong outliers; per-channel W + per-token A helps</td></tr>
    <tr style="background:#caffbf33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">LLM decoder</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">W4A16 (GPTQ/AWQ)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Holds the bulk of params — biggest, lowest-risk memory win; calibrate with image+text</td></tr>
  </tbody>
</table>
</div>


The lever in `llm-compressor` / `autoawq` is the **ignore list** — name the modules you do *not* quantize. Start by quantizing the decoder only; this alone captures most of the memory win because the decoder holds the bulk of the params:


```python
# llm-compressor style: quantize the decoder, leave vision + projector in FP16
recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=[
        "lm_head",
        "re:.*visual.*",   # whole vision tower (ViT blocks)
        "re:.*merger.*",   # the projector / patch merger
    ],
)
```


Only if you still need memory, quantize the vision encoder to W8 and re-measure SQNR. The projector almost never pays off — leave it.


> **Match the processor.** Calibration images must pass through the *same* resize/patchify preprocessor as deployment, or the activation statistics are wrong. This is one of the most common VLM quantization bugs.
{: .prompt-warning }


Reading Qwen2.5-VL-7B through this lens makes the recipe concrete: its `merger` is `ln_q → Linear(5120→5120) → GELU → Linear(5120→3584)`, where `5120 = 1280×4` concatenates a 2×2 block of patches (4× fewer visual tokens) and projects into the 3584-dim Qwen2.5-7B hidden size. That `merger` *is* the projector — the one block to keep in high precision — while the 32 ViT blocks at `d_vision = 1280` are the W8A8 candidates. The VLM has two distributions; the next architecture hides something worse — a discrete switch.


---


## 2. Why is the MoE router the one thing you can't quantize?


*Why is a 47B-parameter MoE that only runs 13B per token still a memory problem quantization must solve?*


A dense model ties parameter count to compute: every token activates all parameters. MoE breaks that link — it scales *total* parameters while keeping *per-token* compute roughly constant by activating only a few experts per token. That gives a MoE two parameter counts that matter, and they pull in opposite directions.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Mixtral 8×7B — total vs. active parameters</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">top-2 of 8 experts routed per token</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:150px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">Total (must store)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">≈ 47B</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:150px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">Active (runs/token)</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:28%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">≈ 13B</span>
    </div>
  </div>
</div>


Read the gap between the two bars: total params (rose) are what we must *store* and quantize; active params (mint) are what runs per token. The gap is the whole point — and the whole problem. Every expert must sit in VRAM even though most are idle for any given token, so the pain is memory/storage, and quantization attacks memory directly. **This is why MoE is a quantization magnet.**


### What changes when an FFN becomes a router plus N experts?


*Why does replacing one FFN with a router and an expert pool change which component is fragile?*


<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="250" fill="#fffffc" rx="10"/>
  <rect x="280" y="20" width="80" height="30" rx="6" fill="#a0c4ff" stroke="#60a5fa"/>
  <text x="320" y="40" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="700">token x</text>
  <!-- router -->
  <rect x="250" y="75" width="140" height="42" rx="8" fill="#ffadad" stroke="#fca5a5"/>
  <text x="320" y="93" text-anchor="middle" font-size="12.5" fill="#7f1d1d" font-weight="700">Router (gate)</text>
  <text x="320" y="109" text-anchor="middle" font-size="10" fill="#7f1d1d">softmax over E · pick top-k · KEEP FP16</text>
  <line x1="320" y1="50" x2="320" y2="73" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar3)"/>
  <!-- experts -->
  <rect x="40" y="155" width="100" height="40" rx="7" fill="#caffbf" stroke="#86efac"/>
  <text x="90" y="179" text-anchor="middle" font-size="11" fill="#14532d" font-weight="700">Expert 3 ✓</text>
  <rect x="160" y="155" width="100" height="40" rx="7" fill="#f1f5f9" stroke="#cbd5e1"/>
  <text x="210" y="179" text-anchor="middle" font-size="11" fill="#94a3b8" font-weight="600">Expert idle</text>
  <rect x="280" y="155" width="100" height="40" rx="7" fill="#f1f5f9" stroke="#cbd5e1"/>
  <text x="330" y="179" text-anchor="middle" font-size="11" fill="#94a3b8" font-weight="600">Expert idle</text>
  <rect x="400" y="155" width="100" height="40" rx="7" fill="#caffbf" stroke="#86efac"/>
  <text x="450" y="179" text-anchor="middle" font-size="11" fill="#14532d" font-weight="700">Expert 7 ✓</text>
  <rect x="520" y="155" width="100" height="40" rx="7" fill="#bdb2ff" stroke="#c4b5fd"/>
  <text x="570" y="173" text-anchor="middle" font-size="10.5" fill="#4c1d95" font-weight="700">Shared</text>
  <text x="570" y="187" text-anchor="middle" font-size="9.5" fill="#4c1d95">always on</text>
  <!-- routes -->
  <line x1="300" y1="117" x2="100" y2="153" stroke="#86efac" stroke-width="2.5" marker-end="url(#ar3g)"/>
  <line x1="340" y1="117" x2="445" y2="153" stroke="#86efac" stroke-width="2.5" marker-end="url(#ar3g)"/>
  <line x1="360" y1="110" x2="560" y2="153" stroke="#c4b5fd" stroke-width="2.5" stroke-dasharray="4,3" marker-end="url(#ar3v)"/>
  <!-- weighted sum -->
  <rect x="250" y="215" width="140" height="28" rx="7" fill="#fdffb6" stroke="#facc15"/>
  <text x="320" y="234" text-anchor="middle" font-size="11" fill="#713f12" font-weight="700">weighted sum → output</text>
  <line x1="90" y1="195" x2="290" y2="214" stroke="#94a3b8" stroke-width="1.5"/>
  <line x1="450" y1="195" x2="350" y2="214" stroke="#94a3b8" stroke-width="1.5"/>
  <defs>
    <marker id="ar3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker>
    <marker id="ar3g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#86efac"/></marker>
    <marker id="ar3v" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c4b5fd"/></marker>
  </defs>
</svg>


The MoE layer replaces the FFN with a **router** (a small linear `d_model → E` plus softmax that picks top-k experts per token) and an **expert pool** (each a standard smaller FFN). Some architectures (DeepSeek-MoE, Qwen-MoE) add a **shared expert** every token uses, in addition to the routed ones. The router introduces something a dense model never had: a discrete switch.


> **The router is a brittle discrete switch.** A tiny perturbation to the gate logits can flip *which* experts a token goes to. A flip is discontinuous and catastrophic — a completely different FFN runs — far worse than the smooth error of quantizing a weight. Keep the router in FP16.
{: .prompt-danger }


### Why does a normal calibration set leave half the experts uncalibrated?


*Why does a calibration set that's fine for a dense model starve a MoE's cold experts?*


<div style="display:flex;flex-direction:column;gap:10px;margin:18px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #f59e0b;border-left:4px solid #f59e0b;border-radius:0 6px 6px 0;padding:12px 16px;">
    <div style="font-weight:700;color:#78350f;margin-bottom:4px;">1 · Calibration imbalance across experts</div>
    <div style="color:#1e293b;line-height:1.6;">Routing is sparse and skewed — a few "hot" experts get hammered, "cold" ones barely move. Cold experts then get garbage scales/Hessians from almost no tokens. Each expert only sees ~k/E of the stream, so you need <em>far more</em> calibration tokens than a dense model, and you track per-expert counts to top up the rare ones.</div>
  </div>
  <div style="background:#fffffc;border:1px solid #fca5a5;border-left:4px solid #fca5a5;border-radius:0 6px 6px 0;padding:12px 16px;">
    <div style="font-weight:700;color:#7f1d1d;margin-bottom:4px;">2 · Router sensitivity — highest stakes</div>
    <div style="color:#1e293b;line-height:1.6;">Keep the gate in FP16 by default. If you must quantize it, use ≥8-bit and validate <em>routing-agreement</em> (fraction of tokens whose top-k is unchanged), not just perplexity.</div>
  </div>
  <div style="background:#fffffc;border:1px solid #86efac;border-left:4px solid #86efac;border-radius:0 6px 6px 0;padding:12px 16px;">
    <div style="font-weight:700;color:#14532d;margin-bottom:4px;">3 · Mixed precision across experts</div>
    <div style="color:#1e293b;line-height:1.6;">Experts aren't equally important. Shared experts (every token) and hot routed experts get more bits; cold experts are under-calibrated, so their errors go uncorrected. You only know by measuring — log a routing histogram.</div>
  </div>
</div>


To allocate bits by importance we need data, not guesses. During calibration, log a **routing histogram**: per expert, count tokens routed and accumulate gate-weight mass. Then allocate precision by an importance score:


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Heuristic · Per-expert importance</div>


$$
\text{importance}(e) \approx \text{frequency}(e) \times \overline{\text{gate-weight}}(e)
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $\text{frequency}(e)$ — fraction of calibration tokens routed to expert $e$ · $\overline{\text{gate-weight}}(e)$ — mean gate probability when it is selected. High score → more bits. Granularity ranges from coarse (whole MoE layer) to fine (per linear within an expert).
  </div>
</div>


### What do people actually ship?


*Why is FP8 round-to-nearest — no calibration at all — the right first move on a big MoE?*


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:30%;">
    <col style="width:22%;">
    <col style="width:48%;">
  </colgroup>
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Component</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Precision</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#ffadad33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Router / gate, lm_head, embeddings</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">FP16</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Never quantize the gate by default — a flip is catastrophic</td></tr>
    <tr style="background:#bdb2ff33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Shared expert</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Bump precision</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Touched by every token → high impact</td></tr>
    <tr style="background:#caffbf33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Routed experts (bulk of bytes)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">FP8 → W4A16</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">FP8 RTN first (no calibration); W4 only when FP8 doesn't fit</td></tr>
  </tbody>
</table>
</div>


FP8 (block/dynamic) is the robust default for big MoEs — DeepSeek-V3, Kimi-K2, and Qwen3-30B-A3B all ship FP8, it's RTN with *no calibration needed*, and it roughly halves VRAM with near-zero quality loss. That's often exactly what makes a big MoE fit on your box at all. Push to W4A16 (GPTQ/AWQ) only under tighter memory pressure, where per-expert calibration coverage becomes the hard part. The gate stays out of it:


```python
recipe = QuantizationModifier(
    scheme="FP8_BLOCK",                     # or W4A16 for tighter memory
    targets="Linear",
    ignore=["lm_head", "re:.*mlp.gate$"],   # <-- gate stays FP16
)
```


> **Validate routing-agreement, not just perplexity.** PPL can look fine while routing silently degrades on tail inputs. Measure the fraction of tokens whose top-k selection is unchanged, plus a task eval.
{: .prompt-tip }


The VLM and MoE both hide a fragile *weight*. The last case is stranger — the fragile tensor isn't a weight at all.


---


## 3. Why does a non-weight tensor become the memory bottleneck?


*Why does a tensor that isn't part of the model weights become the dominant memory consumer at long context?*


In autoregressive decoding, generating token $t$ needs attention over all previous tokens $1..t-1$. Their Keys and Values don't change, so we compute each token's K and V **once** and cache them — turning what would be $O(t^2)$ recomputation into appending one new K/V per step. We trade compute for memory, and that memory is the new problem.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="font-weight:700;color:#1e293b;margin-bottom:4px;">Llama-2-7B — KV cache vs. weights as context grows</div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:14px;">FP16 cache, single sequence · weights ≈ 13 GB</div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:130px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">Weights</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:81%;height:100%;background:#a0c4ff;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">13 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:130px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">KV @ 4k ctx</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:12.5%;height:100%;background:#caffbf;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">2 GB</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="width:130px;text-align:right;font-size:12px;color:#374151;flex-shrink:0;">KV @ 32k ctx</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px;overflow:hidden;">
        <div style="width:100%;height:100%;background:#ffadad;border-radius:4px;"></div>
      </div>
      <span style="width:60px;font-size:12px;font-weight:600;color:#1e293b;flex-shrink:0;">16 GB</span>
      <span style="background:#ffadad;color:#7f1d1d;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid #fca5a5;">&gt; weights</span>
    </div>
  </div>
</div>


Walk down the bars: at 4k context the cache (mint) is a minor 2 GB; at 32k it hits ~16 GB (rose) — *more than the 13 GB of weights*. The cache grows linearly with both sequence length and batch size, which is exactly what makes it explode in long-context and high-throughput serving:


<div style="background:#fffffc;border:1px solid #bdb2ff;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · KV cache size</div>


$$
\text{KV bytes} = 2 \times L \times s \times h_{\text{kv}} \times d_{\text{head}} \times b \times \text{bytes}
$$


  <div style="font-size:12px;color:#1e293b;margin-top:10px;line-height:1.7;">
    $2$ — K and V · $L$ — layers · $s$ — sequence length · $h_{\text{kv}}$ — KV heads · $d_{\text{head}}$ — head dim · $b$ — batch · grows linearly in both $s$ and $b$. Llama-2-7B at 4k: $2\times32\times4096\times32\times128\times1\times2 \approx 2$ GB.
  </div>
</div>


> **GQA/MQA and KV quantization stack.** GQA cuts $h_{\text{kv}}$ (sharing K/V across query heads); quantization cuts the bytes-per-element. They are orthogonal — apply both.
{: .prompt-info }


You *must* quantize the KV cache when context is long (the cache dwarfs weights), when serving large batches (cache caps how many requests you can pack), or when fitting a long-context model on a single GPU. But the cache isn't one uniform tensor — Keys and Values behave differently, and that asymmetry sets the recipe.


### Why can't the same axis serve both Keys and Values?


*Why do Keys and Values demand different quantization axes?*


<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="640" height="250" fill="#fffffc" rx="10"/>
  <text x="20" y="26" font-size="12.5" font-weight="700" fill="#1e293b">K-cache — outliers live in fixed CHANNELS → quantize per-channel (down columns)</text>
  <text x="320" y="44" text-anchor="middle" font-size="10.5" fill="#64748b">head_dim (channels) ──►</text>
  <!-- K grid -->
  <g font-size="12" font-family="monospace">
    <!-- highlight column index 4 -->
    <rect x="190" y="52" width="26" height="78" fill="#ffadad" stroke="#fca5a5"/>
    <text x="60" y="72" fill="#1e293b">tok 0</text>
    <text x="60" y="96" fill="#1e293b">tok 1</text>
    <text x="60" y="120" fill="#1e293b">tok 2</text>
  </g>
  <g font-size="13" font-family="monospace" fill="#475569">
    <text x="115" y="72">k k k k</text><text x="203" y="72" fill="#7f1d1d" font-weight="700">K</text><text x="220" y="72">k k k</text>
    <text x="115" y="96">k k k k</text><text x="203" y="96" fill="#7f1d1d" font-weight="700">K</text><text x="220" y="96">k k k</text>
    <text x="115" y="120">k k k k</text><text x="203" y="120" fill="#7f1d1d" font-weight="700">K</text><text x="220" y="120">k k k</text>
  </g>
  <text x="203" y="146" text-anchor="middle" font-size="10" fill="#7f1d1d">always large</text>
  <line x1="203" y1="150" x2="203" y2="128" stroke="#fca5a5" stroke-width="1.5" marker-end="url(#aru)"/>
  <!-- V grid -->
  <text x="20" y="180" font-size="12.5" font-weight="700" fill="#1e293b">V-cache — no fixed outlier channels → quantize per-token (across each row ►►►)</text>
  <g>
    <rect x="110" y="192" width="180" height="20" fill="#caffbf" stroke="#86efac"/>
    <text x="300" y="207" font-size="10.5" fill="#14532d">◄ new token = new group, O(1) append</text>
  </g>
  <text x="60" y="207" font-size="12" font-family="monospace" fill="#1e293b">tok t</text>
  <defs><marker id="aru" markerWidth="8" markerHeight="8" refX="3" refY="6" orient="auto"><path d="M0,6 L3,0 L6,6 Z" fill="#fca5a5"/></marker></defs>
</svg>


The central empirical finding (KIVI, KVQuant): **Keys and Values have different distributions, so they need different quantization axes.** Key activations have persistent outlier channels — a few feature dimensions are consistently large across all tokens (the rose column above). Quantizing per-token (across channels) would let one outlier channel blow up the scale for the whole token and destroy the small channels; quantizing *per-channel* keeps each outlier in its own group. Values show no such fixed-channel structure, and per-token is both what the attention output $\sum \text{softmax}\cdot v$ tolerates and the natural streaming axis. The rule to memorize: **per-channel Keys, per-token Values.**


> **Quantize Keys pre-RoPE.** RoPE rotates the key vector by a position-dependent angle, mixing channels and smearing the clean per-channel outlier structure. KVQuant stores Keys *before* RoPE and applies rotation after dequant, so the grouping still lines up with the outliers — a real accuracy win at low bits.
{: .prompt-tip }


Two more wrinkles make low-bit work: a **dense-and-sparse** split keeps a tiny fraction of outlier elements in full precision (KVQuant reaches 3-bit with <0.1 PPL loss this way), and a common compromise keeps a small **sliding window of recent tokens in FP16** — they're attended most and most sensitive — quantizing only the older bulk.


### Which precision do you reach for, and where's the knob?


*Why is FP8 KV the default we reach for before anything fancier?*


<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:26%;">
    <col style="width:14%;">
    <col style="width:60%;">
  </colgroup>
  <thead>
    <tr style="background:#f1f5f9;">
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Scheme</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Shrink</th>
      <th style="padding:8px 12px;border:1px solid #e2e8f0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:600;">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#caffbf33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;"><strong>FP8 KV</strong></td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">~2×</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Hardware-supported on Hopper/Blackwell, near-lossless. <strong>Your default</strong> — one flag in vLLM/TRT-LLM.</td></tr>
    <tr style="background:#9bf6ff33;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">INT8 KV</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">~2×</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Generally safe with per-channel K / per-token V.</td></tr>
    <tr style="background:#fdffb633;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">INT4 KV</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">~4×</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Needs the asymmetric axes + grouping to hold accuracy.</td></tr>
    <tr style="background:#ffd6a533;"><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">2–3 bit (KIVI / KVQuant)</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">up to ~4×+</td><td style="padding:7px 12px;border:1px solid #e2e8f0;white-space:normal;overflow-wrap:break-word;">Research-grade, custom CUDA kernels. KIVI 2-bit tuning-free; KVQuant 3-bit → 1M ctx on one A100-80GB. Only if FP8/INT8 don't suffice.</td></tr>
  </tbody>
</table>
</div>


As an engineer you almost never implement the asymmetric per-channel/per-token math yourself — the serving engine does it, and the knob is a config flag. In **vLLM** it's `--kv-cache-dtype fp8` (INT8 via the quantization config); in **TensorRT-LLM** you enable FP8/INT8 KV in the build config, which also needs calibration scales for INT8. Reach for the KIVI/KVQuant repos only when you need sub-4-bit and can absorb the maintenance cost of shipping their custom kernels.


> **Evaluate on long context, not short PPL.** The damage shows up at long range — retrieval/needle-in-a-haystack, RULER, LongBench — and grows with sequence length as error accumulates over the cached history. Short-prompt perplexity can look perfect while 64k-context recall collapses.
{: .prompt-warning }


The decision order is the same shape in all three cases: try the cheap, robust thing first (FP8 KV / FP8 experts / decoder-only VLM), measure the metric that actually catches the failure (long-context recall / routing-agreement / per-stage SQNR), and only spend more bits — or more calibration effort — when that measurement says you must.


---


## References


1. Li, S., Ning, X., Hong, K., et al. (2024). **MBQ: Modality-Balanced Quantization for Large Vision-Language Models.** *arXiv:2412.19509.* [https://arxiv.org/abs/2412.19509](https://arxiv.org/abs/2412.19509) · [code](https://github.com/thu-nics/MBQ)
2. Wang, C., Cheng, G., Cheng, D., et al. (2024). **Q-VLM: Post-training Quantization for Large Vision-Language Models.** *arXiv:2410.08119.* [https://arxiv.org/abs/2410.08119](https://arxiv.org/abs/2410.08119)
3. Li, P., Jin, X., Cheng, Y., & Chen, T. (2024). **Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark.** *arXiv:2406.08155.* [https://arxiv.org/abs/2406.08155](https://arxiv.org/abs/2406.08155)
4. Frantar, E., & Alistarh, D. (2023). **QMoE: Sub-1-Bit Compression of Trillion-Parameter Models.** *arXiv:2310.16795.* [https://arxiv.org/abs/2310.16795](https://arxiv.org/abs/2310.16795)
5. **llm-compressor** (vLLM project) — MoE / FP8 recipes, gate ignore lists. [https://github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
6. Liu, Z., Yuan, J., Jin, H., et al. (2024). **KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache.** *arXiv:2402.02750.* [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)
7. Hooper, C., Kim, S., Mohammadzadeh, H., et al. (2024). **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.** *arXiv:2401.18079.* [https://arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079)



