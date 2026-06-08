---
layout: post
title: "Lab02: AWQ and GPTQ Under the Hood: Inside the Transformer Block Loop"
date: 2026-06-07 09:00:00 +0000
categories: [Model Compression, Quantization, Lab]
tags: [quantization, awq, gptq, wqlinear, int4, w4a16, implementation, qwen3, llm]
math: true
---


<!-- Opening block: questions + prerequisites -->
<div style="display:flex;flex-direction:column;gap:14px;margin:24px 0 28px;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Questions this post answers</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li>How does AutoAWQ group the seven linear layers inside a transformer block, and why does each group use a different loss signal?</li>
      <li>What does <code>WQLinear</code> actually store — how are two INT4 weights packed into one uint8 byte?</li>
      <li>What is AWQ's second optimization pass, and why are Q/K projections excluded from it?</li>
      <li>Why does GPTQ flush column errors in batches rather than one column at a time?</li>
    </ul>
  </div>
  <div style="background:#fffffc;border:1px solid #bdb2ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Prerequisites</div>
    <ul style="margin:0;padding-left:16px;color:#1e293b;line-height:1.8;">
      <li><a href="/posts/gptq-awq/">GPTQ and AWQ: high-level algorithms</a> — read first</li>
      <li>PyTorch <code>nn.Module</code>, <code>register_buffer</code>, forward hooks</li>
      <li>Per-group INT4 quantization: scale, zero-point, <code>group_size=128</code></li>
      <li>Transformer block structure: QKV attention + SwiGLU FFN</li>
    </ul>
  </div>
</div>


The high-level post explained *what* AWQ and GPTQ do. This post traces *how*: the exact per-block execution loop, the bit-level layout of `WQLinear`'s packed buffers, and the calibration sequencing decision.


---


## 1. AWQ: What Happens Inside Each Transformer Block


*Why can't AWQ run an independent scale search on every linear layer — what breaks if it does?*


### Naive approach: per-layer independent scales


From the AWQ paper, the scale formula for each input channel $i$ is:


$$
s_i = \text{mean}(|x_i|)^\alpha, \quad \alpha \in [0, 1]
$$


You could apply this independently to every linear layer — compute each layer's activation statistics, run a grid search over $\alpha$, pick the best scale, absorb it into that layer's preceding operation. The paper derives it per-layer, and it works mathematically.


**But you immediately hit two problems in practice.**


**Runtime overhead.** The AWQ scale $s_i$ must be applied to the input before each GEMM: $\hat{x} = x / s$, then $\hat{x} \cdot (W \cdot s)^\top$. That element-wise divide adds a kernel launch per linear layer. With 7 projections per block and 36 blocks, a 4B model runs 252 extra divide kernels — one before every GEMM. There is no hardware op that fuses a per-channel divide into the INT4 dequantization path, so this overhead is real and unrecoverable.


**The fusion escape.** For the `q_proj`, `k_proj`, and `v_proj` projections, all three read the same `input_layernorm` output. If you apply three different scales — one per projection — each one needs its own correction upstream. That means you'd have to divide the LayerNorm output three different ways, which is not physically possible from a single LayerNorm. The only way to get zero runtime cost is to absorb *one* scale into the LayerNorm's gain weights, so all three projections downstream inherit the correction for free.


**This is about cost, not accuracy.** Fusing the scale into a preceding op and applying it as a runtime divide are mathematically identical: $(x / s) \cdot (W s)^\top = x W^\top$ holds exactly, so fusion changes neither the quantization grid nor the layer output — it only decides whether the $/s$ correction costs an extra kernel per GEMM or nothing at all. The grouping therefore isn't chasing accuracy; it's chasing *fusibility*. A scale can only be folded away for free when something upstream can absorb it, and the only free absorbers are a shared LayerNorm or a single sibling projection. That is why AutoAWQ binds q/k/v to one scale (foldable into `input_layernorm`) and gate/up to one (foldable into `post_attention_layernorm`): the three QKV projections read identical input statistics, so collapsing them to a single shared scale costs little reconstruction error versus three independent scales — while buying the zero-cost fold that independent per-projection scales could never have.


### Block structure: what shares what {#block-structure}


To see which projections can share a scale, you need to know which ones share the same input. A standard decoder block has two fan-out points — the only two places where one normalized tensor fans out to multiple projections.


<div style="background:#fffffc;border:1px solid #d1d5db;border-radius:10px;padding:20px;margin:20px 0;font-family:system-ui,sans-serif;">
  <div style="font-weight:700;color:#1f2328;font-size:13px;margin-bottom:10px;">Decoder block — four AWQ groups, two nonlinearity boundaries</div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;font-size:11px;font-family:system-ui,sans-serif;">
    <span style="background:#dbeafe;color:#1e3a8a;border:1px solid #93c5fd;border-radius:4px;padding:2px 8px;font-weight:600;">Group 1 · QKV</span>
    <span style="background:#ffedd5;color:#7c2d12;border:1px solid #fdba74;border-radius:4px;padding:2px 8px;font-weight:600;">Group 2 · o_proj</span>
    <span style="background:#dcfce7;color:#14532d;border:1px solid #86efac;border-radius:4px;padding:2px 8px;font-weight:600;">Group 3 · gate + up</span>
    <span style="background:#ede9fe;color:#4c1d95;border:1px solid #c4b5fd;border-radius:4px;padding:2px 8px;font-weight:600;">Group 4 · down</span>
  </div>
  <pre style="background:#0d1117;color:#c9d1d9;border-radius:8px;padding:18px 20px;font-size:12px;line-height:1.9;overflow-x:auto;margin:0;">  Residual stream (x)
          │
    input_layernorm (LN1)  ◀── <span style="color:#60a5fa;font-weight:600;">fan-out: q_proj, k_proj, v_proj all read LN1(x)</span>
          │
          ├──────────┬──────────┐
          │          │          │
       <span style="color:#60a5fa;font-weight:600;">q_proj</span>    <span style="color:#60a5fa;font-weight:600;">k_proj</span>    <span style="color:#60a5fa;font-weight:600;">v_proj</span>     <span style="color:#60a5fa;">← Group 1</span>
          │          │          │
          └──────────┴────┬─────┘
                          │
              softmax(Q·Kᵀ/√d) × V   ◀── <span style="color:#ef4444;font-weight:600;">NONLINEARITY — scale fusion boundary</span>
                          │
                       <span style="color:#fb923c;font-weight:600;">o_proj</span>              <span style="color:#fb923c;">← Group 2</span>
          │
      + residual
          │
    post_attn_layernorm (LN2)  ◀── <span style="color:#4ade80;font-weight:600;">fan-out: gate_proj and up_proj both read LN2(x)</span>
          │
          ├──────────────────────┐
          │                      │
       <span style="color:#4ade80;font-weight:600;">gate_proj</span>           <span style="color:#4ade80;font-weight:600;">up_proj</span>       <span style="color:#4ade80;">← Group 3</span>
          │                      │
     SiLU(gate) ───────── × (up)   ◀── <span style="color:#ef4444;font-weight:600;">NONLINEARITY — scale fusion boundary</span>
                          │
                       <span style="color:#c4b5fd;font-weight:600;">down_proj</span>           <span style="color:#c4b5fd;">← Group 4</span>
          │
      + residual → next block</pre>
</div>


The two nonlinearities — softmax attention and SiLU gate — divide the block into three activation zones. A scale can only be fused across projections that share the same input within one zone; it cannot cross a nonlinearity boundary.


### How AutoAWQ groups instead


AWQ's scale trick works because it fuses `s` into the preceding operation (LayerNorm or a sibling projection) so inference needs zero extra ops. That fusion only makes sense when multiple projections share the same input — you can fold one scale into one LayerNorm and have all downstream projections benefit.


AutoAWQ implements this through four fixed groups per transformer block:


<div style="display:flex;flex-direction:column;gap:10px;margin:16px 0;font-family:system-ui,sans-serif;font-size:13px;">


  <div style="border:1px solid #93c5fd;border-left:4px solid #1e3a8a;border-radius:6px;padding:12px 16px;background:#fffffc;">
    <div style="font-weight:700;color:#1e3a8a;font-size:14px;margin-bottom:10px;">Group 1 · QKV &nbsp;<code style="font-size:11px;font-weight:400;">q_proj, k_proj, v_proj</code></div>
    <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:10px;">
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#1e3a8a;border-radius:3px;padding:1px 6px;flex-shrink:0;">Input</span>
        <span style="color:#1e293b;font-size:12px;"><code>input_layernorm</code> output — shared by all three projections; scale folded into LN1 weights</span>
      </div>
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#1e3a8a;border-radius:3px;padding:1px 6px;flex-shrink:0;">Loss</span>
        <span style="color:#1e293b;font-size:12px;"><code>self_attn</code> full output MSE — softmax + V weighting included</span>
      </div>
    </div>
    <div style="color:#475569;line-height:1.6;font-size:12px;border-top:1px solid #e2e8f0;padding-top:8px;"><strong>Why this loss:</strong> Q and K quantization errors interact through $QK^\top$, so they cannot be measured independently — using the full <code>self_attn</code> output MSE captures that interaction where per-projection MSE, measured on Q and K in isolation, would miss it.</div>
  </div>


  <div style="border:1px solid #fdba74;border-left:4px solid #c2410c;border-radius:6px;padding:12px 16px;background:#fffffc;">
    <div style="font-weight:700;color:#7c2d12;font-size:14px;margin-bottom:10px;">Group 2 · o_proj &nbsp;<code style="font-size:11px;font-weight:400;">o_proj</code></div>
    <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:10px;">
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#c2410c;border-radius:3px;padding:1px 6px;flex-shrink:0;">Input</span>
        <span style="color:#1e293b;font-size:12px;">concatenated attention head outputs — downstream of softmax, no shared sibling</span>
      </div>
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#c2410c;border-radius:3px;padding:1px 6px;flex-shrink:0;">Loss</span>
        <span style="color:#1e293b;font-size:12px;"><code>o_proj</code> output MSE directly</span>
      </div>
    </div>
    <div style="color:#475569;line-height:1.6;font-size:12px;border-top:1px solid #e2e8f0;padding-top:8px;"><strong>Why standalone:</strong> no sibling projection shares this input, so there is no LayerNorm to fold into — the scale is absorbed directly into the o_proj weight columns.</div>
  </div>


  <div style="border:1px solid #86efac;border-left:4px solid #14532d;border-radius:6px;padding:12px 16px;background:#fffffc;">
    <div style="font-weight:700;color:#14532d;font-size:14px;margin-bottom:10px;">Group 3 · gate + up &nbsp;<code style="font-size:11px;font-weight:400;">gate_proj, up_proj</code></div>
    <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:10px;">
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#14532d;border-radius:3px;padding:1px 6px;flex-shrink:0;">Input</span>
        <span style="color:#1e293b;font-size:12px;"><code>post_attn_layernorm</code> output — shared by both projections; scale folded into LN2 weights</span>
      </div>
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#14532d;border-radius:3px;padding:1px 6px;flex-shrink:0;">Loss</span>
        <span style="color:#1e293b;font-size:12px;"><code>SiLU(gate_q) × up_q</code> vs <code>SiLU(gate) × up</code> — joint SwiGLU output MSE</span>
      </div>
    </div>
    <div style="color:#475569;line-height:1.6;font-size:12px;border-top:1px solid #e2e8f0;padding-top:8px;"><strong>Why joint loss:</strong> gate_proj selectively zeroes features of up_proj via element-wise multiply — a small error in gate can silently kill high-energy up features; independent per-projection MSE misses this interaction entirely.</div>
  </div>


  <div style="border:1px solid #c4b5fd;border-left:4px solid #4c1d95;border-radius:6px;padding:12px 16px;background:#fffffc;">
    <div style="font-weight:700;color:#4c1d95;font-size:14px;margin-bottom:10px;">Group 4 · down &nbsp;<code style="font-size:11px;font-weight:400;">down_proj</code></div>
    <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:10px;">
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#4c1d95;border-radius:3px;padding:1px 6px;flex-shrink:0;">Input</span>
        <span style="color:#1e293b;font-size:12px;">SwiGLU output (FFN intermediate) — downstream of SiLU gate, no shared sibling</span>
      </div>
      <div style="display:flex;gap:8px;align-items:baseline;">
        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#fff;background:#4c1d95;border-radius:3px;padding:1px 6px;flex-shrink:0;">Loss</span>
        <span style="color:#1e293b;font-size:12px;"><code>down_proj</code> output MSE directly</span>
      </div>
    </div>
    <div style="color:#475569;line-height:1.6;font-size:12px;border-top:1px solid #e2e8f0;padding-top:8px;"><strong>Why standalone:</strong> maps the high-dim FFN intermediate back to hidden dim; no sibling projection shares its input, so the scale is absorbed directly into down_proj weight columns.</div>
  </div>


</div>


Each group runs an independent α grid-search (20 candidates, $\alpha \in [0.0, 0.95]$) to find the per-channel scale $s = \text{mean}(\lvert x \rvert)^\alpha$, normalized so its geometric mean is 1. The scale is then folded into the preceding LayerNorm's gain weights — the division correction emerges from LayerNorm for free at every forward pass.


These four groups are not inferred at runtime — they are hand-written per architecture. Here is the Qwen3 definition verbatim:


```python
# awq/models/qwen3.py
@staticmethod
def get_layers_for_scaling(module, input_feat, module_kwargs):
    layers = []


    # Group 1 — q/k/v share input_layernorm; scale folds into LN1
    layers.append(dict(
        prev_op=module.input_layernorm,
        layers=[module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj],
        inp=input_feat["self_attn.q_proj"],
        module2inspect=module.self_attn, kwargs=module_kwargs,  # full-attention loss
    ))


    # Group 2 — o_proj; only when v and o shapes match (scale folds into v_proj)
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat["self_attn.o_proj"],
        ))


    # Group 3 — gate/up share post_attention_layernorm; joint SwiGLU loss
    layers.append(dict(
        prev_op=module.post_attention_layernorm,
        layers=[module.mlp.gate_proj, module.mlp.up_proj],
        inp=input_feat["mlp.gate_proj"],
        module2inspect=module.mlp,
    ))


    # Group 4 — down_proj; scale folds into up_proj
    layers.append(dict(
        prev_op=module.mlp.up_proj,
        layers=[module.mlp.down_proj],
        inp=input_feat["mlp.down_proj"],
    ))


    return layers
```


Two details map directly onto the loss column above. Groups 1 and 3 set `module2inspect` (`self_attn` and `mlp`) — that is what makes the loss the *full module output* MSE rather than a single projection's. Groups 2 and 4 omit it, so the loss is the projection's own output. And Group 2's `if` guard is why o_proj scaling silently disappears under GQA: when the KV projection is smaller than o_proj, the shapes differ and the group is never appended.


> **Source.** `awq/models/qwen3.py → get_layers_for_scaling()` returns exactly these four group dicts. Each group's actual linear inputs are captured via forward hooks in `awq/quantize/quantizer.py → _get_input_feat()`, not approximated from the layer hidden state.
{: .prompt-info }


---


## 2. WQLinear: How Packed INT4 Is Actually Stored


*Why does the quantized model hold uint8 buffers when the compression target is INT4 — and what does the forward pass look like?*


> For a hands-on walkthrough of INT4 packing and unpacking from scratch, see [Lab 1: W4A16 from Scratch](/posts/lab1-w4a16-scratch/).
{: .prompt-tip }


<svg viewBox="0 0 620 152" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:640px;display:block;margin:16px auto;font-family:system-ui,sans-serif;">
  <rect width="620" height="152" fill="#fffffc" rx="10"/>
  <text x="310" y="26" font-size="13" font-weight="700" fill="#1e293b" text-anchor="middle">WQLinear: two INT4 weights packed into one uint8 byte</text>
  <!-- High nibble block -->
  <rect x="20" y="38" width="280" height="56" fill="#a0c4ff" rx="6" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="160" y="61" font-size="12" font-weight="700" text-anchor="middle" fill="#1e3a8a">HIGH NIBBLE — bits 7:4</text>
  <text x="160" y="77" font-size="11" text-anchor="middle" fill="#1e3a8a">(byte &gt;&gt; 4) &amp; 0xF  →  w_odd</text>
  <text x="160" y="89" font-size="10" text-anchor="middle" fill="#1e3a8a">output column 2j+1</text>
  <!-- divider -->
  <line x1="308" y1="34" x2="308" y2="98" stroke="#64748b" stroke-width="2" stroke-dasharray="5,3"/>
  <!-- Low nibble block -->
  <rect x="320" y="38" width="280" height="56" fill="#caffbf" rx="6" stroke="#86efac" stroke-width="1.5"/>
  <text x="460" y="61" font-size="12" font-weight="700" text-anchor="middle" fill="#14532d">LOW NIBBLE — bits 3:0</text>
  <text x="460" y="77" font-size="11" text-anchor="middle" fill="#14532d">byte &amp; 0xF  →  w_even</text>
  <text x="460" y="89" font-size="10" text-anchor="middle" fill="#14532d">output column 2j</text>
  <!-- pack/unpack formula -->
  <rect x="20" y="104" width="580" height="38" fill="#fdffb6" rx="6"/>
  <text x="310" y="120" font-size="11" text-anchor="middle" fill="#713f12" font-weight="600">Pack:  byte = (w_even &amp; 0xF) | ((w_odd &amp; 0xF) &lt;&lt; 4)</text>
  <text x="310" y="136" font-size="11" text-anchor="middle" fill="#713f12">Unpack sign extension:  if  w &gt; 7  →  w −= 16   (recovers INT4 range [−8, 7])</text>
</svg>


GPU memory does not have a native 4-bit data type, so two INT4 values are co-located in one uint8 byte. `qweight` stores pairs of adjacent output columns: even column `2j` lands in the low nibble, odd column `2j+1` in the high nibble. The matrix shape folds from `(in, out)` to `(in, out/2)` uint8 — exactly half the bytes. After unpacking, values in `[8, 15]` are sign-extended back to `[-8, -1]` by subtracting 16.


The four registered buffers and their roles:


<div style="display:flex;gap:10px;flex-wrap:wrap;margin:16px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="flex:1;min-width:128px;background:#a0c4ff;border:1px solid #60a5fa;border-radius:7px;padding:12px 14px;">
    <div style="font-weight:700;color:#1e3a8a;margin-bottom:5px;font-size:11px;text-transform:uppercase;">qweight</div>
    <div style="color:#1e3a8a;font-size:11px;"><code>(in, out/2)</code> uint8</div>
    <div style="color:#1e3a8a;margin-top:4px;font-size:10px;">packed INT4 pairs; 4 bits per weight</div>
  </div>
  <div style="flex:1;min-width:128px;background:#ffd6a5;border:1px solid #f59e0b;border-radius:7px;padding:12px 14px;">
    <div style="font-weight:700;color:#78350f;margin-bottom:5px;font-size:11px;text-transform:uppercase;">qzeros</div>
    <div style="color:#78350f;font-size:11px;"><code>(in/g, out/2)</code> uint8</div>
    <div style="color:#78350f;margin-top:4px;font-size:10px;">packed zero-points; one per group per output pair</div>
  </div>
  <div style="flex:1;min-width:128px;background:#caffbf;border:1px solid #86efac;border-radius:7px;padding:12px 14px;">
    <div style="font-weight:700;color:#14532d;margin-bottom:5px;font-size:11px;text-transform:uppercase;">scales</div>
    <div style="color:#14532d;font-size:11px;"><code>(in/g, out)</code> float16</div>
    <div style="color:#14532d;margin-top:4px;font-size:10px;">per-group per-output scale; unpacked</div>
  </div>
  <div style="flex:1;min-width:128px;background:#bdb2ff;border:1px solid #c4b5fd;border-radius:7px;padding:12px 14px;">
    <div style="font-weight:700;color:#4c1d95;margin-bottom:5px;font-size:11px;text-transform:uppercase;">s_awq</div>
    <div style="color:#4c1d95;font-size:11px;"><code>(in,)</code> float16</div>
    <div style="color:#4c1d95;margin-top:4px;font-size:10px;">per-input-channel AWQ scale; <code>x ÷ s_awq</code> before GEMM</div>
  </div>
</div>


The four registered buffers below — including `s_awq` — describe the **scratch implementation** this lab builds: it keeps the AWQ scale as an explicit buffer and applies it as a runtime divide on the input. That makes the math easy to follow, but it is *not* how production AutoAWQ stores a quantized layer (see the callout after the formula). The forward pass arithmetic exploits the mathematical identity baked in during quantization — the AWQ scale $s$ was absorbed into the weights as $Q(W \cdot s)$, so dividing the input by $s$ exactly cancels it out:


<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · WQLinear forward pass</div>


$$
y = \underbrace{\!\left(\frac{x}{s_\text{awq}}\right)\!}_{\text{scaled input}} \cdot \underbrace{\left[s_g \cdot \bigl(Q(W \cdot s_\text{awq}) - z_g\bigr)\right]^{\!\top}}_{\text{dequantized stored weights}} \;\approx\; x \cdot W^\top
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $s_\text{awq}$ = per-input-channel AWQ scale · $s_g$ = per-group dequant scale · $z_g$ = per-group zero-point<br>
    The division by $s_\text{awq}$ is a cheap element-wise op on the input vector — negligible cost vs the GEMM that follows.
  </div>
</div>


> **This is the scratch layout — AutoAWQ fuses the scale away.** The `s_awq` buffer and the explicit `x / s_awq` divide above exist only because this lab keeps the scale visible to make the identity easy to read. Production AutoAWQ does *not* store `s_awq` and does *not* divide at runtime: as Section 1 showed, `apply_scale()` folds $s$ into the **preceding** LayerNorm gain (q/k/v, gate/up) or sibling projection weights (o_proj, down_proj) at quantization time. A real `WQLinear` therefore holds only `qweight`, `qzeros`, `scales`, and `bias` — the $/s$ correction has already been baked into whatever runs before it, so the GEMM sees pre-divided activations with zero extra ops. Same math as the formula above; the difference is purely *where* the divide lives.
{: .prompt-info }




> **lm_head stays in FP.** The scratch implementation skips quantizing `lm_head` (the final logit projection). Quantizing it inflates PPL sharply — the lm_head maps from hidden dim to vocabulary size (151,936 tokens for Qwen3) and any rounding error lands directly on the next-token distribution with no downstream layer to absorb it.
{: .prompt-warning }


---


## 3. Calibration Sequencing: Sequential with FP16 Inputs


AutoAWQ quantizes transformer blocks one at a time — block $i$ is fully calibrated and converted to `WQLinear` before block $i+1$ is touched. This sequential order matters because AWQ's `apply_scale()` folds each block's per-channel scale $s$ into the preceding LayerNorm's gain weights, modifying that block's output distribution. Processing blocks in sequence ensures that when block $i+1$'s inputs are collected, they already reflect block $i$'s scale absorption rather than the original unmodified model.


There is a separate question, independent of ordering: when block $i+1$'s calibration inputs are collected, are they the *FP16* activations or the *already-quantized* ones? AutoAWQ and GPTQModel answer this differently, and the contrast is the clearest way to see what each library is optimizing for.


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #a0c4ff;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">AutoAWQ — FP16 inputs</div>
    <p style="margin:0 0 10px;color:#1e293b;font-size:12px;line-height:1.6;">Inside <code>_get_input_feat()</code>, <code>self.inps</code> is advanced by running the layer while it is still FP16; <code>_apply_quant()</code> fires only after <code>self.inps</code> has already been updated. Every block receives a full-precision activation as its calibration input, regardless of how many blocks before it are quantized.</p>
    <div style="background:#dbeafe;border-radius:5px;padding:8px 10px;font-size:11px;color:#1e3a8a;line-height:1.6;">
      <strong>What this buys:</strong> each block's scale search sees a clean signal, decoupled from upstream rounding noise. No cascade — re-tuning one block never forces re-running the blocks after it.
    </div>
  </div>
  <div style="flex:1;min-width:240px;background:#fffffc;border:1px solid #c4b5fd;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#4c1d95;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">GPTQModel — <code>true_sequential=True</code></div>
    <p style="margin:0 0 10px;color:#1e293b;font-size:12px;line-height:1.6;">Propagates <em>quantized</em> activations forward: each layer's Hessian $H = 2X^\top X$ is built from the already-rounded outputs of the layers before it, so the error-compensation update absorbs accumulated upstream quantization error.</p>
    <div style="background:#ede9fe;border-radius:5px;padding:8px 10px;font-size:11px;color:#4c1d95;line-height:1.6;">
      <strong>What this buys:</strong> later layers compensate for the drift their predecessors introduced — at the cost of a hard dependency chain, since the inputs to layer $i$ are only valid once layers $&lt;i$ are final.
    </div>
  </div>
</div>


Same sequential discipline, opposite choice on the input signal. Neither is strictly better: AWQ optimizes weights against a clean reference and keeps blocks independent; GPTQ's `true_sequential` optimizes against the activations the model will actually see at inference. The PPL numbers in Section 6 land within 0.06 of each other, so at W4G128 on this model the choice does not visibly separate them.


> **Where to read it.** AutoAWQ: `awq/quantize/quantizer.py → _get_input_feat()` / `_apply_quant()`. GPTQModel: `gptqmodel/looper/module_looper.py → ModuleLooper._loop_impl()`.
{: .prompt-info }


---


## 4. Weight Clipping: AWQ's Second Optimization Pass


*Why does AWQ need a second optimization pass after scale search — doesn't finding the best $\alpha$ already minimize reconstruction error?*


<div style="display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:13px;">
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #ffd6a5;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#78350f;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Scale search only</div>
    <p style="margin:0 0 10px;color:#1e293b;font-size:12px;line-height:1.6;">Finds per-channel $s$ to minimize output MSE. Group scale = <code>group_max / 7</code>. Within each group, extreme weight values still set the INT4 grid regardless of how rare they are.</p>
    <div style="background:#ffd6a5;border-radius:5px;padding:8px 10px;font-size:11px;color:#78350f;line-height:1.6;">
      <strong>Remaining problem:</strong> an isolated weight outlier within a group forces all 16 INT4 levels to span its full range. The other 127 weights in the group get coarser precision than they need.
    </div>
  </div>
  <div style="flex:1;min-width:220px;background:#fffffc;border:1px solid #caffbf;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#14532d;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-size:11px;">Scale + clipping (AutoAWQ)</div>
    <p style="margin:0 0 10px;color:#1e293b;font-size:12px;line-height:1.6;">After scale search: grid-search <code>clip_ratio</code> ∈ [1.0 → 0.5] in 20 steps (<code>n_grid=20</code>, <code>max_shrink=0.5</code>). For each ratio, hard-clamp weights at <code>ratio × group_max</code> before quantizing and measure reconstruction error.</p>
    <div style="background:#caffbf;border-radius:5px;padding:8px 10px;font-size:11px;color:#14532d;line-height:1.6;">
      <strong>Effect:</strong> outlier weight is clipped; the INT4 grid contracts around the remaining 127 weights, giving them finer effective precision. Small accuracy cost on the outlier, large accuracy gain on the majority.
    </div>
  </div>
</div>


Scale search and clipping are independent levers. Scale search determines *which input channels receive finer quantization* by adjusting per-channel magnitude before quantizing. Clipping determines *what range the INT4 grid spans* for each weight group by trimming extreme values. Both reduce reconstruction error but through different mechanisms — running them sequentially is not redundant.


The Q/K exclusion is a literal name-match guard at the top of the clip loop:


```python
# awq/quantize/quantizer.py → _search_best_clip()
avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]
...
for name in named_linears:
    # due to qk bmm, it is hard to clip precisely
    if any(_ in name for _ in avoid_clipping):
        continue
    ...  # _compute_best_clip() and append to clip_list
```


Any layer whose name contains one of those substrings is skipped entirely — no clip range is searched, and it never enters `clip_list`.


> **Q and K projections are excluded from clipping.** The Q·K batched matrix multiply (`bmm`) inside softmax means you cannot isolate the clip error to one projection. If you clip Q's weights independently, the interaction between clipped-Q and unclipped-K inside `softmax(Q·K^T / sqrt(d))` makes error estimation unreliable. AutoAWQ's `_search_best_clip()` explicitly skips any layer whose name matches `q_`/`k_`/`query`/`key`. Source: `awq/quantize/quantizer.py → _search_best_clip()`.
{: .prompt-danger }


---


## 5. GPTQ: Inside the Column Loop


*Why does GPTQ process weight columns in blocks of 128 rather than column by column — the math is identical, so what changes?*


<div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:10px;padding:18px 20px;margin:20px 0;font-family:system-ui,sans-serif;font-size:12px;">
  <div style="font-weight:700;color:#4c1d95;font-size:13px;margin-bottom:16px;">GPTQ blockwise column quantization — <code>blocksize = B = 128</code></div>
  <div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;">


    <!-- Block 1 -->
    <div style="flex:1;min-width:160px;background:#bdb2ff;border:1px solid #c4b5fd;border-radius:8px;padding:12px;">
      <div style="font-weight:700;color:#4c1d95;margin-bottom:8px;font-size:12px;">Block 1 — cols [0, B)</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;font-size:11px;color:#4c1d95;">① Q( W[:,0:B] ) → Ŵ</div>
        <div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;font-size:11px;color:#4c1d95;">② Err = (W−Ŵ) / diag(H⁻¹)</div>
        <div style="background:#fdffb6;border:1px solid #fcd34d;border-radius:4px;padding:6px 8px;font-size:11px;color:#713f12;font-weight:600;">③ W[:,B:] −= Err @ H⁻¹[0:B, B:]</div>
      </div>
    </div>


    <div style="display:flex;align-items:center;padding-top:30px;font-size:18px;color:#7c3aed;">→</div>


    <!-- Block 2 -->
    <div style="flex:1;min-width:160px;background:#bdb2ff;border:1px solid #c4b5fd;border-radius:8px;padding:12px;opacity:.82;">
      <div style="font-weight:700;color:#4c1d95;margin-bottom:8px;font-size:12px;">Block 2 — cols [B, 2B)</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;font-size:11px;color:#4c1d95;">① Q( W[:,B:2B] ) → Ŵ</div>
        <div style="background:#fffffc;border:1px solid #c4b5fd;border-radius:4px;padding:6px 8px;font-size:11px;color:#4c1d95;">② Err = (W−Ŵ) / diag(H⁻¹)</div>
        <div style="background:#fdffb6;border:1px solid #fcd34d;border-radius:4px;padding:6px 8px;font-size:11px;color:#713f12;font-weight:600;">③ W[:,2B:] −= Err @ H⁻¹[B:2B, 2B:]</div>
      </div>
    </div>


    <div style="display:flex;align-items:center;padding-top:30px;font-size:18px;color:#7c3aed;">→</div>


    <!-- Block N -->
    <div style="flex:1;min-width:160px;background:#a0c4ff;border:1px solid #60a5fa;border-radius:8px;padding:12px;opacity:.72;">
      <div style="font-weight:700;color:#1e3a8a;margin-bottom:8px;font-size:12px;">Block N — last B cols</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="background:#fffffc;border:1px solid #60a5fa;border-radius:4px;padding:6px 8px;font-size:11px;color:#1e3a8a;">① Q( W[:,last B] ) → Ŵ</div>
        <div style="background:#fffffc;border:1px solid #60a5fa;border-radius:4px;padding:6px 8px;font-size:11px;color:#1e3a8a;">② Err = (W−Ŵ) / diag(H⁻¹)</div>
        <div style="background:#caffbf;border:1px solid #86efac;border-radius:4px;padding:6px 8px;font-size:11px;color:#14532d;font-weight:600;">③ No remaining cols — done</div>
      </div>
    </div>
  </div>


  <div style="background:#fdffb6;border-radius:6px;padding:10px 14px;margin-top:14px;font-size:11px;color:#713f12;line-height:1.7;">
    <strong>Why batch?</strong> Step ③ is one GEMM (<code>Err @ H_inv[i1:i2, i2:]</code>) that flushes B column errors simultaneously. The naive per-column loop would issue B separate rank-1 outer products — B CUDA kernel launches, B times the memory round-trips to load <code>H_inv</code>. Same mathematical result; the batched form amortizes all overhead over B columns at once.
  </div>
</div>


<div style="background:#fffffc;border:1px solid #e2e8f0;border-left:3px solid #4c1d95;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4c1d95;margin-bottom:8px;">Formula · GPTQ lazy batch update</div>


$$
W_{:,\; i_2:} \;\leftarrow\; W_{:,\; i_2:} \;-\; \underbrace{E_1}_{\substack{[out \times B]}} \cdot \underbrace{H^{-1}_{i_1:i_2,\; i_2:}}_{\substack{[B \times (in - i_2)]}}
$$


  <div style="font-size:12px;color:#64748b;margin-top:10px;line-height:1.7;">
    $E_{1}[:,j] = \bigl(W_{:,j} - \hat{W}_{:,j}\bigr) / H^{-1}_{jj}$ for $j \in [i_1, i_2)$  &nbsp;·&nbsp;  $B = \texttt{blocksize} = 128$ (default)<br>
    One GEMM replaces $B$ sequential rank-1 updates — memory bandwidth is the bottleneck at 70B scale, so this matters.
  </div>
</div>


Before the column loop, GPTQ computes $H^{-1}$ exactly once per layer using Cholesky decomposition:


```
H  = 2·XᵀX  +  λ·mean(diag H)·I      # λ = damp_percent (default 0.01)
H⁻¹ = cholesky_inverse(cholesky(H))   # numerically stable; never a raw matrix inverse
```


The Cholesky form exploits the fact that $H = 2X^\top X$ is symmetric positive semi-definite — it's 2× cheaper to factor than a general matrix and avoids the catastrophic cancellation that standard LU inversion suffers when any diagonal of $H$ is near zero (dead input channels). Damping adds a floor to every diagonal before factoring.


---


## 6. Lab Results: What the Numbers Show


*Which implementation choices actually move PPL — and by how much?*


<div style="overflow-x:auto;margin:16px 0;">
<p style="font-size:12px;color:#64748b;margin:0 0 8px;font-family:system-ui,sans-serif;">Qwen3-4B · W4G128 · wikitext2 perplexity</p>
<table style="border-collapse:collapse;width:auto;min-width:420px;font-size:13px;font-family:system-ui,sans-serif;">
  <colgroup>
    <col style="width:180px">
    <col style="width:80px">
    <col style="width:80px">
    <col style="width:90px">
  </colgroup>
  <thead>
    <tr style="border-bottom:2px solid #e2e8f0;">
      <th style="padding:8px 14px 8px 0;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;font-weight:600;">Method</th>
      <th style="padding:8px 14px;text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;font-weight:600;">PPL ↓</th>
      <th style="padding:8px 14px;text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;font-weight:600;">Size</th>
      <th style="padding:8px 0 8px 14px;text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;font-weight:600;">Δ PPL</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:9px 14px 9px 0;color:#1e293b;">FP BF16</td>
      <td style="padding:9px 14px;text-align:right;font-weight:700;color:#1e293b;">8.93</td>
      <td style="padding:9px 14px;text-align:right;color:#64748b;">8.04 GB</td>
      <td style="padding:9px 0 9px 14px;text-align:right;color:#94a3b8;">—</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:9px 14px 9px 0;color:#1e293b;">AutoAWQ W4g128</td>
      <td style="padding:9px 14px;text-align:right;font-weight:700;color:#1e293b;">9.16</td>
      <td style="padding:9px 14px;text-align:right;color:#64748b;">2.67 GB</td>
      <td style="padding:9px 0 9px 14px;text-align:right;color:#f59e0b;font-weight:600;">+0.23</td>
    </tr>
    <tr>
      <td style="padding:9px 14px 9px 0;color:#1e293b;">GPTQ W4g128</td>
      <td style="padding:9px 14px;text-align:right;font-weight:700;color:#1e293b;">9.10</td>
      <td style="padding:9px 14px;text-align:right;color:#64748b;">2.67 GB</td>
      <td style="padding:9px 0 9px 14px;text-align:right;color:#f59e0b;font-weight:600;">+0.17</td>
    </tr>
  </tbody>
</table>
</div>


Both libraries land within 0.06 PPL of each other on Qwen3-4B — AWQ at +0.23, GPTQ at +0.17 over the FP16 baseline. That gap is smaller than calibration-data domain effects and well within what a different dataset or `desc_act=True` would close; the two methods are effectively tied at this scale.


The four implementation details this post covered — group structure, packed storage, sequential calibration, weight clipping — are not independent optimizations. Sequential calibration is load-bearing: without it, the other three do not matter. Group structure and clipping are accuracy refinements on top of a correctly working baseline.


---


## References


- Lin et al. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Frantar et al. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **AutoAWQ** source: `awq/quantize/quantizer.py`, `awq/models/qwen3.py`, `awq/quantize/scale.py` — [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- **GPTQModel** source: `gptqmodel/quantization/gptq.py`, `gptqmodel/looper/module_looper.py` — [github.com/ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel)






