# AI Performance Notes

## Wersja 3.0 — Conv rewrite

### Profiling results (profile_run.jl, 1000 iterations, single sample)

Overall: **0.719 ms/pass**, 1390 img/s

| Layer            | Forward  | Share | Backward | Share |
|------------------|----------|-------|----------|-------|
| Conv2 (14×14×16) | 215.8 µs |  78%  | 341.5 µs |  78%  |
| Conv1 (28×28×6)  |  57.4 µs |  21%  |  89.5 µs |  21%  |
| Everything else  |  ~18 µs  |   1%  |   ~4 µs  |   1%  |

Forward/backward split: 38% / 62%.  
Allocations: 84 allocs / pass, ~1.3 KiB — MemoryPool working as intended.

---

### Bottlenecks identified in `clothesolver.jl`

**1. Branch inside `@simd` kills vectorization**  
The boundary check `if 0 < in_h <= H_in && 0 < in_w <= W_in` was placed inside
`@simd for fh`, making vectorization impossible. The CPU cannot emit SIMD instructions
when there is a data-dependent branch inside the loop body. The branch existed only to
handle the 1-pixel border — 108 out of 784 pixels (Conv1) — yet every pixel paid the cost.

**2. `@simd` loop was only 3 iterations long**  
`@simd for fh` ran exactly 3 times for a 3×3 filter. An AVX2 Float32 register holds
8 values. Three iterations fill less than half a register; effective vectorization was
impossible even if the branch were removed.

**3. `h` (28 or 14 elements) was not the innermost loop**  
The h dimension — along which both `x.data` and `out.data` are stride-1 in column-major
`[H,W,C]` layout — was buried in the middle of the loop nest. Moving it innermost exposes
a 26–28 iteration SIMD loop with two stride-1 accesses and one scalar broadcast: the
ideal pattern for auto-vectorization.

**4. Two RMW operations mixed in one backward loop body**  
`adjoint!` wrote to both `w.grad` and `x.grad` in the same `@simd` body. Beyond the
branch problem, this prevented independent optimisation of each gradient and blocked
the compiler from using a scalar register accumulator for the weight gradient.

---

### Changes applied (`clothesolver.jl`)

#### Forward `primal!`

| Before | After |
|---|---|
| Loop order: `c_out → w → h → c_in → fw → [simd] fh` | `c_out → c_in → fw → fh → w → [simd] h` |
| `@simd` over `fh` — 3 iterations | `@simd` over `h` — 26–28 iterations |
| `if` branch every inner iteration | `h_lo`/`h_hi` clamped once per `(fh,fw)` pair |
| `has_bias` ternary inside hot loop | Bias added in a separate branchless pass after accumulation |

Boundary handling: condition `1 ≤ h+fh-2 ≤ H_in` solved analytically once per `(fh,fw)`:
```
h_lo = max(1, 3 - fh)
h_hi = min(H_out, H_in + 2 - fh)
```
The `@simd for h in h_lo:h_hi` loop is unconditional.

#### Backward `adjoint!`

Split into 3 independent passes, each with a single write target:

- **Pass 1 — bias grad:** pure SIMD reduction over the contiguous `out.grad[:,:,c_out]`
  slice into a scalar accumulator; written once to `b.grad[c_out]`.

- **Pass 2 — weight grad:** accumulate into scalar `wgrad_acc` (lives in a register for
  the full valid `(h,w)` region), write once to `w.grad[fh,fw,c_in,c_out]`. Both
  `x.data` and `out.grad` reads are stride-1 over `h`.

- **Pass 3 — input grad:** scalar broadcast `wval = w.data[fh,fw,c_in,c_out]`; both
  `x.grad` and `out.grad` accesses are stride-1 over `h`. No intra-loop write conflicts
  (all `h+fh-2` indices are distinct within one SIMD strip).

---

## Wersja 4.0 — im2col + BLAS GEMM for Conv

### Motivation

After the v3.0 loop rewrite (2.41× overall speedup), Conv2 backward still dominated
at 78% of total time. The input gradient pass (Pass 3) alone held 249/406 profile
samples. The manual SIMD loops, while now branchless, were still running 864 separate
short strips (C_out × C_in × fW × fH = 16×6×3×3) of 12–14 elements each. BLAS
operates on the same arithmetic but with cache-blocking and register tiling tuned
at the assembly level — for matrix shapes that arise from this network it is
consistently faster than even well-written scalar SIMD.

No new dependency is introduced. `mul!` from `LinearAlgebra` (already imported,
already used in `DenseLayer`) calls OpenBLAS under the hood. im2col is purely a
data reorganisation step written in plain Julia.

---

### What im2col does

Convolution computes, for every output pixel (h, w) and every output channel c_out:

```
out[h, w, c_out] = Σ_{fh,fw,c_in}  x[h+fh-2, w+fw-2, c_in] · W[fh, fw, c_in, c_out]
```

That sum is a dot product between a flattened input patch and a flattened filter.
If you lay ALL output-position patches as rows of a matrix (`col_buf`), the entire
convolution for all output channels becomes a single matrix multiply:

```
out_mat [N × C_out]  =  col_buf [N × K]  ×  W_flat [K × C_out]

  N = H_out * W_out          (one row per output spatial position)
  K = fH * fW * C_in         (one column per receptive-field element)
```

`W_flat` and `out_mat` are **zero-copy reshapes** of the existing weight and output
buffers already allocated in the MemoryPool. Only `col_buf` is a new allocation —
done once at `build_layer` time, reused every pass.

For this network:

| Conv  | col_buf shape  | W_flat shape | GEMM (M×K×N)   | Buffer size |
|-------|---------------|--------------|----------------|-------------|
| Conv1 | 784 × 9       | 9 × 6        | 784 × 9 × 6    | ~28 KB      |
| Conv2 | 196 × 54      | 54 × 16      | 196 × 54 × 16  | ~42 KB      |

Both fit in L2 cache.

---

### Changes applied

#### Struct: new `col_buf` field

```julia
# BEFORE
struct ConvLayer{W, B, O} <: Operator
    w::GraphNode{W}; b::GraphNode{B}; out::GraphNode{O}; has_bias::Bool
end

# AFTER
struct ConvLayer{W, B, O, CB} <: Operator
    w::GraphNode{W}; b::GraphNode{B}; out::GraphNode{O}
    col_buf::CB   # Matrix{Float32} [N × K], allocated once at build time
    has_bias::Bool
end
```

`build_layer` allocates it alongside weights and activations:
```julia
col_buf = zeros(Float32, h_out * w_out, fh * fw * c_in)
return ConvLayer(w, b, out, col_buf, bp.bias), (h_out, w_out, c_out)
```

#### Forward `primal!`

```julia
# BEFORE (v3.0) — 864 separate SIMD strips of ~14 elements
@inbounds for c_out in 1:C_out, c_in in 1:C_in
    for fw in 1:fW
        w_lo = max(1, 3 - fw);  w_hi = min(W_out, W_in + 2 - fw)
        for fh in 1:fH
            h_lo = max(1, 3 - fh);  h_hi = min(H_out, H_in + 2 - fh)
            wval = layer.w.data[fh, fw, c_in, c_out]
            for w in w_lo:w_hi
                in_w = w + fw - 2
                @fastmath @simd for h in h_lo:h_hi
                    layer.out.data[h, w, c_out] += x.data[h + fh - 2, in_w, c_in] * wval
                end
            end
        end
    end
end

# AFTER (v4.0) — im2col fill + one mul!
fill!(layer.col_buf, 0.0f0)
@inbounds for c_in in 1:C_in, fw in 1:fW
    w_lo = max(1, 3 - fw);  w_hi = min(W_out, W_in + 2 - fw)
    for fh in 1:fH
        h_lo = max(1, 3 - fh);  h_hi = min(H_out, H_in + 2 - fh)
        k = fh + fH * (fw - 1) + fH * fW * (c_in - 1)
        for w_out in w_lo:w_hi
            in_w   = w_out + fw - 2
            n_base = H_out * (w_out - 1)
            @simd for h_out in h_lo:h_hi
                layer.col_buf[n_base + h_out, k] = x.data[h_out + fh - 2, in_w, c_in]
            end
        end
    end
end
W_flat  = reshape(layer.w.data,   K, C_out)   # zero-copy
out_mat = reshape(layer.out.data, N, C_out)   # zero-copy
mul!(out_mat, layer.col_buf, W_flat)
```

The im2col fill uses the same branchless h_lo/h_hi bounds from v3.0. The [N × K]
layout (rows = output positions) ensures stride-1 writes to `col_buf` and stride-1
reads from `x.data` as `h_out` varies in the inner loop.

#### Backward `adjoint!`

```julia
# BEFORE (v3.0)
# Pass 2: weight grad — scalar accumulator per (fh,fw,c_in,c_out), 864 outer iterations
@inbounds for c_out in 1:C_out, c_in in 1:C_in
    for fw in 1:fW
        ...
        for fh in 1:fH
            wgrad_acc = 0.0f0
            for w in w_lo:w_hi
                @fastmath @simd for h in h_lo:h_hi
                    wgrad_acc += x.data[h+fh-2, in_w, c_in] * layer.out.grad[h, w, c_out]
                end
            end
            layer.w.grad[fh, fw, c_in, c_out] += wgrad_acc
        end
    end
end

# Pass 3: input grad — 864 outer iterations × ~14 SIMD elements
@inbounds for c_out in 1:C_out, c_in in 1:C_in
    for fw in 1:fW
        ...
        for fh in 1:fH
            wval = layer.w.data[fh, fw, c_in, c_out]
            for w in w_lo:w_hi
                @fastmath @simd for h in h_lo:h_hi
                    x.grad[h+fh-2, in_w, c_in] += wval * layer.out.grad[h, w, c_out]
                end
            end
        end
    end
end

# AFTER (v4.0)
# Pass 2: weight grad — one GEMM, accumulates into w.grad (α=1, β=1)
mul!(W_gflat, layer.col_buf', dy_mat, 1f0, 1f0)
#   col_buf' [K × N]  ×  dy_mat [N × C_out]  →  W_gflat [K × C_out]

# Pass 3: input grad — one GEMM to get dx_col, then col2im scatter
mul!(layer.col_buf, dy_mat, W_flat')
#   dy_mat [N × C_out]  ×  W_flat' [C_out × K]  →  col_buf [N × K]

# col2im: scatter col_buf back to x.grad (C_in×fW×fH iterations, no C_out loop)
@inbounds for c_in in 1:C_in, fw in 1:fW
    ...
    for fh in 1:fH
        k = fh + fH*(fw-1) + fH*fW*(c_in-1)
        for w_out in w_lo:w_hi
            @fastmath @simd for h_out in h_lo:h_hi
                x.grad[h_out+fh-2, in_w, c_in] += layer.col_buf[n_base + h_out, k]
            end
        end
    end
end
```

The key improvement in Pass 3: the old approach ran `C_out × C_in × fW × fH = 864`
outer iterations. col2im runs only `C_in × fW × fH = 54` — the C_out reduction was
folded into the GEMM. All three BLAS calls (`mul!` in forward, two in backward) use
the same function already called by `DenseLayer` — no new dependency.

---

### Performance
Before (v3.0): 0.298 ms/pass,  3 355 img/s  
After  (v4.0): 0.055 ms/pass, 18 184 img/s  — **5.42× over v3.0, 13× over original**

| Layer            | Fwd v3.0 | Fwd v4.0 | Δ fwd   | Bwd v3.0  | Bwd v4.0 | Δ bwd    |
|------------------|----------|----------|---------|-----------|----------|----------|
| Conv1            |  9.1 µs  |  2.4 µs  | **3.7×**|  31.1 µs  |  3.0 µs  | **10.3×**|
| Conv2            | 51.5 µs  |  7.4 µs  | **7.0×**| 174.3 µs  |  9.1 µs  | **19.1×**|
| Dense1           |  3.6 µs  |  3.5 µs  |   1.0×  |   8.4 µs  |  8.4 µs  |   1.0×   |
| MaxPool1         |  5.6 µs  |  5.7 µs  |   1.0×  |    —      |    —     |    —     |
| Dense2           |  3.6 µs  | 16.2 µs  | 0.22×†  |   0.7 µs  |  0.7 µs  |   1.0×   |

† Dense2 forward anomaly: at 10×84=840 elements the BLAS call overhead now dominates;
the actual compute is trivial and this will disappear once batching amortises the overhead.

Allocations: 84/pass, 131 KiB per 100 passes — unchanged (MemoryPool still correct).

Profile samples for 2000 passes: 101 (vs 1394 in v1.0, 575 in v3.0). The profiler
barely has time to sample — the entire 2000-pass run completes in ~110 ms.

---

## Wersja 5.0 — True batching + bug fixes

### What v5.0 added

All buffer shapes gained a trailing batch dimension baked in at `build_model` time
(`batch_size::Int` keyword argument). The Flux-like `Chain(...)` / `build_model`
interface is unchanged. Conv loops over B samples calling one `mul!` per sample;
Dense uses a single batched `[out×B] = [out×in] × [in×B]` GEMM. `LogitCrossEntropy`
averages loss over the batch; the gradient is divided by B before accumulation.

### Initial v5.0 results (before bug fixes, batch_size=32)

Per-sample throughput **decreased** vs batch=1 (10,957 vs 12,250 img/s). Three root
causes were identified by profiling:

**Bug 1 — profiler had wrong layer indices.**
`Dense(400=>84, relu)` expands to two blueprints (DenseSpec + ReLUSpec), giving 9
layers, not 8. The old `layer_meta` array had only 8 entries: ReLU was missing, so
"Dropout" was measuring ReLULayer and "Dense2" was measuring DropoutLayer. The actual
Dense2 (84→10) was never benchmarked. This made "Dense2 forward = 144 µs" look like a
regression when in reality it was Dropout being slow (see Bug 2).

**Bug 2 — DropoutLayer had abstract-typed fields (main allocation source).**
```julia
# BEFORE
struct DropoutLayer{O} <: Operator
    p::Float32; mask::Array{Bool}; rand_buf::Array{Float32}; out::GraphNode{O}
end
```
`Array{Bool}` and `Array{Float32}` without a dimension parameter are `UnionAll` types —
not concrete. Julia cannot specialize the hot loop body; intermediate `Bool` values get
boxed on every iteration. Result: **147,504 bytes allocated per forward pass** (1,900×
more than the zero-alloc target). Per-layer measurement: 0 bytes fwd for all other
layers, 147 KB for Dropout alone.
```julia
# AFTER
struct DropoutLayer{M<:AbstractArray{Bool}, R<:AbstractArray{Float32}, O} <: Operator
    p::Float32; mask::M; rand_buf::R; out::GraphNode{O}
end
```
`zeros(Bool, 84, 32)` returns `Matrix{Bool}` and `zeros(Float32, 84, 32)` returns
`Matrix{Float32}` — both concrete, so `M` and `R` are fully specialised. Also replaced
`mask[i] ? expr : 0.0f0` with `ifelse(m, expr, 0.0f0)` in both primal!/adjoint! to
help SIMD vectorisation.

**Bug 3 — MaxPool argmax was 16 bytes/element (Tuple{Int64,Int64}).**
```julia
# BEFORE
argmax_arr = Array{Tuple{Int,Int}}(undef, h_out, w_out, ch, batch_size)
# stored as: layer.argmax[h,w,c,b] = (in_h, in_w)          — 16 bytes/element
```
At batch_size=32:
- MaxPool1 argmax: [14×14×6×32] × 16 bytes = **588 KB**
- MaxPool2 argmax: [7×7×16×32] × 16 bytes = **392 KB**
- Total: **980 KB** — nearly 2× the typical L2 cache (512 KB)

All three buffers (x.data input, out.data output, argmax) for MaxPool1 together
exceeded L2, causing constant evictions and cache thrash.
```julia
# AFTER — packed Int32: lower 8 bits = in_h, upper 8 bits = in_w
argmax_arr = zeros(Int32, h_out, w_out, ch, batch_size)    # 4 bytes/element
# encode: max_idx = Int32(in_h) | (Int32(in_w) << 8)
# decode: in_h = Int(idx & 0xFF);  in_w = Int((idx >> 8) & 0xFF)
```
New argmax buffers: MaxPool1 = 147 KB, MaxPool2 = 98 KB. Total **245 KB — fits in L2**.

### Fixes applied

| Fix | File | Impact |
|-----|------|--------|
| Profiler `layer_meta` + `layer_meta_bwd`: add ReLU entry (index 7), shift Dropout→8, Dense2→9 | `profile_run.jl` | Correct labels and per-layer numbers |
| `DropoutLayer` struct: add type params `M`, `R`; use `ifelse` in hot loops | `clothesolver.jl` | 147 KB/pass → 0 bytes |
| `MaxPoolLayer` argmax: `Array{Tuple{Int,Int}}` → `Array{Int32}` packed encoding | `clothesolver.jl` | 980 KB → 245 KB argmax footprint |
| `ReLULayer adjoint!`: `* Bool` → `ifelse`; local refs for `od/og/xg` | `clothesolver.jl` | SIMD-friendly, zero-alloc confirmed |
| `FlattenLayer adjoint!`: local refs `og/xg` | `clothesolver.jl` | Zero-alloc confirmed |

### v5.0 results after fixes (batch_size=32, 500 iters)

Overall: **2.672 ms/pass, 11,978 img/s** (vs 10,957 before fixes, +9.3%)

| Metric | batch=1 | batch=32 before | batch=32 after |
|--------|---------|-----------------|----------------|
| ms/pass | 0.079 | 2.921 | **2.672** |
| img/s | 12,724 | 10,957 | **11,978** |
| allocs/100 passes | 600 | 1,140,000 | **600** |
| KiB/100 passes | 79.7 | 17,467 | **79.7** |

Forward time breakdown (batch=32, µs/pass):

| Layer | µs/pass | share |
|-------|---------|-------|
| Conv1 | 103 | 7.3% |
| **MaxPool1** | **474** | **33.5%** |
| Conv2 | 437 | 30.8% |
| **MaxPool2** | **351** | **24.7%** |
| Flatten | 5 | 0.4% |
| Dense1 | 40 | 2.8% |
| ReLU | 0.16 | 0.0% |
| Dropout | 0.61 | 0.0% |
| Dense2 | 0.82 | 0.1% |

Backward time breakdown (batch=32, µs/pass):

| Layer | µs/pass | share |
|-------|---------|-------|
| Dense2 | 2 | 0.2% |
| Dropout | 0.2 | 0.0% |
| ReLU | 0.2 | 0.0% |
| Dense1 | 58 | 5.3% |
| Flatten | 1.5 | 0.1% |
| MaxPool2 | 27 | 2.4% |
| **Conv2** | **786** | **70.6%** |
| MaxPool1 | 36 | 3.3% |
| Conv1 | 204 | 18.5% |

### Remaining bottleneck: MaxPool cache pressure

MaxPool forward still degrades 2.3× per sample at B=32 vs B=1 despite the argmax fix.
The argmax fix reduced write pressure (from 980 KB → 245 KB), but the *read* bottleneck
is **x.data itself** — [28×28×6×32] = **602 KB**, which overflows L2 cache. At B=1
the 18 KB input fits in L1 and MaxPool runs at full cache bandwidth; at B=32 every
pass refetches 602 KB from L3. This is a fundamental consequence of the `[H,W,C,B]`
column-major layout: the batch dimension is outermost (stride H×W×C), so iterating
over B in the outer loop accesses non-contiguous 18 KB blocks.

MaxPool now consumes **58% of total forward time**. Conv2 backward dominates backward
at 70.6% but scales linearly with batch size (BLAS). The only structural fix for
MaxPool would be changing to `[B,H,W,C]` layout, which would require rewriting all
layer implementations.

---

### Fix 5 — MaxPool loop restructuring (branchless SIMD, batch as outer index)

**Root cause**: the MaxPool `primal!` loop had a data-dependent branch (`if val > max_val`)
*inside* the `h` SIMD loop. Every comparison caused a branch misprediction ~50% of the time
(the maximum is uniformly distributed over the pool window). At B=32 the argmax buffer is
14×14×6×32 = 37,632 entries, so the misprediction cost scales with batch size.

Additionally, the pool-window loop (`dph`, `dpw`) was nested *inside* the `h` loop. This means
`mv[h]` and `mi[h]` scratch state could not be reused across window iterations — the compiler
was forced to re-initialise them every iteration, and the branch blocked vectorisation entirely.

**Fix**: hoist the pool-window loops *above* the `@simd for h` loop, maintain per-h running max
in scratch buffers `max_val_buf` / `max_idx_buf`, and replace the `if` branch with `ifelse`
(generates a branchless `cmov` instruction, SIMD-compatible).

```julia
# BEFORE — branch inside @simd h, pool window innermost
@inbounds for b in 1:B, c in 1:C, w_out in 1:W_out, h in 1:H_out
    best_val = -Inf32; best_idx = Int32(0)
    for dph in 0:ph-1, dpw in 0:pw-1
        ih = (h - 1) * ph + dph + 1
        iw = (w_out - 1) * pw + dpw + 1
        val = x.data[ih, iw, c, b]
        if val > best_val          # ← DATA-DEPENDENT BRANCH — ~50% misprediction
            best_val = val
            best_idx = Int32(ih) | (Int32(iw) << 8)
        end
    end
    layer.out.data[h, w_out, c, b] = best_val
    layer.argmax[h, w_out, c, b]   = best_idx
end

# AFTER — pool window hoisted, branchless ifelse, @simd over h
mv = layer.max_val_buf;  mi = layer.max_idx_buf
@inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
    @simd for h in 1:H_out; mv[h] = -Inf32; mi[h] = Int32(0); end
    for dpw in 0:pw-1, dph in 0:ph-1          # pool window loops OUTSIDE h
        iw = (w_out - 1) * pw + dpw + 1
        @simd for h in 1:H_out                 # h innermost — stride-1, vectorisable
            ih  = (h - 1) * ph + dph + 1
            val = x.data[ih, iw, c, b]
            upd = val > mv[h]
            mv[h] = ifelse(upd, val,                           mv[h])   # branchless
            mi[h] = ifelse(upd, Int32(ih) | (Int32(iw) << 8), mi[h])
        end
    end
    @simd for h in 1:H_out                     # flush scratch → output once per column
        layer.out.data[h, w_out, c, b] = mv[h]
        layer.argmax[h, w_out, c, b]   = mi[h]
    end
end
```

Two new scratch fields were added to `MaxPoolLayer`:
```julia
struct MaxPoolLayer{O, A} <: Operator
    pool::Tuple{Int,Int}
    argmax::A                     # Array{Int32,4}
    max_val_buf::Vector{Float32}  # length H_out — reused per (b,c,w_out) column
    max_idx_buf::Vector{Int32}
    out::GraphNode{O}
end
```

`adjoint!` needed no structural change (only removed the sentinel `idx == 0` check, which
was no longer needed since all entries are always written during `primal!`).

### v5.0 final results (after fix 5, batch_size=32, 500 iters)

Overall: **2.485 ms/pass, 12,875 img/s** (vs 11,978 before, +7.5%)

| Metric | batch=1 | batch=32 before fix 5 | batch=32 final |
|--------|---------|-----------------------|----------------|
| ms/pass | 0.079 | 2.672 | **2.485** |
| img/s | 12,833 | 11,978 | **12,875** |
| allocs/100 passes | 600 | 600 | **600** |

Batch=32 now **surpasses** batch=1 per-sample throughput (12,875 vs 12,833 img/s).

Forward time breakdown (batch=32, µs/pass, after all fixes):

| Layer | µs/pass | share | vs before fix 5 |
|-------|---------|-------|-----------------|
| Conv1 | 103 | 8.1% | unchanged |
| **MaxPool1** | **282** | **22.1%** | **474 → 282 (1.68×)** |
| Conv2 | 437 | 34.2% | unchanged |
| **MaxPool2** | **184** | **14.4%** | **351 → 184 (1.91×)** |
| Flatten | 5 | 0.4% | unchanged |
| Dense1 | 40 | 3.1% | unchanged |
| ReLU | 0.16 | 0.0% | unchanged |
| Dropout | 0.61 | 0.0% | unchanged |
| Dense2 | 0.82 | 0.1% | unchanged |

MaxPool combined forward share dropped from 58% → 36.5%. Conv2 backward still dominates
backward pass at ~71%.

---

## full_preft_test.jl — Standalone training script

### Purpose

`full_preft_test.jl` is a command-line version of the `test.ipynb` training cells
that can be run directly (`julia --project full_preft_test.jl`) without a Jupyter kernel.

### Bugs found converting from notebook

**1 — Double normalisation.**
MLDatasets ≥ 0.5 returns features as `Float32` already in [0,1]. The initial script
draft added an unconditional `/ 255f0` that squashed all pixel values into [0, 0.004],
making the network unable to learn (accuracy stuck at random-chance ~10%). Fixed with:
```julia
let raw = reshape(train_data.features, 28, 28, 1, :)
    global X_train = Float32.(raw) ./ (eltype(raw) <: Integer ? 255f0 : 1f0)
end
```

**2 — Soft-scope `global` inside `@elapsed for`.**
Julia's soft-scope rules mean an assignment inside a `for` block creates a new *local*
variable; it does not update the outer binding. Fix: `global best_acc = max(best_acc, acc)`.
Without this the loop crashes with `UndefVarError: best_acc not defined in local scope`.

### Why the notebook is ~2× slower than the script

The notebook `train!` function passes `batch_size` as an **untyped keyword argument**:
```julia
function train!(model, X, Y; batch_size, η)   # batch_size::Any
    for b_start in 1:batch_size:N             # range step = Any → dynamic dispatch
```
Julia cannot infer the type of `batch_size` at the call site. Every arithmetic result
(`b_end - b_start + 1`, `actual_bs < batch_size`, …) is boxed on the heap for all
1 875 iterations per epoch. Measured overhead: **~2×** (10 s/epoch in notebook vs
5 s/epoch in script).

The script avoids this by using `BATCH_SIZE` — a module-level `const Int64` that Julia
specialises the loop on at compile time, with zero boxing.

**Fix in notebook:** annotate the kwargs:
```julia
function train!(model::CompiledModel, X, Y; batch_size::Int, η::Float32)
```

### Final script hyperparameters and results

| Parameter | Value |
|-----------|-------|
| Epochs    | 3     |
| η         | 0.02  |
| Batch size | 32   |

Training results (measured on this machine):

| Epoch | Loss   | Test acc | Train time | Eval time | Total  |
|-------|--------|----------|------------|-----------|--------|
| 1     | 0.6575 | 83.87%   | 7.53 s †   | 0.70 s    | 8.23 s |
| **2** | 0.4753 | **86.13%** | 4.93 s   | 0.42 s    | 5.36 s |
| **3** | 0.4221 | **86.58%** | 5.23 s   | 0.42 s    | 5.65 s |

† epoch 1 is ~2 s longer due to first-call JIT compilation.

**Best test accuracy: 86.58%.** Target (>85%) met from epoch 2 onward, comfortably
above target at epoch 3. Steady-state compute: **~5 s/epoch** (4.93 s train + 0.42 s eval).