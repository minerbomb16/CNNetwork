# ==============================================================================
# clothesolver.jl - API BIBLIOTEKI I LOGIKA WARSTW
# ==============================================================================
include("autodiff.jl")
using LinearAlgebra
using Random

function he_uniform(shape...; fan_in)
    return (rand(Float32, shape...) .- 0.5f0) .* Float32(sqrt(24.0 / fan_in))
end

abstract type Operator end
abstract type Blueprint end

# --- API Użytkownika (Blueprints) ---
# Identical to Flux: Chain(Conv((3,3), 1=>6), MaxPool((2,2)), Dense(400=>84, relu), ...)
struct DenseSpec <: Blueprint in_out::Pair{Int, Int} end
struct ReLUSpec  <: Blueprint end
const relu = ReLUSpec()

Dense(pair::Pair{Int, Int})               = DenseSpec(pair)
Dense(pair::Pair{Int, Int}, ::ReLUSpec)   = (DenseSpec(pair), ReLUSpec())

struct FlattenSpec <: Blueprint end
Flatten() = FlattenSpec()

struct DropoutSpec <: Blueprint p::Float32 end
Dropout(p::Real) = DropoutSpec(Float32(p))

struct ConvSpec <: Blueprint
    filter::Tuple{Int, Int}
    ch::Pair{Int, Int}
    bias::Bool
end
Conv(filter, ch; bias=true) = ConvSpec(filter, ch, bias)

struct MaxPoolSpec <: Blueprint pool::Tuple{Int, Int} end
MaxPool(pool) = MaxPoolSpec(pool)

struct ChainDef blueprints::Tuple end
function Chain(args...)
    flat = []
    for a in args
        a isa Tuple ? push!(flat, a...) : push!(flat, a)
    end
    return ChainDef(Tuple(flat))
end

# ==============================================================================
# LAYER IMPLEMENTATIONS
# Every build_layer receives batch_size and allocates output buffers with a
# trailing batch dimension [spatial..., channels, batch_size].
# primal!/adjoint! read B from the live tensor sizes so they need no changes
# when called with a different-sized input at test vs train time.
# ==============================================================================

# --- Dense ---
struct DenseLayer{W, B, O} <: Operator w::GraphNode{W}; b::GraphNode{B}; out::GraphNode{O} end
function build_layer(bp::DenseSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    insize, outsize = in_shape[1], bp.in_out[2]
    w = alloc_weight!(pool, outsize, insize)
    w.data .= he_uniform(outsize, insize; fan_in=insize)
    b   = alloc_weight!(pool, outsize)
    out = alloc_act!(pool, outsize, batch_size)   # [outsize × B]
    return DenseLayer(w, b, out), (outsize,)
end
function primal!(layer::DenseLayer, x::GraphNode, is_training::Bool)
    # [outsize × B] = [outsize × insize] × [insize × B]  — one batched GEMM
    mul!(layer.out.data, layer.w.data, x.data)
    # Broadcast bias [outsize] across all B columns
    @inbounds for b in axes(layer.out.data, 2)
        @simd for i in axes(layer.b.data, 1)
            layer.out.data[i, b] += layer.b.data[i]
        end
    end
end
function adjoint!(layer::DenseLayer, x::GraphNode, is_training::Bool)
    # Weight grad: [out × in] += [out × B] × [B × in]
    mul!(layer.w.grad, layer.out.grad, x.data', 1, 1)
    # Bias grad: sum over batch dimension
    @inbounds for b in axes(layer.out.grad, 2)
        @simd for i in axes(layer.b.grad, 1)
            layer.b.grad[i] += layer.out.grad[i, b]
        end
    end
    # Input grad: [in × B] += [in × out] × [out × B]
    mul!(x.grad, layer.w.data', layer.out.grad, 1, 1)
end

# --- ReLU ---
struct ReLULayer{O} <: Operator out::GraphNode{O} end
function build_layer(bp::ReLUSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    return ReLULayer(alloc_act!(pool, in_shape..., batch_size)), in_shape
end
# eachindex covers all spatial+batch elements — no changes needed for batching.
function primal!(layer::ReLULayer, x::GraphNode, is_training::Bool)
    @inbounds @simd for i in eachindex(layer.out.data) layer.out.data[i] = max(0f0, x.data[i]) end
end
function adjoint!(layer::ReLULayer, x::GraphNode, is_training::Bool)
    od = layer.out.data; og = layer.out.grad; xg = x.grad
    @inbounds @simd for i in eachindex(xg)
        xg[i] += ifelse(od[i] > 0f0, og[i], 0f0)
    end
end

# --- Flatten ---
struct FlattenLayer{O} <: Operator out::GraphNode{O} end
function build_layer(bp::FlattenSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    flat_size = prod(in_shape)
    return FlattenLayer(alloc_act!(pool, flat_size, batch_size)), (flat_size,)
end
# copyto! is a flat memcopy. [H,W,C,B] and [H*W*C,B] share the same linear layout
# in column-major memory, so no reordering occurs — this is correct for all B.
function primal!(layer::FlattenLayer, x::GraphNode, is_training::Bool)
    copyto!(layer.out.data, x.data)
end
function adjoint!(layer::FlattenLayer, x::GraphNode, is_training::Bool)
    og = layer.out.grad; xg = x.grad
    @inbounds @simd for i in eachindex(xg) xg[i] += og[i] end
end

# --- Conv (im2col + BLAS GEMM) ---
#
# col_buf [N × K] where N = H_out*W_out, K = fH*fW*C_in.
# Allocated once per layer; reused for every sample in every forward/backward call.
# With batching, the outer forward/backward loops iterate over B samples, calling
# mul! once per sample. Dense layers use a single batched GEMM; Conv loops over B
# because [H,W,C_out,B] storage doesn't allow a zero-copy [N*B × C_out] reshape
# without interleaving channels (which would break the weight matrix layout).
struct ConvLayer{W, B, O, CB} <: Operator w::GraphNode{W}; b::GraphNode{B}; out::GraphNode{O}; col_buf::CB; has_bias::Bool end
function build_layer(bp::ConvSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h_in, w_in, c_in = in_shape; fh, fw = bp.filter; c_out = bp.ch[2]
    h_out, w_out = h_in + 3 - fh, w_in + 3 - fw
    w = alloc_weight!(pool, fh, fw, c_in, c_out)
    w.data .= he_uniform(fh, fw, c_in, c_out; fan_in=fh*fw*c_in)
    b       = alloc_weight!(pool, bp.bias ? c_out : 0)
    out     = alloc_act!(pool, h_out, w_out, c_out, batch_size)  # [H,W,C,B]
    col_buf = zeros(Float32, h_out * w_out, fh * fw * c_in)      # [N × K], reused per sample
    return ConvLayer(w, b, out, col_buf, bp.bias), (h_out, w_out, c_out)
end

function primal!(layer::ConvLayer, x::GraphNode, is_training::Bool)
    H_in  = size(x.data, 1);  W_in  = size(x.data, 2);  C_in  = size(x.data, 3)
    H_out = size(layer.out.data, 1); W_out = size(layer.out.data, 2); C_out = size(layer.out.data, 3)
    B     = size(x.data, 4)
    fH, fW = size(layer.w.data, 1), size(layer.w.data, 2)
    K, N   = fH * fW * C_in, H_out * W_out
    W_flat = reshape(layer.w.data, K, C_out)  # zero-copy [K × C_out]

    for b in 1:B
        # im2col: fill col_buf [N × K] for sample b.
        # Stride-1 writes (h_out varies in inner loop → row index n = h_out + H_out*(w_out-1)
        # increments by 1). Boundary handled by pre-computed h_lo/h_hi/w_lo/w_hi — no branch.
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
                        layer.col_buf[n_base + h_out, k] = x.data[h_out + fh - 2, in_w, c_in, b]
                    end
                end
            end
        end
        # GEMM: out_b [N × C_out] = col_buf [N × K] × W_flat [K × C_out]
        # out_b is a zero-copy reshape of the contiguous sample-b slice of out.data.
        out_b = reshape(@view(layer.out.data[:, :, :, b]), N, C_out)
        mul!(out_b, layer.col_buf, W_flat)
    end

    # Bias: branchless pass, has_bias check outside hot loops.
    if layer.has_bias
        @inbounds for b in 1:B, c_out in 1:C_out
            bval = layer.b.data[c_out]
            for w in 1:W_out
                @fastmath @simd for h in 1:H_out
                    layer.out.data[h, w, c_out, b] += bval
                end
            end
        end
    end
end

function adjoint!(layer::ConvLayer, x::GraphNode, is_training::Bool)
    H_in  = size(x.data, 1);  W_in  = size(x.data, 2);  C_in  = size(x.data, 3)
    H_out = size(layer.out.grad, 1); W_out = size(layer.out.grad, 2); C_out = size(layer.out.grad, 3)
    B     = size(x.data, 4)
    fH, fW = size(layer.w.data, 1), size(layer.w.data, 2)
    K, N   = fH * fW * C_in, H_out * W_out
    W_flat  = reshape(layer.w.data, K, C_out)
    W_gflat = reshape(layer.w.grad, K, C_out)

    # Pass 1: bias gradient — pure reduction, no col_buf needed.
    if layer.has_bias
        @inbounds for b in 1:B, c_out in 1:C_out
            acc = 0.0f0
            for w in 1:W_out
                @fastmath @simd for h in 1:H_out
                    acc += layer.out.grad[h, w, c_out, b]
                end
            end
            layer.b.grad[c_out] += acc
        end
    end

    # Passes 2+3 are fused into one loop over B to fill col_buf only once per sample.
    for b in 1:B
        # Re-fill col_buf for sample b (same branchless im2col as in primal!).
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
                        layer.col_buf[n_base + h_out, k] = x.data[h_out + fh - 2, in_w, c_in, b]
                    end
                end
            end
        end

        dy_b = reshape(@view(layer.out.grad[:, :, :, b]), N, C_out)

        # Pass 2: weight gradient — accumulate across batch (α=1, β=1).
        # dW_flat [K × C_out] += col_buf^T [K × N] × dy_b [N × C_out]
        mul!(W_gflat, layer.col_buf', dy_b, 1f0, 1f0)

        # Pass 3: input gradient — GEMM then col2im scatter.
        # dx_col [N × K] = dy_b [N × C_out] × W_flat^T [C_out × K]
        # Reuse col_buf as output (im2col data for this sample no longer needed).
        mul!(layer.col_buf, dy_b, W_flat')
        @inbounds for c_in in 1:C_in, fw in 1:fW
            w_lo = max(1, 3 - fw);  w_hi = min(W_out, W_in + 2 - fw)
            for fh in 1:fH
                h_lo = max(1, 3 - fh);  h_hi = min(H_out, H_in + 2 - fh)
                k = fh + fH * (fw - 1) + fH * fW * (c_in - 1)
                for w_out in w_lo:w_hi
                    in_w   = w_out + fw - 2
                    n_base = H_out * (w_out - 1)
                    @fastmath @simd for h_out in h_lo:h_hi
                        x.grad[h_out + fh - 2, in_w, c_in, b] += layer.col_buf[n_base + h_out, k]
                    end
                end
            end
        end
    end
end

# --- MaxPool ---
#
# argmax: packed Int32 — lower 8 bits = in_h, upper 8 bits = in_w (max dim 255).
# max_val_buf / max_idx_buf: H_out-length scratch vectors reused every (b,c,w) column.
#
# Loop structure (b outer → c → w → [pool window dph,dpw] → @simd h inner):
#   • batch is the outermost loop so each b's 18 KB x.data slice stays in L1 while
#     we sweep all (c, w_out) columns for that batch item.
#   • pool window dimensions (dph, dpw) are hoisted ABOVE h so the inner @simd loop
#     runs over all H_out positions with a single fixed (in_w, dph) — no data-
#     dependent branch inside the SIMD body; branchless ifelse handles the max update.
#   • Writing max_val_buf[h] / max_idx_buf[h] is stride-1 in h — ideal for SIMD.
struct MaxPoolLayer{O, A} <: Operator
    pool::Tuple{Int,Int}
    argmax::A
    max_val_buf::Vector{Float32}
    max_idx_buf::Vector{Int32}
    out::GraphNode{O}
end
function build_layer(bp::MaxPoolSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h_in, w_in, ch = in_shape; ph, pw = bp.pool
    h_out, w_out   = h_in ÷ ph, w_in ÷ pw
    out            = alloc_act!(pool, h_out, w_out, ch, batch_size)
    argmax_arr     = zeros(Int32, h_out, w_out, ch, batch_size)
    return MaxPoolLayer(bp.pool, argmax_arr,
                        Vector{Float32}(undef, h_out),
                        Vector{Int32}(undef, h_out),
                        out), (h_out, w_out, ch)
end
function primal!(layer::MaxPoolLayer, x::GraphNode, is_training::Bool)
    H_out = size(layer.out.data, 1); W_out = size(layer.out.data, 2)
    C     = size(layer.out.data, 3); B     = size(layer.out.data, 4)
    ph, pw = layer.pool
    mv = layer.max_val_buf   # length H_out scratch — reused every (b,c,w) column
    mi = layer.max_idx_buf

    @inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
        # Initialise scratch for this column
        @simd for h in 1:H_out; mv[h] = -Inf32; mi[h] = Int32(0); end

        # Pool-window scan: dph/dpw outer, h inner → one @simd strip per window position
        for dpw in 0:pw-1, dph in 0:ph-1
            iw = (w_out - 1) * pw + dpw + 1
            @simd for h in 1:H_out
                ih  = (h - 1) * ph + dph + 1
                val = x.data[ih, iw, c, b]
                upd = val > mv[h]
                mv[h] = ifelse(upd, val,                            mv[h])
                mi[h] = ifelse(upd, Int32(ih) | (Int32(iw) << 8),  mi[h])
            end
        end

        # Flush scratch → output (stride-1 in h)
        @simd for h in 1:H_out
            layer.out.data[h, w_out, c, b] = mv[h]
            layer.argmax[h, w_out, c, b]   = mi[h]
        end
    end
end
function adjoint!(layer::MaxPoolLayer, x::GraphNode, is_training::Bool)
    H_out = size(layer.out.grad, 1); W_out = size(layer.out.grad, 2)
    C     = size(layer.out.grad, 3); B     = size(layer.out.grad, 4)
    @inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
        for h in 1:H_out
            idx  = layer.argmax[h, w_out, c, b]
            in_h = Int(idx & 0xFF); in_w = Int((idx >> 8) & 0xFF)
            x.grad[in_h, in_w, c, b] += layer.out.grad[h, w_out, c, b]
        end
    end
end

# --- Dropout ---
#
# mask and rand_buf are now fully typed (concrete M and R).
# Previously mask::Array{Bool} / rand_buf::Array{Float32} had abstract dimensionality
# (UnionAll), which caused boxing of intermediate Bool values inside the hot loop —
# ~147 KB allocation per forward pass. Concrete type parameters eliminate this.
struct DropoutLayer{M<:AbstractArray{Bool}, R<:AbstractArray{Float32}, O} <: Operator
    p::Float32; mask::M; rand_buf::R; out::GraphNode{O}
end
function build_layer(bp::DropoutSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    mask     = zeros(Bool,    in_shape..., batch_size)   # concrete: Matrix{Bool}
    rand_buf = zeros(Float32, in_shape..., batch_size)   # concrete: Matrix{Float32}
    return DropoutLayer(bp.p, mask, rand_buf, alloc_act!(pool, in_shape..., batch_size)), in_shape
end
# eachindex covers all elements including the batch dimension — unchanged.
function primal!(layer::DropoutLayer, x::GraphNode, is_training::Bool)
    if is_training
        rand!(layer.rand_buf); scale = 1.0f0 / (1.0f0 - layer.p)
        p = layer.p; mask = layer.mask; rb = layer.rand_buf
        xd = x.data; od = layer.out.data
        @inbounds @simd for i in eachindex(od)
            m = rb[i] > p
            mask[i] = m
            od[i] = ifelse(m, xd[i] * scale, 0.0f0)
        end
    else
        copyto!(layer.out.data, x.data)
    end
end
function adjoint!(layer::DropoutLayer, x::GraphNode, is_training::Bool)
    if is_training
        scale = 1.0f0 / (1.0f0 - layer.p)
        mask = layer.mask; og = layer.out.grad; xg = x.grad
        @inbounds @simd for i in eachindex(xg)
            xg[i] += ifelse(mask[i], og[i] * scale, 0.0f0)
        end
    else
        @inbounds @simd for i in eachindex(x.grad) x.grad[i] += layer.out.grad[i] end
    end
end

# --- Loss Function ---
#
# primal! computes the numerically stable softmax cross-entropy averaged over the batch.
# adjoint! back-propagates (prob - y_true) / B, scaled by the upstream gradient.
struct LogitCrossEntropy{P, O} probs::GraphNode{P}; out::GraphNode{O} end
function LogitCrossEntropy(pool::MemoryPool, num_classes::Int, batch_size::Int=1)
    return LogitCrossEntropy(alloc_act!(pool, num_classes, batch_size), alloc_act!(pool, 1))
end
function primal!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    C = size(y_pred.data, 1)
    B = size(y_pred.data, 2)
    total_loss = 0.0f0
    @inbounds for b in 1:B
        m = -Inf32
        @simd for i in 1:C; m = max(m, y_pred.data[i, b]) end
        sum_exp = 0.0f0
        @fastmath @simd for i in 1:C
            layer.probs.data[i, b] = exp(y_pred.data[i, b] - m)
            sum_exp += layer.probs.data[i, b]
        end
        l = 0.0f0
        @fastmath @simd for i in 1:C
            layer.probs.data[i, b] /= sum_exp
            l -= y_true.data[i, b] * log(layer.probs.data[i, b] + 1f-10)
        end
        total_loss += l
    end
    layer.out.data[1] = total_loss / B
    return total_loss / B
end
function adjoint!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    B = size(y_pred.data, 2)
    g = layer.out.grad[1] / B   # divide by B: gradient of average-loss w.r.t. each sample
    @fastmath @inbounds @simd for i in eachindex(y_pred.grad)
        y_pred.grad[i] += (layer.probs.data[i] - y_true.data[i]) * g
    end
end

# --- Kompilator Sieci ---
#
# CompiledModel owns the input/target/loss nodes so callers never need to
# allocate them separately or declare a BATCH_SIZE constant.  batch_size is
# stored as a concrete Int field so any function that reads model.batch_size
# is fully type-stable without keyword-arg type annotations.
struct CompiledModel{T, I, TG, P, O}
    chain::StaticChain{T}
    pool::MemoryPool
    input::GraphNode{I}       # [input_shape..., batch_size]
    target::GraphNode{TG}     # [n_classes, batch_size]
    loss::LogitCrossEntropy{P, O}
    batch_size::Int
end

# build_model allocates all buffers (weights, activations, input/target/loss)
# and infers n_classes from the final layer's output shape.
function build_model(def::ChainDef, input_shape::Tuple; batch_size::Int=1)
    pool = MemoryPool()
    compiled_layers = []
    current_shape = input_shape
    for bp in def.blueprints
        layer, current_shape = build_layer(bp, pool, current_shape, batch_size)
        push!(compiled_layers, layer)
    end
    chain       = StaticChain(Tuple(compiled_layers))
    n_classes   = current_shape[1]          # last layer output size
    input_node  = alloc_act!(pool, input_shape..., batch_size)
    target_node = alloc_act!(pool, n_classes, batch_size)
    loss_fn     = LogitCrossEntropy(pool, n_classes, batch_size)
    return CompiledModel(chain, pool, input_node, target_node, loss_fn, batch_size)
end
