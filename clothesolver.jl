include("autodiff.jl")
using LinearAlgebra
using Random

function he_uniform(shape...; fan_in)
    return (rand(Float32, shape...) .- 0.5f0) .* Float32(sqrt(24.0 / fan_in))
end

abstract type Operator end
abstract type Blueprint end

struct DenseSpec <: Blueprint
    in_out::Pair{Int, Int}
end
struct ReLUSpec <: Blueprint end
const relu = ReLUSpec()

Dense(pair::Pair{Int, Int}) = DenseSpec(pair)
Dense(pair::Pair{Int, Int}, ::ReLUSpec) = (DenseSpec(pair), ReLUSpec())

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

struct MaxPoolSpec <: Blueprint
    pool::Tuple{Int, Int}
end
MaxPool(pool) = MaxPoolSpec(pool)

struct ChainDef blueprints::Tuple end
function Chain(args...)
    flat = []
    for a in args
        a isa Tuple ? push!(flat, a...) : push!(flat, a)
    end
    return ChainDef(Tuple(flat))
end

# ========================= Dense =========================

struct DenseLayer{W, B, O} <: Operator
    w::GraphNode{W}
    b::GraphNode{B}
    out::GraphNode{O}
end

function build_layer(bp::DenseSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    insize = in_shape[1]
    outsize = bp.in_out[2]
    w = alloc_weight!(pool, outsize, insize)
    w.data .= he_uniform(outsize, insize; fan_in=insize)
    b = alloc_weight!(pool, outsize)
    out = alloc_act!(pool, outsize, batch_size)
    return DenseLayer(w, b, out), (outsize,)
end

function primal!(layer::DenseLayer, x::GraphNode)
    mul!(layer.out.data, layer.w.data, x.data)
    @inbounds for b in axes(layer.out.data, 2)
        for i in axes(layer.b.data, 1)
            layer.out.data[i, b] += layer.b.data[i]
        end
    end
end

function adjoint!(layer::DenseLayer, x::GraphNode)
    mul!(layer.w.grad, layer.out.grad, x.data', 1, 1)
    @inbounds for b in axes(layer.out.grad, 2)
        for i in axes(layer.b.grad, 1)
            layer.b.grad[i] += layer.out.grad[i, b]
        end
    end
    mul!(x.grad, layer.w.data', layer.out.grad, 1, 1)
end

# ========================= ReLU =========================

struct ReLULayer{O} <: Operator
    out::GraphNode{O}
end

function build_layer(bp::ReLUSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    return ReLULayer(alloc_act!(pool, in_shape..., batch_size)), in_shape
end

function primal!(layer::ReLULayer, x::GraphNode)
    @inbounds for i in eachindex(layer.out.data)
        layer.out.data[i] = max(0f0, x.data[i])
    end
end

function adjoint!(layer::ReLULayer, x::GraphNode)
    od = layer.out.data
    og = layer.out.grad
    xg = x.grad
    @inbounds for i in eachindex(xg)
        xg[i] += ifelse(od[i] > 0f0, og[i], 0f0)
    end
end

# ========================= Flatten =========================

struct FlattenLayer{O} <: Operator
    out::GraphNode{O}
    aliased::Bool
end

function build_layer(bp::FlattenSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    flat_size = prod(in_shape)
    return FlattenLayer(alloc_act!(pool, flat_size, batch_size), false), (flat_size,)
end

function build_layer(bp::FlattenSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int, prev_out::GraphNode)
    flat_size = prod(in_shape)
    out = GraphNode(reshape(prev_out.data, flat_size, batch_size),
                    reshape(prev_out.grad, flat_size, batch_size))
    return FlattenLayer(out, true), (flat_size,)
end

function primal!(layer::FlattenLayer, x::GraphNode)
    layer.aliased && return nothing
    copyto!(layer.out.data, x.data)
end

function adjoint!(layer::FlattenLayer, x::GraphNode)
    layer.aliased && return nothing
    og = layer.out.grad
    xg = x.grad
    @inbounds for i in eachindex(xg)
        xg[i] += og[i]
    end
end

# ========================= Conv =========================

@inline function im2col!(col_buf, x_data, H_in::Int, W_in::Int, C_in::Int, H_out::Int, W_out::Int, fH::Int, fW::Int, b::Int)
    rb = (b - 1) * H_out * W_out
    @inbounds for c_in in 1:C_in, fw in 1:fW
        w_lo = max(1, 3 - fw)
        w_hi = min(W_out, W_in + 2 - fw)
        for fh in 1:fH
            h_lo = max(1, 3 - fh)
            h_hi = min(H_out, H_in + 2 - fh)
            k = fh + fH * (fw - 1) + fH * fW * (c_in - 1)
            for w_out in w_lo:w_hi
                in_w = w_out + fw - 2
                n_base = H_out * (w_out - 1) + rb
                for h_out in h_lo:h_hi
                    col_buf[n_base + h_out, k] = x_data[h_out + fh - 2, in_w, c_in, b]
                end
            end
        end
    end
end

@inline function col2im!(x_grad, dx_col, H_in::Int, W_in::Int, C_in::Int, H_out::Int, W_out::Int, fH::Int, fW::Int, b::Int)
    rb = (b - 1) * H_out * W_out
    @inbounds for c_in in 1:C_in, fw in 1:fW
        w_lo = max(1, 3 - fw)
        w_hi = min(W_out, W_in + 2 - fw)
        for fh in 1:fH
            h_lo = max(1, 3 - fh)
            h_hi = min(H_out, H_in + 2 - fh)
            k = fh + fH * (fw - 1) + fH * fW * (c_in - 1)
            for w_out in w_lo:w_hi
                in_w = w_out + fw - 2
                n_base = H_out * (w_out - 1) + rb
                for h_out in h_lo:h_hi
                    x_grad[h_out + fh - 2, in_w, c_in, b] += dx_col[n_base + h_out, k]
                end
            end
        end
    end
end

struct ConvLayer{W, B, O} <: Operator
    w::GraphNode{W}
    b::GraphNode{B}
    out::GraphNode{O}
    col_buf::Matrix{Float32}
    big_buf::Matrix{Float32}
    dx_col::Matrix{Float32}
    has_bias::Bool
end

function build_layer(bp::ConvSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h_in, w_in, c_in = in_shape
    fh, fw = bp.filter
    c_out = bp.ch[2]
    h_out = h_in + 3 - fh
    w_out = w_in + 3 - fw
    w = alloc_weight!(pool, fh, fw, c_in, c_out)
    w.data .= he_uniform(fh, fw, c_in, c_out; fan_in=fh*fw*c_in)
    b = alloc_weight!(pool, bp.bias ? c_out : 0)
    out = alloc_act!(pool, h_out, w_out, c_out, batch_size)
    nb = h_out * w_out * batch_size
    k = fh * fw * c_in
    col_buf = zeros(Float32, nb, k)
    big_buf = zeros(Float32, nb, c_out)
    dx_col = zeros(Float32, nb, k)
    return ConvLayer(w, b, out, col_buf, big_buf, dx_col, bp.bias), (h_out, w_out, c_out)
end

function primal!(layer::ConvLayer, x::GraphNode)
    H_in = size(x.data, 1)
    W_in = size(x.data, 2)
    C_in = size(x.data, 3)
    H_out = size(layer.out.data, 1)
    W_out = size(layer.out.data, 2)
    C_out = size(layer.out.data, 3)
    B = size(x.data, 4)
    fH = size(layer.w.data, 1)
    fW = size(layer.w.data, 2)
    K = fH * fW * C_in
    N = H_out * W_out
    NB = N * B
    W_flat = reshape(layer.w.data, K, C_out)

    for b in 1:B
        im2col!(layer.col_buf, x.data, H_in, W_in, C_in, H_out, W_out, fH, fW, b)
    end
 
    mul!(layer.big_buf, layer.col_buf, W_flat)
    od = layer.out.data
    bb = layer.big_buf
    @inbounds for b in 1:B, c in 1:C_out
        doff = N*(c-1) + N*C_out*(b-1)
        soff = N*(b-1) + NB*(c-1)
        for n in 1:N
            od[doff + n] = bb[soff + n]
        end
    end

    if layer.has_bias
        @inbounds for b in 1:B, c_out in 1:C_out
            bval = layer.b.data[c_out]
            for w in 1:W_out
                for h in 1:H_out
                    layer.out.data[h, w, c_out, b] += bval
                end
            end
        end
    end
end

function adjoint!(layer::ConvLayer, x::GraphNode)
    H_in = size(x.data, 1)
    W_in = size(x.data, 2)
    C_in = size(x.data, 3)
    H_out = size(layer.out.grad, 1)
    W_out = size(layer.out.grad, 2)
    C_out = size(layer.out.grad, 3)
    B = size(x.data, 4)
    fH = size(layer.w.data, 1)
    fW = size(layer.w.data, 2)
    K = fH * fW * C_in
    N = H_out * W_out
    NB = N * B
    W_flat = reshape(layer.w.data, K, C_out)
    W_gflat = reshape(layer.w.grad, K, C_out)

    if layer.has_bias
        @inbounds for b in 1:B, c_out in 1:C_out
            acc = 0.0f0
            for w in 1:W_out
                for h in 1:H_out
                    acc += layer.out.grad[h, w, c_out, b]
                end
            end
            layer.b.grad[c_out] += acc
        end
    end

    og = layer.out.grad
    bb = layer.big_buf
    @inbounds for b in 1:B, c in 1:C_out
        doff = N*(b-1) + NB*(c-1)
        soff = N*(c-1) + N*C_out*(b-1)
        for n in 1:N
            bb[doff + n] = og[soff + n]
        end
    end
    mul!(W_gflat, layer.col_buf', layer.big_buf, 1f0, 1f0)
    mul!(layer.dx_col, layer.big_buf, W_flat')
    for b in 1:B
        col2im!(x.grad, layer.dx_col, H_in, W_in, C_in, H_out, W_out, fH, fW, b)
    end
end

# ========================= MaxPool =========================

struct MaxPoolLayer{O, A} <: Operator
    pool::Tuple{Int,Int}
    argmax::A
    max_val_buf::Vector{Float32}
    max_idx_buf::Vector{Int32}
    out::GraphNode{O}
end

function build_layer(bp::MaxPoolSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h_in, w_in, ch = in_shape
    ph, pw = bp.pool
    h_out = h_in ÷ ph
    w_out = w_in ÷ pw
    out = alloc_act!(pool, h_out, w_out, ch, batch_size)
    argmax_arr = zeros(Int32, h_out, w_out, ch, batch_size)
    return MaxPoolLayer(bp.pool, argmax_arr, Vector{Float32}(undef, h_out), Vector{Int32}(undef, h_out), out), (h_out, w_out, ch)
end

function primal!(layer::MaxPoolLayer, x::GraphNode)
    H_out = size(layer.out.data, 1)
    W_out = size(layer.out.data, 2)
    C = size(layer.out.data, 3)
    B = size(layer.out.data, 4)
    ph, pw = layer.pool
    mv = layer.max_val_buf   
    mi = layer.max_idx_buf

    @inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
        for h in 1:H_out
            mv[h] = -Inf32
            mi[h] = Int32(0)
        end

        for dpw in 0:pw-1, dph in 0:ph-1
            iw = (w_out - 1) * pw + dpw + 1
            for h in 1:H_out
                ih = (h - 1) * ph + dph + 1
                val = x.data[ih, iw, c, b]
                upd = val > mv[h]
                mv[h] = ifelse(upd, val, mv[h])
                mi[h] = ifelse(upd, Int32(ih) | (Int32(iw) << 16),  mi[h])
            end
        end

        for h in 1:H_out
            layer.out.data[h, w_out, c, b] = mv[h]
            layer.argmax[h, w_out, c, b] = mi[h]
        end
    end
end

function adjoint!(layer::MaxPoolLayer, x::GraphNode)
    H_out = size(layer.out.grad, 1)
    W_out = size(layer.out.grad, 2)
    C = size(layer.out.grad, 3)
    B = size(layer.out.grad, 4)
    @inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
        for h in 1:H_out
            idx = layer.argmax[h, w_out, c, b]
            in_h = Int(idx & 0xFFFF)
            in_w = Int((idx >> 16) & 0xFFFF)
            x.grad[in_h, in_w, c, b] += layer.out.grad[h, w_out, c, b]
        end
    end
end

# ========================= Dropout =========================

struct DropoutLayer{R<:AbstractArray{Float32}, O} <: Operator
    p::Float32
    rand_buf::R
    out::GraphNode{O}
end

function build_layer(bp::DropoutSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    rand_buf = zeros(Float32, in_shape..., batch_size)
    return DropoutLayer(bp.p, rand_buf, alloc_act!(pool, in_shape..., batch_size)), in_shape
end

function primal_train!(layer::DropoutLayer, x::GraphNode)
    rand!(layer.rand_buf)

    scale = 1.0f0 / (1.0f0 - layer.p)
    p = layer.p
    rb = layer.rand_buf
    xd = x.data
    od = layer.out.data

    @inbounds for i in eachindex(od)
        od[i] = ifelse(rb[i] > p, xd[i] * scale, 0.0f0)
    end
end

function primal_test!(layer::DropoutLayer, x::GraphNode)
    copyto!(layer.out.data, x.data)
end

function adjoint!(layer::DropoutLayer, x::GraphNode)
    scale = 1.0f0 / (1.0f0 - layer.p)
    p = layer.p
    rb = layer.rand_buf
    og = layer.out.grad
    xg = x.grad

    @inbounds for i in eachindex(xg)
        xg[i] += ifelse(rb[i] > p, og[i] * scale, 0.0f0)
    end
end

# ========================= LogitCrossEntropy =========================

struct LogitCrossEntropy{P, O}
    probs::GraphNode{P}
    out::GraphNode{O}
end

function LogitCrossEntropy(pool::MemoryPool, num_classes::Int, batch_size::Int=1)
    return LogitCrossEntropy(alloc_act!(pool, num_classes, batch_size), alloc_act!(pool, 1))
end

function primal!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    C = size(y_pred.data, 1)
    B = size(y_pred.data, 2)
    total_loss = 0.0f0
    @inbounds for b in 1:B
        m = -Inf32
        for i in 1:C
            m = max(m, y_pred.data[i, b])
        end
        sum_exp = 0.0f0
        for i in 1:C
            layer.probs.data[i, b] = exp(y_pred.data[i, b] - m)
            sum_exp += layer.probs.data[i, b]
        end
        lse = m + log(sum_exp)
        inv_se = 1.0f0 / sum_exp
        l = 0.0f0
        for i in 1:C
            layer.probs.data[i, b] *= inv_se   
            l += y_true.data[i, b] * (lse - y_pred.data[i, b])
        end
        total_loss += l
    end
    layer.out.data[1] = total_loss / B
    return total_loss / B
end

function adjoint!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    B = size(y_pred.data, 2)
    g = layer.out.grad[1] / B 
    @inbounds for i in eachindex(y_pred.grad)
        y_pred.grad[i] += (layer.probs.data[i] - y_true.data[i]) * g
    end
end

# ========================= CompiledModel =========================

struct CompiledModel{T, I, TG, P, O}
    chain::StaticChain{T}
    pool::MemoryPool
    input::GraphNode{I}
    target::GraphNode{TG}
    loss::LogitCrossEntropy{P, O}
    batch_size::Int
end

build_layer(bp::Blueprint, pool::MemoryPool, in_shape::Tuple, batch_size::Int, prev_out) =
    build_layer(bp, pool, in_shape, batch_size)

function build_model(def::ChainDef, input_shape::Tuple; batch_size::Int=1)
    pool = MemoryPool()
    compiled_layers = []
    current_shape = input_shape
    prev_out = nothing  
    for bp in def.blueprints
        layer, current_shape = build_layer(bp, pool, current_shape, batch_size, prev_out)
        push!(compiled_layers, layer)
        prev_out = layer.out
    end
    chain = StaticChain(Tuple(compiled_layers))
    n_classes = current_shape[1]
    input_node = alloc_act!(pool, input_shape..., batch_size)
    target_node = alloc_act!(pool, n_classes, batch_size)
    loss_fn = LogitCrossEntropy(pool, n_classes, batch_size)
    return CompiledModel(chain, pool, input_node, target_node, loss_fn, batch_size)
end
