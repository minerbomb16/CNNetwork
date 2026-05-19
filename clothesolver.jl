include("autodiff.jl")

global IS_TRAINING = true

# function glorot_uniform(shape...; fan_in, fan_out)
#     scale = sqrt(24.0 / (fan_in + fan_out))
#     return (rand(shape...) .- 0.5) .* scale
# end

function he_uniform(shape...; fan_in, fan_out=nothing)
    return (rand(shape...) .- 0.5) .* sqrt(24.0 / fan_in)
end

abstract type Operator end

function Chain(operators...)
    function flatten_ops(x::Tuple)
        y = Vector{Operator}()
        for v in x
            if v isa Tuple
                push!(y, v...)
            else
                push!(y, v)
            end
        end
        return y
    end
    return flatten_ops(operators)
end

function (chain::Vector{Operator})(x::GraphNode)
    node = x
    for op in chain
        node = op(node)
    end
    return node
end

# ============================= DENSE =============================

struct Dense <: Operator
    insize::Int
    outsize::Int
end
Dense(pair::Pair{Int, Int}) = Dense(first(pair), last(pair))
Dense(pair::Pair{Int, Int}, activation) = tuple(Dense(pair), activation())

function (layer::Dense)(x::GraphNode)
    n, m = layer.insize, layer.outsize
    W_data = he_uniform(m, n; fan_in=n, fan_out=m)
    b_data = zeros(m)
    W = GraphNode(W_data, true)
    b = GraphNode(b_data, true)
    return GraphNode(:dense, (W, b, x), zeros(m))
end

function primal!(y::GraphNode{:dense, 3})
    W, b, x = y.args
    y.data .= W.data * x.data .+ b.data
end

function adjoint!(y::GraphNode{:dense, 3})
    W, b, x = y.args
    W.grad .+= y.grad * x.data'
    b.grad .+= y.grad
    x.grad .+= W.data' * y.grad
end

# ============================= RELU =============================

struct ReLU <: Operator end
relu() = ReLU()

function (layer::ReLU)(x::GraphNode)
    return GraphNode(:relu, (x,), zeros(size(x.data)))
end

function primal!(y::GraphNode{:relu, 1})
    x, = y.args
    y.data .= max.(0, x.data)
end

function adjoint!(y::GraphNode{:relu, 1})
    x, = y.args
    x.grad .+= y.grad .* (y.data .> 0)
end

# ============================= DROPOUT =============================

struct Dropout <: Operator
    p::Float64
end

function (layer::Dropout)(x::GraphNode)
    return GraphNode(:dropout, (x,), zeros(size(x.data)), 
                     params=Dict(:p => layer.p, :mask => zeros(Bool, size(x.data))))
end

function primal!(y::GraphNode{:dropout, 1})
    x, = y.args
    if IS_TRAINING
        p = y.params[:p]
        mask = rand(size(x.data)...) .> p
        y.params[:mask] .= mask
        y.data .= (x.data .* mask) ./ (1.0 - p)
    else
        y.data .= x.data
    end
end

function adjoint!(y::GraphNode{:dropout, 1})
    x, = y.args
    if IS_TRAINING
        p = y.params[:p]
        mask = y.params[:mask]
        x.grad .+= (y.grad .* mask) ./ (1.0 - p)
    else
        x.grad .+= y.grad
    end
end

# ============================= FLATTEN =============================

struct Flatten <: Operator end

function (layer::Flatten)(x::GraphNode)
    flat_size = prod(size(x.data))
    return GraphNode(:flatten, (x,), zeros(flat_size))
end

function primal!(y::GraphNode{:flatten, 1})
    x, = y.args
    y.data .= vec(x.data)
end

function adjoint!(y::GraphNode{:flatten, 1})
    x, = y.args
    x.grad .+= reshape(y.grad, size(x.grad))
end

# ============================= CONV =============================

struct Conv <: Operator
    filter::Tuple{Int, Int}
    ch::Pair{Int, Int}
    bias::Bool
end

Conv(filter, ch; bias=true) = Conv(filter, ch, bias)

function (layer::Conv)(x::GraphNode)
    filter_height, filter_width = layer.filter
    channels_in, channels_out = layer.ch[1], layer.ch[2]

    fan_in = filter_height * filter_width * channels_in
    fan_out = filter_height * filter_width * channels_out
    W_data = he_uniform(filter_height, filter_width, channels_in, channels_out; fan_in=fan_in, fan_out=fan_out)
    b_data = layer.bias ? zeros(channels_out) : zeros(0)

    W = GraphNode(W_data, true)
    b = GraphNode(b_data, true)

    height, width, _ = size(x.data)
    height_out = height + 3 - filter_height
    width_out = width + 3 - filter_width

    return GraphNode(:conv, (W, b, x), zeros(height_out, width_out, channels_out))
end

function primal!(y::GraphNode{:conv, 3})
    W, b, x = y.args

    function conv_primal_work!(y_data, W_data, b_data, x_data)
        height, width, channels_in = size(x_data)
        filter_height, filter_width, _, channels_out = size(W_data)
        y_data .= 0.0 
        
        @inbounds for c_out in 1:channels_out
            for h in axes(y_data, 1)
                for w in axes(y_data, 2)
                    sum_val = 0.0
                    for c_in in 1:channels_in
                        for fh in 1:filter_height
                            for fw in 1:filter_width
                                in_h = h + fh - 2
                                in_w = w + fw - 2
                                if in_h > 0 && in_h <= height && in_w > 0 && in_w <= width
                                    sum_val += x_data[in_h, in_w, c_in] * W_data[fh, fw, c_in, c_out]
                                end
                            end
                        end
                    end
                    bias_val = length(b_data) > 0 ? b_data[c_out] : 0.0
                    y_data[h, w, c_out] = sum_val + bias_val
                end
            end
        end
    end
    conv_primal_work!(y.data, W.data, b.data, x.data)
end

function adjoint!(y::GraphNode{:conv, 3})
    W, b, x = y.args

    function conv_adjoint_work!(y_grad, W_data, W_grad, b_data, b_grad, x_data, x_grad)
        height, width, channels_in = size(x_data)
        filter_height, filter_width, _, channels_out = size(W_data)

        @inbounds for c_out in 1:channels_out
            for h in axes(y_grad, 1)
                for w in axes(y_grad, 2)
                    dy = y_grad[h, w, c_out]
                    if length(b_data) > 0
                        b_grad[c_out] += dy
                    end
                    
                    for c_in in 1:channels_in
                        for fh in 1:filter_height
                            for fw in 1:filter_width
                                in_h = h + fh - 2
                                in_w = w + fw - 2
                                if in_h > 0 && in_h <= height && in_w > 0 && in_w <= width
                                    W_grad[fh, fw, c_in, c_out] += x_data[in_h, in_w, c_in] * dy
                                    x_grad[in_h, in_w, c_in] += W_data[fh, fw, c_in, c_out] * dy
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    conv_adjoint_work!(y.grad, W.data, W.grad, b.data, b.grad, x.data, x.grad)
end

# ============================= MAXPOOL =============================

struct MaxPool <: Operator
    pool::Tuple{Int, Int}
end

function (layer::MaxPool)(x::GraphNode)
    pool_height, pool_width = layer.pool
    height, width, channels = size(x.data)
    
    height_out = floor(Int, height / pool_height)
    width_out = floor(Int, width / pool_width)
    
    argmax_mask = Array{Tuple{Int,Int}}(undef, height_out, width_out, channels)
    
    return GraphNode(:maxpool, (x,), zeros(height_out, width_out, channels), 
                     params=Dict(:pool => layer.pool, :argmax => argmax_mask))
end

function primal!(y::GraphNode{:maxpool, 1})
    x, = y.args
    
    function maxpool_primal_work!(y_data, x_data, argmax_mask, pool)
        pool_height, pool_width = pool
        height, width, channels = size(x_data)
        height_out, width_out = size(y_data, 1), size(y_data, 2)

        @inbounds for c in 1:channels
            for h in 1:height_out
                for w in 1:width_out
                    max_val = -Inf
                    max_idx = (0, 0)
                    for ph in 1:pool_height
                        for pw in 1:pool_width
                            in_h = (h - 1) * pool_height + ph
                            in_w = (w - 1) * pool_width + pw
                            if in_h <= height && in_w <= width
                                val = x_data[in_h, in_w, c]
                                if val > max_val
                                    max_val = val
                                    max_idx = (in_h, in_w)
                                end
                            end
                        end
                    end
                    y_data[h, w, c] = max_val
                    argmax_mask[h, w, c] = max_idx
                end
            end
        end
    end
    maxpool_primal_work!(y.data, x.data, y.params[:argmax], y.params[:pool])
end

function adjoint!(y::GraphNode{:maxpool, 1})
    x, = y.args
    
    function maxpool_adjoint_work!(y_grad, x_grad, argmax_mask)
        channels = size(y_grad, 3)
        height_out, width_out = size(y_grad, 1), size(y_grad, 2)

        @inbounds for c in 1:channels
            for h in 1:height_out
                for w in 1:width_out
                    max_idx = argmax_mask[h, w, c]
                    in_h, in_w = max_idx
                    x_grad[in_h, in_w, c] += y_grad[h, w, c]
                end
            end
        end
    end
    maxpool_adjoint_work!(y.grad, x.grad, y.params[:argmax])
end

# ============================= LOGIT CROSS ENTROPY =============================

struct LogitCrossEntropy <: Operator end

function (layer::LogitCrossEntropy)(y_pred::GraphNode, y_true::GraphNode)
    return GraphNode(:crossentropy, (y_pred, y_true), zeros(1), 
                     params=Dict(:probs => zeros(size(y_pred.data))))
end

function primal!(node::GraphNode{:crossentropy, 2})
    y_pred, y_true = node.args
    
    m = maximum(y_pred.data)
    exp_scores = exp.(y_pred.data .- m)
    probs = exp_scores ./ sum(exp_scores)
    node.params[:probs] .= probs
    
    node.data[1] = -sum(y_true.data .* log.(probs .+ 1e-10))
end

function adjoint!(node::GraphNode{:crossentropy, 2})
    y_pred, y_true = node.args
    probs = node.params[:probs]
    
    y_pred.grad .+= (probs .- y_true.data) .* node.grad[1]
end