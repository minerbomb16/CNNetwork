mutable struct MemoryPool
    weights::Vector{Float32}
    w_grad::Vector{Float32}
    w_offset::Int

    acts::Vector{Float32}
    a_grad::Vector{Float32}
    a_offset::Int
end

MemoryPool() = MemoryPool(Float32[], Float32[], 1, Float32[], Float32[], 1)

struct GraphNode{T}
    data::T
    grad::T
end

function alloc_weight!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.w_offset
    pool.w_offset += len
    append!(pool.weights, zeros(Float32, len))
    append!(pool.w_grad, zeros(Float32, len))
    return GraphNode(reshape(view(pool.weights, start:(start+len-1)), dims), 
                     reshape(view(pool.w_grad, start:(start+len-1)), dims))
end

function alloc_act!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.a_offset
    pool.a_offset += len
    append!(pool.acts, zeros(Float32, len))
    append!(pool.a_grad, zeros(Float32, len))
    return GraphNode(reshape(view(pool.acts, start:(start+len-1)), dims), 
                     reshape(view(pool.a_grad, start:(start+len-1)), dims))
end

struct StaticChain{T <: Tuple}
    layers::T
end
StaticChain(layers...) = StaticChain(layers)

primal_train!(layer, x) = primal!(layer, x)
primal_test!(layer, x) = primal!(layer, x)

@generated function forward_train!(chain::StaticChain{T}, x::GraphNode) where T
    N = length(T.parameters)
    exprs = Expr[:(curr_x = x)]
    for i in 1:N
        layer_expr = :(getfield(chain.layers, $i))
        push!(exprs, :(primal_train!($layer_expr, curr_x)))
        push!(exprs, :(curr_x = $layer_expr.out))
    end
    push!(exprs, :(return curr_x))
    return Expr(:block, exprs...)
end

@generated function forward_test!(chain::StaticChain{T}, x::GraphNode) where T
    N = length(T.parameters)
    exprs = Expr[:(curr_x = x)]
    for i in 1:N
        layer_expr = :(getfield(chain.layers, $i)) 
        push!(exprs, :(primal_test!($layer_expr, curr_x))) 
        push!(exprs, :(curr_x = $layer_expr.out))
    end
    push!(exprs, :(return curr_x))
    return Expr(:block, exprs...)
end

@generated function backward!(chain::StaticChain{T}, x::GraphNode) where T
    N = length(T.parameters)
    exprs = Expr[]
    for i in N:-1:1
        layer_expr = :(getfield(chain.layers, $i))
        input_expr = i == 1 ? :(x) : :(getfield(chain.layers, $(i-1)).out)
        push!(exprs, :(adjoint!($layer_expr, $input_expr)))
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

function zero_w_grad!(pool::MemoryPool)
    fill!(pool.w_grad, 0.0f0)
end
function zero_a_grad!(pool::MemoryPool)
    fill!(pool.a_grad, 0.0f0)
end

function optimize!(pool::MemoryPool, η::Float32)
    @inbounds for i in eachindex(pool.weights)
        pool.weights[i] -= η * pool.w_grad[i]
    end
end