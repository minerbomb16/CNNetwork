# ==============================================================================
# autodiff.jl - PŁASKA PAMIĘĆ I GENERACJA KODU
# ==============================================================================
struct MemoryPool
    weights::Vector{Float32}
    w_grad::Vector{Float32}
    w_offset::Base.RefValue{Int}
    
    acts::Vector{Float32}
    a_grad::Vector{Float32}
    a_offset::Base.RefValue{Int}
end

MemoryPool() = MemoryPool(Float32[], Float32[], Ref(1), Float32[], Float32[], Ref(1))

struct GraphNode{T}
    data::T
    grad::T
end

function alloc_weight!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.w_offset[]
    pool.w_offset[] += len
    append!(pool.weights, zeros(Float32, len))
    append!(pool.w_grad, zeros(Float32, len))
    return GraphNode(reshape(view(pool.weights, start:(start+len-1)), dims), 
                     reshape(view(pool.w_grad, start:(start+len-1)), dims))
end

function alloc_act!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.a_offset[]
    pool.a_offset[] += len
    append!(pool.acts, zeros(Float32, len))
    append!(pool.a_grad, zeros(Float32, len))
    return GraphNode(reshape(view(pool.acts, start:(start+len-1)), dims), 
                     reshape(view(pool.a_grad, start:(start+len-1)), dims))
end

struct StaticChain{T <: Tuple} layers::T end
StaticChain(layers...) = StaticChain(layers)

# is_training jest wstrzykiwane bezpośrednio przez kompilator w locie
@generated function forward!(chain::StaticChain{T}, x::GraphNode, is_training::Bool) where T
    N = length(T.parameters)
    exprs = Expr[:(curr_x = x)]
    for i in 1:N
        push!(exprs, :(primal!(chain.layers[$i], curr_x, is_training)))
        push!(exprs, :(curr_x = chain.layers[$i].out))
    end
    push!(exprs, :(return curr_x))
    return Expr(:block, exprs...)
end

@generated function backward!(chain::StaticChain{T}, x::GraphNode, is_training::Bool) where T
    N = length(T.parameters)
    exprs = Expr[]
    for i in N:-1:1
        input_expr = i == 1 ? :(x) : :(chain.layers[$(i-1)].out)
        push!(exprs, :(adjoint!(chain.layers[$i], $input_expr, is_training)))
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

function zero_w_grad!(pool::MemoryPool) fill!(pool.w_grad, 0.0f0) end
function zero_a_grad!(pool::MemoryPool) fill!(pool.a_grad, 0.0f0) end

function optimize!(pool::MemoryPool, η::Float32)
    @inbounds @simd for i in eachindex(pool.weights)
        pool.weights[i] -= η * pool.w_grad[i]
    end
end