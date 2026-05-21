abstract type AbstractNode end

mutable struct GraphNode{OP, N, T} <: AbstractNode
    args :: NTuple{N, AbstractNode}
    grad :: T
    data :: T
    params :: Any
end

const GraphWeight{T} = GraphNode{:weight, 0, T}
const GraphTensor{T} = GraphNode{:tensor, 0, T}
function GraphNode(data::T, trainable=false; params=nothing) where T
    if trainable
        return GraphNode{:weight, 0, T}((), zero(data), data, params)
    else
        return GraphNode{:tensor, 0, T}((), zero(data), data, params)
    end
end

function GraphNode(op::Symbol, args::Tuple, data::T; params=nothing) where T
    N = length(args)
    grad = similar(data)
    grad .= 0
    return GraphNode{op, N, T}(args, grad, data, params)
end

function graph(node)
    function visit!(node::GraphNode, visited, ordered)
        if !(node in visited)
            push!(visited, node)
            for arg in node.args
                visit!(arg, visited, ordered)
            end
            push!(ordered, node)
        end
        return nothing
    end
    ordered = Vector{AbstractNode}()
    visited = Set{AbstractNode}()
    visit!(node, visited, ordered)
    return ordered
end

function zerograd!(order :: Vector{AbstractNode})
    for node in order
        node.grad .= 0
    end
end

function primal!(tensor::GraphTensor)  end
function primal!(weight::GraphWeight)  end
function tangent!(tensor::GraphTensor) end
function tangent!(weight::GraphWeight) end
function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end
function forward!(order::Vector{AbstractNode}, pairs...)
    for (tensor, data) in pairs
        tensor.data .= data
    end
    for node in order
        primal!(node)
    end
end

function backward!(order::Vector{AbstractNode})
	seed = last(order)
    seed_grad = (seed::GraphNode{:crossentropy, 2, Vector{Float32}}).grad
	seed.grad .= 1.0f0
    for node in Iterators.reverse(order)
        adjoint!(node)
    end
end

function optimize!(graph, η)
    function optimize_work!(node::GraphNode{OP, N, T}, η) where {OP, N, T}
        @. node.data -= η * node.grad
    end

    for node in graph
        if node isa GraphWeight
            optimize_work!(node, η)
        end
    end
end