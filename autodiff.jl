mutable struct GraphNode{OP, N}
    args :: NTuple{N, GraphNode}
    grad :: Any
    data :: Any
    params :: Any
end

const GraphWeight = GraphNode{:weight, 0}
const GraphTensor = GraphNode{:tensor, 0}
function GraphNode(data::T, trainable=false; params=nothing) where T
    if trainable
        return GraphNode{:weight, 0}((), zero(data), data, params)
    else
        return GraphNode{:tensor, 0}((), zero(data), data, params)
    end
end

function GraphNode(op::Symbol, args::Tuple, data::T; params=nothing) where T
    N = length(args)
    grad = similar(data)
    grad .= 0
    return GraphNode{op, N}(args, grad, data, params)
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
    ordered = Vector{GraphNode}()
    visited = Set{GraphNode}()
    visit!(node, visited, ordered)
    return ordered
end

function zerograd!(order :: Vector{GraphNode})
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
function forward!(order::Vector{GraphNode}, pairs...)
    for (tensor, data) in pairs
        tensor.data .= data
    end
    for node in order
        primal!(node)
    end
end

function backward!(order::Vector{GraphNode})
	seed = last(order)
	seed.grad .= 1
    for node in reverse(order)
        adjoint!(node)
    end
end

function optimize!(graph, η)
    for node in graph
        if node isa GraphWeight
            node.data .-= η .* node.grad
        end
    end
end