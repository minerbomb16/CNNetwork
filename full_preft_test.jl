# ==============================================================================
# full_preft_test.jl — Standalone training & evaluation for CNNetwork
# Translated from test.ipynb (cells 767c64e7, 325ec456) for command-line use.
# Target: >85% test accuracy in exactly 3 epochs.
#
# Why the notebook ran ~10 s/epoch while this script runs ~5 s/epoch:
#   Old notebook defined:
#       function train!(model, X, Y; batch_size, η)
#   Untyped keyword args default to Any.  Inside the loop body:
#       for b_start in 1:batch_size:N        # range step = Any
#   Julia cannot infer the type, so every arithmetic result is boxed on the heap
#   across 1875 iterations per epoch — ~2× slowdown.
#
#   Fix (applied here and in clothesolver.jl):
#   build_model stores batch_size as a concrete Int field.  train!/evaluate
#   read it as  bs = model.batch_size  (inferred as Int at compile time).
#   No const declaration required at the call site.
#
# Run with: julia --project full_preft_test.jl
# ==============================================================================

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using MLDatasets
using Flux: onehotbatch
using Random
using Printf

include("clothesolver.jl")

sep(c='─', n=62) = println(c^n)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading FashionMNIST... ")
train_data = MLDatasets.FashionMNIST(split=:train)
test_data  = MLDatasets.FashionMNIST(split=:test)

# MLDatasets ≥ 0.5 already returns Float32 in [0,1].
# Older versions return UInt8. This handles both.
let raw = reshape(train_data.features, 28, 28, 1, :)
    global X_train = Float32.(raw) ./ (eltype(raw) <: Integer ? 255f0 : 1f0)
end
let raw = reshape(test_data.features, 28, 28, 1, :)
    global X_test  = Float32.(raw) ./ (eltype(raw) <: Integer ? 255f0 : 1f0)
end

Y_train_raw = train_data.targets
Y_test_raw  = test_data.targets
Y_train = Float32.(onehotbatch(Y_train_raw, 0:9))
Y_test  = Float32.(onehotbatch(Y_test_raw,  0:9))

println("done  (train=$(size(X_train,4))  test=$(size(X_test,4))  " *
        "range=[$(minimum(X_train)), $(maximum(X_train))])")

# ── Hyperparameters ───────────────────────────────────────────────────────────
const EPOCHS = 3
const LR     = 0.02f0

# ── Model ─────────────────────────────────────────────────────────────────────
# No BATCH_SIZE constant needed — build_model stores it in model.batch_size.
my_model = build_model(
    Chain(
        Conv((3, 3), 1 => 6,  bias=false),
        MaxPool((2, 2)),
        Conv((3, 3), 6 => 16, bias=false),
        MaxPool((2, 2)),
        Flatten(),
        Dense(400 => 84, relu),
        Dropout(0.4),
        Dense(84 => 10)
    ),
    (28, 28, 1);
    batch_size = 32
)

println("Model built  (batch_size=$(my_model.batch_size), weights=$(length(my_model.pool.weights)))")
println()

# ── Evaluate ──────────────────────────────────────────────────────────────────
function evaluate(model::CompiledModel, X, Y_raw)
    correct     = 0
    N           = size(X, 4)
    bs          = model.batch_size        # concrete Int — type-stable
    input_node  = model.input
    final_layer = model.chain.layers[end]

    for i in 1:bs:N
        b_end    = min(i + bs - 1, N)
        actual_b = b_end - i + 1
        fill!(input_node.data, 0f0)
        copyto!(@view(input_node.data[:, :, :, 1:actual_b]),
                @view(X[:, :, :, i:b_end]))
        forward!(model.chain, input_node, false)
        for b in 1:actual_b
            correct += (argmax(@view(final_layer.out.data[:, b])) - 1 == Y_raw[i + b - 1])
        end
    end
    return correct / N * 100.0
end

# ── Train one epoch ───────────────────────────────────────────────────────────
function train_epoch!(model::CompiledModel, X, Y; η::Float32)
    bs          = model.batch_size        # concrete Int — fully inferred
    input_node  = model.input
    target_node = model.target
    loss_fn     = model.loss
    N           = size(X, 4)
    indices     = randperm(N)
    total_loss  = 0.0
    n_batches   = 0

    for b_start in 1:bs:N
        b_end     = min(b_start + bs - 1, N)
        b_end - b_start + 1 < bs && break   # skip incomplete last batch

        batch_idx = indices[b_start:b_end]
        zero_w_grad!(model.pool)
        zero_a_grad!(model.pool)
        copyto!(input_node.data,  X[:, :, :, batch_idx])
        copyto!(target_node.data, Y[:, batch_idx])

        preds      = forward!(model.chain, input_node, true)
        batch_loss = primal!(loss_fn, preds, target_node)
        loss_fn.out.grad[1] = 1.0f0
        adjoint!(loss_fn, preds, target_node)
        backward!(model.chain, input_node, true)
        optimize!(model.pool, η)

        total_loss += batch_loss
        n_batches  += 1
    end
    return total_loss / n_batches
end

# ── Run ───────────────────────────────────────────────────────────────────────
sep('═')
@printf("  TRAINING  (%d epochs, batch=%d, η=%.4f)\n", EPOCHS, my_model.batch_size, LR)
sep('═')
@printf("  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s\n",
        "Epoch", "Loss", "Acc%", "Train(s)", "Eval(s)", "Total(s)")
sep()

best_acc   = 0.0
total_time = @elapsed for ep in 1:EPOCHS
    t_train = @elapsed loss = train_epoch!(my_model, X_train, Y_train; η=LR)
    t_eval  = @elapsed acc  = evaluate(my_model, X_test, Y_test_raw)
    global best_acc = max(best_acc, acc)
    ok = acc > 85.0 ? "✓" : " "
    @printf("  %s ep %d   %6.4f    %6.2f%%  %6.2fs    %5.2fs    %6.2fs\n",
            ok, ep, loss, acc, t_train, t_eval, t_train + t_eval)
end

println()
sep('═')
@printf("  Total wall time  : %.2f s\n", total_time)
@printf("  Best accuracy    : %.2f%%\n", best_acc)
println()
if best_acc > 85.0
    @printf("  ✓  TARGET MET  (%.2f%% > 85%% within %d epochs)\n", best_acc, EPOCHS)
else
    @printf("  ✗  TARGET MISSED  (%.2f%% ≤ 85%%) — increase η or EPOCHS\n", best_acc)
end
sep('═')
