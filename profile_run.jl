# ==============================================================================
# profile_run.jl — Text-based profiling for CNNetwork (v5.0 batched)
# Run with: julia --project profile_run.jl
# ==============================================================================

using MLDatasets
using Flux: onehotbatch
using Profile
using Printf
using LinearAlgebra
using Random

include("clothesolver.jl")

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading FashionMNIST... ")
train_data = MLDatasets.FashionMNIST(split=:train)
X_train    = Float32.(reshape(train_data.features, 28, 28, 1, :))
Y_train    = Float32.(onehotbatch(train_data.targets, 0:9))
println("done ($(size(X_train, 4)) samples)")

sep(c='─', n=62) = println(c^n)

# ── Build + benchmark helper ──────────────────────────────────────────────────
function build_and_bench(batch_size::Int, n_iters::Int; do_profile=false)
    my_net_def = Chain(
        Conv((3, 3), 1 => 6,  bias=false),
        MaxPool((2, 2)),
        Conv((3, 3), 6 => 16, bias=false),
        MaxPool((2, 2)),
        Flatten(),
        Dense(400 => 84, relu),
        Dropout(0.4),
        Dense(84 => 10)
    )

    my_model = build_model(my_net_def, (28, 28, 1); batch_size)
    # input/target/loss are owned by the model — no separate alloc_act! needed.
    input_node  = my_model.input
    target_node = my_model.target
    loss_fn     = my_model.loss

    # Grab a fixed batch from training data for benchmarking
    X_batch = X_train[:, :, :, 1:batch_size]
    Y_batch = Y_train[:, 1:batch_size]

    function one_pass!()
        zero_a_grad!(my_model.pool)
        zero_w_grad!(my_model.pool)
        copyto!(input_node.data,  X_batch)
        copyto!(target_node.data, Y_batch)
        preds = forward!(my_model.chain, input_node, true)
        primal!(loss_fn, preds, target_node)
        loss_fn.out.grad[1] = 1.0f0
        adjoint!(loss_fn, preds, target_node)
        backward!(my_model.chain, input_node, true)
    end

    # Warm-up
    for _ in 1:10; one_pass!() end

    # Overall timing
    t = @elapsed for _ in 1:n_iters; one_pass!() end
    ms_per_pass    = t / n_iters * 1e3
    ms_per_sample  = ms_per_pass / batch_size
    throughput     = batch_size * n_iters / t

    sep('═')
    @printf("  batch_size = %d   (%d iters)\n", batch_size, n_iters)
    sep('═')
    @printf("  Total      : %.3f s\n",     t)
    @printf("  Per pass   : %.3f ms\n",    ms_per_pass)
    @printf("  Per sample : %.4f ms\n",    ms_per_sample)
    @printf("  Throughput : %.0f img/s\n\n", throughput)

    # Forward vs backward
    preds = forward!(my_model.chain, input_node, true)
    primal!(loss_fn, preds, target_node)
    t_fwd  = @elapsed for _ in 1:n_iters
        zero_a_grad!(my_model.pool); copyto!(input_node.data, X_batch)
        forward!(my_model.chain, input_node, true)
    end
    t_loss = @elapsed for _ in 1:n_iters; primal!(loss_fn, preds, target_node) end
    t_bwd  = @elapsed for _ in 1:n_iters
        zero_a_grad!(my_model.pool); loss_fn.out.grad[1] = 1.0f0
        adjoint!(loss_fn, preds, target_node)
        backward!(my_model.chain, input_node, true)
    end
    tot = t_fwd + t_loss + t_bwd
    @printf("  Forward  : %6.3f ms  (%4.1f%%)\n", t_fwd /n_iters*1e3, t_fwd /tot*100)
    @printf("  Loss     : %6.3f ms  (%4.1f%%)\n", t_loss/n_iters*1e3, t_loss/tot*100)
    @printf("  Backward : %6.3f ms  (%4.1f%%)\n", t_bwd /n_iters*1e3, t_bwd /tot*100)
    println()

    # Per-layer forward
    layers = my_model.chain.layers
    layer_meta = [
        ("Conv1    (28×28×1  → 28×28×6)",   layers[1], input_node,    200),
        ("MaxPool1 (28×28×6  → 14×14×6)",   layers[2], layers[1].out, 2000),
        ("Conv2    (14×14×6  → 14×14×16)",  layers[3], layers[2].out, 200),
        ("MaxPool2 (14×14×16 →  7×7×16)",   layers[4], layers[3].out, 2000),
        ("Flatten  (7×7×16   → 400)",        layers[5], layers[4].out, 5000),
        ("Dense1   (400      → 84)",         layers[6], layers[5].out, 5000),
        ("ReLU     (84)",                    layers[7], layers[6].out, 5000),
        ("Dropout  (84)",                    layers[8], layers[7].out, 5000),
        ("Dense2   (84       → 10)",         layers[9], layers[8].out, 5000),
    ]
    forward!(my_model.chain, input_node, true)
    total_fwd = sum(@elapsed(for _ in 1:n; primal!(l,inp,true) end)/n for (_,l,inp,n) in layer_meta)

    sep()
    @printf("  %-42s  %8s  %8s\n", "Layer (forward)", "µs/pass", "share")
    sep()
    for (name, layer, inp, n) in layer_meta
        t_l = @elapsed for _ in 1:n; primal!(layer, inp, true) end
        us = t_l/n*1e6; pct = (t_l/n)/total_fwd*100
        @printf("  %-42s  %8.2f  %7.1f%%\n", name, us, pct)
    end
    sep()
    @printf("  %-42s  %8.2f\n\n", "Total", total_fwd*1e6)

    # Per-layer backward
    layer_meta_bwd = [
        ("Dense2   adj",   layers[9], layers[8].out, 5000),
        ("Dropout  adj",   layers[8], layers[7].out, 5000),
        ("ReLU     adj",   layers[7], layers[6].out, 5000),
        ("Dense1   adj",   layers[6], layers[5].out, 5000),
        ("Flatten  adj",   layers[5], layers[4].out, 5000),
        ("MaxPool2 adj",   layers[4], layers[3].out, 2000),
        ("Conv2    adj",   layers[3], layers[2].out, 200),
        ("MaxPool1 adj",   layers[2], layers[1].out, 2000),
        ("Conv1    adj",   layers[1], input_node,    200),
    ]
    total_bwd = sum(@elapsed(for _ in 1:n; adjoint!(l,inp,true) end)/n for (_,l,inp,n) in layer_meta_bwd)

    sep()
    @printf("  %-42s  %8s  %8s\n", "Layer (backward)", "µs/pass", "share")
    sep()
    for (name, layer, inp, n) in layer_meta_bwd
        t_l = @elapsed for _ in 1:n; adjoint!(layer, inp, true) end
        us = t_l/n*1e6; pct = (t_l/n)/total_bwd*100
        @printf("  %-42s  %8.2f  %7.1f%%\n", name, us, pct)
    end
    sep()
    @printf("  %-42s  %8.2f\n\n", "Total", total_bwd*1e6)

    # Allocation report
    sep('═'); println("  ALLOCATION REPORT  (100 passes)"); sep('═')
    println()
    @time for _ in 1:100; one_pass!() end
    println()

    if do_profile
        sep('═'); println("  PROFILE SAMPLES  (2000 passes, mincount=3)"); sep('═'); println()
        Profile.clear()
        @profile for _ in 1:2000; one_pass!() end
        Profile.print(maxdepth=8, mincount=3, sortedby=:count, groupby=:thread)
    end
end

# ── Run benchmarks ────────────────────────────────────────────────────────────
println()
println("══════════════════════════════════════════════════════════════")
println("  BATCH SIZE = 1   (baseline, comparable to v4.0 numbers)")
println("══════════════════════════════════════════════════════════════")
build_and_bench(1, 1000; do_profile=false)

println()
println("══════════════════════════════════════════════════════════════")
println("  BATCH SIZE = 32  (true batching)")
println("══════════════════════════════════════════════════════════════")
build_and_bench(32, 500; do_profile=false)

println()
println("══════════════════════════════════════════════════════════════")
println("  PROFILE  (batch_size=32, 2000 passes)")
println("══════════════════════════════════════════════════════════════")
build_and_bench(32, 2000; do_profile=true)
