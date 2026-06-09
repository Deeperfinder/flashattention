import argparse
import math
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Your kernel
from sm120attn import fa_5090

# FA3
try:
    import flash_attn_interface
    flash_attn_func_fa3 = flash_attn_interface.flash_attn_func
    FA3_AVAILABLE = True
except ImportError:
    FA3_AVAILABLE = False

# FA4
try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_fa4
    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False

# FA2
try:
    from flash_attn import flash_attn_func as flash_attn_func_fa2
    FA2_AVAILABLE = True
except ImportError:
    FA2_AVAILABLE = False

# cuDNN frontend python
try:
    import cudnn
    CUDNN_AVAILABLE = True
except ImportError:
    cudnn = None
    CUDNN_AVAILABLE = False


def get_tflops(b, h, lq, lk, d, ms):
    return (4 * b * h * lq * lk * d) / (ms * 1e9)


def benchmark_kernel(func, iterations, warmup):
    for _ in range(warmup):
        _ = func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    out = None
    for _ in range(iterations):
        out = func()
    end_event.record()

    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / iterations
    return out, ms


def normalize_output(name, out):
    if out is None:
        return None

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise TypeError(f"{name} returned unsupported type: {type(out)}")

    if out.dim() != 4:
        raise ValueError(f"{name} returned shape {tuple(out.shape)}, expected 4D tensor")

    return out


def compute_error(a, b):
    diff = (a - b).abs()
    return diff.mean().item(), diff.max().item()


def print_error_block(title, a, b):
    diff, max_diff = compute_error(a, b)
    print("-" * 50)
    print(f"Mean Absolute Error vs {title}: {diff:.6e}")
    print(f"Max Absolute Error vs {title}:  {max_diff:.6e}")
    return diff, max_diff


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError(f"Unsupported tensor data type for cuDNN: {torch_type}")


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=-1):
    """
    Input q/k/v expected shape: [B, H, L, D]
    Returns a callable run() that executes cuDNN SDPA and returns output [B, H, Lq, D]
    """
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, headdim_k = k.shape
    _, nheads_v, seqlen_v, headdim_v = v.shape

    assert headdim == headdim_k, "cuDNN SDPA requires q and k head dim match"
    assert nheads_k == nheads_v, "k/v nheads mismatch"
    assert seqlen_k == seqlen_v, "k/v seqlen mismatch"
    assert headdim == headdim_v, "This helper currently assumes V dim == Q/K dim"
    assert cudnn is not None, "CUDNN is not available"

    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty_like(q_gpu)
    stats_gpu = torch.empty(
        b, nheads, seqlen_q, 1,
        dtype=torch.float32,
        device=q.device
    )

    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_t = graph.tensor_like(q_gpu.detach())
    k_t = graph.tensor_like(k_gpu.detach())
    v_t = graph.tensor_like(v_gpu.detach())

    o_t, stats_t = graph.sdpa(
        name="sdpa",
        q=q_t,
        k=k_t,
        v=v_t,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    o_t.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q_t: q_gpu,
        k_t: k_gpu,
        v_t: v_gpu,
        o_t: o_gpu,
        stats_t: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run():
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-B', type=int, default=1)
    parser.add_argument('--heads', '-H', type=int, default=5)
    parser.add_argument('--qlen', type=int, default=144169)
    parser.add_argument('--klen', type=int, default=144169)
    parser.add_argument('--dim', '-D', type=int, default=128)
    parser.add_argument('--iters', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16

    # Base tensors: [B, H, L, D]
    q = torch.randn((args.batch, args.heads, args.qlen, args.dim), device=device, dtype=dtype)
    k = torch.randn((args.batch, args.heads, args.klen, args.dim), device=device, dtype=dtype)
    v = torch.randn((args.batch, args.heads, args.klen, args.dim), device=device, dtype=dtype)

    results = {}
    outputs = {}

    # --------------------------------------------------
    # Test 1: FlashAttention-2
    # expects [B, L, H, D]
    # --------------------------------------------------
    if FA2_AVAILABLE:
        q_fa2 = q.transpose(1, 2).contiguous()
        k_fa2 = k.transpose(1, 2).contiguous()
        v_fa2 = v.transpose(1, 2).contiguous()

        def run_fa2():
            out = flash_attn_func_fa2(q_fa2, k_fa2, v_fa2, causal=args.causal)
            return out.transpose(1, 2).contiguous()

        out_fa2, ms_fa2 = benchmark_kernel(run_fa2, args.iters, args.warmup)
        out_fa2 = normalize_output("FlashAttn2", out_fa2)
        results["FlashAttn2"] = ms_fa2
        outputs["FlashAttn2"] = out_fa2
    else:
        out_fa2 = None
        print("FlashAttention-2 not available.")

    # --------------------------------------------------
    # Test 2: FlashAttention-3
    # expects [B, L, H, D]
    # --------------------------------------------------
    if FA3_AVAILABLE:
        q_fa3 = q.transpose(1, 2).contiguous()
        k_fa3 = k.transpose(1, 2).contiguous()
        v_fa3 = v.transpose(1, 2).contiguous()

        def run_fa3():
            out = flash_attn_func_fa3(q_fa3, k_fa3, v_fa3, causal=args.causal)
            return out.transpose(1, 2).contiguous()

        out_fa3, ms_fa3 = benchmark_kernel(run_fa3, args.iters, args.warmup)
        out_fa3 = normalize_output("FlashAttn3", out_fa3)
        results["FlashAttn3"] = ms_fa3
        outputs["FlashAttn3"] = out_fa3
    else:
        out_fa3 = None
        print("FlashAttention-3 not available.")

    # --------------------------------------------------
    # Test 3: FlashAttention-4
    # assume [B, H, L, D]
    # --------------------------------------------------
    if FA4_AVAILABLE:
        def run_fa4():
            out = flash_attn_func_fa4(q, k, v, causal=args.causal)
            return out

        out_fa4, ms_fa4 = benchmark_kernel(run_fa4, args.iters, args.warmup)
        out_fa4 = normalize_output("FlashAttn4", out_fa4)
        results["FlashAttn4"] = ms_fa4
        outputs["FlashAttn4"] = out_fa4
    else:
        out_fa4 = None
        print("FlashAttention-4 not available.")

    # --------------------------------------------------
    # Test 4: cuDNN SDPA
    # expects [B, H, L, D]
    # --------------------------------------------------
    if CUDNN_AVAILABLE:
        try:
            run_cudnn = cudnn_spda_setup(q, k, v, causal=args.causal)
            out_cudnn, ms_cudnn = benchmark_kernel(run_cudnn, args.iters, args.warmup)
            out_cudnn = normalize_output("cuDNN", out_cudnn)
            results["cuDNN"] = ms_cudnn
            outputs["cuDNN"] = out_cudnn
        except Exception as e:
            out_cudnn = None
            print(f"cuDNN SDPA not available or unsupported for this config: {e}")
    else:
        out_cudnn = None
        print("cuDNN Python frontend not available.")

    # --------------------------------------------------
    # Test 5: Your Kernel
    # --------------------------------------------------
    def run_fa5090():
        return fa_5090(q, k, v, attn_mask=None, is_causal=args.causal)

    out_custom, ms_custom = benchmark_kernel(run_fa5090, args.iters, args.warmup)
    out_custom = normalize_output("FA_5090", out_custom)
    results["FA_5090"] = ms_custom
    outputs["FA_5090"] = out_custom

    # --------------------------------------------------
    # Test 6: PyTorch SDPA
    # --------------------------------------------------
    def run_sdpa():
        return sdpa(q, k, v, is_causal=args.causal)

    out_sdpa, ms_sdpa = benchmark_kernel(run_sdpa, args.iters, args.warmup)
    out_sdpa = normalize_output("SDPA", out_sdpa)
    results["SDPA"] = ms_sdpa
    outputs["SDPA"] = out_sdpa

    # --------------------------------------------------
    # Print benchmark report
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print(f"Benchmark: B={args.batch}, H={args.heads}, Lq={args.qlen}, Lk={args.klen}, D={args.dim}, Causal={args.causal}")
    print(f"Precision: {args.dtype}")
    print("-" * 50)

    for name, ms in results.items():
        tflops = get_tflops(args.batch, args.heads, args.qlen, args.klen, args.dim, ms)
        print(f"{name:<15}: {ms:>8.3f} ms | {tflops:>8.2f} TFLOPS")

    # --------------------------------------------------
    # Error check
    # --------------------------------------------------
    all_diffs = []

    if out_fa2 is not None:
        diff, _ = print_error_block("FlashAttn2", out_custom, out_fa2)
        all_diffs.append(diff)

    if out_fa3 is not None:
        diff, _ = print_error_block("FlashAttn3", out_custom, out_fa3)
        all_diffs.append(diff)

    if out_fa4 is not None:
        diff, _ = print_error_block("FlashAttn4", out_custom, out_fa4)
        all_diffs.append(diff)

    if out_cudnn is not None:
        diff, _ = print_error_block("cuDNN", out_custom, out_cudnn)
        all_diffs.append(diff)

    diff_sdpa, _ = print_error_block("SDPA", out_custom, out_sdpa)
    all_diffs.append(diff_sdpa)

    if out_fa2 is not None:
        diff_fa2_sdpa, _ = print_error_block("SDPA (FA2)", out_fa2, out_sdpa)
        all_diffs.append(diff_fa2_sdpa)

    if out_fa3 is not None:
        diff_fa3_sdpa, _ = print_error_block("SDPA (FA3)", out_fa3, out_sdpa)
        all_diffs.append(diff_fa3_sdpa)

    if out_fa4 is not None:
        diff_fa4_sdpa, _ = print_error_block("SDPA (FA4)", out_fa4, out_sdpa)
        all_diffs.append(diff_fa4_sdpa)

    if out_cudnn is not None:
        diff_cudnn_sdpa, _ = print_error_block("SDPA (cuDNN)", out_cudnn, out_sdpa)
        all_diffs.append(diff_cudnn_sdpa)

    if out_fa2 is not None and out_fa3 is not None:
        diff_fa2_fa3, _ = print_error_block("FlashAttn3", out_fa2, out_fa3)
        all_diffs.append(diff_fa2_fa3)

    if out_fa2 is not None and out_fa4 is not None:
        diff_fa2_fa4, _ = print_error_block("FlashAttn4", out_fa2, out_fa4)
        all_diffs.append(diff_fa2_fa4)

    if out_fa3 is not None and out_fa4 is not None:
        diff_fa3_fa4, _ = print_error_block("FlashAttn4", out_fa3, out_fa4)
        all_diffs.append(diff_fa3_fa4)

    if out_fa2 is not None and out_cudnn is not None:
        diff_fa2_cudnn, _ = print_error_block("cuDNN", out_fa2, out_cudnn)
        all_diffs.append(diff_fa2_cudnn)

    if out_fa3 is not None and out_cudnn is not None:
        diff_fa3_cudnn, _ = print_error_block("cuDNN", out_fa3, out_cudnn)
        all_diffs.append(diff_fa3_cudnn)

    if out_fa4 is not None and out_cudnn is not None:
        diff_fa4_cudnn, _ = print_error_block("cuDNN", out_fa4, out_cudnn)
        all_diffs.append(diff_fa4_cudnn)

    # --------------------------------------------------
    # Status
    # --------------------------------------------------
    threshold = 1e-3
    print("-" * 50)
    if all(diff < threshold for diff in all_diffs):
        print("Status: \033[92mPASS\033[0m")
    else:
        print("Status: \033[91mFAIL (Significant Difference)\033[0m")

    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
