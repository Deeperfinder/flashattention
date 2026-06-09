"""Simple accuracy comparison between FA_5090 implementation and FlashAttention.

The script creates random query/key/value tensors in either bfloat16 or float16,
runs both attention kernels, and reports absolute/relative error statistics.  It is
primarily used for quick sanity checks during development.

Usage::

    python test_accuracy.py

The inputs are hardcoded in this example; modify the tensor shapes or data type
list as needed.
"""

from sm120attn import fa_5090
from flash_attn import flash_attn_func
import torch


def calculate_diff(x: torch.Tensor, y: torch.Tensor, name: str = "") -> tuple[float, float]:
    """Compute and print absolute/relative errors between two tensors.

    Both inputs are first cast to ``float32`` to eliminate differences due to
    reduced precision.  The function prints max/mean absolute and relative
    errors, and returns ``(max_abs, max_rel)``.  A simple heuristic determines
    pass/fail based on max absolute error.
    """

    # cast to float32 to avoid dtype-induced discrepancies
    x = x.float()
    y = y.float()

    abs_diff = (x - y).abs()
    rel_diff = abs_diff / (y.abs() + 1e-8)  # avoid division-by-zero

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    print(f"--- {name} Comparison ---")
    print(f"  Max Abs Diff: {max_abs:.6e}")
    print(f"  Mean Abs Diff: {mean_abs:.6e}")
    print(f"  Max Rel Diff: {max_rel:.6e}")
    print(f"  Mean Rel Diff: {mean_rel:.6e}")

    if max_abs < 1e-3:
        print("Status: \033[92mPASS\033[0m")
    else:
        print("Status: \033[91mFAIL (Significant Difference)\033[0m")

    return max_abs, max_rel

def run_random_tests(batch: int = 1,
                     heads: int = 8,
                     seq_len: int = 9450,
                     dim: int = 128) -> None:
    """Generate random tensors and compare attention outputs.

    Parameters are hard-coded defaults matching the original script.  The
    input shape for ``fa_5090`` is ``[batch, heads, seq_len, dim]``, while
    ``flash_attn_func`` expects ``[batch, seq_len, heads, dim]`` so a
    transpose is required.
    """

    for tensor_type in (torch.bfloat16, torch.float16):
        # create random inputs on CUDA
        q = torch.randn((batch, heads, seq_len, dim),
                        device="cuda",
                        dtype=tensor_type)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # run FA_5090; output shape is [batch, heads, seq_len, dim]
        out_5090 = fa_5090(q, k, v)

        print("Running compare FA_5090 with FlashAttn...")
        print("=" * 95)

        # prepare inputs for FlashAttention: transpose to [b, s, h, d]
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        scale = q.shape[-1] ** -0.5
        out_fa = flash_attn_func(q_fa, k_fa, v_fa, causal=False, softmax_scale=scale)
        out_fa = out_fa.transpose(1, 2).contiguous()  # back to [b, h, s, d]

        # FA_5090 returns an extra leading batch dimension sometimes; take [0]
        out_5090_res = out_5090[0]

        calculate_diff(out_5090_res,
                       out_fa,
                       f"Random tensor ({tensor_type})")

        diff = (out_5090_res - out_fa).abs().mean().item()
        max_diff = (out_5090_res - out_fa).abs().max().item()

        print("-" * 50)
        print(f"Mean absolute error vs FlashAttn: {diff:.6e}")
        print(f"Max absolute error  vs FlashAttn: {max_diff:.6e}")
        print("=" * 95)


def main() -> None:
    # single default test invocation; you may replace with argument parsing in
    # future if interactive configuration is needed.
    run_random_tests()


if __name__ == "__main__":
    main()
