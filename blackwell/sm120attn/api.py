import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from typing import Tuple
import fa5090_cuda

def fa_5090(q, k, v, attn_mask=None, is_causal=False, **kwargs):
    #  b, h, s, d
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal=is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    is_causal = False
    delta_s = torch.empty_like(q)
    softmax_scale = (q.shape[-1]) ** (-0.5)
    return fa5090_cuda.fwd(q, k, v, delta_s, KL, None, softmax_scale, is_causal, is_bf16)