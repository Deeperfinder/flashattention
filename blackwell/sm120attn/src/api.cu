#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/numeric_types.h>

#include "params.h"
#include "launch.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x "must have shape (" #__VA_ARGS__ ") ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,  
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t unpadded_seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor delta_s,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool is_bf16,
                      bool seqlenq_ngroups_swapped=false){
    // Reset the parameters
    params = {};                  
    // Set the pointers and strides
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.delta_s_ptr = delta_s.data_ptr();
    
    // All stride are in elements, not bytes
    params.q_row_stride = q.stride(-2);
    params.k_row_stride = k.stride(-2);      // b, h, s, d
    params.v_row_stride = v.stride(-2);
    params.q_head_stride = q.stride(-3);
    params.k_head_stride = k.stride(-3);
    params.v_head_stride = v.stride(-3);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-2);
    params.o_head_stride = out.stride(-3);
    
    // // 是否是变长sequence
    // if (cu_seqlens_q_d == nullptr) {
    //     params.q_batch_stride = q.stride(0) * 2;
    //     params.k_batch_stride = k.stride(0) * 2;
    //     params.v_batch_stride = v.stride(0) * 2;
    //     params.ds_batch_stride = delta_s.stride(0);
    //     params.o_batch_stride = out.stride(0);
    //     if (seqlenq_ngroups_swapped) {
    //         params.q_batch_stride *= seqlen_q;
    //         params.k_batch_stride *= seqlen_q;
    //     }
    // }

    // params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    // params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    // params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // set the dimensions
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.unpadded_seqlen_k = unpadded_seqlen_k;
    // params.seqlen_q_rounded = seqlen_q_rounded;
    // params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.head_divmod = cutlass::FastDivmod(int(h));

    // Set the different scale values
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    params.seqlen_s = 128; // size of BLOCK_M
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    // params.is_seqlens_k_cumulative = true;
    params.is_bf16 = is_bf16;
    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
}

template<bool IsBF16>
void run_mha_fwd_dispatch_dtype(Flash_fwd_params &params, cudaStream_t stream) { 
  using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
  if(params.d == 64) {
    run_mha_fwd_<OType, 64, OType>(params, stream);
  } else if (params.d == 128){
    run_mha_fwd_<OType, 128, OType>(params, stream);
  }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
  BOOL_SWITCH(params.is_bf16, IsBF16, ([&]{
    run_mha_fwd_dispatch_dtype<IsBF16>(params, stream);
  }));
}

std::vector<at::Tensor> 
mha_fwd(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &delta_s,
        int unpadded_k,
        c10::optional<at::Tensor> &out_,
        const float softmax_scale,
        bool is_causal,
        bool is_bf16
){
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm120 = dprops->major == 12 && dprops->minor == 0;
  bool is_sm121 = dprops->major == 12 && dprops->minor == 1;
  TORCH_CHECK(is_sm120 || is_sm121, "only support Blackwell GPUs with sm_120 or sm_121")

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
              "FlashAttention only supports fp16, bf16 data type");
  TORCH_CHECK(k.scalar_type() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_dtype, "query and value must have the same dtype");
  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(delta_s.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  TORCH_CHECK(q.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "Input tensor must be contiguous");

  const auto sizes = q.sizes(); // b h s d
  auto opts = q.options();
  const int batch_size = sizes[0];
  int seqlen_q = sizes[2];
  int num_heads = sizes[1];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(2);
  const int num_heads_k = k.size(1);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size <=256, "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(num_heads == num_heads_k, "We do not support MQA/GQA yet")

  CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size);
  CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size);
  CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size);
  TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");

  auto dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
  at::Tensor out = torch::empty({batch_size, num_heads, seqlen_q, head_size}, opts.dtype(dtype));

  auto round_multiple = [](int x, int m){ return (x + m -1) / m * m;}; 
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor p;

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   seqlen_q, seqlen_k, unpadded_k,
                   seqlen_q_rounded,
                   num_heads, num_heads_k,
                   head_size, head_size,
                   q, k, v, delta_s, out,
                   /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr,
                   /*seqused_k=*/nullptr,
                   nullptr,
                   softmax_lse.data_ptr(),
                   /*p_dropout*/0.f,
                   softmax_scale,
                   /*window_size_left=*/-1,
                   /*window_size_right=*/is_causal ? 0 : -1,
                   is_bf16);
  auto tile_count_semaphore = is_causal ? torch::full({1}, 132, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
  params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

  if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
  } else {
    // If seqlen_k is 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
}