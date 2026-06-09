#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

////////////////////////////////////////////////////////////////////////////////////
struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void * __restrict__ q_ptr;
    void * __restrict__ k_ptr;
    void * __restrict__ v_ptr;
    void *__restrict__ delta_s_ptr;
    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

struct Flash_fwd_params : public Qkv_params {
    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;
    void * __restrict__ s_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, unpadded_seqlen_k;
    cutlass::FastDivmod head_divmod, m_block_divmod;
    int total_blocks;
    int seqlen_s;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;

    // Random state.
    at::PhiloxCudaState philox_args;
    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;
    int * __restrict__ tile_count_semaphore;
};