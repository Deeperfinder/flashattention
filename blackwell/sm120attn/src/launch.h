#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "params.h"
#include "tile_scheduler.h"
#include "kernel_ws.h"
#include "kernel_traits.h"

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream){
    using Element = typename Kernel_traits::Element;
    using ElementOut = typename Kernel_traits::ElementOut;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;
    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits>;
    // using Scheduler = flash::SingleTileScheduler;
    using Scheduler = flash::StaticPersistentTileScheduler;
    typename CollectiveMainloop::Params mainloop_params = 
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),
            {params.seqlen_q, params.d, params.h, params.b}, // shape_Q
            {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},
            static_cast<Element const*>(params.k_ptr),
            {params.seqlen_k, params.d, params.h_k, params.b}, // shape_K
            {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
            static_cast<Element const*>(params.v_ptr),
            {params.d, params.seqlen_k, params.h_k, params.b}, // Shape_Vt
            {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_Vt
            params.scale_softmax_log2
        });
        
    typename CollectiveEpilogue::Params epilogue_params = 
        CollectiveEpilogue::to_underlying_arguments({
            static_cast<ElementOut*>(params.o_ptr),
            {params.seqlen_q, params.d, params.h, params.b}, // shape_O
            {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride}, // stride_O
            static_cast<float *>(params.softmax_lse_ptr),
            {_1{}, params.seqlen_q, params.h*params.seqlen_q}, // stride_LSE
        });
        
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);
    // Get the ptr to kernel function
    void * kernel;
    kernel = (void *) flash::compute_attn_ws<Kernel_traits, Is_causal, Scheduler>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    params.m_block_divmod = cutlass::FastDivmod(num_blocks_m);
    params.total_blocks = num_blocks_m * params.h * params.b;
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, 170);
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if constexpr (Headdim == 64 || Headdim == 128) {
            run_flash_fwd <
                Flash_fwd_kernel_traits<Headdim, 128, 32, 2, 1, T, O>,
                Is_causal
                >(params, stream);
        } else {
            static_assert(Headdim == 64 || Headdim ==128, "Unsupported headdim");
        }
    });
}