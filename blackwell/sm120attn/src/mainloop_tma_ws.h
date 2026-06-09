#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "utils.h"
#include "named_barrier.h"

namespace flash{
using namespace cute;
template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {
    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int BlockMean = Ktraits::BlockMean;

    using GmemTiledCopy = typename Ktraits::GmemTiledCopy;
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head, batch) 
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using LayoutP = typename Ktraits::LayoutP;
    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));
    
    using TMA_KV = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        _1{}));
    using TMA_Vt = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideQKV{})),
        take<0,2>(SmemLayoutVt{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));
    using SmemCopyAtomQK = typename Ktraits::SmemCopyAtomQK;
    using SmemCopyAtomV = typename Ktraits::SmemCopyAtomV;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;
    static constexpr int NumMmaThreads = size(TiledMmaQK{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size((SmemLayoutQ{})) * sizeof_bits<Element>::value));
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutK{})) * sizeof_bits<Element>::value));
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutVt{})) * sizeof_bits<Element>::value));
    
    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* ptr_Vt;
        ShapeQKV const shape_Vt;
        StrideQKV const stride_Vt;
        float const softmax_scale_log2;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_Vt;
        TMA_Q tma_load_Q;
        TMA_KV tma_load_K;
        TMA_Vt tma_load_Vt;
        float const softmax_scale_log2;
    };

    static Params
    to_underlying_arguments(Arguments const& args){
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopy{},
            mQ,
            SmemLayoutQ{},
            select<0,2>(TileShape_MNK{}),
            _1{});
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_KV tma_load_K = make_tma_copy(
            GmemTiledCopy{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1,2>(TileShape_MNK{}),
            _1{}); 
        Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, select<1,0,2,3>(args.stride_Vt));
        TMA_Vt tma_load_Vt = make_tma_copy(
            GmemTiledCopy{},
            mVt,
            SmemLayoutVt{}(_, _, _0{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{});
        // auto [Seqlen_Q, Seqlen_K, HeadNum, Batch] = args.shape_ds;

        return {
            args.shape_Q, args.shape_K, args.shape_Vt,
            tma_load_Q, tma_load_K, tma_load_Vt,
            args.softmax_scale_log2};
    };

    //// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params){
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_n_block_max(Params const & mainloop_params, int m_block){
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    template <typename SchedulerParams, typename SharedStorage, typename WorkTileInfo>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         SchedulerParams const& scheduler_params,
         MainloopPipelineQ pipeline_q,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineStateQ& smem_pipe_write_q,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage& shared_storage,
         WorkTileInfo work_tile_info,
         int& work_idx,
         int& tile_count_semaphore){

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
        
        int n_block_max = get_n_block_max(mainloop_params, m_block);

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
   
        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
        Tensor mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);

        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0,2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // 按block_m切分，找到block的全局index [BLK_M, BLK_K, k]
        Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1,2>(TileShape_MNK{}), make_coord(_, _0{}));     // （ N, K, _)
        Tensor gVt = local_tile(mVt(_, _, bidh, bidb), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));

        auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{}); // 当前的thread，获取当前thread需要copy的在block中的tile
        Tensor tQgQ = block_tma_q.partition_S(gQ); // [CPY, CPY_M, CPY_K], 传入source tensor， 获取当前线程需要copy的source tensor有哪些
        Tensor tQsQ = block_tma_q.partition_D(sQ); // des tensor
        auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
        auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
        Tensor tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));   // [TMA, k, batch]
        Tensor tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));   // [TMA, PIPE]
        uint16_t mcast_mask_kv = 0;

        int n_block = n_block_max - 1;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write_q);
            copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgQ, tQsQ);
            ++smem_pipe_write_q;
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                  tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                  tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        }

        n_block--;
        if (lane_predicate) {
            #pragma unroll 2
            for( ; n_block >= 0; --n_block){
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                      tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                      tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        ++work_idx;
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q,
              MainloopPipeline pipeline_k, 
              MainloopPipeline pipeline_v,
              PipelineStateQ& smem_pipe_write_q,
              PipelineState& smem_pipe_write_k, 
              PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
          pipeline_q.producer_tail(smem_pipe_write_q);
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO,
        SoftmaxFused& softmax_fused,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, " O tensor must be rmem resident.");
        
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockK = get<1>(TileShape_MNK{});
        static constexpr int kBlockN = get<2>(TileShape_MNK{});
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor CKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ); // tile : [128,128] -> ((_2,_2,_2),_1,_8):((_1,_2,_4),_0,_8)
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_,_, Int<0>{}));
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_,_, Int<0>{}));
        // Tensor tOrP = make_tensor_like<Element>(LayoutP{});

        // copy qk and sf from smem to rmem
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomV{}, tiled_mma_pv);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
        Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt); 

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read){
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block = n_block_count -1;
        auto copy_k_block = [&](auto block_id) {
            auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
            copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
        };
        auto copy_v_block = [&](auto block_id) {
            auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
            copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
        };

        consumer_wait(pipeline_q, smem_pipe_read_q);
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read_k);
        copy_k_block(_0{});

        CUTLASS_PRAGMA_UNROLL
        for(int k_block=0; k_block < size<2>(tSrQ); ++k_block){
            cute::gemm(tiled_mma_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), tSrS);
            if (k_block < size<2>(tSrQ) - 1){
                copy_k_block(k_block + 1);
            }else {
                pipeline_k.consumer_release(smem_pipe_read_k);
                ++smem_pipe_read_k;
            }
        }
        Tensor scores_scale = softmax_fused.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
        softmax_fused.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
        Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaQK>(tSrS.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        convert_type_out(tOrP_acc, tOrP);
        consumer_wait(pipeline_v, smem_pipe_read_v);
        copy_v_block(Int<0>{});
        // Need to initialize tOrO in the case of RescaleOBeforeGemm where we will scale tOrO even in the 1st iter
        clear(tOrO);

        CUTLASS_PRAGMA_UNROLL
        for (int v_block=0; v_block < size<2>(tOrP); ++v_block){
            cute::gemm(tiled_mma_pv, tOrP(_, _, v_block), tOrVt(_, _, v_block), tOrO);
            if (v_block < size<2>(tOrP) - 1){
                copy_v_block(v_block + 1);
            } else {
                pipeline_v.consumer_release(smem_pipe_read_v);
                ++smem_pipe_read_v;
            }
        }
        n_block--;
        #pragma unroll 1
        for(; n_block >=0; --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

            consumer_wait(pipeline_k, smem_pipe_read_k);
            copy_k_block(_0{});
            CUTLASS_PRAGMA_UNROLL
            for (int k_block=0; k_block < size<2>(tSrQ); ++k_block){
                cute::gemm(tiled_mma_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), tSrS);
                if (k_block < size<2>(tSrQ) -1){
                    copy_k_block(k_block + 1);
                } else {
                    pipeline_k.consumer_release(smem_pipe_read_k);
                    ++smem_pipe_read_k;
                }
            }

            cute::copy(softmax_fused.template max_get_scale</*Is_first=*/false, /*Check_inf=*/true>(tSrS), scores_scale);
            softmax_fused.rescale_o(tOrO, scores_scale);
            softmax_fused.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS);
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaQK>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);

            consumer_wait(pipeline_v, smem_pipe_read_v);
            copy_v_block(Int<0>{});
            CUTLASS_PRAGMA_UNROLL
            for (int v_block=0; v_block < size<2>(tOrP); ++v_block) {
                cute::gemm(tiled_mma_pv, tOrP(_, _, v_block), tOrVt(_, _, v_block), tOrO);
                if (v_block < size<2>(tOrP) - 1){
                    copy_v_block(v_block + 1);
                } else {
                    pipeline_v.consumer_release(smem_pipe_read_v);
                    ++smem_pipe_read_v;
                }
            }
        }
        cute::copy(softmax_fused.finalize(1.0f), scores_scale);
        softmax_fused.rescale_o(tOrO, scores_scale);
        return;
    }

  }; // mainloop

} // namespace flash