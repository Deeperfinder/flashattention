#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "named_barrier.h"

using namespace cute;

template <
    int kStages,
    int EpiStages,
    typename Elements,
    typename OutputType,
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutO
>
struct SharedStorageQKVO : cute::aligned_struct<128, _0>{
    alignas(1024) cute::ArrayEngine<Elements, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(1024) cute::ArrayEngine<Elements, cute::cosize_v<SmemLayoutK>> smem_k;
    alignas(1024) cute::ArrayEngine<Elements, cute::cosize_v<SmemLayoutV>> smem_v;
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    
    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename fa::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };    
};

// 128 x 128
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    typename elem_type = cutlass::half_t,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits {
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == 64 || kBlockM == 128);
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;
    static constexpr int NumSFQK = kHeadDim / 16;
    static constexpr int NumSFPV = kBlockN / 16;

    using Element = elem_type;
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
    
    using PermutationTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
    using PermutationTileN = _32;
    using PermutationTileK = _16;
    using MMA_P_T = Tile<PermutationTileM, PermutationTileN, PermutationTileK>;

    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>
                                            >;                         //excute unit  repeat
    static_assert(kBlockM == 128, "kBlockM must be 128");
    using MMA_Atom_Arch = std::conditional_t<std::is_same_v<Element, cutlass::half_t>,
                                            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                                            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
                                            >;

    using TiledMmaQK = decltype(cute::make_tiled_mma(
        MMA_Atom_Arch{},
        AtomLayoutMNK{},
        MMA_P_T{}
    ));
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        MMA_Atom_Arch{},
        AtomLayoutMNK{},
        MMA_P_T{}
    ));

    using GmemTiledCopy = SM90_TMA_LOAD;

    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::MN, Element,
                                      decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        Step<_1, _2, _3>{}));
    // using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    //     decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    // using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0,1>(TileShape_MNK{})));

    
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element); //64
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    // using SmemLayoutAtomQKV = decltype(
    //     composition(Swizzle<3, 4, 3>{},
    //                 Layout<Shape<_8, Int<kBlockKGmem>>,   // 8， 64
    //                        Stride<Int<kBlockKGmem>, _1>>{}));
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // using SmemLayoutV = decltype(tile_to_shape(
    //     SmemLayoutAtomQKV{},
    //     make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    // using SmemLayoutVt = decltype(
    //     composition(SmemLayoutV{},
    //                 make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
    //                                     Step<_2, _1, _3>{})));      // (K , N)
    // using SmemLayoutVt = decltype(tile_to_shape(
    //     SmemLayoutAtomQKV{},
    //     make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})
    // ));
    // using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
    //     decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))
    ));
    using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomV = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    // using SmemCopyAtomQ = Copy_Atom<UniversalCopy<uint128_t>, Element>;
    // using SmemCopyAtomKV = Copy_Atom<UniversalCopy<uint128_t>, Element>;
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 16>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );

    using SharedStorage = SharedStorageQKVO<kStages, EpiStages, Element, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutVt, SmemLayoutO>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;
    using EpilogueBarrier = typename fa::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};
