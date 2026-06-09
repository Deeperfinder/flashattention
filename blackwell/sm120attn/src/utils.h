#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flash {

using namespace cute;

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<bool Transposed=false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));    // (nrow = (2, MMA_M), ncol = (2, MMA_N))
        } else {
            return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));    //（nrow = (2, MMA_N)， ncol = (2, MMA_M)）
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
// For SM90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
// For SM90, FP8, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using X = Underscore;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
        if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
            return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
        } else {
            static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
            static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
            static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{});  // (((2, 2), N / 32))
            // This combines the first two modes (<0, 0> and <0, 1>) into one mode.
            // Will require register shuffling later to be correct.
            return make_layout(make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
                               get<1>(acc_layout),
                               coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));  // ((4, 2, 2), MMA_M, N / 32 * MMA_N)
            // This combination is right but doesn't work with register shuffling.
            // return make_layout(make_layout(coalesce(make_layout(get<0, 0>(acc_layout), get<0, 0, 0>(l))), get<0, 1>(acc_layout), get<0, 0, 1>(l)),
            //                    get<1>(acc_layout),
            //                    coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
        }
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}
}