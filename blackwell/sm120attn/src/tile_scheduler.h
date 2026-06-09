#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

namespace flash{

struct SingleTileScheduler{

public:
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int *const tile_count_semaphore = nullptr;
    };

    struct Params {};

    static Params
    to_underlying_arguments(Arguments const& args){
        return {};
    };

    static dim3
    get_grid_dim(Arguments const& args, int num_sm){
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
    }

    struct WorkTileInfo {
        int M_idx = 0;
        int N_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, N_idx, B_idx};
        }

        CUTLASS_DEVICE
        WorkTileInfo
        get_next_work(Params const& params) const {
            return {-1, -1, -1, false};
        }
    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }
};

class StaticPersistentTileScheduler {

public:
    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore = nullptr;
    };

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(Arguments const& args){
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod{args.num_blocks_m}, cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm){
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
            return {m_block, bidh, bidb};
        }
    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }
};

} // namespace fa
