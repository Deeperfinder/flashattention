#include "common.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <iostream>

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_kernel(
    const nv_bfloat16 *Q, // [bs, len_q, dim]
    const nv_bfloat16 *K, // [bs, len_kv, dim]
    const nv_bfloat16 *V, // [bs, len_kv, dim]
    nv_bfloat16 *O, //[bs, len_q, dim]
    int bs,
    int len_q,
    int len_kv
){
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    // 1 CTA handles 1 BLOCK_Q
    // grid = bs * (len_q / BLOCK_Q)
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
     
    Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;  // Q += (bs * len_q * DIM +  q_block_id * BLOCK_Q * DIM)
    K += bs_id * len_kv * DIM;
    V += bs_id * len_kv * DIM;
    O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem; // double buffer K
    const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];  // col major
    
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    // pre-compute address and swizzling for ldmatrix
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        // A tile
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile trans
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const float softmaxScale = rsqrtf(static_cast<float>(DIM));

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }
    // load Q [BLOCK_Q, DIM]
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // shared - > registers
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16); // row
            addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16); // col
            ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }
    }
    __syncthreads();

    const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

    auto load_K = [&](int kv_id) {
        if (kv_id < num_kv_iter) {
        // double buffer for K
        const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);
        K += BLOCK_KV * DIM;
        }
        asm volatile("cp.async.commit_group;");
    };
    auto load_V = [&](int kv_id) {
        // single buffer for V
        const uint32_t dst = V_smem;
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid);
        V += BLOCK_KV * DIM;
        asm volatile("cp.async.commit_group;");
    };

    // prefetch K
    load_K(0);

    for(int kv_id = 0; kv_id < num_kv_iter; kv_id ++){
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
        // prefetch V
        __syncthreads();
        load_V(kv_id);

        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // shared -> registers
        for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d+=2) {
                uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
                addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);  // row
                addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);  // col
                ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
            }
        }
        // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
                    mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                                 K_rmem[mma_id_kv][mma_id_d],
                                 S_rmem[mma_id_q][mma_id_kv]);
        // prefetch K
        load_K(kv_id + 1);

        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for(int reg_id = 0; reg_id<4; reg_id++)
                    S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmaxScale;
            
            // rowmax
            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));   // C0 and C1
                this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));   // C2 and C3
            }

            // butterfly reduction with 4 threads. T0. T1. T2. T3
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

            // new rowmax
            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);
            // rescale for previous O
            float rescale[2];
            rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            // save new rowmax
            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            // rowsumexp
            float this_rowsumexp[2] = {};
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]); // c0
                regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]); // c1
                regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]); // c0
                regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]); // c1

                this_rowsumexp[0] += regs[0] + regs[1];
                this_rowsumexp[1] += regs[2] + regs[3];

                // pack to P registers for next MMA
                nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
                this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
            }

            // butterfly reduction
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

            // accumulate to total rowsumexp
            rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
        }

        // load V [BLOCK_KV, DIM]
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // shared -> registers
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d+=2) {
            uint32_t addr = V_smem_thread;
            addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);  // row
            addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);  // col
            // 这里使用trans改变的是ldmatrix 结果中的thread layout， 由row-major 变为了 col-major, 便于按col进行seq_len 维度的乘法
            ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        }

        //MMA_O += P @ V
        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
                for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                    mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                                 V_rmem[mma_id_kv][mma_id_d],
                                 O_rmem[mma_id_q][mma_id_d]);
    }

    // write to O
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + lane_id / 4;
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            //divide by softmax denominator
            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];

            reinterpret_cast<nv_bfloat162 *>(O + (row+0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
            reinterpret_cast<nv_bfloat162 *>(O + (row+8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
        }
}

torch::Tensor attention_sm120(
    const torch::Tensor& Q, 
    const torch::Tensor& K,
    const torch::Tensor& V
){

    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be of bfloat16 dtype");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be of bfloat16 dtype");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be of bfloat16 dtype");
    torch::Tensor O = torch::empty_like(Q);
    TORCH_CHECK(O.dtype() == torch::kBFloat16, "O must be of bfloat16 dtype")

    const int bs = Q.size(0) * Q.size(1);
    const int len_q = Q.size(2);
    const int len_kv = K.size(2);
    const int dim = Q.size(3);

    if (dim!=128){
        std::cerr <<"Unsupported dim=" << dim << std::endl;
        exit(1);
    }


    auto Q_ptr = reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr());
    auto K_ptr = reinterpret_cast<const nv_bfloat16 *>(K.data_ptr());
    auto V_ptr = reinterpret_cast<const nv_bfloat16 *>(V.data_ptr());
    auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

    const int BLOCK_Q = 64;
    const int BLOCK_KV = 64;
    const int DIM = 128;
    const int NUM_WARPS = 4;
    const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
    const int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(nv_bfloat16);

    auto kernel = attention_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
    launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, Q_ptr, K_ptr, V_ptr, O_ptr, bs, len_q, len_kv);
    
    return O;
}