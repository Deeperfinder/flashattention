#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace fa{
    
template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierVarGroupSizeSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  Barrier barrier_[SequenceDepth][SequenceLength];
};

template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrierVarGroupSize {
public:
  static constexpr int SequenceDepth = SequenceDepth_;
  static constexpr int SequenceLength = SequenceLength_;
  using Barrier = cutlass::arch::ClusterBarrier;
  using SharedStorage = fa::OrderedSequenceBarrierVarGroupSizeSharedStorage<SequenceDepth, SequenceLength>;
  

  struct Params {
    uint32_t group_id;
    uint32_t* group_size_list;
  };

private :
  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier *barrier_ptr_;
  cutlass::PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrierVarGroupSize() = delete;
  OrderedSequenceBarrierVarGroupSize(const OrderedSequenceBarrierVarGroupSize&) = delete;
  OrderedSequenceBarrierVarGroupSize(OrderedSequenceBarrierVarGroupSize&&) = delete;
  OrderedSequenceBarrierVarGroupSize& operator=(const OrderedSequenceBarrierVarGroupSize&) = delete;
  OrderedSequenceBarrierVarGroupSize& operator=(OrderedSequenceBarrierVarGroupSize&&) = delete;
  ~OrderedSequenceBarrierVarGroupSize() = default;

  CUTLASS_DEVICE
  OrderedSequenceBarrierVarGroupSize(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 - starts with an opposite phase
      stage_({0, params.group_id == 0, 0}) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by the one elected thread of the block
    if (warp_idx == 0 && lane_predicate) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(*(params.group_size_list + l));
        }
      }
    }
    cutlass::arch::fence_barrier_init();
  }

  // Wait on a stage to be unlocked
  CUTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  CUTLASS_DEVICE
  void advance() {
    ++stage_;
  }

private:

  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

} // namepsace fa