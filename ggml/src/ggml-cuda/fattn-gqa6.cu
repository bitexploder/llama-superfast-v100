/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// prompt attention with all 6 query heads of a KV head packed into the rows of two tensor-core GEMMs (Volta).
//
// Ported from 1Cat-vLLM 99bbedf, csrc/attention/sm70_v37/ (SPDX BSD-3-Clause, notice above, which those files carry):
// - EpilogueVisitorSoftmaxGqa6 is epilogue_visitor_with_softmax.h's EpilogueVisitorSoftmaxV37, with ggml's mask added to the
//   scaled logits (read once per token and shared by its 6 rows), a fully masked tile giving zeros and a max of -inf, and the
//   source operand C and its private tile-zero slab dropped;
// - gqa6_qk_kernel is the batched path of CUTLASS example 35's GemmWithEpilogueVisitor::operator() (gemm_with_softmax.h's
//   kernel), batched over the KV heads (grid.z);
// - gqa6_block_update is reduce_softmax_final.h's ApplySoftmaxFinalReductionV37 fused with prefill.cu's prepare_prefix_update;
// - PVTransformA, PVOutputOp and gqa6_pv_kernel are prefill.cu's ExpRowSumTransformA, PVOutputOp and the PV GEMM, with split-K 2
//   (each split keeps its own FP32 running output, which is linear in the shared per-row scales, so the merge sums them);
// - gqa6_merge replaces merge_prefix_accumulator_tail: there is no causal tail (the mask covers the current chunk), and it writes
//   ggml's F32 dst directly.
// Not ported: tail.cu, bridge.cu, register.cpp and everything that uses torch. The workspace comes from ggml's pool: no side
// stream, no host sync, no cudaMemcpyToSymbol, so it is safe under CUDA graph capture.
//
// Per chunk of at most GQA6_ROW_CHUNK tokens (the chunks run in order on the stream), for each KV head h (6 query
// heads 6h..6h+5), with T tokens, R = 6T packed rows (row = token*6 + head), padded to rows_pad (a multiple of the 128-row tile
// with zero rows that are never written to dst):
// 1. pack Q to FP16 as [h][row][256];
// 2. for each block of Nb keys: QK GEMM + softmax epilogue (FP16 probabilities against each 256-key tile's max, FP32 tile max and
//    sum), the block update (block max, tile scales, running max and sum), the PV GEMM (A rescaled per tile on load, FP32
//    accumulation, split-K 2, running output rescaled in the epilogue);
// 3. merge the two splits, normalise and write dst.

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"

#include "fattn-gqa6.cuh"

#include <atomic>

// route thresholds (the route is faster at every bench cell: 256 to 2,048 rows, 4,096 to 245,760 keys);
// LLAMA_FA_GQA6_MIN_ROWS / LLAMA_FA_GQA6_MIN_KV override them
#define GQA6_MIN_ROWS 256
#define GQA6_MIN_KV   4096

// each call runs its tokens in chunks of at most this many (rows_pad, Nb and the workspace follow the chunk);
// LLAMA_FA_GQA6_ROW_CHUNK overrides it, and 0 runs the whole call as one chunk
#define GQA6_ROW_CHUNK 1024

namespace cutlass {
namespace epilogue {
namespace threadblock {

// 1Cat's EpilogueVisitorSoftmaxV37 with ggml's mask. For each row of a 256-key tile it stores the FP16 probabilities
// exp(logit - tile max), the FP32 tile max (-inf if every key is masked) and the FP32 sum of the rounded probabilities.
template <typename ThreadblockShape_, int ThreadCount, typename OutputTileIterator_, typename ElementAccumulator_,
          typename ElementNorm_, typename ElementSum_, typename ElementSoftmaxCompute_>
class EpilogueVisitorSoftmaxGqa6 {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = ElementAccumulator_;

  using ElementNorm = ElementNorm_;
  using ElementSum = ElementSum_;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using SoftmaxFragment = Array<ElementSoftmaxCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth;
  static int const kColumnIterations = OutputTileIterator::ThreadMap::Iterations::kColumn;

  static_assert(kElementsPerAccess == 8, "the mask is read 8 halves (16 bytes) at a time");

  struct Params {
    float scale;          // ggml's KQ scale
    half const* mask;     // ggml's F16 mask, at this key block's first column
    int64_t mask_stride;  // halves per token row of the mask
    int rows;             // real packed rows (6 x tokens); rows beyond are padding
  };
  using Arguments = Params;

  struct SharedStorage {};

 private:
  Params const& params_;
  MatrixCoord extent_;

  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementNorm* ptr_Max_;
  ElementSum* ptr_Sum_;

  int column_offset_;

  ElementSoftmaxCompute accum_max_;
  // Keep unrounded logits until the complete tile-row maximum is known.
  SoftmaxFragment row_logits_[kColumnIterations];
  int first_row_fragment_;

  MatrixCoord thread_offset_;

 public:
  CUTLASS_DEVICE
  EpilogueVisitorSoftmaxGqa6(Params const& params, SharedStorage& shared_storage, cutlass::MatrixCoord const& problem_size,
                             int thread_idx, int warp_idx, int lane_idx, typename OutputTileIterator::Params params_D,
                             typename OutputTileIterator::Element* ptr_D, ElementNorm* ptr_Max, ElementSum* ptr_Sum,
                             cutlass::MatrixCoord const& threadblock_offset, int column_offset)
      : params_(params),
        extent_(problem_size),
        iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        ptr_Max_(ptr_Max),
        ptr_Sum_(ptr_Sum),
        column_offset_(column_offset) {
    GGML_UNUSED(shared_storage);
    GGML_UNUSED(warp_idx);
    GGML_UNUSED(lane_idx);
  }

  CUTLASS_DEVICE
  void set_k_partition(int split_k_index, int split_k_slices) {
    GGML_UNUSED(split_k_index);
    GGML_UNUSED(split_k_slices);
  }

  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) { GGML_UNUSED(batch_idx); }

  CUTLASS_DEVICE
  void begin_epilogue() {}

  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    GGML_UNUSED(step_idx);
    fragment_D_.clear();
  }

  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    GGML_UNUSED(row_idx);
    accum_max_ = -INFINITY;
  }

  CUTLASS_DEVICE
  void visit(int iter_idx, int row_idx, int column_idx, int frag_idx, AccumulatorFragment const& accum) {
    GGML_UNUSED(iter_idx);
    GGML_UNUSED(row_idx);

    thread_offset_ = iterator_D_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    bool const column_guard = thread_offset_.column() < extent_.column();
    int const row = thread_offset_.row();

    SoftmaxFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      result[i] = ElementSoftmaxCompute(accum[i]) * params_.scale;
    }
    // ggml's mask: one row per token, shared by its 6 packed rows
    if (column_guard && row < params_.rows) {
      uint4 const m = *reinterpret_cast<uint4 const*>(params_.mask + int64_t(row / 6) * params_.mask_stride + thread_offset_.column());
      half2 const* m2 = reinterpret_cast<half2 const*>(&m);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElementsPerAccess / 2; ++i) {
        float2 const f = __half22float2(m2[i]);
        result[2 * i + 0] += f.x;
        result[2 * i + 1] += f.y;
      }
    }
    if (!column_idx) first_row_fragment_ = frag_idx;
    row_logits_[column_idx] = result;

    if (column_guard) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElementsPerAccess; ++i) {
        accum_max_ = fmaxf(accum_max_, result[i]);
      }
    }
  }

  CUTLASS_DEVICE
  void end_row(int row_idx) {
    GGML_UNUSED(row_idx);

    bool const is_first_thread_in_tile = ((threadIdx.x % kThreadsPerRow) == 0);
    bool const row_guard = thread_offset_.row() < extent_.row();
    bool const is_write_thread = row_guard && is_first_thread_in_tile;

    accum_max_ = warp_reduce_max_(accum_max_);
    // a fully masked tile row: probabilities 0 (exp(-inf)), max -inf
    ElementSoftmaxCompute const centre = accum_max_ == -INFINITY ? ElementSoftmaxCompute(0) : accum_max_;

    NumericArrayConverter<ElementOutput, ElementSoftmaxCompute, kElementsPerAccess> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kColumnIterations; ++c) {
      SoftmaxFragment values;
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kElementsPerAccess; ++e) {
        values[e] = fast_exp(row_logits_[c][e] - centre);
      }
      reinterpret_cast<OutputVector*>(&fragment_D_)[first_row_fragment_ + c] = convert(values);
    }
    // Match the mass of the probabilities actually consumed by PV.
    ElementSoftmaxCompute rounded_sum = ElementSoftmaxCompute(0);
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kColumnIterations; ++c) {
      int const fragment = first_row_fragment_ + c;
      int const column = iterator_D_.thread_start().column() + OutputTileIterator::ThreadMap::iteration_offset(fragment).column();
      OutputVector const& probabilities = reinterpret_cast<OutputVector const*>(&fragment_D_)[fragment];
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kElementsPerAccess; ++e) {
        if (column + e < extent_.column()) {
          rounded_sum += ElementSoftmaxCompute(probabilities[e]);
        }
      }
    }
    ElementSoftmaxCompute const accum_sum = warp_reduce_sum_(rounded_sum);

    ElementNorm* curr_ptr_max = ptr_Max_ + thread_offset_.row() + column_offset_;
    ElementSum* curr_ptr_sum = ptr_Sum_ + thread_offset_.row() + column_offset_;

    arch::global_store<ElementNorm, sizeof(ElementNorm)>(ElementNorm(accum_max_), (void*)curr_ptr_max, is_write_thread);
    arch::global_store<ElementSum, sizeof(ElementSum)>(ElementSum(accum_sum), (void*)curr_ptr_sum, is_write_thread);
  }

  CUTLASS_DEVICE
  void end_step(int step_idx) {
    GGML_UNUSED(step_idx);
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  CUTLASS_DEVICE
  void end_epilogue() {}

 private:
  CUTLASS_DEVICE
  ElementSoftmaxCompute warp_reduce_sum_(ElementSoftmaxCompute sum_) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = kThreadsPerRow >> 1; i > 0; i >>= 1) {
      sum_ += __shfl_xor_sync(0xFFFFFFFF, sum_, i);
    }
    return sum_;
  }

  CUTLASS_DEVICE
  ElementSoftmaxCompute warp_reduce_max_(ElementSoftmaxCompute max_) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = kThreadsPerRow >> 1; i > 0; i >>= 1) {
      max_ = fmaxf(max_, __shfl_xor_sync(0xFFFFFFFF, max_, i));
    }
    return max_;
  }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

namespace {

using Element = cutlass::half_t;

constexpr int kHeadDim = 256;
constexpr int kQPerKV = 6;
constexpr int kTileKeys = 256;  // keys per QK tile (the N tile), one max/sum per row each

// both GEMMs: v37's 128x256x32 threadblock tiles, 64x64x32 warp tiles, Volta's 8x8x4 mma, 2 stages
using TBShape = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstShape = cutlass::gemm::GemmShape<8, 8, 4>;
static_assert(TBShape::kN == kTileKeys, "one QK tile is one softmax tile");

// ---- QK: scores = softmax tiles of (Q K^T * scale + mask) ----

using QKLinearOp = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
using QKDefault = typename cutlass::gemm::kernel::DefaultGemm<
    Element, cutlass::layout::RowMajor, 8, Element, cutlass::layout::ColumnMajor, 8, Element, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, TBShape, WarpShape, InstShape, QKLinearOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, Element,
                                                             Element, Element, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;
using QKMma = typename QKDefault::Mma;
using QKVisitor = cutlass::epilogue::threadblock::EpilogueVisitorSoftmaxGqa6<
    TBShape, QKDefault::kThreadCount, typename QKDefault::Epilogue::OutputTileIterator, float, float, float, float>;
using QKEpilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    QKVisitor, typename QKDefault::Epilogue>::Epilogue;

union QKSharedStorage {
  typename QKMma::SharedStorage main_loop;
  struct {
    typename QKEpilogue::SharedStorage epilogue;
    typename QKVisitor::SharedStorage visitor;
  } epilogue;
};

struct QKParams {
  int rows_pad;  // GEMM M
  int width;     // GEMM N: keys in this block
  typename QKMma::IteratorA::Params params_A;
  typename QKMma::IteratorB::Params params_B;
  typename QKVisitor::OutputTileIterator::Params params_D;
  Element* q;  // packed Q, [head][rows_pad][256]
  Element* k;  // K at this block's first key
  int64_t k_head_stride;
  Element* scores;  // [head][rows_pad][score_stride]
  int64_t scores_head_stride;
  float* tile_max;  // [head][tile][rows_pad]
  float* tile_sum;
  int64_t stats_head_stride;
  typename QKVisitor::Params visitor;
};

static __global__ void __launch_bounds__(QKDefault::kThreadCount, 1) gqa6_qk_kernel(const QKParams p) {
  extern __shared__ int4 gqa6_smem_qk[];
  QKSharedStorage& shared = *reinterpret_cast<QKSharedStorage*>(gqa6_smem_qk);

  const int tile_m = blockIdx.x;
  const int tile_n = blockIdx.y;
  const int head = blockIdx.z;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const int lane_idx = threadIdx.x % 32;

  typename QKMma::IteratorA iterator_A(p.params_A, p.q + head * int64_t(p.rows_pad) * kHeadDim, {p.rows_pad, kHeadDim},
                                       thread_idx, {tile_m * TBShape::kM, 0});
  typename QKMma::IteratorB iterator_B(p.params_B, p.k + head * p.k_head_stride, {kHeadDim, p.width}, thread_idx,
                                       {0, tile_n * TBShape::kN});

  QKMma mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
  typename QKMma::FragmentC accumulators;
  accumulators.clear();
  mma(kHeadDim / TBShape::kK, accumulators, iterator_A, iterator_B, accumulators);

  QKVisitor visitor(p.visitor, shared.epilogue.visitor, {p.rows_pad, p.width}, thread_idx, warp_idx, lane_idx, p.params_D,
                    p.scores + head * p.scores_head_stride, p.tile_max + head * p.stats_head_stride,
                    p.tile_sum + head * p.stats_head_stride, {tile_m * TBShape::kM, tile_n * TBShape::kN},
                    tile_n * p.rows_pad);
  QKEpilogue epilogue(shared.epilogue.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(visitor, accumulators);
}

// ---- block update: block max, tile scales, running max and sum (ApplySoftmaxFinalReductionV37 + prepare_prefix_update) ----

static __global__ void gqa6_block_update(const float* __restrict__ tile_max, const float* __restrict__ tile_sum,
                                         float* __restrict__ tile_scale, float* __restrict__ run_max,
                                         float* __restrict__ run_sum, float* __restrict__ old_scale,
                                         float* __restrict__ block_scale, const int rows_pad, const int n_tiles,
                                         const int64_t stats_head_stride, const bool first) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int head = blockIdx.y;
  if (row >= rows_pad) {
    return;
  }
  const float* tm = tile_max + head * stats_head_stride + row;
  const float* ts = tile_sum + head * stats_head_stride + row;
  float* tsc = tile_scale + head * stats_head_stride + row;

  float block_max = -INFINITY;
  for (int t = 0; t < n_tiles; ++t) {
    block_max = fmaxf(block_max, tm[t * int64_t(rows_pad)]);
  }
  float block_mass = 0.0f;
  for (int t = 0; t < n_tiles; ++t) {
    const float m = tm[t * int64_t(rows_pad)];
    const float scale = m == -INFINITY ? 0.0f : expf(m - block_max);
    tsc[t * int64_t(rows_pad)] = scale;
    block_mass += ts[t * int64_t(rows_pad)] * scale;
  }

  const int64_t i = head * int64_t(rows_pad) + row;
  if (first) {
    old_scale[i] = 0.0f;
    block_scale[i] = 1.0f;
    run_max[i] = block_max;
    run_sum[i] = block_mass;
    return;
  }
  const float prev_max = run_max[i];
  const float new_max = fmaxf(prev_max, block_max);
  const float prev_scale = prev_max == -INFINITY ? 0.0f : expf(prev_max - new_max);
  const float next_scale = block_max == -INFINITY ? 0.0f : expf(block_max - new_max);
  old_scale[i] = prev_scale;
  block_scale[i] = next_scale;
  run_max[i] = new_max;
  run_sum[i] = run_sum[i] * prev_scale + block_mass * next_scale;
}

// ---- PV: running output = running output * old scale + (P * tile scale) V * block scale ----

using PVLayout = cutlass::layout::RowMajor;
using PVLinearOp = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;

CUTLASS_DEVICE int pv_output_row(int sequence);

struct PVOutputOp : PVLinearOp {
  using Base = PVLinearOp;
  struct Params : Base::Params {
    float const* old_scales;
    float const* block_scales;
    bool initialize;
    CUTLASS_HOST_DEVICE
    Params() : Base::Params(1.0f, 0.0f), old_scales(nullptr), block_scales(nullptr), initialize(true) {}
    CUTLASS_HOST_DEVICE
    Params(float const* old_, float const* next_, bool init)
        : Base::Params(1.0f, init ? 0.0f : 1.0f), old_scales(old_), block_scales(next_), initialize(init) {}
  };
  float const* old_scales;
  float const* block_scales;
  bool initialize;
  mutable int sequence = 0;
  CUTLASS_HOST_DEVICE
  explicit PVOutputOp(Params const& params)
      : Base(params), old_scales(params.old_scales), block_scales(params.block_scales), initialize(params.initialize) {}
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return !initialize; }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accum, FragmentSource const& source) const {
    const int row = pv_output_row(sequence++);
    FragmentOutput result;
    const float a = old_scales[row];
    const float b = block_scales[row];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      result[i] = fmaf(source[i], a, accum[i] * b);
    }
    return result;
  }
  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accum) const {
    const int row = pv_output_row(sequence++);
    FragmentOutput result;
    const float b = block_scales[row];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      result[i] = accum[i] * b;
    }
    return result;
  }
};

using PVDefault = typename cutlass::gemm::kernel::DefaultGemm<
    Element, PVLayout, 8, Element, PVLayout, 8, float, PVLayout, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    TBShape, WarpShape, InstShape, PVOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using PVDefaultMma = typename PVDefault::Mma;
using PVEpilogue = typename PVDefault::Epilogue;
using PVOutputIterator = typename PVEpilogue::OutputTileIterator;

// The epilogue calls the output operator once per access, in order; recover the row of each call (prefill.cu's pv_output_row).
CUTLASS_DEVICE int pv_output_row(int sequence) {
  using Map = typename PVOutputIterator::ThreadMap;
  constexpr int kAccesses = PVOutputIterator::Fragment::kElements / PVOutputIterator::kElementsPerAccess;
  const int step = sequence / kAccesses;
  const int fragment = sequence % kAccesses;
  PVOutputIterator iterator(typename PVOutputIterator::Params(PVLayout(kHeadDim)), nullptr, {TBShape::kM, kHeadDim},
                            threadIdx.x, {int(blockIdx.x) * TBShape::kM, 0});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < PVOutputIterator::kIterations; ++i) {
    if (i < step) {
      ++iterator;
    }
  }
  return iterator.thread_start().row() + Map::iteration_offset(fragment).row();
}

using PVIteratorA = typename PVDefaultMma::IteratorA;
using PVIteratorB = typename PVDefaultMma::IteratorB;
using PVSmemIteratorA = typename PVDefaultMma::SmemIteratorA;
using PVSmemIteratorB = typename PVDefaultMma::SmemIteratorB;

// Rescales each loaded probability by its 256-key tile's scale exp(tile max - block max) (ExpRowSumTransformA). The next
// fragment's scales are loaded one call ahead, when it starts a new tile.
struct PVTransformA {
  using InputFragment = typename PVIteratorA::Fragment;
  using OutputFragment = cutlass::Array<typename PVSmemIteratorA::Element, InputFragment::kElements>;
  using ThreadMap = typename PVIteratorA::ThreadMap;
  static constexpr int kAccessesPerVector = PVIteratorA::UnderlyingIterator::kAccessesPerVector;
  static constexpr int kContiguousIterations = ThreadMap::Iterations::kContiguous;
  static constexpr int kStridedIterations = ThreadMap::Iterations::kStrided;

  float const* scale;  // this head's tile scales, [tile][rows_pad]
  int rows_pad;
  int last_tile;
  int k_offset;  // key (within the block) of the next fragment
  int tile;      // tile of the scales held in tile_scale
  int row[kStridedIterations];
  float tile_scale[kStridedIterations];

  CUTLASS_DEVICE
  PVTransformA(float const* scale_, int rows_pad_, int n_tiles, int k_begin, int row_begin)
      : scale(scale_), rows_pad(rows_pad_), last_tile(n_tiles - 1), k_offset(k_begin) {
    const auto thread_offset = ThreadMap::initial_offset(threadIdx.x);
    tile = min(k_offset / kTileKeys, last_tile);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kStridedIterations; ++s) {
      row[s] = row_begin + thread_offset.strided() + s * ThreadMap::Delta::kStrided;
      tile_scale[s] = scale[tile * int64_t(rows_pad) + row[s]];
    }
  }

  CUTLASS_DEVICE
  OutputFragment operator()(InputFragment const& input) {
    OutputFragment output;
    constexpr int kElementsPerAccess = PVIteratorA::AccessType::kElements;
    auto const* input_access = reinterpret_cast<typename PVIteratorA::AccessType const*>(&input);
    auto* output_access = reinterpret_cast<typename PVIteratorA::AccessType*>(&output);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kStridedIterations; ++s) {
      const float probability_scale = tile_scale[s];
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kContiguousIterations; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          const int index = v + kAccessesPerVector * (c + s * kContiguousIterations);
          typename PVIteratorA::AccessType transformed;
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kElementsPerAccess; ++e) {
            transformed[e] = Element(static_cast<float>(input_access[index][e]) * probability_scale);
          }
          output_access[index] = transformed;
        }
      }
    }
    // The double-buffered pipeline transforms one final masked look-ahead fragment (all zeros); the clamp keeps its scale
    // load in bounds.
    k_offset += TBShape::kK;
    const int next_tile = min(k_offset / kTileKeys, last_tile);
    if (next_tile != tile) {
      tile = next_tile;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kStridedIterations; ++s) {
        tile_scale[s] = scale[tile * int64_t(rows_pad) + row[s]];
      }
    }
    return output;
  }
};

using PVTransformB =
    cutlass::NumericArrayConverter<typename PVSmemIteratorB::Element, typename PVIteratorB::Element, PVIteratorB::Fragment::kElements>;
using PVMma = cutlass::gemm::threadblock::MmaPipelined<typename PVDefaultMma::Shape, PVIteratorA, PVSmemIteratorA, PVIteratorB,
                                                       PVSmemIteratorB, float, PVLayout, typename PVDefaultMma::Policy,
                                                       PVTransformA, PVTransformB>;

union PVSharedStorage {
  typename PVMma::SharedStorage main_loop;
  typename PVEpilogue::SharedStorage epilogue;
};

constexpr int kSplitK = 2;

struct PVParams {
  int rows_pad;  // GEMM M
  int width;     // GEMM K: keys in this block
  int k_split;   // first key of the second split
  int n_tiles;   // 256-key tiles in this block
  typename PVIteratorA::Params params_A;
  typename PVIteratorB::Params params_B;
  typename PVOutputIterator::Params params_D;
  Element* scores;  // [head][rows_pad][score_stride]
  int64_t scores_head_stride;
  Element* v;  // V at this block's first key
  int64_t v_head_stride;
  float const* tile_scale;  // [head][tile][rows_pad]
  int64_t stats_head_stride;
  float* out;  // [head][split][rows_pad][256]
  float const* old_scale;  // [head][rows_pad]
  float const* block_scale;
  bool first;
};

static __global__ void __launch_bounds__(PVDefault::kThreadCount, 1) gqa6_pv_kernel(const PVParams p) {
  extern __shared__ int4 gqa6_smem_pv[];
  PVSharedStorage& shared = *reinterpret_cast<PVSharedStorage*>(gqa6_smem_pv);

  const int tile_m = blockIdx.x;
  const int head = blockIdx.z / kSplitK;
  const int split = blockIdx.z % kSplitK;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const int lane_idx = threadIdx.x % 32;

  const int k_begin = split == 0 ? 0 : p.k_split;
  const int k_end = split == 0 ? p.k_split : p.width;

  typename PVMma::IteratorA iterator_A(p.params_A, p.scores + head * p.scores_head_stride, {p.rows_pad, k_end}, thread_idx,
                                       {tile_m * TBShape::kM, k_begin});
  typename PVMma::IteratorB iterator_B(p.params_B, p.v + head * p.v_head_stride, {k_end, kHeadDim}, thread_idx, {k_begin, 0});

  PVTransformA transform_A(p.tile_scale + head * p.stats_head_stride, p.rows_pad, p.n_tiles, k_begin, tile_m * TBShape::kM);
  PVMma mma(shared.main_loop, thread_idx, warp_idx, lane_idx, transform_A);
  typename PVMma::FragmentC accumulators;
  accumulators.clear();
  mma((k_end - k_begin + TBShape::kK - 1) / TBShape::kK, accumulators, iterator_A, iterator_B, accumulators);

  PVOutputOp output_op(typename PVOutputOp::Params(p.old_scale + head * int64_t(p.rows_pad),
                                                   p.block_scale + head * int64_t(p.rows_pad), p.first));
  float* out = p.out + int64_t(blockIdx.z) * p.rows_pad * kHeadDim;
  PVOutputIterator iterator_D(p.params_D, out, {p.rows_pad, kHeadDim}, thread_idx, {tile_m * TBShape::kM, 0});
  PVOutputIterator iterator_C(p.params_D, out, {p.rows_pad, kHeadDim}, thread_idx, {tile_m * TBShape::kM, 0});

  PVEpilogue epilogue(shared.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

// ---- pack Q and merge ----

// Q (F32, [token][head][256] through its strides) -> FP16 [kv head][token*6 + h][256]; pad rows are zero.
static __global__ void gqa6_pack_q(const char* __restrict__ Q, half* __restrict__ q16, const int rows, const int rows_pad,
                                   const int64_t nb1, const int64_t nb2) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int head = blockIdx.y;
  if (row >= rows_pad) {
    return;
  }
  const int d = threadIdx.x * 8;
  uint4 packed = make_uint4(0, 0, 0, 0);
  if (row < rows) {
    const float* src = (const float*) (Q + (row / kQPerKV) * nb1 + (head * kQPerKV + row % kQPerKV) * nb2) + d;
    const float4 a = *(const float4*) (src + 0);
    const float4 b = *(const float4*) (src + 4);
    half2* h2 = (half2*) &packed;
    h2[0] = __floats2half2_rn(a.x, a.y);
    h2[1] = __floats2half2_rn(a.z, a.w);
    h2[2] = __floats2half2_rn(b.x, b.y);
    h2[3] = __floats2half2_rn(b.z, b.w);
  }
  *(uint4*) (q16 + (int64_t(head) * rows_pad + row) * kHeadDim + d) = packed;
}

// dst (F32, [token][head][256] through its strides) = (split 0 + split 1) / running sum; a row with no unmasked key gives 0
static __global__ void gqa6_merge(const float* __restrict__ out, const float* __restrict__ run_sum, char* __restrict__ dst,
                                  const int rows, const int rows_pad, const int64_t nb1, const int64_t nb2) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int head = blockIdx.y;
  if (row >= rows) {
    return;
  }
  const int d = threadIdx.x * 4;
  const float sum = run_sum[int64_t(head) * rows_pad + row];
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  const float4 a = *(const float4*) (out + ((int64_t(head) * kSplitK + 0) * rows_pad + row) * kHeadDim + d);
  const float4 b = *(const float4*) (out + ((int64_t(head) * kSplitK + 1) * rows_pad + row) * kHeadDim + d);
  float4 r;
  r.x = (a.x + b.x) * inv;
  r.y = (a.y + b.y) * inv;
  r.z = (a.z + b.z) * inv;
  r.w = (a.w + b.w) * inv;
  *(float4*) ((float*) (dst + (row / kQPerKV) * nb2 + (head * kQPerKV + row % kQPerKV) * nb1) + d) = r;
}

static bool gqa6_aligned(const ggml_tensor* t) {
  if ((uintptr_t) t->data % 16 != 0) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (t->nb[i] % 16 != 0) {
      return false;
    }
  }
  return true;
}

static int gqa6_env(const char* name, int def) {
  const char* e = getenv(name);
  return e ? atoi(e) : def;
}

}  // namespace

bool ggml_cuda_fattn_gqa6_applies(const ggml_tensor* dst) {
  static const bool enabled = gqa6_env("LLAMA_FA_GQA6", 1) != 0;
  static const int min_rows = gqa6_env("LLAMA_FA_GQA6_MIN_ROWS", GQA6_MIN_ROWS);
  static const int min_kv = gqa6_env("LLAMA_FA_GQA6_MIN_KV", GQA6_MIN_KV);
  if (!enabled || dst->op != GGML_OP_FLASH_ATTN_EXT) {
    return false;
  }
  const ggml_tensor* Q = dst->src[0];
  const ggml_tensor* K = dst->src[1];
  const ggml_tensor* V = dst->src[2];
  const ggml_tensor* mask = dst->src[3];
  const ggml_tensor* sinks = dst->src[4];

  float max_bias = 0.0f;
  float logit_softcap = 0.0f;
  memcpy(&max_bias, (const float*) dst->op_params + 1, sizeof(float));
  memcpy(&logit_softcap, (const float*) dst->op_params + 2, sizeof(float));
  if (!mask || mask->type != GGML_TYPE_F16 || max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr ||
      ggml_get_op_params_i32(dst, 4) != 0) {
    return false;
  }
  if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
    return false;
  }
  if (Q->ne[0] != kHeadDim || K->ne[0] != kHeadDim || V->ne[0] != kHeadDim || K->ne[2] == 0 || Q->ne[2] != kQPerKV * K->ne[2] ||
      V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1] || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
    return false;
  }
  if (Q->ne[1] < min_rows || K->ne[1] < min_kv) {
    return false;
  }
  // every key block is a whole number of 256-key tiles; the mask covers every key and token
  if (K->ne[1] % kTileKeys != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) {
    return false;
  }
  // contiguous rows, and 16-byte loads of Q, K, V, the mask and dst
  if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || mask->nb[0] != sizeof(half) ||
      dst->nb[0] != sizeof(float)) {
    return false;
  }
  if (!gqa6_aligned(Q) || !gqa6_aligned(K) || !gqa6_aligned(V) || !gqa6_aligned(mask) || !gqa6_aligned(dst)) {
    return false;
  }
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true)) {
    GGML_LOG_WARN("fattn: 6-head prompt attention (LLAMA_FA_GQA6): taken, rows %lld, keys %lld, min rows %d, min keys %d\n",
                  (long long) Q->ne[1], (long long) K->ne[1], min_rows, min_kv);
  }
  return true;
}

// Nb for a chunk of rows_pad packed rows: the largest multiple of 256 keeping the scores (heads x rows_pad x Nb x 2 bytes)
// within 128 MiB, at least 256 (and no more keys than the call has)
static int gqa6_block_n(const int n_heads, const int rows_pad, const int n_kv) {
  const size_t score_budget = size_t(128) << 20;
  int block_n = int(std::min<size_t>(score_budget / (size_t(n_heads) * rows_pad * sizeof(half)), size_t(INT32_MAX / 2)));
  block_n = std::max(block_n / kTileKeys * kTileKeys, kTileKeys);
  return std::min(block_n, n_kv);
}

void ggml_cuda_flash_attn_ext_gqa6(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
  const ggml_tensor* Q = dst->src[0];
  const ggml_tensor* K = dst->src[1];
  const ggml_tensor* V = dst->src[2];
  const ggml_tensor* mask = dst->src[3];

  float scale = 1.0f;
  memcpy(&scale, (const float*) dst->op_params + 0, sizeof(float));

  const int id = ggml_cuda_get_device();
  cudaStream_t stream = ctx.stream();
  ggml_cuda_pool& pool = ctx.pool();

  const int n_heads = K->ne[2];
  const int n_kv = K->ne[1];
  const int n_tokens = Q->ne[1];

  // the call's tokens in chunks of at most GQA6_ROW_CHUNK, in order on the stream; 0 is one chunk
  static const int row_chunk = gqa6_env("LLAMA_FA_GQA6_ROW_CHUNK", GQA6_ROW_CHUNK);
  const int chunk_tokens = row_chunk > 0 ? std::min(row_chunk, n_tokens) : n_tokens;

  // the workspace, allocated once for the largest chunk's needs (the full chunk and the last, shorter one) and reused
  size_t q16_size = 0;
  size_t scores_size = 0;
  size_t stats_size = 0;
  for (const int t : {chunk_tokens, n_tokens % chunk_tokens}) {
    if (t == 0) {
      continue;
    }
    const int rows_pad = GGML_PAD(t * kQPerKV, TBShape::kM);
    const int block_n = gqa6_block_n(n_heads, rows_pad, n_kv);
    q16_size = std::max(q16_size, size_t(n_heads) * rows_pad * kHeadDim);
    scores_size = std::max(scores_size, size_t(n_heads) * rows_pad * block_n);
    stats_size = std::max(stats_size, 3 * size_t(n_heads) * (block_n / kTileKeys) * rows_pad + 4 * size_t(n_heads) * rows_pad);
  }
  ggml_cuda_pool_alloc<half> q16(pool, q16_size);
  ggml_cuda_pool_alloc<half> scores(pool, scores_size);
  ggml_cuda_pool_alloc<float> stats(pool, stats_size);
  ggml_cuda_pool_alloc<float> out(pool, kSplitK * q16_size);

  const int qk_smem = int(sizeof(QKSharedStorage));
  const int pv_smem = int(sizeof(PVSharedStorage));
  static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
  if (!smem_set[id]) {
    CUDA_CHECK(cudaFuncSetAttribute(gqa6_qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qk_smem));
    CUDA_CHECK(cudaFuncSetAttribute(gqa6_pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pv_smem));
    smem_set[id] = true;
  }

  const int64_t k_row = K->nb[1] / sizeof(half);
  const int64_t v_row = V->nb[1] / sizeof(half);

  for (int t0 = 0; t0 < n_tokens; t0 += chunk_tokens) {
    const int rows = std::min(chunk_tokens, n_tokens - t0) * kQPerKV;
    const int rows_pad = GGML_PAD(rows, TBShape::kM);
    const int block_n = gqa6_block_n(n_heads, rows_pad, n_kv);
    const int max_tiles = block_n / kTileKeys;

    const int64_t stats_head_stride = int64_t(max_tiles) * rows_pad;

    float* tile_max = stats.get();
    float* tile_sum = tile_max + n_heads * stats_head_stride;
    float* tile_scale = tile_sum + n_heads * stats_head_stride;
    float* run_max = tile_scale + n_heads * stats_head_stride;
    float* run_sum = run_max + size_t(n_heads) * rows_pad;
    float* old_scale = run_sum + size_t(n_heads) * rows_pad;
    float* block_scale = old_scale + size_t(n_heads) * rows_pad;

    {
      const dim3 block(32, 8);
      const dim3 grid((rows_pad + block.y - 1) / block.y, n_heads);
      gqa6_pack_q<<<grid, block, 0, stream>>>((const char*) Q->data + t0 * Q->nb[1], q16.get(), rows, rows_pad, Q->nb[1],
                                              Q->nb[2]);
      CUDA_CHECK(cudaGetLastError());
    }

    QKParams qk = {};
    qk.rows_pad = rows_pad;
    qk.params_A = typename QKMma::IteratorA::Params(cutlass::layout::RowMajor(kHeadDim));
    qk.params_B = typename QKMma::IteratorB::Params(cutlass::layout::ColumnMajor(int(k_row)));
    qk.params_D = typename QKVisitor::OutputTileIterator::Params(cutlass::layout::RowMajor(block_n));
    qk.q = (Element*) q16.get();
    qk.k_head_stride = K->nb[2] / sizeof(half);
    qk.scores = (Element*) scores.get();
    qk.scores_head_stride = int64_t(rows_pad) * block_n;
    qk.tile_max = tile_max;
    qk.tile_sum = tile_sum;
    qk.stats_head_stride = stats_head_stride;
    qk.visitor.scale = scale;
    qk.visitor.mask_stride = mask->nb[1] / sizeof(half);
    qk.visitor.rows = rows;

    PVParams pv = {};
    pv.rows_pad = rows_pad;
    pv.params_A = typename PVIteratorA::Params(PVLayout(block_n));
    pv.params_B = typename PVIteratorB::Params(PVLayout(int(v_row)));
    pv.params_D = typename PVOutputIterator::Params(PVLayout(kHeadDim));
    pv.scores = (Element*) scores.get();
    pv.scores_head_stride = int64_t(rows_pad) * block_n;
    pv.v_head_stride = V->nb[2] / sizeof(half);
    pv.tile_scale = tile_scale;
    pv.stats_head_stride = stats_head_stride;
    pv.out = out.get();
    pv.old_scale = old_scale;
    pv.block_scale = block_scale;

    // this chunk's mask rows
    const half* mask_rows = (const half*) ((const char*) mask->data + t0 * mask->nb[1]);

    for (int begin = 0; begin < n_kv; begin += block_n) {
      const int width = std::min(block_n, n_kv - begin);
      const int n_tiles = width / kTileKeys;

      qk.width = width;
      qk.k = (Element*) ((char*) K->data + begin * K->nb[1]);
      qk.visitor.mask = mask_rows + begin;
      gqa6_qk_kernel<<<dim3(rows_pad / TBShape::kM, n_tiles, n_heads), QKDefault::kThreadCount, qk_smem, stream>>>(qk);
      CUDA_CHECK(cudaGetLastError());

      gqa6_block_update<<<dim3((rows_pad + 255) / 256, n_heads), 256, 0, stream>>>(
          tile_max, tile_sum, tile_scale, run_max, run_sum, old_scale, block_scale, rows_pad, n_tiles, stats_head_stride,
          begin == 0);
      CUDA_CHECK(cudaGetLastError());

      pv.width = width;
      pv.k_split = GGML_PAD(width / 2, TBShape::kK);
      pv.n_tiles = n_tiles;
      pv.v = (Element*) ((char*) V->data + begin * V->nb[1]);
      pv.first = begin == 0;
      gqa6_pv_kernel<<<dim3(rows_pad / TBShape::kM, 1, n_heads * kSplitK), PVDefault::kThreadCount, pv_smem, stream>>>(pv);
      CUDA_CHECK(cudaGetLastError());
    }

    {
      const dim3 block(kHeadDim / 4, 4);
      const dim3 grid((rows + block.y - 1) / block.y, n_heads);
      gqa6_merge<<<grid, block, 0, stream>>>(out.get(), run_sum, (char*) dst->data + t0 * dst->nb[2], rows, rows_pad,
                                             dst->nb[1], dst->nb[2]);
      CUDA_CHECK(cudaGetLastError());
    }
  }
}
