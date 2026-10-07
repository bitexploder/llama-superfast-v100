// MUL_MAT_ID's routed expert products on Volta's FP16 tensor cores, for Flash-Next's seven expert formats.
//
// A port of Strata-V100 (https://github.com/jmnargi/Strata-V100 at a4b679b): the kernel is expert_kernel_wmma of
// src/prefill/moe_fp16tc.cu, the weight decoders are load_unit<T> / convert<T> of src/prefill/moe_fused_iq.cu.
//
//   MIT License
//
//   Copyright (c) 2026 Niko1221 and the Strata contributors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// The block. 64 routed rows of one expert by 128 output columns, 256 threads (8 warps: 2 row halves x 4 column
// quarters, each a 32 x 32 tile of 2 x 2 WMMA m16n16k16 fragments, FP16 inputs and FP32 accumulators). K advances 64
// values a step: the weights (128 rows x 64) and the activations (64 rows x 64) are decoded to FP16 in shared memory,
// then multiplied. The next step's raw weight and activation bytes are loaded into registers before the current
// step's products, so their latency hides behind the tensor cores.
//
// The weights. Each thread decodes one 32-value sub-block of one output row a step (row tid / 2, sub-block tid & 1):
// load_unit<T> reads its block bytes with 16-bit loads (the GGUF blocks are 2-byte aligned), convert<T> turns them into
// 32 int8 codes q and the sub-block's scales d_w (one per 16 values for IQ2_XS and IQ2_S), and the tile gets the FP16
// product q * d_w (q exact in FP16, d_w rounded to FP16, one rounding of the product). Q2_0 keeps moe_fp16tc.cu's own
// decode, (code - 1) * d in half2. The i-quant codebooks are copied to shared memory at the block's start. Strata's
// sign step (__vcmpne4 / __vsub4, emulated on Volta) is replaced by a byte spread and an add: every codebook byte is in
// 1..127, so (g ^ 0xFF) + 1 = -g never carries into the next byte.
//
// The activations are MMQ's own: the compact, expert-sorted q8_1 buffer (block_q8_1_mmq with the D4 layout, 4 float
// scales and 128 int8 codes, laid out [k / 128][row]), quantized once by ggml_cuda_mul_mat_q for all seven types. As
// in Strata's dequant_act, each code becomes exact in FP16 by a byte permute and is multiplied by its scale rounded to
// FP16. Rows past the expert's last are zero and never read. Outputs are not bit-identical to MMQ's int8 dot products.
//
// The grid. MMQ's expert_bounds give each expert's rows; a one-block kernel lists the (expert, first row) of every
// 64-row tile in a pool buffer, at most ceil(rows / 64) + min(experts, rows) of them, and the main grid is (output
// columns / 128) x that bound, its unused slots exiting at once. No host sync, no side stream; CUDA-graph safe.
//
// The output. Each warp stages its fragments through shared memory (WMMA does not specify the fragment element map)
// and writes FP32 to dst[ids_dst[row] * stride_dst + column], exactly where MMQ writes.
#include "mmid-fp16tc.cuh"

#include <mma.h>

#include <atomic>
#include <climits>
#include <cstdlib>

// the smallest token count (ne12) that takes the route; LLAMA_MMID_FP16TC_MIN_TOKENS overrides it
#define MMID_FP16TC_MIN_TOKENS 1024

namespace {

constexpr int BM    = 64;    // routed rows per block (2 warp row halves of 32)
constexpr int BN    = 128;   // output columns per block (4 warp column quarters of 32)
constexpr int KS    = 64;    // K per step
constexpr int NT    = 256;   // threads
constexpr int AWMMA = 80;    // activation tile leading dimension (halves, a multiple of 8 for WMMA)
constexpr int WWMMA = 72;    // weight tile leading dimension (halves)
constexpr int ACTB  = 144;   // block_q8_1_mmq (mmq.cuh): 4 float scales + 128 int8

static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_nl) == 18 &&
              sizeof(block_q2_0) == 18, "the block layouts this file decodes");

// block bytes, values per block, codebook bytes in shared memory
static __host__ __device__ constexpr int tc_block_bytes(ggml_type t) {
    return t == GGML_TYPE_IQ2_XXS ? 66 : t == GGML_TYPE_IQ2_XS ? 74 : t == GGML_TYPE_IQ2_S ? 82 :
           t == GGML_TYPE_IQ3_XXS ? 98 : t == GGML_TYPE_IQ3_S ? 110 : 18;
}
static __host__ __device__ constexpr int tc_block_values(ggml_type t) {
    return t == GGML_TYPE_IQ4_NL ? 32 : t == GGML_TYPE_Q2_0 ? 64 : 256;
}
static __host__ __device__ constexpr int tc_grid_bytes(ggml_type t) {
    return t == GGML_TYPE_IQ2_XXS ? 256 * 8 : t == GGML_TYPE_IQ2_XS ? 512 * 8 : t == GGML_TYPE_IQ2_S ? 1024 * 8 :
           t == GGML_TYPE_IQ3_XXS ? 256 * 4 : t == GGML_TYPE_IQ3_S ? 512 * 4 : 0;
}

__device__ __forceinline__ half2 h2_from_u32(const uint32_t u) {
    half2 h;
    memcpy(&h, &u, 4);
    return h;
}
__device__ __forceinline__ uint32_t u32_from_h2(const half2 h) {
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

// ---- Strata's load stage (moe_fused_iq.cu): a 32-value sub-block's bytes into registers, then int8 codes and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(const uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything); the sign byte
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return v ^ (p << 7);
}
// 8 codebook bytes (gx: values 0-3, gy: 4-7, each byte in 1..127) with sign byte s (bit i: value i negative), as int8.
// The sign bits spread to the bytes' low bits (s * 0x00204081: four non-overlapping copies), then (g ^ 0xFF) + 1 = -g.
__device__ __forceinline__ void signed8(const uint32_t gx, const uint32_t gy, const uint32_t s, uint32_t & qx, uint32_t & qy) {
    const uint32_t lx = ((s & 0x0F) * 0x00204081u) & 0x01010101u;
    const uint32_t ly = (((s >> 4) & 0x0F) * 0x00204081u) & 0x01010101u;
    qx = (gx ^ (lx * 0xFFu)) + lx;
    qy = (gy ^ (ly * 0xFFu)) + ly;
}
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in lo, the high ones' in hi
__device__ __forceinline__ void table16(const uint32_t q4, const uint32_t (&t)[4], uint32_t & lo, uint32_t & hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
template <ggml_type T> __device__ __forceinline__ void load_unit(const uint8_t * bp, const int ib, uint32_t (&w)[5]) {
    if constexpr (T == GGML_TYPE_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == GGML_TYPE_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == GGML_TYPE_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == GGML_TYPE_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == GGML_TYPE_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31). Not used for Q2_0.
template <ggml_type T>
__device__ __forceinline__ void convert(const uint32_t (&w)[5], const uint8_t * grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float & s0, float & s1) {
    if constexpr (T == GGML_TYPE_IQ2_XXS) {
        const uint2 * g = (const uint2 *) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == GGML_TYPE_IQ2_XS) {
        const uint2 * g = (const uint2 *) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == GGML_TYPE_IQ2_S) {
        const uint2 * g = (const uint2 *) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, (w[1] >> (8 * l)) & 255, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == GGML_TYPE_IQ3_XXS) {
        const uint32_t * g = (const uint32_t *) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == GGML_TYPE_IQ3_S) {
        const uint32_t * g = (const uint32_t *) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    (w[2] >> (8 * l)) & 255, q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = half_at(w[4]);
    } else {
        static_assert(T == GGML_TYPE_IQ4_NL, "Q2_0 has its own decode");
    }
}

template <ggml_type T> __device__ __forceinline__ const void * grid_src() {
    if constexpr (T == GGML_TYPE_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == GGML_TYPE_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == GGML_TYPE_IQ2_S) return iq2s_grid;
    else if constexpr (T == GGML_TYPE_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == GGML_TYPE_IQ3_S) return iq3s_grid;
    else return nullptr;
}

// 4 int8 codes (one word) times the half2 scale s2: 2 half2. Sign-flip every byte, then PRMT each pair into {0x64, byte}
// halves = 1024 + (byte ^ 128); subtracting 1152 leaves the signed code exactly (Strata's dequant_act).
__device__ __forceinline__ void i8x4_to_h2(const uint32_t q, const half2 s2, uint32_t & lo, uint32_t & hi) {
    const half2 c1152 = h2_from_u32(0x64806480u);
    const uint32_t v = q ^ 0x80808080u;
    lo = u32_from_h2(__hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), s2));
    hi = u32_from_h2(__hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), s2));
}

// One thread's 32 weights of a step into the tile (32 halves at `o`, 16-byte aligned).
template <ggml_type T>
__device__ __forceinline__ void put_weights(const uint32_t (&w)[5], const uint8_t * grid, const uint32_t (&kv)[4], half * o) {
    uint4 * o4 = (uint4 *) o;
    if constexpr (T == GGML_TYPE_Q2_0) {
        // moe_fp16tc.cu's dequant_weights: pair k is codes 2k (bits 4k) and 2k+1 (bits 4k+2); the bits are 1024 + q in
        // each half, and 1024 + q - 1025 = q - 1
        const half2 d2 = __half2half2(__ushort_as_half((unsigned short) w[2]));
        const half2 k1025 = h2_from_u32(0x64016401u);
        uint32_t h[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t t = (w[k >> 3] >> (4 * (k & 7))) & 0xFu;
            const uint32_t p = (t & 3u) | ((t & 0xCu) << 14);
            h[k] = u32_from_h2(__hmul2(__hsub2(h2_from_u32(0x64006400u | p), k1025), d2));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) o4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
    } else {
        uint32_t q[8];
        float s0, s1;
        convert<T>(w, grid, kv, q, s0, s1);
        const half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(s1);
        uint32_t h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) i8x4_to_h2(q[i], i < 4 ? sa : sb, h[2 * i], h[2 * i + 1]);
#pragma unroll
        for (int i = 0; i < 4; ++i) o4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
    }
}

// The (expert, first row) of every 64-row tile, expert-major; the rest of the `max_tiles` slots get expert -1.
static __global__ void __launch_bounds__(NT, 1)
mmid_fp16tc_tiles(const int32_t * __restrict__ bounds, const int n_experts, int2 * __restrict__ tiles, const int max_tiles) {
    __shared__ int warp_sum[NT / WARP_SIZE];
    __shared__ int carry;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    if (tid == 0) {
        carry = 0;
    }
    __syncthreads();
    for (int base = 0; base < n_experts; base += NT) {
        const int z = base + tid;
        const int lo = z < n_experts ? bounds[z] : 0;
        const int nt = z < n_experts ? (bounds[z + 1] - lo + BM - 1) / BM : 0;
        int x = nt;   // inclusive scan in the warp
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFF, x, o);
            x += lane >= o ? y : 0;
        }
        if (lane == WARP_SIZE - 1) {
            warp_sum[warp] = x;
        }
        __syncthreads();
        int before = carry;
        for (int w = 0; w < warp; ++w) {
            before += warp_sum[w];
        }
        before += x - nt;
        for (int r = 0; r < nt && before + r < max_tiles; ++r) {
            tiles[before + r] = make_int2(z, lo + r * BM);
        }
        __syncthreads();
        if (tid == 0) {
            int total = 0;
            for (int w = 0; w < NT / WARP_SIZE; ++w) {
                total += warp_sum[w];
            }
            carry += total;
        }
        __syncthreads();
    }
    for (int i = carry + tid; i < max_tiles; i += NT) {
        tiles[i] = make_int2(-1, 0);
    }
}

template <ggml_type T>
static __global__ void __launch_bounds__(NT, 2)
mmid_fp16tc_kernel(const char * __restrict__ x, const int64_t nb01, const int64_t nb02,
                   const char * __restrict__ act, const int64_t act_rows,
                   const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds, const int2 * __restrict__ tiles,
                   float * __restrict__ dst, const int64_t stride_dst, const int k_reduce) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using namespace nvcuda;
    constexpr int BS  = tc_block_bytes(T);
    constexpr int SPB = tc_block_values(T) / 32;   // 32-value sub-blocks per block
    constexpr int GB  = tc_grid_bytes(T);

    const int2 tl = tiles[blockIdx.y];
    const int z = tl.x;
    if (z < 0) {
        return;
    }
    const int row0  = tl.y;
    const int local = min(bounds[z + 1] - row0, BM);
    const int out_base = blockIdx.x * BN;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int m_base = (warp >> 2) * 32, n_base = (warp & 3) * 32;

    __shared__ __align__(32) union {
        struct { half a[BM * AWMMA]; half w[BN * WWMMA]; } ab;
        float c[NT / WARP_SIZE][16 * 16];   // per warp, one fragment at a time
    } sm;
    __shared__ __align__(16) uint32_t grid_s[GB > 0 ? GB / 4 : 1];
    __shared__ int ids_s[BM];

    if (tid < BM) {
        ids_s[tid] = tid < local ? ids_dst[row0 + tid] : 0;
    }
    if constexpr (GB > 0) {
        const uint32_t * gsrc = (const uint32_t *) grid_src<T>();
        for (int i = tid; i < GB / 4; i += NT) {
            grid_s[i] = gsrc[i];
        }
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }

    // this thread's weight unit: output row out_base + wcol, sub-block wsub of each step
    const int wcol = tid >> 1, wsub = tid & 1;
    const uint8_t * wrow = (const uint8_t *) x + (int64_t) z * nb02 + (int64_t) (out_base + wcol) * nb01;
    // this thread's activation unit: tile row am, values 16 apart (16 * apart) of each step
    const int am = tid >> 2, apart = tid & 3;
    const bool a_on = am < local;
    const uint8_t * arow = (const uint8_t *) act + (int64_t) (row0 + am) * ACTB;
    const int64_t akb = act_rows * ACTB;   // bytes per 128 values of K

    uint32_t wraw[5] = {0, 0, 0, 0, 0};
    uint4 araw = make_uint4(0, 0, 0, 0);
    float ad = 0.0f;
    auto load = [&](const int k0) {
        const int sb = k0 / 32 + wsub;
        load_unit<T>(wrow + (int64_t) (sb / SPB) * BS, sb % SPB, wraw);
        if (a_on) {
            const uint8_t * b = arow + (int64_t) (k0 / 128) * akb;
            const int off = (k0 % 128) + apart * 16;
            ad   = ((const float *) b)[off / 32];
            araw = *(const uint4 *) (b + 16 + off);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[r][c], 0.0f);

    load(0);
    for (int k0 = 0; k0 < k_reduce; k0 += KS) {
        __syncthreads();   // the previous step's products are done with the tiles (first step: ids_s and the codebook)
        put_weights<T>(wraw, (const uint8_t *) grid_s, kv, sm.ab.w + wcol * WWMMA + wsub * 32);
        {
            const half2 d2 = __float2half2_rn(ad);
            uint32_t h[8];
            i8x4_to_h2(araw.x, d2, h[0], h[1]);
            i8x4_to_h2(araw.y, d2, h[2], h[3]);
            i8x4_to_h2(araw.z, d2, h[4], h[5]);
            i8x4_to_h2(araw.w, d2, h[6], h[7]);
            uint4 * o4 = (uint4 *) (sm.ab.a + am * AWMMA + apart * 16);
            o4[0] = make_uint4(h[0], h[1], h[2], h[3]);
            o4[1] = make_uint4(h[4], h[5], h[6], h[7]);
        }
        __syncthreads();
        if (k0 + KS < k_reduce) {
            load(k0 + KS);
        }
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
#pragma unroll
            for (int r = 0; r < 2; ++r)
                wmma::load_matrix_sync(af[r], sm.ab.a + (m_base + 16 * r) * AWMMA + k16 * 16, AWMMA);
#pragma unroll
            for (int c = 0; c < 2; ++c)
                wmma::load_matrix_sync(bf[c], sm.ab.w + (n_base + 16 * c) * WWMMA + k16 * 16, WWMMA);
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
    }
    __syncthreads();   // the tiles are dead: their storage stages the accumulators

    float * stage = sm.c[warp];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            wmma::store_matrix_sync(stage, acc[r][c], 16, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int i = lane; i < 16 * 16; i += WARP_SIZE) {
                const int lr = m_base + 16 * r + i / 16;
                if (lr < local) {
                    dst[(int64_t) ids_s[lr] * stride_dst + out_base + n_base + 16 * c + i % 16] = stage[i];
                }
            }
            __syncwarp();
        }
    }
#else
    GGML_UNUSED_VARS(x, nb01, nb02, act, act_rows, ids_dst, bounds, tiles, dst, stride_dst, k_reduce);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
}

template <ggml_type T>
static void launch_type(const dim3 grid, const ggml_tensor * src0, const void * act, const int64_t act_rows,
                        const int32_t * ids_dst, const int32_t * bounds, const int2 * tiles, float * dst,
                        const int64_t stride_dst, cudaStream_t stream) {
    mmid_fp16tc_kernel<T><<<grid, NT, 0, stream>>>((const char *) src0->data, src0->nb[1], src0->nb[2],
        (const char *) act, act_rows, ids_dst, bounds, tiles, dst, stride_dst, (int) src0->ne[0]);
}

} // namespace

bool ggml_cuda_mmid_fp16tc(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const int64_t n_tokens,
        const void * src1_q8_1, const int64_t act_rows, const int32_t * ids_dst, const int32_t * expert_bounds,
        float * dst, const int64_t stride_dst, cudaStream_t stream) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_MMID_FP16TC");
        return e == nullptr || atoi(e) != 0;
    }();
    static const int min_tokens = [] {
        const char * e = getenv("LLAMA_MMID_FP16TC_MIN_TOKENS");
        return e ? atoi(e) : MMID_FP16TC_MIN_TOKENS;
    }();
    if (!enabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_TURING ||
            ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q2_0:
            break;
        default:
            return false;
    }
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2];
    if (ne00 % KS != 0 || ne01 % BN != 0 || n_tokens < min_tokens) {
        return false;
    }
    // weight rows are read with 16-bit loads
    if (src0->nb[1] % 2 != 0 || src0->nb[2] % 2 != 0 || (uintptr_t) src0->data % 2 != 0) {
        return false;
    }
    const int64_t max_tiles = (act_rows + BM - 1) / BM + std::min(ne02, act_rows);
    if (max_tiles > 65535 || ne00 > INT_MAX) {
        return false;
    }

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("mmid: experts on FP16 tensor cores (LLAMA_MMID_FP16TC): taken, type %s, tokens %lld, min tokens %d\n",
                      ggml_type_name(src0->type), (long long) n_tokens, min_tokens);
    }

    ggml_cuda_pool_alloc<int2> tiles(ctx.pool(), max_tiles);
    mmid_fp16tc_tiles<<<1, NT, 0, stream>>>(expert_bounds, (int) ne02, tiles.get(), (int) max_tiles);

    const dim3 grid((unsigned) (ne01 / BN), (unsigned) max_tiles, 1);
    switch (src0->type) {
        case GGML_TYPE_IQ2_XXS: launch_type<GGML_TYPE_IQ2_XXS>(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        case GGML_TYPE_IQ2_XS:  launch_type<GGML_TYPE_IQ2_XS >(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        case GGML_TYPE_IQ2_S:   launch_type<GGML_TYPE_IQ2_S  >(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        case GGML_TYPE_IQ3_XXS: launch_type<GGML_TYPE_IQ3_XXS>(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        case GGML_TYPE_IQ3_S:   launch_type<GGML_TYPE_IQ3_S  >(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        case GGML_TYPE_IQ4_NL:  launch_type<GGML_TYPE_IQ4_NL >(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
        default:                launch_type<GGML_TYPE_Q2_0   >(grid, src0, src1_q8_1, act_rows, ids_dst, expert_bounds, tiles.get(), dst, stride_dst, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
