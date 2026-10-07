#pragma once

// IQ decode for the MMVQ kernels (LLAMA_MMVQ_IQ_DECODE, default on).
//
// The vec-dot functions in vecdotq.cuh negate the bytes of each grid entry with __vcmpne4/__vsub4, which
// Volta emulates with about ten integer instructions per four weights, and these kernels are issue-bound on
// that integer work. Here the signs are applied as
//     sum_i s_i*g_i*u_i  =  dp4a(g, u) - 2*dp4a(g & m, u),     m = 0xFF in the bytes whose sign is negative,
// with m built by one multiply that moves each sign bit to the top bit of its byte and one byte permute that
// replicates it. This is exact: every grid byte is in [1, 127], so the negated byte of the original is -g_i
// as an int8, and the integer sums are equal. Everything after the integer sum (scales, float products) is
// unchanged, so the result is bit-identical to the vecdotq.cuh functions.
//
// The grouped expert kernel (mul_mat_vec_q_moe) also keeps the grid in shared memory: each of its blocks
// decodes 2 rows for every token, so the copy is amortized and the gathers no longer go through the L1.

#include "common.cuh"
#include "vecdotq.cuh"

// byte j of the result is 0xFF when bit j of s is set, for bits 0-3 (lo) and 4-7 (hi) of s
static __device__ __forceinline__ uint32_t mmvq_iq_replicate_msb(const uint32_t x) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return __vcmpne4(x & 0x80808080, 0); // not used: mmvq_iq_has_fast is false there
#else
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(r) : "r"(x)); // each byte = its top bit replicated
    return r;
#endif
}
static __device__ __forceinline__ uint2 mmvq_iq_mask8(const uint32_t s) {
    // the multiplies place bit j (resp. 4+j) at bit 8j+7 and every other copy at a distinct lower bit, so there are no carries
    return make_uint2(mmvq_iq_replicate_msb((s & 0x0F) * 0x10204080u), mmvq_iq_replicate_msb((s & 0xF0) * 0x01020408u));
}
// the 7 stored sign bits plus their parity bit, as unpack_ksigns
static __device__ __forceinline__ uint2 mmvq_iq_mask7(const uint8_t v) {
    return mmvq_iq_mask8(v ^ (__popc(v) & 1) << 7);
}

// the codebook of one type
template <ggml_type type> struct mmvq_iq_grid {
    static constexpr int n_grid = 0;
    using grid_t = uint32_t;
};
template <> struct mmvq_iq_grid<GGML_TYPE_IQ2_XXS> {
    static constexpr int n_grid = 256;  using grid_t = uint64_t;
    static __device__ const grid_t * grid() { return iq2xxs_grid; }
};
template <> struct mmvq_iq_grid<GGML_TYPE_IQ2_XS> {
    static constexpr int n_grid = 512;  using grid_t = uint64_t;
    static __device__ const grid_t * grid() { return iq2xs_grid; }
};
template <> struct mmvq_iq_grid<GGML_TYPE_IQ2_S> {
    static constexpr int n_grid = 1024; using grid_t = uint64_t;
    static __device__ const grid_t * grid() { return iq2s_grid; }
};
template <> struct mmvq_iq_grid<GGML_TYPE_IQ3_XXS> {
    static constexpr int n_grid = 256;  using grid_t = uint32_t;
    static __device__ const grid_t * grid() { return iq3xxs_grid; }
};
template <> struct mmvq_iq_grid<GGML_TYPE_IQ3_S> {
    static constexpr int n_grid = 512;  using grid_t = uint32_t;
    static __device__ const grid_t * grid() { return iq3s_grid; }
};

// CUDA only: the sign masks use PTX prmt; HIP and MUSA keep the vecdotq.cuh decode.
static constexpr __host__ __device__ bool mmvq_iq_has_fast(ggml_type type) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(type);
    return false;
#else
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
#endif
}

// Copies the grid into shared memory with every thread of the block, then synchronizes the block.
template <ggml_type type>
static __device__ __forceinline__ void mmvq_iq_load_grid(typename mmvq_iq_grid<type>::grid_t * grid) {
    using T = mmvq_iq_grid<type>;
    const int nthreads = blockDim.x*blockDim.y;
    const int tid      = threadIdx.y*blockDim.x + threadIdx.x;
    const typename T::grid_t * src = T::grid();
    for (int i = tid; i < T::n_grid; i += nthreads) {
        grid[i] = src[i];
    }
    __syncthreads();
}

// sp += g.u over 8 bytes, sn += (g & m).u over 8 bytes
static __device__ __forceinline__ void mmvq_iq_dot8(const uint2 g, const uint2 m, const int u0, const int u1, int & sp, int & sn) {
    sp = ggml_cuda_dp4a((int) g.x, u0, sp);
    sp = ggml_cuda_dp4a((int) g.y, u1, sp);
    sn = ggml_cuda_dp4a((int) (g.x & m.x), u0, sn);
    sn = ggml_cuda_dp4a((int) (g.y & m.y), u1, sn);
}

static __device__ __forceinline__ uint2 mmvq_iq_u64(const uint64_t v) {
    return make_uint2((uint32_t) v, (uint32_t) (v >> 32));
}

// Same contract as vec_dot_iq2_xxs_q8_1.
static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint64_t * __restrict__ grid) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sp = 0;
    int sn = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 g = mmvq_iq_u64(grid[aux8[k0/2]]);
        const uint2 m = mmvq_iq_mask7(aux32 >> (7 * k0 / 2));
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        mmvq_iq_dot8(g, m, u0, u1, sp, sn);
    }
    int sumi = sp - 2*sn;

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// Same contract as vec_dot_iq2_xs_q8_1.
static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint64_t * __restrict__ grid) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sp0 = 0;
    int sn0 = 0;
    int sp1 = 0;
    int sn1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 g = mmvq_iq_u64(grid[q2[l0/2] & 0x1FF]);
        const uint2 m = mmvq_iq_mask7(q2[l0/2] >> 9);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) {
            mmvq_iq_dot8(g, m, u0, u1, sp0, sn0);
        } else {
            mmvq_iq_dot8(g, m, u0, u1, sp1, sn1);
        }
    }
    const int sumi0 = sp0 - 2*sn0;
    const int sumi1 = sp1 - 2*sn1;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// Same contract as vec_dot_iq2_s_q8_1.
static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint64_t * __restrict__ grid) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sp0 = 0;
    int sn0 = 0;
    int sp1 = 0;
    int sn1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 g = mmvq_iq_u64(grid[qs[l0/2] | ((qh << (8-l0)) & 0x300)]);
        const uint2 m = mmvq_iq_mask8(signs_packed_8[l0/2]);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) {
            mmvq_iq_dot8(g, m, u0, u1, sp0, sn0);
        } else {
            mmvq_iq_dot8(g, m, u0, u1, sp1, sn1);
        }
    }
    const int sumi0 = sp0 - 2*sn0;
    const int sumi1 = sp1 - 2*sn1;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// Same contract as vec_dot_iq3_xxs_q8_1.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint32_t * __restrict__ grid) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sp = 0;
    int sn = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 g = make_uint2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint2 m = mmvq_iq_mask7(aux32 >> (7*l0/2));
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        mmvq_iq_dot8(g, m, u0, u1, sp, sn);
    }
    int sumi = sp - 2*sn;

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// Same contract as vec_dot_iq3_s_q8_1.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint32_t * __restrict__ grid) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sp = 0;
    int sn = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 g = make_uint2(
            grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const uint2 m = mmvq_iq_mask8(signs_packed_8[l0/2]);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        mmvq_iq_dot8(g, m, u0, u1, sp, sn);
    }
    int sumi = sp - 2*sn;

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_iq_q8_1_fast(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const typename mmvq_iq_grid<type>::grid_t * __restrict__ grid) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1_fast(vbq, bq8_1, kbx, iqs, grid);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        return vec_dot_iq2_xs_q8_1_fast(vbq, bq8_1, kbx, iqs, grid);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return vec_dot_iq2_s_q8_1_fast(vbq, bq8_1, kbx, iqs, grid);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return vec_dot_iq3_xxs_q8_1_fast(vbq, bq8_1, kbx, iqs, grid);
    } else {
        static_assert(type == GGML_TYPE_IQ3_S, "no fast IQ decode for this type");
        return vec_dot_iq3_s_q8_1_fast(vbq, bq8_1, kbx, iqs, grid);
    }
}
