#pragma once

// The prepared input of the tensor-core weight products (mmvq-qpn.cu): the ONE definition of
// its layout and its arithmetic. qpn_prep_kernel calls it, and so does every kernel that writes a product's input
// directly. Write every byte through the functions below, so a layout change here reaches every writer.
//
// For T tokens (1..8) and K = 256*nsb columns, per 256-column slice sb and token t:
//   xh : fp16 of x * 2^-sh, 8 consecutive columns per 16-byte fragment, in [sb][g = 0..31][t][8] order
//        (column sb*256 + g*8 + i at xh[((sb*32 + g)*T + t)*8 + i])
//   xsc: 2^sh, the scale that undoes it; 0 if the slice holds a non-finite value (xh is then 0)
//   xs : (Q4_K/Q5_K consumers) fp16 of (the sum of the slice's 32 fp16 values of sub-block j)/32, in [sb][t][j = 0..7]
// sh = max(e - 15, -100) for the slice's max |x| = m*2^e, m in [0.5, 1), so max |x*2^-sh| is in [2^14, 2^15); sh = 0 for
// an all-zero or non-finite slice. The 32-value sum is taken in this order: each 8-column group adds its values in
// column order from 0.0f, then groups (4j, 4j + 1) and (4j + 2, 4j + 3) are added, then those two sums.

#include "common.cuh"

#include <cfloat>

// the slice's range scale, from its max |x| and whether all its values are finite (max and and are order-free)
struct qpn_prep_scale {
    float inv;  // 2^-sh, multiplies x
    float undo; // xsc: 2^sh, or 0 if not finite
    bool  fin;
};

static __device__ __forceinline__ qpn_prep_scale qpn_prep_scale_make(const float amax, const bool fin) {
    int e = 0;
    frexpf(amax, &e);
    const int sh = (!fin || amax == 0.0f) ? 0 : max(e - 15, -100);
    qpn_prep_scale s;
    s.inv  = __int_as_float((127 - sh) << 23); // exact
    s.undo = fin ? __int_as_float((127 + sh) << 23) : 0.0f;
    s.fin  = fin;
    return s;
}

// one value's contribution to the slice's (amax, fin)
static __device__ __forceinline__ void qpn_prep_range_add(const float v, float & amax, bool & fin) {
    fin  = fin && fabsf(v) <= FLT_MAX;
    amax = fmaxf(amax, fabsf(v));
}

// one 8-column group: its fp16 fragment and the sum of its fp16 values, in column order
static __device__ __forceinline__ uint4 qpn_prep_group(const float * v, const qpn_prep_scale & s, float & sum8) {
    half h[8];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        h[i] = s.fin ? __float2half_rn(v[i] * s.inv) : __float2half(0.0f);
        acc += __half2float(h[i]);
    }
    sum8 = acc;
    return *(const uint4 *) h;
}

// the per-32 value of sub-block j from its 4 groups' sums (groups 4j .. 4j + 3)
static __device__ __forceinline__ half qpn_prep_sub(const float s0, const float s1, const float s2, const float s3) {
    return __float2half_rn(((s0 + s1) + (s2 + s3)) * (1.0f/32.0f));
}

// where each piece goes (xh as uint4 fragments)
static __device__ __forceinline__ void qpn_prep_store_xh(half * xh, const int sb, const int g, const int t, const int T, const uint4 f) {
    *(uint4 *) (xh + (((int64_t) sb*32 + g)*T + t)*8) = f;
}
static __device__ __forceinline__ void qpn_prep_store_xs(half * xs, const int sb, const int j, const int t, const int T, const half v) {
    xs[((int64_t) sb*T + t)*8 + j] = v;
}
static __device__ __forceinline__ void qpn_prep_store_xsc(float * xsc, const int sb, const int t, const int T, const float v) {
    xsc[(int64_t) sb*T + t] = v;
}

// The whole slice (sb, t) by one warp: lane = group g holds columns sb*256 + 8*lane .. + 7 in v. xs may be nullptr.
static __device__ __forceinline__ void qpn_prep_slice_warp(const float * v, const int sb, const int t, const int T,
        half * xh, half * xs, float * xsc) {
    const int lane = threadIdx.x % WARP_SIZE;
    float amax = 0.0f;
    bool  fin  = true;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qpn_prep_range_add(v[i], amax, fin);
    }
    fin  = __all_sync(0xFFFFFFFF, fin);
    amax = warp_reduce_max(amax);
    const qpn_prep_scale s = qpn_prep_scale_make(amax, fin);

    float sum8;
    qpn_prep_store_xh(xh, sb, lane, t, T, qpn_prep_group(v, s, sum8));
    if (xs != nullptr) {
        // lane 4j gets (s[4j] + s[4j+1]) + (s[4j+2] + s[4j+3]), which is qpn_prep_sub's order
        const float s01 = sum8 + __shfl_xor_sync(0xFFFFFFFF, sum8, 1);
        const float s23 = __shfl_xor_sync(0xFFFFFFFF, s01, 2);
        if (lane % 4 == 0) {
            qpn_prep_store_xs(xs, sb, lane/4, t, T, __float2half_rn((s01 + s23) * (1.0f/32.0f)));
        }
    }
    if (lane == 0) {
        qpn_prep_store_xsc(xsc, sb, t, T, s.undo);
    }
}
