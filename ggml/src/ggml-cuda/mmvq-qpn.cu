// Volta tensor-core products for the dense K-quants (Q4_K, Q5_K, Q6_K) at 1 to 8 tokens, on weights repacked into
// mma fragment order once at load (toggle LLAMA_MMVQ_QPN).
//
// The design follows NInfer's QPN kernels (github.com/... ninfer-v100, Apache-2.0: ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh,
// which credits the "v100-skinny" project): tokens are the 8-row M side of mma.sync.m8n8k4 (zero-padded), each quadpair
// owns 8 output rows, a warp owns 32, the inner dimension is split across the warps of a block (split-K) with a
// shared-memory reduction only at the end, and the weights are read as one contiguous 16-byte chunk per lane
// (512 bytes per warp) with no shared-memory staging, because they were put in fragment order at load. No NInfer code
// is copied: its kernels decode NVFP4/FP8, these decode K-quant codes.
//
// Layout. A tensor of N rows and K = 256*nsb columns becomes N/32 row tiles; tile t is a contiguous run of nsb
// tile-superblocks (32 rows x 256 columns, 32*sizeof(block) bytes, so the size is unchanged and the repack is in place).
// A tile-superblock is a list of 512-byte chunks; in each chunk lane l's 16 bytes belong to row l:
//   Q4_K (9 chunks)  : [d, dmin, scales[12]] verbatim, then one chunk per 32-column sub-block j: 32 4-bit codes
//   Q5_K (11 chunks) : the same header, 2 chunks of high-bit words (word j: sub-block j's 32 fifth bits), 8 code chunks
//   Q6_K (13 chunks + 64 bytes): 8 code chunks (low 4 bits), 4 chunks of high-bit words (2 bits per code, one word per
//                      16 codes), the 16 int8 scales verbatim, then the 32 rows' d, stored so rows r and r + 2 are adjacent
// A code word holds 8 codes of consecutive columns k0..k0+7: slot s at nibble (s even ? s/2 : 4 + s/2). Masking
// (w << 4), w, (w >> 4), (w >> 8) with 0x00F000F0 and or-ing 0x6400 puts slots (2x, 2x+1) into a half2 as exactly
// 1024 + 16*q; the high bits for pair x sit in the high-bit word so that one rotate and one mask put them at bits 8
// (and 9) of each half. One HFMA2 then gives sc*q exactly for Q4_K/Q5_K (sc/16 and -64*sc) and sc*(q - 32) with a
// single rounding for Q6_K (sc/16 and -96*sc). The superblock scale d, Q4_K/Q5_K's mins (dmin * sum_j m_j * sum_j x,
// from per-32 activation sums through one more mma) and the activations' power-of-two range scale are applied to the
// fp32 accumulators once per superblock.
//
// IQ4_XS and IQ4_NL hold 4-bit indices into the 16-entry int8 codebook kvalues_iq4nl. Their blocks (136
// and 18 bytes) are not 16-byte aligned, so they too are only read repacked; a tile-superblock keeps its size:
//   IQ4_XS (8 chunks + 256 bytes): one chunk per sub-block j, then each row's 8-byte GGUF header (d, scales_h,
//                      scales_l) verbatim, in row order
//   IQ4_NL (9 chunks): the same 8 code chunks, then one chunk of each row's 8 fp16 block scales d_j
// Here a code word holds the codes of columns k0..k0+7 in nibbles 0..7. PRMT looks up 4 codes at a time: its low 3 bits
// pick one of 8 bytes of kv + 128 from each codebook half, its bit 3 which half (a third PRMT), and a fourth puts two
// bytes under 0x64, exactly 1024 + 128 + kv. One HFMA2 then gives kv*(ls - 32) with a single rounding for IQ4_XS
// (ls - 32 and -1152*(ls - 32), both exact; d in fp32 per superblock), and one HSUB2 and one HMUL2 give kv*d_j for
// IQ4_NL, rounded once exactly as ggml's fp16 dequantization rounds it.
//
// Q3_K and Q8_0, also in place and the same size:
//   Q3_K (6 chunks + 448 bytes): 4 chunks of the codes' low 2 bits, 2 chunks of their high bits (hmask), then each row's
//                      12 scale bytes verbatim (row r at 12r, so 4-byte aligned), then the 32 rows' d as Q6_K stores them
//                      (rows r and r + 2 adjacent). Low-bit word w (16 columns 16w..16w+15, one 16-column sub-block)
//                      holds pair e = 4f + i (fragment f of the word, columns 2i, 2i+1 of it) at bits 2e + p + {0,1} and
//                      16 + 2e + p + {0,1} (mod 32), p = w & 1; high-bit word v (32 columns, low-bit words 2v, 2v+1) holds
//                      that pair's high bits at 2e + p + 2 and 16 + 2e + p + 2 (mod 32). One LOP3 selects the pair's 4 low
//                      bits from w and its 2 high bits from v, one rotate puts the 3-bit codes c = q2 | h << 2 at bits 2-4
//                      of each half, and under 0x64 they are 1024 + 4c. One HFMA2 with S = (s - 32)/4 and Z = -260*(s - 32)
//                      gives (c - 4)*(s - 32) exactly (an integer of at most 128 in magnitude), and d is applied in fp32
//                      per superblock, so a weight is d*(s - 32)*(c - 4) with no rounding, as ggml's dequantize_row_q3_K
//   Q8_0 (17 chunks)  : 16 code chunks (chunk c: columns 16c..16c+15 of the row, each code stored as q + 128), then one chunk
//                      of each row's 8 fp16 block scales d_j (IQ4_NL's). PRMT puts two codes under 0x64, exactly
//                      1024 + 128 + q; one HSUB2 gives q and one HMUL2 by d_j rounds q*d_j once, as ggml's fp16
//                      dequantization (dequantize_block_q8_0_f16) does
//
// Activations are fp16, prepared once per input (qpn-prep.cuh): each 256-column slice of a token is scaled by a power
// of two so its max |x| is below 2^15 (exact; undone in fp32). A slice holding a non-finite value marks its token, and
// that token's column is then computed in fp32 from the dequantized weights (qpn_fallback), as the generic path would.
//
// Nothing else may read a repacked tensor: it carries GGML_TENSOR_FLAG_BACKEND_LAYOUT, ggml_cuda_mul_mat sends it here
// (or, above GGML_CUDA_QPN_MAX_TOKENS tokens, through ggml_cuda_qpn_to_fp16 to cuBLAS, as a quantized weight already
// goes on this GPU), fusion never takes it, and ggml_cuda_compute_forward aborts if any other op reads it.

#include "mmvq-qpn.cuh"
#include "qpn-prep.cuh"
#include "dequantize.cuh"
#include "mma.cuh"
#include "allreduce-p2p.cuh"

#include <cfloat>
#include <unordered_set>
#include <cstdlib>

using namespace ggml_cuda_mma;

#define QPN_MAX_WARPS  8   // warps per block (split-K x row tiles)
#define QPN_MIN_BLOCKS 2   // so at most 128 registers: 16 warps per SM
#define QPN_MAX_TILES  16384 // row tiles a product split across blocks may have (the counters)
#define QPN_CHUNK      512 // bytes per chunk: 32 lanes x 16
#define QPN_MAGIC      0x64006400u

template <ggml_type type> struct qpn_t;
template <> struct qpn_t<GGML_TYPE_Q4_K> {
    typedef block_q4_K block;
    static constexpr int  NCH  = 9;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = true;
    static constexpr int  bit  = GGML_CUDA_QPN_Q4_K;
};
template <> struct qpn_t<GGML_TYPE_Q5_K> {
    typedef block_q5_K block;
    static constexpr int  NCH  = 11;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = true;
    static constexpr int  bit  = GGML_CUDA_QPN_Q5_K;
};
template <> struct qpn_t<GGML_TYPE_Q6_K> {
    typedef block_q6_K block;
    static constexpr int  NCH  = 13;
    static constexpr int  TAIL = 64;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q6_K;
};
template <> struct qpn_t<GGML_TYPE_IQ4_XS> {
    typedef block_iq4_xs block;
    static constexpr int  NCH  = 8;
    static constexpr int  TAIL = 256;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_IQ4_XS;
};
// IQ4_NL: the 8 blocks of a 256-column span of one row
struct qpn_iq4_nl_sb { block_iq4_nl b[QK_K/QK4_NL]; };
template <> struct qpn_t<GGML_TYPE_IQ4_NL> {
    typedef qpn_iq4_nl_sb block;
    static constexpr int  NCH  = 9;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_IQ4_NL;
};
template <> struct qpn_t<GGML_TYPE_Q3_K> {
    typedef block_q3_K block;
    static constexpr int  NCH  = 6;
    static constexpr int  TAIL = 448;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q3_K;
};
// Q8_0: the 8 blocks of a 256-column span of one row
struct qpn_q8_0_sb { block_q8_0 b[QK_K/QK8_0]; };
template <> struct qpn_t<GGML_TYPE_Q8_0> {
    typedef qpn_q8_0_sb block;
    static constexpr int  NCH  = 17;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q8_0;
};
template <ggml_type type> static constexpr __host__ __device__ int qpn_tsb() { return 32*(int) sizeof(typename qpn_t<type>::block); }
static_assert(qpn_tsb<GGML_TYPE_Q4_K>()   ==  9*QPN_CHUNK,       "Q4_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q5_K>()   == 11*QPN_CHUNK,       "Q5_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q6_K>()   == 13*QPN_CHUNK + 64,  "Q6_K layout");
static_assert(qpn_tsb<GGML_TYPE_IQ4_XS>() ==  8*QPN_CHUNK + 256, "IQ4_XS layout");
static_assert(qpn_tsb<GGML_TYPE_IQ4_NL>() ==  9*QPN_CHUNK,       "IQ4_NL layout");
static_assert(qpn_tsb<GGML_TYPE_Q3_K>()   ==  6*QPN_CHUNK + 448, "Q3_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q8_0>()   == 17*QPN_CHUNK,       "Q8_0 layout");
template <ggml_type type> static constexpr __host__ __device__ bool qpn_iq4() { return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL; }
// the types which run only the streaming loop (qpn_stream_loop), at every width
template <ggml_type type> static constexpr __host__ __device__ bool qpn_new() { return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q8_0; }

// ------------------------------------------------------------------------------------------------------------------
// layout: where each code and high bit goes (used by the repack, the inverse and the fp32 fallback)

// nibble of a code word that holds slot s (column k0 + s)
static __host__ __device__ __forceinline__ int qpn_nib(const int s) { return (s & 1) ? 4 + s/2 : s/2; }
// Q5_K: the bit of sub-block j's high-bit word that holds the fifth bit of word q, slot s
static __host__ __device__ __forceinline__ int qpn_hb5(const int q, const int s) { return (8 + 16*(s & 1) + 4*(s/2) + q) & 31; }
// Q6_K: the lower of the two bits (in high-bit word 2c + q/2) that hold bits 4-5 of chunk c's word q, slot s
static __host__ __device__ __forceinline__ int qpn_hb6(const int q, const int s) { return (8 + 16*(s & 1) + 4*(s/2) + 2*(q & 1)) & 31; }
// Q6_K: position of row r's d in the tail (rows r and r + 2 adjacent, r & 2 == 0)
static __host__ __device__ __forceinline__ int qpn_dpos(const int r) { return (r & ~3) | ((r & 1) << 1) | ((r >> 1) & 1); }

// code of column k (0..255) in a GGUF block
static __device__ __forceinline__ int qpn_code(const block_q4_K * b, const int k) {
    return (b->qs[32*(k/64) + k % 32] >> (4*((k % 64)/32))) & 0xF;
}
static __device__ __forceinline__ int qpn_code(const block_q5_K * b, const int k) {
    return ((b->qs[32*(k/64) + k % 32] >> (4*((k % 64)/32))) & 0xF) | (((b->qh[k % 32] >> (k/32)) & 1) << 4);
}
static __device__ __forceinline__ int qpn_code(const block_q6_K * b, const int k) {
    const int ip = k/128, g = (k % 128)/32, l = k % 32;
    const int lo = (b->ql[64*ip + l + 32*(g & 1)] >> (4*(g/2))) & 0xF;
    const int hi = (b->qh[32*ip + l] >> (2*g)) & 3;
    return lo | (hi << 4);
}
static __device__ __forceinline__ void qpn_set_code(block_q4_K * b, const int k, const int q) {
    uint8_t & v = b->qs[32*(k/64) + k % 32];
    v |= q << (4*((k % 64)/32));
}
static __device__ __forceinline__ void qpn_set_code(block_q5_K * b, const int k, const int q) {
    uint8_t & v = b->qs[32*(k/64) + k % 32];
    v |= (q & 0xF) << (4*((k % 64)/32));
    b->qh[k % 32] |= (q >> 4) << (k/32);
}
static __device__ __forceinline__ void qpn_set_code(block_q6_K * b, const int k, const int q) {
    const int ip = k/128, g = (k % 128)/32, l = k % 32;
    b->ql[64*ip + l + 32*(g & 1)] |= (q & 0xF) << (4*(g/2));
    b->qh[32*ip + l]              |= (q >> 4) << (2*g);
}
// IQ4_XS, IQ4_NL: in each 32-column block, byte l of qs holds column l (low nibble) and column l + 16 (high nibble)
static __device__ __forceinline__ int qpn_code(const block_iq4_xs * b, const int k) {
    return (b->qs[16*(k/32) + k % 16] >> (4*((k % 32)/16))) & 0xF;
}
static __device__ __forceinline__ int qpn_code(const qpn_iq4_nl_sb * b, const int k) {
    return (b->b[k/32].qs[k % 16] >> (4*((k % 32)/16))) & 0xF;
}
static __device__ __forceinline__ void qpn_set_code(block_iq4_xs * b, const int k, const int q) {
    b->qs[16*(k/32) + k % 16] |= q << (4*((k % 32)/16));
}
static __device__ __forceinline__ void qpn_set_code(qpn_iq4_nl_sb * b, const int k, const int q) {
    b->b[k/32].qs[k % 16] |= q << (4*((k % 32)/16));
}
// Q3_K: the 3-bit code c = q2 | h << 2 of column k, the value being d*(s - 32)*(c - 4)
static __device__ __forceinline__ int qpn_code(const block_q3_K * b, const int k) {
    const int q2 = (b->qs[32*(k/128) + k % 32] >> (2*((k % 128)/32))) & 3;
    const int h  = (b->hmask[k % 32] >> (k/32)) & 1;
    return q2 | (h << 2);
}
static __device__ __forceinline__ void qpn_set_code(block_q3_K * b, const int k, const int c) {
    b->qs[32*(k/128) + k % 32] |= (c & 3) << (2*((k % 128)/32));
    b->hmask[k % 32]           |= (c >> 2) << (k/32);
}
// Q3_K: the bit of low-bit word k/16 that holds bit 0 of column k's code (bit 1 is the next one, mod 32), and the bit of
// high-bit word k/32 that holds its high bit
static __host__ __device__ __forceinline__ int qpn_q3_lpos(const int k) {
    const int s = k % 8, e = 4*((k/8) & 1) + s/2;
    return (2*e + ((k/16) & 1) + 16*(s & 1)) & 31;
}
static __host__ __device__ __forceinline__ int qpn_q3_hpos(const int k) { return (qpn_q3_lpos(k) + 2) & 31; }
// Q3_K: the 16 six-bit sub-block scales from the 12 GGUF scale bytes a0..a2, one per byte (sub-block j: byte j % 4 of
// word j/4), as dequantize_row_q3_K unpacks them
static __device__ __forceinline__ void qpn_q3_scales(const uint32_t a0, const uint32_t a1, const uint32_t a2, uint32_t * s) {
    s[0] = ( a0       & 0x0F0F0F0Fu) | (( a2       & 0x03030303u) << 4);
    s[1] = ( a1       & 0x0F0F0F0Fu) | (((a2 >> 2) & 0x03030303u) << 4);
    s[2] = ((a0 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 4) & 0x03030303u) << 4);
    s[3] = ((a1 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 6) & 0x03030303u) << 4);
}

// one row's GGUF block -> its 16 bytes of every chunk (w[ch*4 + i]) and its tail entry tl (Q6_K: d in tl.x; IQ4_XS:
// the 8-byte header in tl.x, tl.y; Q3_K: the 12 scale bytes in tl.x..tl.z and d in tl.w)
template <ggml_type type>
static __device__ void qpn_pack_row(const typename qpn_t<type>::block * b, uint32_t * w, uint4 & tl) {
    constexpr int NCH = qpn_t<type>::NCH;
    for (int i = 0; i < NCH*4; ++i) {
        w[i] = 0;
    }
    tl = make_uint4(0, 0, 0, 0);
    if constexpr (type == GGML_TYPE_Q3_K) {
        for (int k = 0; k < 256; ++k) {
            const int c = qpn_code(b, k), p = qpn_q3_lpos(k);
            w[k/16]      |= (uint32_t) (c & 1)        << p;
            w[k/16]      |= (uint32_t) ((c >> 1) & 1) << ((p + 1) & 31);
            w[16 + k/32] |= (uint32_t) (c >> 2)       << qpn_q3_hpos(k);
        }
        memcpy(&tl, b->scales, 12);
        memcpy(&tl.w, &b->d, 2);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        for (int k = 0; k < 256; ++k) {
            w[(k/16)*4 + (k % 16)/4] |= (uint32_t) (uint8_t) (b->b[k/32].qs[k % 32] + 128) << (8*(k % 4));
        }
        for (int j = 0; j < QK_K/QK8_0; ++j) {
            memcpy((char *) (w + 16*4) + 2*j, &b->b[j].d, 2);
        }
    } else if constexpr (qpn_iq4<type>()) {
        for (int k = 0; k < 256; ++k) {
            w[(k/32)*4 + (k % 32)/8] |= (uint32_t) qpn_code(b, k) << (4*(k % 8));
        }
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            memcpy(&tl, b, 8); // d, scales_h, scales_l
        } else {
            for (int j = 0; j < QK_K/QK4_NL; ++j) {
                memcpy((char *) (w + 8*4) + 2*j, &b->b[j].d, 2);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        memcpy(w, b, 16); // d, dmin, scales
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        for (int k = 0; k < 256; ++k) {
            const int j = k/32, q = (k % 32)/8, s = k % 8, c = qpn_code(b, k);
            w[(ch0 + j)*4 + q] |= (uint32_t) (c & 0xF) << (4*qpn_nib(s));
            if constexpr (type == GGML_TYPE_Q5_K) {
                w[4 + j] |= (uint32_t) (c >> 4) << qpn_hb5(q, s);
            }
        }
    } else {
        for (int k = 0; k < 256; ++k) {
            const int ch = k/32, q = (k % 32)/8, s = k % 8, c = qpn_code(b, k);
            w[ch*4 + q]               |= (uint32_t) (c & 0xF) << (4*qpn_nib(s));
            w[8*4 + 2*ch + q/2]       |= (uint32_t) (c >> 4)  << qpn_hb6(q, s);
        }
        memcpy(w + 12*4, b->scales, 16);
        memcpy(&tl.x, &b->d, 2);
    }
}

// the inverse: one row's chunk words and tail entry -> its GGUF block
template <ggml_type type>
static __device__ void qpn_unpack_row(const uint32_t * w, const uint4 tl, typename qpn_t<type>::block * b) {
    memset(b, 0, sizeof(*b));
    if constexpr (type == GGML_TYPE_Q3_K) {
        for (int k = 0; k < 256; ++k) {
            const int p = qpn_q3_lpos(k);
            const int c = ((w[k/16] >> p) & 1) | (((w[k/16] >> ((p + 1) & 31)) & 1) << 1) | (((w[16 + k/32] >> qpn_q3_hpos(k)) & 1) << 2);
            qpn_set_code(b, k, c);
        }
        memcpy(b->scales, &tl, 12);
        memcpy(&b->d, &tl.w, 2);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        for (int k = 0; k < 256; ++k) {
            b->b[k/32].qs[k % 32] = (int8_t) (((w[(k/16)*4 + (k % 16)/4] >> (8*(k % 4))) & 0xFF) - 128);
        }
        for (int j = 0; j < QK_K/QK8_0; ++j) {
            memcpy(&b->b[j].d, (const char *) (w + 16*4) + 2*j, 2);
        }
    } else if constexpr (qpn_iq4<type>()) {
        for (int k = 0; k < 256; ++k) {
            qpn_set_code(b, k, (w[(k/32)*4 + (k % 32)/8] >> (4*(k % 8))) & 0xF);
        }
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            memcpy(b, &tl, 8);
        } else {
            for (int j = 0; j < QK_K/QK4_NL; ++j) {
                memcpy(&b->b[j].d, (const char *) (w + 8*4) + 2*j, 2);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        memcpy(b, w, 16);
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        for (int k = 0; k < 256; ++k) {
            const int j = k/32, q = (k % 32)/8, s = k % 8;
            int c = (w[(ch0 + j)*4 + q] >> (4*qpn_nib(s))) & 0xF;
            if constexpr (type == GGML_TYPE_Q5_K) {
                c |= ((w[4 + j] >> qpn_hb5(q, s)) & 1) << 4;
            }
            qpn_set_code(b, k, c);
        }
    } else {
        for (int k = 0; k < 256; ++k) {
            const int ch = k/32, q = (k % 32)/8, s = k % 8;
            const int c = ((w[ch*4 + q] >> (4*qpn_nib(s))) & 0xF) | (((w[8*4 + 2*ch + q/2] >> qpn_hb6(q, s)) & 3) << 4);
            qpn_set_code(b, k, c);
        }
        memcpy(b->scales, w + 12*4, 16);
        memcpy(&b->d, &tl.x, 2);
    }
}

// where a tile-superblock's tail entry of row r lives (Q6_K: its d, 2 bytes; IQ4_XS: its header, 8 bytes; Q3_K: its 12
// scale bytes at 12r, then its d at 384 + 2*qpn_dpos(r))
template <ggml_type type>
static __device__ __forceinline__ void qpn_store_tail(char * tsb, const int r, const uint4 tl) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        ((uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[qpn_dpos(r)] = (uint16_t) tl.x;
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        ((uint2 *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[r] = make_uint2(tl.x, tl.y);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        uint32_t * s = (uint32_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK) + 3*r;
        s[0] = tl.x; s[1] = tl.y; s[2] = tl.z;
        ((uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK + 384))[qpn_dpos(r)] = (uint16_t) tl.w;
    }
}
template <ggml_type type>
static __device__ __forceinline__ uint4 qpn_load_tail(const char * tsb, const int r) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        return make_uint4(((const uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[qpn_dpos(r)], 0, 0, 0);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const uint2 h = ((const uint2 *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[r];
        return make_uint4(h.x, h.y, 0, 0);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        const uint32_t * s = (const uint32_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK) + 3*r;
        return make_uint4(s[0], s[1], s[2], ((const uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK + 384))[qpn_dpos(r)]);
    }
    return make_uint4(0, 0, 0, 0);
}

// ------------------------------------------------------------------------------------------------------------------
// repack (load time) and the fp16 dequantization of a repacked tensor (prefill)

// one block of 32 threads per tile-superblock; src holds the GGUF rows of tiles [0, ntiles) of this pass
template <ggml_type type>
static __global__ void qpn_repack_kernel(const char * __restrict__ src, char * __restrict__ dst, const int nsb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), BS = sizeof(block);
    __shared__ block blk[32];
    const int lane = threadIdx.x;
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;

    // stage the 32 rows' blocks of this superblock (2-byte units, Q6_K blocks are only 2-byte aligned)
    uint16_t * s16 = (uint16_t *) blk;
    for (int i = lane; i < 32*BS/2; i += 32) {
        const int r = i / (BS/2), o = i % (BS/2);
        s16[i] = ((const uint16_t *) (src + ((int64_t) (tile*32 + r)*nsb + sb)*BS))[o];
    }
    __syncwarp();

    uint32_t w[NCH*4];
    uint4    tl;
    qpn_pack_row<type>(&blk[lane], w, tl);

    char * out = dst + ((int64_t) tile*nsb + sb)*TSB;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        *(uint4 *) (out + ch*QPN_CHUNK + lane*16) = make_uint4(w[ch*4 + 0], w[ch*4 + 1], w[ch*4 + 2], w[ch*4 + 3]);
    }
    qpn_store_tail<type>(out, lane, tl);
}

// row lane's chunk words of one tile-superblock
template <ggml_type type>
static __device__ __forceinline__ void qpn_load_row(const char * tsb, const int lane, uint32_t * w, uint4 & tl) {
    constexpr int NCH = qpn_t<type>::NCH;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        const uint4 v = *(const uint4 *) (tsb + ch*QPN_CHUNK + lane*16);
        w[ch*4 + 0] = v.x; w[ch*4 + 1] = v.y; w[ch*4 + 2] = v.z; w[ch*4 + 3] = v.w;
    }
    tl = qpn_load_tail<type>(tsb, lane);
}

// Rebuilds the GGUF blocks of one tile-superblock in shared memory and dequantizes them with ggml's own functions
// (dequantize.cuh), so the values are those ggml's to_fp16 gives for the original tensor.
template <ggml_type type>
static __global__ void qpn_to_fp16_kernel(const char * __restrict__ W, half * __restrict__ y, const int nsb, const int64_t K) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    __shared__ block blk[32];
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    if (threadIdx.x < 32) {
        uint32_t w[NCH*4];
        uint4    tl;
        qpn_load_row<type>(W + ((int64_t) tile*nsb + sb)*TSB, threadIdx.x, w, tl);
        qpn_unpack_row<type>(w, tl, &blk[threadIdx.x]);
    }
    __syncthreads();
    for (int i = 0; i < 32; ++i) {
        half * yi = y + (int64_t) (tile*32 + i)*K + sb*QK_K;
        if constexpr (type == GGML_TYPE_Q4_K) {
            dequantize_q4_K(blk, i, yi, threadIdx.x); // 32 threads
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            dequantize_q5_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            dequantize_q6_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            dequantize_iq4_xs(blk, i, yi, threadIdx.x); // 32 threads
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            dequantize_q3_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            // 32 threads, 8 columns each; the arithmetic of dequantize_block_q8_0_f16 (convert.cu), ggml's Q8_0 -> fp16
            const block_q8_0 & bq = blk[i].b[threadIdx.x/4];
            const half2 d2 = __half2half2(bq.d);
#pragma unroll
            for (int l = 0; l < 8; l += 2) {
                const int c = 8*(threadIdx.x % 4) + l;
                ((half2 *) yi)[(8*threadIdx.x + l)/2] = __hmul2(make_half2(bq.qs[c], bq.qs[c + 1]), d2);
            }
        } else {
            dequantize_iq4_nl(blk, i, yi, threadIdx.x); // 32 threads; block i*8 .. i*8 + 7
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// the product

static __device__ __forceinline__ half2 qpn_u2h(const uint32_t u) { return *(const half2 *) &u; }

// weights: read once, keep them out of L1 (the activations live there)
static __device__ __forceinline__ uint4 qpn_ldg_w(const char * p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
static __device__ __forceinline__ uint2 qpn_ldg_w2(const char * p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
// the 128-byte line holding p into L2, for a later load (no registers)
static __device__ __forceinline__ void qpn_prefetch_l2(const char * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
static __device__ __forceinline__ uint4 qpn_ldg_x(const half * p) {
    return __ldg((const uint4 *) p);
}
// half2 (b, b) of byte i of w, exact: 0x64 above the byte is 1024 + b
static __device__ __forceinline__ half2 qpn_byte_h2(const uint32_t w, const int i) {
    return __hsub2(qpn_u2h(__byte_perm(w, 0x64, (i) | (4 << 4) | ((i) << 8) | (4 << 12))), qpn_u2h(QPN_MAGIC));
}
// half2 (byte i, byte i + 1) of w, exact
static __device__ __forceinline__ half2 qpn_bytes_h2(const uint32_t w, const int i) {
    return __hsub2(qpn_u2h(__byte_perm(w, 0x64, (i) | (4 << 4) | ((i + 1) << 8) | (4 << 12))), qpn_u2h(QPN_MAGIC));
}

// the A fragment of one code word: sc*q (Q4_K, Q5_K) or sc*(q - 32) (Q6_K) for its 8 columns, from s16 = sc/16 and
// z = -64*sc or -96*sc; hw holds the high bits of this word already rotated by its word offset
// mg: QPN_MAGIC. The streaming loop passes it from a register the compiler cannot fold into an immediate, so that
// "x & mask | magic" is one LOP3 (two register operands, one immediate) and a half2's two ORs are two LOP3 instead of three; same values
template <ggml_type type>
static __device__ __forceinline__ void qpn_decode(tile<32, 4, half2> & A, const uint32_t w, const uint32_t hw, const half2 s16, const half2 z,
        const uint32_t mg = QPN_MAGIC) {
    uint32_t g[4];
    if constexpr (type == GGML_TYPE_Q4_K) {
        g[0] = g[1] = g[2] = g[3] = mg;
    } else {
        constexpr uint32_t hm = type == GGML_TYPE_Q5_K ? 0x01000100u : 0x03000300u;
        g[0] = ( hw                          & hm) | mg;
        g[1] = (__funnelshift_r(hw, hw,  4)  & hm) | mg;
        g[2] = (__funnelshift_r(hw, hw,  8)  & hm) | mg;
        g[3] = (__funnelshift_r(hw, hw, 12)  & hm) | mg;
    }
    A.x[0] = __hfma2(qpn_u2h(((w << 4) & 0x00F000F0u) | g[0]), s16, z);
    A.x[1] = __hfma2(qpn_u2h(( w       & 0x00F000F0u) | g[1]), s16, z);
    A.x[2] = __hfma2(qpn_u2h(((w >> 4) & 0x00F000F0u) | g[2]), s16, z);
    A.x[3] = __hfma2(qpn_u2h(((w >> 8) & 0x00F000F0u) | g[3]), s16, z);
}

// PRMT as the hardware does it: selector bit 3 fills the byte with the selected byte's sign (__byte_perm clears it)
static __device__ __forceinline__ uint32_t qpn_prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// kvalues_iq4nl + 128 as bytes: codes 0-7 (all below 128) and 8-15
#define QPN_IQ4_T0 0x3F2D1801u
#define QPN_IQ4_T1 0x766A5D4Fu
#define QPN_IQ4_T2 0xA6998D81u
#define QPN_IQ4_T3 0xF1D9C5B5u

// the 4 codes of s's low 16 bits -> their kv + 128 as bytes, in order. sm = s without bit 3 of each code, sel = the
// codes' bit 3 at bit 2 over 0x3210 (the low half of the codebook where it is 0, from lo; else from hi)
static __device__ __forceinline__ uint32_t qpn_iq4_lookup(const uint32_t s, const uint32_t sm, const uint32_t sel) {
    const uint32_t lo = qpn_prmt(QPN_IQ4_T0, QPN_IQ4_T1, s); // a code with bit 3 set gets a sign fill here, never taken
    const uint32_t hi = qpn_prmt(QPN_IQ4_T2, QPN_IQ4_T3, sm);
    return qpn_prmt(lo, hi, sel);
}

// the A fragment of one IQ4 code word (columns k0..k0+7 in nibbles 0..7): 1024 + 128 + kv as fp16, exactly, then
// IQ4_XS: kv*s by one HFMA2 with S = s and Z = -1152*s; IQ4_NL: kv*d by HSUB2 and HMUL2 with S = d
template <ggml_type type>
static __device__ __forceinline__ void qpn_decode_iq4(tile<32, 4, half2> & A, const uint32_t w, const half2 S, const half2 Z) {
    const uint32_t wm  = w & 0x77777777u;
    const uint32_t sel = ((w >> 1) & 0x44444444u) | 0x32103210u;
    const uint32_t ra  = qpn_iq4_lookup(w,       wm,       sel);
    const uint32_t rb  = qpn_iq4_lookup(w >> 16, wm >> 16, sel >> 16);
    const uint32_t x[4] = {
        __byte_perm(ra, 0x64646464u, 0x4140), __byte_perm(ra, 0x64646464u, 0x4342),
        __byte_perm(rb, 0x64646464u, 0x4140), __byte_perm(rb, 0x64646464u, 0x4342),
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            A.x[i] = __hfma2(qpn_u2h(x[i]), S, Z);
        } else {
            A.x[i] = __hmul2(__hsub2(qpn_u2h(x[i]), make_half2(1152.0f, 1152.0f)), S);
        }
    }
}

// Partial fp32 dot products, over this warp's superblocks, for the columns whose token held a non-finite value: the
// dequantized weights (the same expressions as dequantize.cuh) times the fp32 activations. lane = row; fb[c] for bad c.
template <ggml_type type>
static __device__ __noinline__ void qpn_fallback(
        const char * __restrict__ Wt, const int sb0, const int sb1, const int S, const unsigned badmask, const int T,
        const float * __restrict__ x, const int64_t stride_col_x, float * fb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    const int lane = threadIdx.x;
    for (int c = 0; c < 16; ++c) { // up to 16 columns
        fb[c] = 0.0f;
    }
    for (int sb = sb0; sb < sb1; sb += S) {
        uint32_t w[NCH*4];
        uint4    tl;
        block b;
        qpn_load_row<type>(Wt + (int64_t) sb*TSB, lane, w, tl);
        qpn_unpack_row<type>(w, tl, &b);
        float v[QK_K];
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            for (int j = 0; j < 8; ++j) {
                const float d = (float) b.d * ((((b.scales_l[j/2] >> 4*(j%2)) & 0xf) | (((b.scales_h >> 2*j) & 3) << 4)) - 32);
                for (int l = 0; l < 32; ++l) {
                    v[32*j + l] = d * kvalues_iq4nl[qpn_code(&b, 32*j + l)];
                }
            }
        } else if constexpr (type == GGML_TYPE_IQ4_NL) {
            for (int k = 0; k < QK_K; ++k) {
                v[k] = (float) b.b[k/32].d * kvalues_iq4nl[qpn_code(&b, k)];
            }
        } else if constexpr (type == GGML_TYPE_Q3_K) { // dequantize_row_q3_K: (d*(s - 32))*(c - 4)
            uint32_t a[3], s[4];
            memcpy(a, b.scales, 12);
            qpn_q3_scales(a[0], a[1], a[2], s);
            const float d = b.d;
            for (int k = 0; k < QK_K; ++k) {
                const int sc = (s[k/64] >> (8*((k/16) % 4))) & 0xFF;
                v[k] = (d * (sc - 32)) * (qpn_code(&b, k) - 4);
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) { // dequantize_row_q8_0: q*d
            for (int k = 0; k < QK_K; ++k) {
                v[k] = b.b[k/32].qs[k % 32] * (float) b.b[k/32].d;
            }
        } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            const float dall = __low2half(b.dm), dmin = __high2half(b.dm);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                get_scale_min_k4(j, b.scales, sc, m);
                const float d1 = dall * sc, m1 = dmin * m;
                for (int l = 0; l < 32; ++l) {
                    v[32*j + l] = d1 * qpn_code(&b, 32*j + l) - m1;
                }
            }
        } else {
            const float d = b.d;
            for (int k = 0; k < QK_K; ++k) {
                v[k] = d * b.scales[k/16] * ((int8_t) qpn_code(&b, k) - 32);
            }
        }
        for (int c = 0; c < T; ++c) {
            if ((badmask >> c) & 1) {
                const float * xc = x + c*stride_col_x + (int64_t) sb*QK_K;
                float acc = 0.0f;
                for (int k = 0; k < QK_K; ++k) {
                    acc += v[k] * xc[k];
                }
                fb[c] += acc;
            }
        }
    }
}

// C elements of a lane that can hold a real token at T tokens: one 8-token B tile (4 or 8), or two at T = 16 (9 to 16
// tokens, the second tile's at 8 .. 15)
template <int T> static constexpr __host__ __device__ int qpn_nl() { return T > 8 ? 16 : T > 4 ? 8 : 4; }

// One superblock of one row tile: buf holds this lane's 16 bytes of every chunk (dp: Q6_K's d of rows r0, r0 + 2 in
// dp.x; IQ4_XS: the lane's row's header),
// xk this superblock's activation fragments of token nB (fragment of column group g at xk[g*T]; shared memory if
// STAGE). Adds the superblock's contribution to the lane's C elements D and marks tokens whose slice is not finite.
// A register ring: as soon as a chunk of buf (or dp) has been read for the last time, the same chunk of
// the next superblock (at Wn; if more) is requested into it, so each load has about one superblock of lead with no
// second buffer. The arithmetic is unchanged, in the same order. Without RING the caller holds a second
// buffer and loads the whole next superblock itself, as before.
#define QPN_NEXT(c) do { if (RING && more) { buf[c] = qpn_ldg_w(Wn + (c)*QPN_CHUNK); } } while (0)
template <ggml_type type> static constexpr __host__ __device__ int qpn_ring_min_t() {
    // measured (microbenchmark on the real tensors, card 1, SM clock 1380 MHz): Q6_K's spills go (23-32
    // LDL in the loop -> 1-6) and every shape is 22-40% faster at T = 3, 4, 8; Q5_K is 9-25% faster at T = 8 but up
    // to 15% slower at T = 3, 4 (ffn down, GDN qkv, GDN out); Q4_K and IQ4_XS are 0-6% slower
    return type == GGML_TYPE_Q6_K ? 1 : type == GGML_TYPE_Q5_K ? 5 : 99;
}
template <ggml_type type, int T, int NL, bool STAGE, bool RING>
static __device__ __forceinline__ void qpn_superblock(
        uint4 * buf, uint2 & dp, const char * Wn, const bool more, const int dofs, const int sb, const uint4 * xk,
        const half * __restrict__ xs, const float * __restrict__ xsc, const int nB, const bool hasB, const int r0,
        const int c0, float * D, unsigned & bad) {
#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;

    // this superblock's column scales (0: the token's slice is not finite)
    float cs[NL/2];
#pragma unroll
    for (int i = 0; i < NL/2; ++i) {
        const int col = c0 + (i & 1) + 4*(i/2);
        cs[i] = col < T ? xsc[sb*T + col] : 1.0f;
        bad |= (col < T && cs[i] == 0.0f) ? 1u << col : 0u;
    }

    tile_C Ds[2];

    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        const uint4 hdr = buf[0];
        uint4 qh[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[0] = buf[1];
            qh[1] = buf[2];
        }
        // 6-bit scales and mins, a byte each (get_scale_min_k4)
        const uint32_t scl[2] = { hdr.y & 0x3F3F3F3Fu, (hdr.w & 0x0F0F0F0Fu) | ((hdr.y >> 2) & 0x30303030u) };
        const uint32_t mnl[2] = { hdr.z & 0x3F3F3F3Fu, ((hdr.w >> 4) & 0x0F0F0F0Fu) | ((hdr.z >> 2) & 0x30303030u) };
        // RING: the rows' d and dmin now, so the header chunk can be requested for the next superblock
        float2 dm0, dm2;
        if constexpr (RING) {
            dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
        }
        QPN_NEXT(0);

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 cw = buf[ch0 + j];
            const half2 sc  = qpn_byte_h2(scl[j/4], j % 4);
            const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
            const half2 z   = __hmul2(sc, make_half2(-64.0f, -64.0f));
            uint32_t H = 0;
            if constexpr (type == GGML_TYPE_Q5_K) {
                const uint4 hq = qh[j/4];
                H = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
            }
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                tile_A A;
                qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, q), s16, z);
                tile_B B; // a lane of a padding token (nB >= T) feeds only output columns that are never written
                *(uint4 *) B.x = STAGE ? xk[(4*j + q)*T] : __ldg(xk + (4*j + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(ch0 + j);
            if constexpr (type == GGML_TYPE_Q5_K) {
                if (j % 4 == 3) {
                    QPN_NEXT(1 + j/4);
                }
            }
        }

        // mins: sum_j m_j * (per-32 sums of the activations)/32 through one more mma
        tile_A Am;
        Am.x[0] = qpn_bytes_h2(mnl[0], 0);
        Am.x[1] = qpn_bytes_h2(mnl[0], 2);
        Am.x[2] = qpn_bytes_h2(mnl[1], 0);
        Am.x[3] = qpn_bytes_h2(mnl[1], 2);
        tile_B Bm;
        *(uint4 *) Bm.x = __ldg((const uint4 *) xs + ((int64_t) sb*T + nB));
        tile_C M;
        mma(M, Am, Bm);

        if constexpr (!RING) {
            dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float2 dm = (l & 2) ? dm2 : dm0;
            const float  c  = cs[(l & 1) + ((l & 4) >> 1)];
            D[l] += c * (dm.x * (Ds[0].x[l] + Ds[1].x[l]) - 32.0f*dm.y * M.x[l]);
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const uint4 scw = buf[12];
        // int8 scales, offset to unsigned so the 0x64 byte trick applies: 1024 + sc + 128
        const uint32_t scu[4] = {scw.x ^ 0x80808080u, scw.y ^ 0x80808080u, scw.z ^ 0x80808080u, scw.w ^ 0x80808080u};
        uint4 hq[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            hq[h] = buf[8 + h];
        }
        float2 d02;
        if constexpr (RING) {
            d02 = __half22float2(qpn_u2h(dp.x));
        }
        QPN_NEXT(12);
        if (RING && more) {
            dp.x = __ldg((const uint32_t *) Wn + dofs);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint4 cw = buf[c];
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
            const uint4    hc     = hq[c/2];
            const uint32_t hcv[2] = {(c & 1) ? hc.z : hc.x, (c & 1) ? hc.w : hc.y};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int   sbi = 2*c + q/2; // 16-column sub-block
                const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                          make_half2(1152.0f, 1152.0f));
                const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                const uint32_t H = hcv[q/2];
                tile_A A;
                qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z);
                tile_B B;
                *(uint4 *) B.x = STAGE ? xk[(4*c + q)*T] : __ldg(xk + (4*c + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(c);
            if (c & 1) {
                QPN_NEXT(8 + c/2);
            }
        }
        if constexpr (!RING) {
            d02 = __half22float2(qpn_u2h(dp.x));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float d = (l & 2) ? d02.y : d02.x;
            const float c = cs[(l & 1) + ((l & 4) >> 1)];
            D[l] += c * d * (Ds[0].x[l] + Ds[1].x[l]);
        }
    } else { // IQ4_XS, IQ4_NL
        // IQ4_XS: the 8 six-bit sub-block scales ls_j of this lane's row as bytes, even j in ls02, odd j in ls13
        // (dp.x = d | scales_h << 16, dp.y = scales_l); IQ4_NL: its 8 fp16 block scales
        uint32_t ls02 = 0, ls13 = 0;
        uint4    dnl  = make_uint4(0, 0, 0, 0);
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            const uint32_t h = __byte_perm(dp.x, 0, 0x3322); // scales_h bytes 0, 0, 1, 1
            ls02 = ( dp.y       & 0x0F0F0F0Fu) | ((h << 4) & 0x00300030u) | ( h       & 0x30003000u);
            ls13 = ((dp.y >> 4) & 0x0F0F0F0Fu) | ((h << 2) & 0x00300030u) | ((h >> 2) & 0x30003000u);
        } else {
            dnl = buf[8];
        }
        float d0 = 1.0f, d2 = 1.0f;
        if constexpr (type == GGML_TYPE_IQ4_XS && RING) {
            d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0)));
            d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0 + 2)));
            if (more) {
                dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wn + dofs));
            }
        } else if constexpr (type == GGML_TYPE_IQ4_NL) {
            QPN_NEXT(8);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 cw = buf[j];
            half2 S, Z;
            if constexpr (type == GGML_TYPE_IQ4_XS) {
                const int b = j/2;
                S = __hsub2(qpn_u2h(__byte_perm(j & 1 ? ls13 : ls02, 0x64, b | (4 << 4) | (b << 8) | (4 << 12))),
                            make_half2(1056.0f, 1056.0f)); // ls_j - 32, exact
                Z = __hmul2(S, make_half2(-1152.0f, -1152.0f));
            } else {
                const uint32_t dw = (j/2) == 0 ? dnl.x : (j/2) == 1 ? dnl.y : (j/2) == 2 ? dnl.z : dnl.w;
                S = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                Z = S;
            }
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                tile_A A;
                qpn_decode_iq4<type>(A, cwv[q], S, Z);
                tile_B B;
                *(uint4 *) B.x = STAGE ? xk[(4*j + q)*T] : __ldg(xk + (4*j + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(j);
        }
        if constexpr (type == GGML_TYPE_IQ4_XS && !RING) {
            d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0)));
            d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0 + 2)));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float c = cs[(l & 1) + ((l & 4) >> 1)];
            if constexpr (type == GGML_TYPE_IQ4_XS) {
                D[l] += c * ((l & 2) ? d2 : d0) * (Ds[0].x[l] + Ds[1].x[l]);
            } else {
                D[l] += c * (Ds[0].x[l] + Ds[1].x[l]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(buf, dp, Wn, more, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// grid = (row tile groups, P), blockDim = (32, R*S). Block (x, p) takes row tiles x*R .. x*R + R - 1 over superblocks
// [nsb*p/P, nsb*(p + 1)/P); its warp y takes tile x*R + y/S and every S-th superblock of that range from y%S on.
// STAGE: the block's activations are copied to shared memory first. The S warps of a tile add up in shared memory;
// with P > 1 the P blocks of a tile leave their partial sums in ws and the last one to finish (a per-stream, per-tile
// counter) adds them in the order p = 0 .. P-1, so the result does not depend on which block finishes last.
// One register buffer: each chunk of the next superblock is requested once the current one has been read
// (inside qpn_superblock, chunk by chunk).
// Streaming: the superblock loop with only what a superblock needs
// throughout (the header or scale chunk, the tail entry) and a lookahead of two code chunks and one high-bit chunk in
// registers (about 24 weight registers instead of 88), so the kernel fits 64-80 registers and 24-32 warps per SM; latency
// is hidden with warps, not with register buffers. The arithmetic is qpn_superblock's, expression for expression and in
// the same order, so at the same split the outputs are bit-identical.
//   Q4_K, Q5_K: header chunk 0, Q5_K's high-bit chunks 1-2 (one per 4 code chunks), codes from chunk 1 or 3
//   Q6_K      : scale chunk 12, d in the tail, high-bit chunks 8-11 (one per 2 code chunks), codes from chunk 0
//   IQ4_XS    : the row's 8-byte header in the tail, codes from chunk 0
//   IQ4_NL    : the d_j chunk 8, codes from chunk 0
//   Q3_K      : the scales and d in the tail, high-bit chunks 4-5 (one per 4 units), low-bit chunks from 0 (one per
//               2 units; so the lookahead of two code chunks is 4 units)
//   Q8_0      : the d_j chunk 16, codes from chunk 0 (two per unit)
// A unit j is the superblock's 32 columns 32j..32j+31 (4 fragments of 8); a type has CPU2 code chunks per 2 units.
template <ggml_type type> struct qpn_stream_t {
    static constexpr int CH0 = type == GGML_TYPE_Q4_K ? 1 : type == GGML_TYPE_Q5_K ? 3 : 0;          // first code chunk
    static constexpr int HDR = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 0 :
                               type == GGML_TYPE_Q6_K ? 12 : type == GGML_TYPE_IQ4_NL ? 8 :
                               type == GGML_TYPE_Q8_0 ? 16 : -1;                                        // held chunk
    static constexpr int NHB = type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q3_K ? 2 : type == GGML_TYPE_Q6_K ? 4 : 0; // high-bit chunks
    static constexpr int HB0 = type == GGML_TYPE_Q5_K ? 1 : type == GGML_TYPE_Q3_K ? 4 : 8;
    static constexpr int JPH = type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q3_K ? 4 : 2;              // units per high-bit chunk
    static constexpr bool TAIL = type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q3_K;
    static constexpr int CPU2 = type == GGML_TYPE_Q3_K ? 1 : type == GGML_TYPE_Q8_0 ? 4 : 2;          // code chunks per 2 units
};

// what the lanes of padding tokens (T <= token < 8) read instead of a real token's fragments:
// zeros, so they no longer depend on real rows; their outputs are never written, and a real token's C column depends
// only on its own B column, so the outputs are unchanged. It is the same one load per lane (a warp-wide LDG.128 either way).
static __device__ uint4 qpn_zero_x[32*16]; // 16 tokens (fragment stride t <= 16)
template <ggml_type type, int T, int NL, bool STAGE, int LA = 2, bool FULL = false>
static __device__ __forceinline__ void qpn_stream_loop(
        const char * Wp, const int64_t step, int sb, const int sbB, const int S, const bool active, const uint4 * xsm, const int sbA, const half * __restrict__ xh,
        const half * __restrict__ xs, const float * __restrict__ xsc, const int nB_, const int r0, const int c0, const int dofs, float * D, unsigned & bad,
        const int t_, const int ts_) {
#if defined(VOLTA_MMA_AVAILABLE)
    // t is the block's token count, ts the token stride of the prepared input (t, or a paired block's whole pass, 9 to 16)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;
    typedef qpn_stream_t<type> st;
    constexpr int CH0 = st::CH0, HDR = st::HDR, NHB = st::NHB, HB0 = st::HB0, JPH = st::JPH;
    // FULL, the pass holds exactly T tokens (t = ts = T, every lane's token real), so every activation address is a constant
    // offset from one base (the second tile's at +8 fragments' tokens) and needs no arithmetic of its own
    static_assert(!FULL || !STAGE, "full: unstaged");
    const int t  = FULL ? T : t_, ts = FULL ? T : ts_;
    const int nB = FULL ? tile_B::get_i(0) : nB_;
    // LA, the lookahead in units: unit j requests code chunk j + LA, and the next superblock's first LA code chunks (and
    // the high-bit chunks its first LA units read) are requested in this superblock's last LA units. LA = 2 is the default;
    // LA = 4 holds twice the weight bytes in flight per warp (only load timing moves: the arithmetic is unchanged)
    static_assert(LA == 2 || (LA == 4 && st::CPU2 == 2), "LA = 4: the types with two code chunks per two units");
    constexpr int HBH = LA == 2 || NHB == 0 ? 1 : (LA + JPH - 1)/JPH; // high-bit chunks requested with the next superblock's head
    // at T = 16 (9 to 16 tokens) two 8-token B tiles share each decoded A fragment, so one weight read serves both;
    // tile b's C elements are D[8*b .. 8*b + 7], summed in the same order as a pass of 8 (each column bit-identical to it)
    constexpr int NBT = T > 8 ? 2 : 1;
    static_assert(NBT == 1 || (!STAGE && NL == 16), "16 tokens: unstaged, 16 C elements a lane");
    // QPN_MAGIC for qpn_decode, as a register: qpn_zero_x is never written, so the OR leaves the value unchanged
    constexpr bool MGR = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
    const uint32_t mg = MGR ? (QPN_MAGIC | __ldg(&qpn_zero_x[0].x)) : QPN_MAGIC;

    uint4 hdrn = make_uint4(0, 0, 0, 0), hbn = make_uint4(0, 0, 0, 0), c0n = make_uint4(0, 0, 0, 0), c1n = make_uint4(0, 0, 0, 0);
    uint4 cxn[LA > 2 ? LA - 2 : 1], hbxn[HBH > 1 ? HBH - 1 : 1]; // LA = 4: the next superblock's code chunks 2, 3 and high-bit chunks 1 ..
    uint2 tln  = make_uint2(0, 0);
    uint4 q3n  = make_uint4(0, 0, 0, 0); // Q3_K: the row's 12 scale bytes and the d of rows r0, r0 + 2
    // this superblock's held chunk, first high-bit chunk, tail entry and first two code chunks, from Wb
    auto head = [&](const char * Wb) {
        if constexpr (HDR >= 0) {
            hdrn = qpn_ldg_w(Wb + HDR*QPN_CHUNK);
        }
        if constexpr (NHB > 0) {
            hbn = qpn_ldg_w(Wb + HB0*QPN_CHUNK);
        }
        if constexpr (type == GGML_TYPE_Q6_K) {
            tln.x = __ldg((const uint32_t *) Wb + dofs);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            tln = qpn_ldg_w2((const char *) ((const uint32_t *) Wb + dofs));
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            // dofs: the u32 index of this lane's scales from Wb; its d pair follows the 32 rows' scales like Q6_K's
            const uint32_t * sp = (const uint32_t *) Wb + dofs;
            q3n.x = __ldg(sp); q3n.y = __ldg(sp + 1); q3n.z = __ldg(sp + 2);
            q3n.w = __ldg(sp - 3*(threadIdx.x % 32) + 96 + (r0 >> 2)*2 + (r0 & 1));
        }
    };
    if (active && sb < sbB) {
        head(Wp);
        c0n = qpn_ldg_w(Wp + CH0*QPN_CHUNK);
        c1n = qpn_ldg_w(Wp + (CH0 + 1)*QPN_CHUNK);
        if constexpr (LA > 2) {
#pragma unroll
            for (int m = 2; m < LA; ++m) {
                cxn[m - 2] = qpn_ldg_w(Wp + (CH0 + m)*QPN_CHUNK);
            }
#pragma unroll
            for (int h = 1; h < HBH; ++h) {
                hbxn[h - 1] = qpn_ldg_w(Wp + (HB0 + h)*QPN_CHUNK);
            }
        }
    }
    for (; active && sb < sbB; sb += S) {
        const bool more = sb + S < sbB;
        const char * Wn = Wp + step;
        const uint4 hdr = hdrn;
        const uint2 tl  = tln;
        const uint4 q3  = q3n;
        uint4 hb[NHB > 0 ? NHB : 1];
        hb[0] = hbn;
        uint4 cc[4*st::CPU2];
        cc[0] = c0n;
        cc[1] = c1n;
        if constexpr (LA > 2) {
#pragma unroll
            for (int m = 2; m < LA; ++m) {
                cc[m] = cxn[m - 2];
            }
#pragma unroll
            for (int h = 1; h < HBH; ++h) {
                hb[h] = hbxn[h - 1];
            }
        }
        // padding-token lanes read qpn_zero_x (staged: the block's shared copy of a real token, as before)
        const int nBr = tile_B::get_i(0);
        const uint4 * xk = STAGE ? xsm + ((sb - sbA)*32*t + nB) : (FULL || nBr < t) ? (const uint4 *) xh + ((int64_t) sb*32*ts + nB) : qpn_zero_x + nBr;
        const uint4 * xm = (const uint4 *) xs + ((int64_t) sb*ts + nB);
        // the second tile's tokens 8 .. t - 1 (lanes of padding tokens read qpn_zero_x; their sums a real token's)
        const uint4 * xk1 = NBT == 1 ? xk : FULL ? xk + 8 : nBr + 8 < t ? (const uint4 *) xh + ((int64_t) sb*32*ts + nBr + 8) : qpn_zero_x + nBr;
        const uint4 * xm1 = FULL ? xm + 8 : (const uint4 *) xs + ((int64_t) sb*ts + min(nBr + 8, t - 1));
// one decoded fragment A (fragment f of the superblock) into accumulator acc of every B tile
#define QPN_MMA2(A, f, acc) do {                                                                   \
            tile_B B;                                                                              \
            *(uint4 *) B.x = STAGE ? xk[(f)*t] : __ldg(xk + (f)*ts);                               \
            mma(Ds[0][acc], A, B);                                                                 \
            if constexpr (NBT == 2) {                                                              \
                tile_B B1;                                                                         \
                *(uint4 *) B1.x = __ldg(xk1 + (f)*ts);                                             \
                mma(Ds[NBT - 1][acc], A, B1);                                                      \
            }                                                                                      \
        } while (0)

        float cs[NL/2];
        if constexpr (NL == 4) {
#pragma unroll
        for (int i = 0; i < NL/2; ++i) {
            const int col = c0 + (i & 1) + 4*(i/2);
            cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
            bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
        }
        }
        tile_C Ds[NBT][2];

        // per-superblock scales: Q4_K/Q5_K 6-bit scales and mins; Q6_K int8 scales + 0x80; IQ4_XS 6-bit ls_j
        uint32_t scl[2] = {0, 0}, mnl[2] = {0, 0}, scu[4] = {0, 0, 0, 0}, ls02 = 0, ls13 = 0;
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            scl[0] = hdr.y & 0x3F3F3F3Fu;
            scl[1] = (hdr.w & 0x0F0F0F0Fu) | ((hdr.y >> 2) & 0x30303030u);
            mnl[0] = hdr.z & 0x3F3F3F3Fu;
            mnl[1] = ((hdr.w >> 4) & 0x0F0F0F0Fu) | ((hdr.z >> 2) & 0x30303030u);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            scu[0] = hdr.x ^ 0x80808080u; scu[1] = hdr.y ^ 0x80808080u; scu[2] = hdr.z ^ 0x80808080u; scu[3] = hdr.w ^ 0x80808080u;
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            const uint32_t h = __byte_perm(tl.x, 0, 0x3322); // scales_h bytes 0, 0, 1, 1
            ls02 = ( tl.y       & 0x0F0F0F0Fu) | ((h << 4) & 0x00300030u) | ( h       & 0x30003000u);
            ls13 = ((tl.y >> 4) & 0x0F0F0F0Fu) | ((h << 2) & 0x00300030u) | ((h >> 2) & 0x30003000u);
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            qpn_q3_scales(q3.x, q3.y, q3.z, scu); // six-bit s_j, sub-block j at byte j % 4 of scu[j/4]
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (st::CPU2 == 2) {
                if (j + LA < 8) {
                    cc[j + LA] = qpn_ldg_w(Wp + (CH0 + j + LA)*QPN_CHUNK);
                }
            } else if constexpr (st::CPU2 == 4) { // Q8_0: the next unit's two code chunks
                if (j + 1 < 8) {
                    cc[2*j + 2] = qpn_ldg_w(Wp + (CH0 + 2*j + 2)*QPN_CHUNK);
                    cc[2*j + 3] = qpn_ldg_w(Wp + (CH0 + 2*j + 3)*QPN_CHUNK);
                }
            } else { // Q3_K: the low-bit chunk of units j + 4, j + 5
                if (j % 2 == 0 && j/2 + 2 < 4) {
                    cc[j/2 + 2] = qpn_ldg_w(Wp + (CH0 + j/2 + 2)*QPN_CHUNK);
                }
            }
            if constexpr (NHB > 0) {
                if ((j + LA) % JPH == 0 && (j + LA)/JPH < NHB && (j + LA)/JPH >= HBH) {
                    hb[(j + LA)/JPH] = qpn_ldg_w(Wp + (HB0 + (j + LA)/JPH)*QPN_CHUNK);
                }
            }
            if constexpr (LA == 2) {
                if (j == 6 && more) {
                    head(Wn);
                }
                if (j == 7 && more) {
                    c0n = qpn_ldg_w(Wn + CH0*QPN_CHUNK);
                    c1n = qpn_ldg_w(Wn + (CH0 + 1)*QPN_CHUNK);
                }
            } else { // the next superblock's chunk m in unit 8 - LA + m, its head with chunk 0
                if (j == 8 - LA && more) {
                    head(Wn);
                    c0n = qpn_ldg_w(Wn + CH0*QPN_CHUNK);
                }
                if (j == 9 - LA && more) {
                    c1n = qpn_ldg_w(Wn + (CH0 + 1)*QPN_CHUNK);
                }
#pragma unroll
                for (int m = 2; m < LA; ++m) {
                    if (j == 8 - LA + m && more) {
                        cxn[m - 2] = qpn_ldg_w(Wn + (CH0 + m)*QPN_CHUNK);
                    }
                }
#pragma unroll
                for (int h = 1; h < HBH; ++h) {
                    if (j == 8 - LA + h*JPH && more) {
                        hbxn[h - 1] = qpn_ldg_w(Wn + (HB0 + h)*QPN_CHUNK);
                    }
                }
            }
            const uint4 cw = cc[st::CPU2 == 2 ? j : st::CPU2 == 4 ? 2*j : j/2];
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
            if constexpr (type == GGML_TYPE_Q3_K) {
                // low-bit words 2j, 2j + 1 (sub-blocks 2j, 2j + 1; parity p = 0, 1), high-bit word j
                const uint4    hq = hb[j/4];
                const uint32_t H  = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const uint32_t L  = cwv[2*(j % 2) + p];
                    const int      sb = 2*j + p;
                    const half2 v = qpn_u2h(__byte_perm(scu[sb/4], 0x64, (sb % 4) | (4 << 4) | ((sb % 4) << 8) | (4 << 12))); // 1024 + s
                    const half2 S2 = __hfma2(v, make_half2(0.25f, 0.25f), make_half2(-264.0f, -264.0f)); // (s - 32)/4, exact
                    const half2 Z  = __hmul2(S2, make_half2(-1040.0f, -1040.0f));                        // -260*(s - 32), exact
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        tile_A A;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int e = 4*f + i;
                            const uint32_t K = __funnelshift_l(0x00030003u, 0x00030003u, 2*e + p); // this pair's low bits
                            const uint32_t C = (L & K) | (H & ~K);
                            const uint32_t x = (__funnelshift_r(C, C, (2*e + p + 30) & 31) & 0x001C001Cu) | QPN_MAGIC; // 1024 + 4c
                            A.x[i] = __hfma2(qpn_u2h(x), S2, Z);
                        }
                        const int q = 2*p + f;
                        QPN_MMA2(A, 4*j + q, q & 1);
                    }
                }
            } else if constexpr (type == GGML_TYPE_Q8_0) {
                // chunks 2j, 2j + 1: fragments 4j .. 4j + 3, two words each; block j's d_j
                const uint32_t dw = (j/2) == 0 ? hdr.x : (j/2) == 1 ? hdr.y : (j/2) == 2 ? hdr.z : hdr.w;
                const half2    S2 = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                const uint4    cw1 = cc[2*j + 1];
                const uint32_t w8[8] = {cw.x, cw.y, cw.z, cw.w, cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t x = __byte_perm(w8[2*q + i/2], 0x64646464u, (i & 1) ? 0x4342 : 0x4140); // 1024 + 128 + q
                        A.x[i] = __hmul2(__hsub2(qpn_u2h(x), make_half2(1152.0f, 1152.0f)), S2);
                    }
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                const half2 sc  = qpn_byte_h2(scl[j/4], j % 4);
                const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                const half2 z   = __hmul2(sc, make_half2(-64.0f, -64.0f));
                uint32_t H = 0;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    const uint4 hq = hb[j/4];
                    H = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
                    qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, q), s16, z, mg);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else if constexpr (type == GGML_TYPE_Q6_K) {
                const uint4    hc     = hb[j/2];
                const uint32_t hcv[2] = {(j & 1) ? hc.z : hc.x, (j & 1) ? hc.w : hc.y};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int   sbi = 2*j + q/2; // 16-column sub-block
                    const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                              make_half2(1152.0f, 1152.0f));
                    const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                    const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                    const uint32_t H = hcv[q/2];
                    tile_A A;
                    qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z, mg);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else {
                half2 S2, Z;
                if constexpr (type == GGML_TYPE_IQ4_XS) {
                    const int b = j/2;
                    S2 = __hsub2(qpn_u2h(__byte_perm(j & 1 ? ls13 : ls02, 0x64, b | (4 << 4) | (b << 8) | (4 << 12))),
                                 make_half2(1056.0f, 1056.0f)); // ls_j - 32, exact
                    Z  = __hmul2(S2, make_half2(-1152.0f, -1152.0f));
                } else {
                    const uint32_t dw = (j/2) == 0 ? hdr.x : (j/2) == 1 ? hdr.y : (j/2) == 2 ? hdr.z : hdr.w;
                    S2 = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                    Z  = S2;
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
                    qpn_decode_iq4<type>(A, cwv[q], S2, Z);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            }
        }

        // 5 to 8 tokens: the range scales are read after the products, so they hold no registers through them
        // (at 80 registers this takes the loop's spills from 2-17 LDL to 0 for Q4_K, IQ4_XS and IQ4_NL, 7 for Q5_K and Q6_K)
        if constexpr (NL >= 8) {
#pragma unroll
            for (int i = 0; i < NL/2; ++i) {
                const int col = c0 + (i & 1) + 4*(i/2);
                cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
                bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
            }
        }
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            tile_A Am;
            Am.x[0] = qpn_bytes_h2(mnl[0], 0);
            Am.x[1] = qpn_bytes_h2(mnl[0], 2);
            Am.x[2] = qpn_bytes_h2(mnl[1], 0);
            Am.x[3] = qpn_bytes_h2(mnl[1], 2);
            tile_C M[NBT];
            {
                tile_B Bm;
                *(uint4 *) Bm.x = __ldg(xm);
                mma(M[0], Am, Bm);
            }
            if constexpr (NBT == 2) {
                tile_B Bm1;
                *(uint4 *) Bm1.x = __ldg(xm1);
                mma(M[NBT - 1], Am, Bm1);
            }
            const float2 dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            const float2 dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float2 dm = (l & 2) ? dm2 : dm0;
                const float  c  = cs[(l & 1) + ((l & 4) >> 1) + ((l & 8) >> 1)];
                D[l] += c * (dm.x * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]) - 32.0f*dm.y * M[l >> 3].x[l & 7]);
            }
        } else if constexpr (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q3_K) {
            const float2 d02 = __half22float2(qpn_u2h(type == GGML_TYPE_Q6_K ? tl.x : q3.w));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float d = (l & 2) ? d02.y : d02.x;
                const float c = cs[(l & 1) + ((l & 4) >> 1) + ((l & 8) >> 1)];
                D[l] += c * d * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            const float d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, tl.x, r0)));
            const float d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, tl.x, r0 + 2)));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float c = cs[(l & 1) + ((l & 4) >> 1) + ((l & 8) >> 1)];
                D[l] += c * ((l & 2) ? d2 : d0) * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        } else {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float c = cs[(l & 1) + ((l & 4) >> 1) + ((l & 8) >> 1)];
                D[l] += c * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        }
        Wp = Wn;
    }
#undef QPN_MMA2
#else
    GGML_UNUSED_VARS(Wp, step, sb, sbB, S, active, xsm, sbA, xh, xs, xsc, nB_, r0, c0, dofs, D, bad, t_, ts_);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// MODE 0: the whole-superblock loop at QPN_MIN_BLOCKS blocks per SM; MODE 2, 3 or 4: qpn_stream_loop, with the launch bounds of
// MODE blocks of 8 warps per SM. The streaming kernels at T = 8 take
// any of 5 to 8 tokens, nt, at run time; every other kernel takes exactly T.
// MODE 5, the streaming loop with a lookahead of 4 units (LA = 4) for the products whose grid leaves registers free (160
// blocks of 4 warps, about 9 warps per SM): blocks of at most 4 warps, 3 per SM, so at most 168 registers (one wave of 216 blocks)
// QPN_FULL added to a streaming mode (2, 3 or 5) at T = 8 or 16: the kernel takes exactly T tokens, so the token count and
// the input's stride are constants and every activation address is a constant offset from one base (qpn_stream_loop's FULL)
#define QPN_FULL 100
static constexpr __host__ __device__ bool qpn_mode_full(const int mode) { return mode >= QPN_FULL; }
static constexpr __host__ __device__ int  qpn_mode_m(const int mode) { return mode % QPN_FULL; }
static constexpr __host__ __device__ int qpn_min_blocks(const int mode) { return mode == 0 ? QPN_MIN_BLOCKS : qpn_mode_m(mode) == 5 ? 3 : qpn_mode_m(mode); }
static constexpr __host__ __device__ int qpn_max_threads(const int mode) { return qpn_mode_m(mode) == 5 ? 4*WARP_SIZE : QPN_MAX_WARPS*WARP_SIZE; }

// the two one-tile members' outputs, strides and stored rows (CLIP), and the member of the calling block, found at the store
// from a fresh read of blockIdx.x (as the fallback reads blockIdx.y), so no choice made before the loop is live across it
struct qpn_clip { float * dst0; int64_t stride_col_dst0; int nrow0; float * dst1; int64_t stride_col_dst1; int nrow1; int nb0; };
static __device__ __forceinline__ void qpn_clip_member(const qpn_clip & c, float * & dst, int64_t & stride_col_dst, int & nrow) {
    unsigned bx;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bx));
    const bool second = (int) bx >= c.nb0;
    dst            = second ? c.dst1 : c.dst0;
    stride_col_dst = second ? c.stride_col_dst1 : c.stride_col_dst0;
    nrow           = second ? c.nrow1 : c.nrow0;
}

// One block's work on one product (a segment): grid = (row tile groups, P), blockDim = (32, R*S); bx is the block's index
// among the product's row tile groups. The caller has zeroed *sbad and staged the activations (STAGE), then synchronized.
// t: the product's token count. It is T (a constant) except in the streaming kernels instantiated at T = 8, which take 5 to 8
// tokens at run time: their activations are laid out for t tokens, and lanes of tokens t .. 7 read qpn_zero_x.
// CLIP (qpn_mul4ab_kernel only): the block is one of two one-tile members (blocks [0, nb0) the first, the rest the second),
// each storing only the first nrow rows of its tile into its own output; dst and stride_col_dst are unused and the store takes the member's
// from clip (qpn_clip_member). Every other instantiation stores all 32 rows to dst, with no test.
template <ggml_type type, int T, bool STAGE, int MODE, bool PUSH = false, bool CLIP = false>
static __device__ __forceinline__ void qpn_mul_block(const int bx,
        const char * __restrict__ W, const int nsb, const int ntiles, const int S, const int P,
        const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
        const float * __restrict__ x, const int64_t stride_col_x, float * __restrict__ dst, const int64_t stride_col_dst,
        float * __restrict__ ws, unsigned * __restrict__ cnt, const uint4 * qpn_xsm, float * red, unsigned & sbad, const int t, const int ts,
        const ggml_cuda_p2p_push push = {}, const qpn_clip clip = {}) {
#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    constexpr int NL  = qpn_nl<T>(); // C elements of a lane that can hold a real token

    const int lane = threadIdx.x, wy = threadIdx.y;
    const int R    = blockDim.y / S;
    const int tile = bx*R + wy/S, sp = wy % S;
    const int p    = blockIdx.y;
    const int sbA  = nsb*p/P, sbB = nsb*(p + 1)/P;
    const bool active = tile < ntiles;
    const int  nB   = min(tile_B::get_i(0), t - 1); // token of this lane's B fragment (a real one for lanes of padding)
    const bool hasB = true;
    const int  r0   = lane & ~2;          // rows of this lane's C: r0, r0 + 2 (tile_C::get_i)
    const int  c0   = lane & 2;           // its columns: c0 + (l & 13) (tile_C::get_j; l & 8 is the second B tile)

    float    D[NL] = {0.0f};
    unsigned bad   = 0;

    const char * Wt = W + (int64_t) tile*nsb*TSB;
    const char * Wp = Wt + (int64_t) (sbA + sp)*TSB + lane*16;
    const int64_t step = (int64_t) S*TSB;
    // u32 index from Wp of Q6_K's d(r0), d(r0 + 2), of the IQ4_XS header of row lane, or of the Q3_K scales of row lane
    const int dofs = type == GGML_TYPE_Q6_K ? NCH*QPN_CHUNK/4 - lane*4 + (r0 >> 2)*2 + (r0 & 1) :
                     type == GGML_TYPE_Q3_K ? NCH*QPN_CHUNK/4 - lane : NCH*QPN_CHUNK/4 - lane*2;

    if constexpr (MODE != 0 || qpn_new<type>()) { // qpn_new types have only the streaming loop
        qpn_stream_loop<type, T, NL, STAGE, qpn_mode_m(MODE) == 5 && qpn_stream_t<type>::CPU2 == 2 ? 4 : 2, qpn_mode_full(MODE)>(Wp, step, sbA + sp, sbB, S, active, qpn_xsm, sbA, xh, xs, xsc, nB, r0, c0, dofs, D, bad, t, ts);
    } else {
    uint4 buf[NCH];
    uint2 dp = make_uint2(0, 0);
    int sb = sbA + sp;
    if (active && sb < sbB) {
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            buf[ch] = qpn_ldg_w(Wp + ch*QPN_CHUNK);
        }
        if constexpr (type == GGML_TYPE_Q6_K) {
            dp.x = __ldg((const uint32_t *) Wp + dofs);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wp + dofs));
        }
    }
    constexpr bool RING = T >= qpn_ring_min_t<type>();
    for (; active && sb < sbB; sb += S) {
        const bool more = sb + S < sbB;
        const uint4 * xk = STAGE ? qpn_xsm + ((sb - sbA)*32*T + nB) : (const uint4 *) xh + ((int64_t) sb*32*T + nB);
        if constexpr (RING) {
            Wp += step;
            qpn_superblock<type, T, NL, STAGE, true>(buf, dp, Wp, more, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
        } else {
            uint4 cur[NCH];
            uint2 dcur = dp;
#pragma unroll
            for (int ch = 0; ch < NCH; ++ch) {
                cur[ch] = buf[ch];
            }
            Wp += step;
            if (more) {
#pragma unroll
                for (int ch = 0; ch < NCH; ++ch) {
                    buf[ch] = qpn_ldg_w(Wp + ch*QPN_CHUNK);
                }
                if constexpr (type == GGML_TYPE_Q6_K) {
                    dp.x = __ldg((const uint32_t *) Wp + dofs);
                } else if constexpr (type == GGML_TYPE_IQ4_XS) {
                    dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wp + dofs));
                }
            }
            qpn_superblock<type, T, NL, STAGE, false>(cur, dcur, Wp, false, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
        }
    }

    } // MODE

    // the S warps of each tile
    if (bad) {
        atomicOr(&sbad, bad);
    }
    if (S > 1) {
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            red[(wy*NL + l)*WARP_SIZE + lane] = D[l];
        }
    }
    __syncthreads();
    const unsigned badmask = sbad;
    if (sp == 0) {
        for (int i = 1; i < S; ++i) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                D[l] += red[((wy + i)*NL + l)*WARP_SIZE + lane];
            }
        }
    }
    // The block's result goes out now, to dst or (P > 1) to its slice of ws, and the non-finite fallback below only
    // overwrites the columns it recomputes, so D is not live across the fallback's call (a D live across that
    // __noinline__ call gets a stack home for the whole kernel, which put it in local memory in every superblock).
    float * wsp = P > 1 ? ws + ((int64_t) tile*P + p)*NL*WARP_SIZE : nullptr;
    if (active && sp == 0) {
        float * dstm = dst;
        int64_t sdm  = stride_col_dst;
        int     nrow = 32;
        if constexpr (CLIP) {
            qpn_clip_member(clip, dstm, sdm, nrow);
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const int row = tile*32 + r0 + (l & 2);
            const int col = c0 + (l & 13);
            if (P > 1) {
                wsp[l*WARP_SIZE + lane] = D[l];
            } else if (col < t && (!CLIP || r0 + (l & 2) < nrow)) {
                dstm[col*sdm + row] = D[l];
            }
        }
    }
    if (badmask) { // block-uniform: those columns again, in fp32 from the dequantized weights
        float fb[16];
        if (active) {
            // the split's bounds again, from a fresh read of blockIdx.y, so the loop's copies are not live across the call
            unsigned py;
            asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(py));
            qpn_fallback<type>(Wt, nsb*(int) py/P + sp, nsb*((int) py + 1)/P, S, badmask, t, x, stride_col_x, fb);
        } else {
            for (int c = 0; c < 16; ++c) {
                fb[c] = 0.0f;
            }
        }
        for (int c = 0; c < t; ++c) { // one bad column at a time through red
            if (!((badmask >> c) & 1)) {
                continue;
            }
            __syncthreads();
            red[wy*WARP_SIZE + lane] = fb[c];
            __syncthreads();
            if (active && sp == 0) {
                float * dstm = dst;
                int64_t sdm  = stride_col_dst;
                int     nrow = 32;
                if constexpr (CLIP) {
                    qpn_clip_member(clip, dstm, sdm, nrow);
                }
#pragma unroll
                for (int l = 0; l < NL; ++l) {
                    if (c0 + (l & 13) == c) {
                        float v = 0.0f;
                        for (int i = 0; i < S; ++i) {
                            v += red[(wy + i)*WARP_SIZE + r0 + (l & 2)];
                        }
                        if (P > 1) {
                            wsp[l*WARP_SIZE + lane] = v;
                        } else if (!CLIP || r0 + (l & 2) < nrow) {
                            dstm[c*sdm + tile*32 + r0 + (l & 2)] = v;
                        }
                    }
                }
            }
        }
    }
    // with --split-mode tensor, the finished tile goes to the peer's landing slot of the ALLREDUCE that reduces dst
    if constexpr (PUSH) {
        if (active && sp == 0 && P == 1) {
            ggml_cuda_p2p_push_tile(push, dst, stride_col_dst, (int64_t) tile*32, t);
        }
    }
    if (!active || sp != 0 || P == 1 || CLIP) { // CLIP: P = 1
        return;
    }

    // P > 1: the last of the tile's P blocks adds their partial sums in the order p = 0 .. P-1
    __threadfence();
    unsigned * cp = cnt + tile;
    unsigned prev = 0;
    if (lane == 0) {
        prev = atomicAdd(cp, 1u);
    }
    prev = __shfl_sync(0xFFFFFFFF, prev, 0);
    if (prev != (unsigned) P - 1) {
        return;
    }
    __threadfence();
    const float * ws0 = ws + (int64_t) tile*P*NL*WARP_SIZE;
#pragma unroll
    for (int l = 0; l < NL; ++l) {
        float v = 0.0f;
        for (int i = 0; i < P; ++i) {
            v += __ldcg(ws0 + (i*NL + l)*WARP_SIZE + lane);
        }
        const int row = tile*32 + r0 + (l & 2);
        const int col = c0 + (l & 13);
        if (col < t) {
            dst[col*stride_col_dst + row] = v;
        }
    }
    if constexpr (PUSH) {
        ggml_cuda_p2p_push_tile(push, dst, stride_col_dst, (int64_t) tile*32, t);
    }
    if (lane == 0) {
        *cp = 0; // ready for the next product on this stream
    }
#else
    GGML_UNUSED_VARS(bx, W, nsb, ntiles, S, P, xh, xs, xsc, x, stride_col_x, dst, stride_col_dst, ws, cnt, qpn_xsm, red, sbad, t, push, clip);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// the block's shared state and staged activations (the same input for every segment of a launch)
template <bool STAGE>
static __device__ __forceinline__ void qpn_block_start(const half * __restrict__ xh, const int nsb, const int P, uint4 * qpn_xsm, unsigned & sbad, const int t) {
    const int lane = threadIdx.x, wy = threadIdx.y;
    if (threadIdx.x == 0 && wy == 0) {
        sbad = 0;
    }
    if constexpr (STAGE) {
        const int p   = blockIdx.y;
        const int sbA = nsb*p/P, sbB = nsb*(p + 1)/P;
        const uint4 * src = (const uint4 *) xh + (int64_t) sbA*32*t;
        const int n = (sbB - sbA)*32*t;
        for (int i = wy*WARP_SIZE + lane; i < n; i += blockDim.y*WARP_SIZE) {
            qpn_xsm[i] = __ldg(src + i);
        }
    }
    __syncthreads();
}

template <int T, int MODE> static constexpr __host__ __device__ bool qpn_wide() { return (T == 8 || T == 16) && MODE != 0; }
// PAIR: the paired 8-token blocks are their own instantiation, so every other kernel keeps one token count
// (t, which is also the stride): with the pairing a run-time branch, t and ts were two values in every 8-token kernel (Q5_K down,
// 8 tokens, 55.6 -> 61.0 us with the pairing)
template <ggml_type type, int T, bool STAGE, int MODE = 0, bool PUSH = false, bool PAIR = false>
static __global__ void __launch_bounds__(qpn_max_threads(MODE), qpn_min_blocks(MODE))
qpn_mul_kernel(const char * __restrict__ W, const int nsb, const int ntiles, const int S, const int P,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, float * __restrict__ dst, const int64_t stride_col_dst,
               float * __restrict__ ws, unsigned * __restrict__ cnt, const int nt, const int nh, ggml_cuda_p2p_push push) {
    constexpr int NL = qpn_nl<T>();
    int t = qpn_wide<T, MODE>() ? nt : T, ts = t, bx = blockIdx.x;
    // nh = 2 runs a pass of 9 to 16 tokens (nt, the prepared input's token stride) as paired 8-token blocks of this kernel at
    // T = 8: block x takes tokens 8*(x % 2) .. of tile group x/2, exactly as an 8-token call computes them, and the two blocks of a pair
    // stream the same weights side by side (the second read can come from L2)
    static_assert(!PAIR || (qpn_wide<T, MODE>() && T == 8 && !STAGE), "paired blocks: the unstaged 8-token kernel");
    if constexpr (PAIR) {
        const int hf = blockIdx.x % nh;
        bx  = blockIdx.x / nh;
        t   = min(8, nt - 8*hf);
        xh  += 8*hf*8;          // uint4 (8 halves) per token in [sb][fragment][token]
        xs  += xs ? 8*hf*8 : 0; // uint4 per token in [sb][token]
        xsc += 8*hf;            // float per token in [sb][token]
        x   += 8*hf*stride_col_x;
        dst += 8*hf*stride_col_dst;
        if constexpr (PUSH) {
            ggml_cuda_p2p_push_advance(push, 8*hf*stride_col_dst); // paired blocks advance in wire elements
        }
        if (P > 1) {
            ws  += (int64_t) hf*ntiles*P*NL*WARP_SIZE;
            cnt += hf*ntiles;
        }
    }
    extern __shared__ uint4 qpn_xsm[];
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<STAGE>(xh, nsb, P, qpn_xsm, sbad, t);
    qpn_mul_block<type, T, STAGE, MODE, PUSH>(bx, W, nsb, ntiles, S, P, xh, xs, xsc, x, stride_col_x, dst, stride_col_dst, ws, cnt,
        qpn_xsm, red, sbad, t, ts, push);
}

// Two products of the same input in one launch: blocks [0, nb0) take product 0,
// the rest product 1, each with its own weights, type, rows and output, and the same K, S, R and P = 1. Each product's
// arithmetic and order are those of its own launch, so the outputs are bit-identical to two launches.
struct qpn_seg { const char * W; int ntiles; float * dst; int64_t stride_col_dst; };
template <ggml_type type0, ggml_type type1, int T, bool STAGE, int MODE>
static __global__ void __launch_bounds__(qpn_max_threads(MODE), qpn_min_blocks(MODE))
qpn_mul2_kernel(const qpn_seg g0, const qpn_seg g1, const int nb0, const int nsb, const int S,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    constexpr int NL = qpn_nl<T>();
    const int t = qpn_wide<T, MODE>() ? nt : T;
    extern __shared__ uint4 qpn_xsm[];
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<STAGE>(xh, nsb, 1, qpn_xsm, sbad, t);
    if ((int) blockIdx.x < nb0) {
        qpn_mul_block<type0, T, STAGE, MODE>(blockIdx.x, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, qpn_xsm, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, STAGE, MODE>(blockIdx.x - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, qpn_xsm, red, sbad, t, t);
    }
}

// Three products of the same input in one launch (the attention input, q + gate, k and v, whose 1024-row k and v
// alone are 32 row tiles that cannot fill 72 SMs): blocks [0, nb0) take product 0, [nb0, nb0 + nb1) product 1, the rest product 2,
// each as in its own launch at P = 1, unstaged: q + gate at its own split (S, R), k and v at split-K S12 with blockDim.y/S12 tiles a block.
template <ggml_type type0, ggml_type type1, ggml_type type2, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul3_kernel(const qpn_seg g0, const qpn_seg g1, const qpn_seg g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S,
               const int S12, const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    constexpr int NL = qpn_nl<T>();
    const int t = qpn_wide<T, MODE>() ? nt : T;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    // products 1 and 2 (k, v) take the first blocks, with their own split-K S12 (the block's rows are then blockDim.y/S12 tiles)
    const int bx = blockIdx.x;
    if (bx < nb1) {
        qpn_mul_block<type1, T, false, MODE>(bx, g1.W, nsb, g1.ntiles, S12, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else if (bx < nb1 + nb2) {
        qpn_mul_block<type2, T, false, MODE>(bx - nb1, g2.W, nsb, g2.ntiles, S12, 1, xh, xs, xsc, x, stride_col_x, g2.dst, g2.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type0, T, false, MODE>(bx - nb1 - nb2, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
    GGML_UNUSED(nb0);
}

// The GDN layer's three products of one input in one launch at 5 to 8 tokens: the merged alpha/beta weight ab (Q8_0, 96 rows =
// 3 row tiles) first, then in-proj (product 0) and z (product 1) exactly as qpn_mul2_kernel runs them. ab's blocks are the pair's shape
// (S warps a tile, R tiles a block), unstaged, streaming loop at T = 8 with the token count at run time (or at T = 16, 9 to 16 tokens).
template <ggml_type type0, ggml_type type1, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul2ab_kernel(const qpn_seg gab, const qpn_seg g0, const qpn_seg g1, const int nbab, const int nb0, const int nsb, const int S,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    static_assert(T == 8 || T == 16, "the wide kernels");
    constexpr int NL = qpn_nl<T>();
    const int t = nt;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    const int bx = blockIdx.x;
    if (bx < nbab) {
        qpn_mul_block<GGML_TYPE_Q8_0, T, false, MODE>(bx, gab.W, nsb, gab.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, gab.dst, gab.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else if (bx < nbab + nb0) {
        qpn_mul_block<type0, T, false, MODE>(bx - nbab, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, false, MODE>(bx - nbab - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
}

// Under --split-mode tensor, the GDN layer's in-proj and z with beta and alpha, each a device's 24-row slice as a padded one-tile
// private copy (Q8_0): beta's nbb blocks, alpha's nba blocks, then in-proj (product 0) and z (product 1) as qpn_mul2ab_kernel runs them. A
// member stores the first nrowb (nrowa) rows of its tile. Only the weights are chosen before the loop; the store finds its member again from a
// fresh read of blockIdx.x (qpn_clip), so the member's output, stride and row count stay kernel parameters and are not live across the loop.
template <ggml_type type0, ggml_type type1, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul4ab_kernel(const qpn_seg gb, const qpn_seg ga, const int nrowb, const int nrowa, const qpn_seg g0, const qpn_seg g1, const int nbb,
               const int nba, const int nb0, const int nsb, const int S, const half * __restrict__ xh, const half * __restrict__ xs,
               const float * __restrict__ xsc, const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    static_assert(T == 8 || T == 16, "the wide kernels");
    constexpr int NL = qpn_nl<T>();
    const int t = nt;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    const int bx  = blockIdx.x;
    const int nbq = nbb + nba;
    if (bx < nbq) {
        const bool second = bx >= nbb;
        const qpn_clip clip = { gb.dst, gb.stride_col_dst, nrowb, ga.dst, ga.stride_col_dst, nrowa, nbb };
        qpn_mul_block<GGML_TYPE_Q8_0, T, false, MODE, false, true>(second ? bx - nbb : bx, second ? ga.W : gb.W, nsb, second ? ga.ntiles : gb.ntiles,
            S, 1, xh, xs, xsc, x, stride_col_x, nullptr, 0, nullptr, nullptr, nullptr, red, sbad, t, t, {}, clip);
    } else if (bx < nbq + nb0) {
        qpn_mul_block<type0, T, false, MODE>(bx - nbq, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, false, MODE>(bx - nbq - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
}

// One warp per (256-column slice, token); the arithmetic and the layout are qpn-prep.cuh's.
static __global__ void qpn_prep_kernel(const float * __restrict__ x, const int64_t stride_col_x, const int nsb, const int T,
        half * __restrict__ xh, half * __restrict__ xs, float * __restrict__ xsc) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int g    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (g >= nsb*T) {
        return;
    }
    const int sb = g / T, t = g % T;
    const float4 * xp = (const float4 *) (x + t*stride_col_x + (int64_t) sb*QK_K + lane*8);
    const float4 a = xp[0], b = xp[1];
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    qpn_prep_slice_warp(v, sb, t, T, xh, xs, xsc);
}

// ------------------------------------------------------------------------------------------------------------------
// routing and launch

static int ggml_cuda_qpn_mask() {
    static const int mask = [] {
        const char * e = getenv("LLAMA_MMVQ_QPN");
        return e == nullptr ? GGML_CUDA_QPN_ALL : atoi(e);
    }();
    return mask;
}

// LLAMA_MMVQ_QPN_FORCE: 1 = every enabled type, any other nonzero value = a bitmask of the types to force (as
// LLAMA_MMVQ_QPN's), e.g. 24 = IQ4_XS and IQ4_NL, the others keeping their routes
static bool ggml_cuda_qpn_force(const int bit) {
    static const int force = [] {
        const char * e = getenv("LLAMA_MMVQ_QPN_FORCE");
        const int v = e == nullptr ? 0 : atoi(e);
        return v == 1 ? GGML_CUDA_QPN_ALL : v;
    }();
    return (force & bit) != 0;
}

static int ggml_cuda_qpn_type_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return GGML_CUDA_QPN_Q4_K;
        case GGML_TYPE_Q5_K: return GGML_CUDA_QPN_Q5_K;
        case GGML_TYPE_Q6_K:   return GGML_CUDA_QPN_Q6_K;
        case GGML_TYPE_IQ4_XS: return GGML_CUDA_QPN_IQ4_XS;
        case GGML_TYPE_IQ4_NL: return GGML_CUDA_QPN_IQ4_NL;
        case GGML_TYPE_Q3_K:   return GGML_CUDA_QPN_Q3_K;
        case GGML_TYPE_Q8_0:   return GGML_CUDA_QPN_Q8_0;
        default:               return 0;
    }
}

// How a product is split (see qpn_mul_kernel): S warps per row tile, R row tiles per block, P blocks per row tile
// along K, stage: the block's activations in shared memory. Per shape (N rows, K), type-independent: the best of
// a sweep at 3 and 4 tokens, and the nearest swept shape for the two it did not sweep.
// Any other shape takes { 4, 2, 1, 1 }, or { 1, 4, 1, 0 } at 65536 rows and more (the output head).
struct qpn_cfg       { int S, R, P, stage, mode = 0; }; // mode: 0, or 2/3 = qpn_stream_loop at that many blocks per SM
// the (N, K) that keys a repacked weight's launch configuration (the split tables below), its pairs and its triple: its own,
// or for one device's slice of a weight under --split-mode tensor the whole weight's, recorded when the per-slice repack ran. Every
// table here is keyed by the model's shapes, so a slice (8704 x 5120 of the 17408-row ffn gate, 5120 x 8704 of the down projection, ...)
// otherwise missed them all: it ran the default split ({4, 2}, too few blocks for the card) and never formed a pair or a triple. Keyed
// by the whole weight, a slice of rows runs the one-card split (the same S, R, P and mode, so each output row sums in the one-card
// order), a slice along K runs the one-card split over its half of K, and the planner groups slices as it groups the whole weights.
// Written at load only (empty on one card), read by each device's compute thread afterwards.
static std::unordered_map<const void *, std::pair<int64_t, int64_t>> qpn_slice_keys;

// the slices a draft model's route repacked (kind 1 of ggml_backend_cuda_qpn_repack_slice), written at load only
static std::unordered_set<const void *> qpn_draft_slices;

void ggml_cuda_qpn_mark_draft_slice(const ggml_tensor * slice) {
    qpn_draft_slices.insert(slice->data);
}

void ggml_cuda_qpn_set_slice_key(const ggml_tensor * slice, const int64_t n, const int64_t k) {
    qpn_slice_keys[slice->data] = { n, k };
}

static inline int64_t qpn_key_n(const ggml_tensor * t) {
    if (!qpn_slice_keys.empty()) {
        const auto it = qpn_slice_keys.find(t->data);
        if (it != qpn_slice_keys.end()) {
            return it->second.first;
        }
    }
    return t->ne[1];
}

// a device's slice of a weight under --split-mode tensor (its key recorded)
static inline bool qpn_is_slice(const ggml_tensor * t) {
    return !qpn_slice_keys.empty() && qpn_slice_keys.count(t->data) != 0;
}
static inline bool qpn_is_draft_slice(const ggml_tensor * t) {
    return !qpn_draft_slices.empty() && qpn_draft_slices.count(t->data) != 0;
}

static inline int64_t qpn_key_k(const ggml_tensor * t) {
    if (!qpn_slice_keys.empty()) {
        const auto it = qpn_slice_keys.find(t->data);
        if (it != qpn_slice_keys.end()) {
            return it->second.second;
        }
    }
    return t->ne[0];
}

struct qpn_shape_cfg { int64_t n, k; qpn_cfg cfg; };
static const qpn_shape_cfg qpn_cfgs[] = {
    { 10240,  5120, { 2, 2, 1, 0 } }, // GDN qkv
    { 12288,  5120, { 2, 2, 1, 0 } }, // attention q + gate (not swept; as 10240)
    { 17408,  5120, { 2, 4, 1, 0 } }, // ffn gate, ffn up
    {  5120, 17408, { 4, 1, 1, 0 } }, // ffn down
    {  6144,  5120, { 4, 1, 1, 0 } }, // GDN z (attn_gate)
    {  5120,  6144, { 4, 1, 1, 0 } }, // GDN out (ssm_out), attention out
    {  1024,  5120, { 4, 1, 1, 0 } }, // attention k, v (not swept; as the other small grids)
};

// Routed shapes, repacked at load by default: each (type, N, K) whose products, prep included, took less time per
// round in the graph on the repacked weights than on dp4a by more than the drift of that capture (the
// Qwen3.8-27B, n-max 3, 4-token verifies, node-level capture of both arms; SM clock 1380 MHz throughout, untouched
// kernels within 2%). Per round, dp4a -> QPN. Not routed: the 1024-row k/v (+64% to +125%) and every other Q6_K
// shape (+2% to +18%; four of them are routed below). Anything else keeps the GGUF layout; LLAMA_MMVQ_QPN_FORCE=1
// repacks every eligible shape.
struct qpn_route { ggml_type type; int64_t n, k; };
static const qpn_route qpn_routes[] = {
    { GGML_TYPE_Q5_K,  17408,  5120 }, // ffn gate, up      6.32 -> 4.50 ms
    { GGML_TYPE_Q5_K,   5120, 17408 }, // ffn down          3.91 -> 3.33
    { GGML_TYPE_Q5_K,   5120,  6144 }, // GDN out, attn out 1.94 -> 1.83
    { GGML_TYPE_Q5_K,   6144,  5120 }, // GDN z             1.79 -> 1.67
    { GGML_TYPE_Q5_K,  10240,  5120 }, // GDN qkv           1.67 -> 1.35
    { GGML_TYPE_Q5_K,  12288,  5120 }, // attn q + gate     0.75 -> 0.56
    { GGML_TYPE_Q4_K,  17408,  5120 }, // ffn gate, up      2.29 -> 1.44
    { GGML_TYPE_Q4_K,  10240,  5120 }, // GDN qkv           1.76 -> 1.34
    { GGML_TYPE_Q4_K,   6144,  5120 }, // GDN z             0.71 -> 0.59
    { GGML_TYPE_Q4_K,   5120, 17408 }, // ffn down          0.54 -> 0.45
    { GGML_TYPE_Q4_K,  12288,  5120 }, // attn q + gate     0.54 -> 0.37
    { GGML_TYPE_Q4_K,   5120,  6144 }, // GDN out           0.05 -> 0.04
    { GGML_TYPE_Q6_K, 248320,  5120 }, // output head       1.86 -> 1.72
    // the same rule (n-max 3, rejection sampling, the 98,304-token draft vocabulary; LLAMA_MMVQ_QPN=7 against
    // LLAMA_MMVQ_QPN_FORCE=24, SM clock 1380 MHz in both, untouched items within 3.5%). Not routed: IQ4_XS ffn down (+7.7%)
    // and attn out (+9.1%), and every IQ4_NL shape (gate/up -2.8%, inside the drift; down +9.0%; GDN qkv +15.2%)
    { GGML_TYPE_IQ4_XS, 17408, 5120 }, // ffn gate, up      4.43 -> 3.44
    { GGML_TYPE_IQ4_XS, 12288, 5120 }, // attn q + gate     0.149 -> 0.117
    { GGML_TYPE_IQ4_XS,  6144, 5120 }, // GDN z             0.149 -> 0.138
    { GGML_TYPE_IQ4_XS, 10240, 5120 }, // GDN qkv           0.131 -> 0.123
};

// the same rule after the register ring and the shared prep (n-max 3, rejection sampling, the 98,304-token
// draft vocabulary; defaults against LLAMA_MMVQ_QPN_FORCE=1, SM clock recorded). LLAMA_QPN_ROUTES_0086=0 leaves them
// out. Drift 3.25% (k_bin_bcast); per round, dp4a -> QPN with prep. Not routed: Q6_K 5120x6144 (-2.8%,
// inside the drift), IQ4_XS ffn down (-0.4%) and attn out (+7.7%), IQ4_NL ffn down (+8.9%) and GDN qkv (+15.9%), the 1024-row k/v
// (+42% to +87%).
static const qpn_route qpn_routes_0086[] = {
    { GGML_TYPE_Q6_K,  17408,  5120 }, // ffn gate, up      0.834 -> 0.610 ms
    { GGML_TYPE_Q6_K,   5120, 17408 }, // ffn down          0.660 -> 0.561
    { GGML_TYPE_Q6_K,  10240,  5120 }, // GDN qkv           0.091 -> 0.078
    { GGML_TYPE_Q6_K,   6144,  5120 }, // GDN z             0.062 -> 0.059
    { GGML_TYPE_IQ4_NL, 17408, 5120 }, // ffn gate, up      0.265 -> 0.255
};

// the same rule on the finished kernel (n-max 3, rejection sampling, the 98,304-token draft vocabulary; defaults
// against LLAMA_MMVQ_QPN_FORCE=1, node-level, card 0, drift 1.67% (flash_attn_tile); per round, dp4a -> QPN with prep).
// LLAMA_QPN_ROUTES_0091=0 leaves them out. Not routed: IQ4_XS ffn down (+0.5%) and the 1024-row k/v (+62% to +92%).
static const qpn_route qpn_routes_0091[] = {
    { GGML_TYPE_Q6_K,   5120,  6144 }, // GDN out, attn out 1.236 -> 0.982 ms
    { GGML_TYPE_IQ4_NL, 5120, 17408 }, // ffn down          0.179 -> 0.170
    { GGML_TYPE_IQ4_XS, 5120,  6144 }, // attn out          0.042 -> 0.041
    { GGML_TYPE_IQ4_NL, 10240, 5120 }, // GDN qkv           0.056 -> 0.055 (then grouped with its z)
};

// the same rule over the widths in use, for the new types' keys: node-level captures at
// n-max 3 and 7 (4- and 8-token verifies) on poetry, CSV and the code prompt (17.7K deep), each width's pair in one session;
// arm O = LLAMA_QPN_DRAFT=0 LLAMA_QPN_ROUTES_0096=0, arm F = LLAMA_MMVQ_QPN_FORCE=96 LLAMA_QPN_DRAFT_FORCE=1;
// card 0; drift 4.0-5.1% at width 4, 5.7-6.4% at width 8 (flash_attn_tile, or a 0.13 ms IQ4_XS item). Per round,
// dp4a -> QPN, width 4 / width 8, the same on all three prompts to 1%.
// LLAMA_QPN_ROUTES_0096=0 leaves them out. Not routed: Q8_0 1024x5120 (attn v, and one k: +20% / +78%; 32 row tiles, the
// three-product grouping is its route).
static const qpn_route qpn_routes_0096[] = {
    { GGML_TYPE_Q3_K, 17408,  5120 }, // ffn gate, up (3)  0.506 -> 0.179 / 0.944 -> 0.281 ms
    { GGML_TYPE_Q8_0,  5120,  6144 }, // GDN out (1)       0.050 -> 0.046 / 0.070 -> 0.053
};

static int ggml_cuda_qpn_wide_mask(); // LLAMA_QPN_WIDE, below

// LLAMA_QPN_DRAFT_KV (default OFF: no gain measured, -1 us a step; 1 = on): the draft's attention k and v (Q6_K, 1,024 x 5,120) are repacked (qpn_routes_draft) and, at one
// token, run with q as one launch (q + k pair, and the triple with v)
static bool ggml_cuda_qpn_draft_kv_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_DRAFT_KV"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

// LLAMA_QPN_GDN_GROUP (default on; 0 = off): qwen35.cpp puts the GDN layer's in-proj and z (and ab) side by side in the graph, and the
// sibling planner runs in-proj + z as one qpn_mul2 launch at 3 to 8 tokens (outputs bit-identical to two launches).
// LLAMA_QPN_GDN_AB (default ON, 0 = off: not bit-identical, KL 0.000021; -0.6 ms per verify graph at 16K and 64K): the merged alpha/beta weight ab (Q8_0, 96 x 5,120) is
// repacked and runs in the same launch at 5 to 8 tokens (ggml_cuda_mul_mat_qpn2ab), on the tensor cores instead of dp4a
static bool ggml_cuda_qpn_gdn_group_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_GDN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    return on;
}
static bool ggml_cuda_qpn_gdn_ab_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_GDN_AB"); return e == nullptr || atoi(e) != 0; }() && ggml_cuda_qpn_gdn_group_on();
    return on;
}

// the same rule over the widths in use: a key routes if it is no slower at width 4 within the capture's drift and faster
// at width 8, on poetry, CSV and code (n-max 3 and 7, T=0, generated tokens 100-220, node-level, card 0; defaults against
// LLAMA_MMVQ_QPN_FORCE=1 in one session, drift at most 4.75% (flash_attn_tile); per round, dp4a -> QPN with prep).
// LLAMA_QPN_ROUTES_0095=0 leaves them out. Every earlier route still wins against dp4a at both widths (LLAMA_MMVQ_QPN=0).
static const qpn_route qpn_routes_0095[] = {
    { GGML_TYPE_IQ4_XS, 5120, 17408 }, // ffn down          1.342 -> 1.300 ms at width 4, 2.667 -> 1.478 at width 8 (poetry; CSV, code within 0.3%)
    // the 1024-row attention k, v: only in one launch with their q + gate (ggml_cuda_mul_mat_qpn3, or the pair when v is Q8_0), so
    // only with LLAMA_QPN_GROUP and LLAMA_QPN_WIDE bit 8 on. The attention input's products together (q + gate, k, v, prep):
    // 1.418 -> 1.280 ms at width 4, 1.749 -> 1.465 at width 8 (poetry; CSV, code within 0.4%)
    { GGML_TYPE_Q6_K,   1024,  5120 },
    { GGML_TYPE_Q5_K,   1024,  5120 },
    { GGML_TYPE_Q4_K,   1024,  5120 },
};

static const qpn_route * ggml_cuda_qpn_find_route(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_route & r : qpn_routes) {
        if (r.type == type && r.n == n && r.k == k) {
            return &r;
        }
    }
    static const bool routes_0086 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0086"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0086) {
        for (const qpn_route & r : qpn_routes_0086) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0091 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0091"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0091) {
        for (const qpn_route & r : qpn_routes_0091) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0096 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0096"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0096) {
        for (const qpn_route & r : qpn_routes_0096) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0095 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0095"); return e == nullptr || atoi(e) != 0; }();
    static const bool grouped3    = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }() &&
                                    (ggml_cuda_qpn_wide_mask() & 8);
    if (routes_0095) {
        for (const qpn_route & r : qpn_routes_0095) {
            if (r.type == type && r.n == n && r.k == k && (n != 1024 || grouped3)) {
                return &r;
            }
        }
    }
    // the merged GDN alpha/beta weight, whose only reader is the 8-row verify's grouped launch (5 to 8 tokens) or, alone, the
    // QPN kernel at any other width (build_layer_attn_linear reads it through ab at every token count once it is repacked)
    if (type == GGML_TYPE_Q8_0 && n == 96 && k == 5120 && ggml_cuda_qpn_gdn_ab_on()) {
        static const qpn_route r_ab = { GGML_TYPE_Q8_0, 96, 5120 };
        return &r_ab;
    }
    return nullptr;
}

// Per type, at 1 to 4 tokens. LLAMA_QPN_STREAM: bitmask, default 15.
// 1 = ffn gate/up (Q4_K/Q5_K), streamed with
// staged activations at the same S (outputs bit-identical); 2 = GDN qkv (Q4_K/Q5_K) at S = 4 instead of 2 (a different split-K
// summation order); 4 = qpn_stream_loop, 80 registers, for every
// other key at 1 to 4 tokens, at the key's split (outputs bit-identical); 8 = the splits of qpn_cfgs_0091 (new
// summation orders), which then replace bits 1 and 2. 0 = the original whole-superblock kernel and splits; 5 = every
// type on the streaming loop at those splits, bit-identical to it. Microbenchmark on the
// real tensors, card 1, SM 1380 MHz, T = 4, us: Q5_K gate/up 77.4 -> 73.1, Q4_K 63.8 -> 59.0; GDN qkv
// Q5_K 52.7 -> 48.1, Q4_K 47.0 -> 39.2.
struct qpn_type_cfg { ggml_type type; int64_t n, k; int bit; qpn_cfg cfg; };
static const qpn_type_cfg qpn_type_cfgs[] = {
    { GGML_TYPE_Q5_K, 17408, 5120, 1, { 2, 4, 1, 1, 3 } },
    { GGML_TYPE_Q4_K, 17408, 5120, 1, { 2, 4, 1, 1, 2 } },
    { GGML_TYPE_Q5_K, 10240, 5120, 2, { 4, 1, 1, 0, 3 } },
    { GGML_TYPE_Q4_K, 10240, 5120, 2, { 4, 1, 1, 0, 3 } },
};

static int ggml_cuda_qpn_stream_mask() {
    static const int mask = [] { const char * e = getenv("LLAMA_QPN_STREAM"); return e == nullptr ? 15 : atoi(e); }();
    return mask;
}

static const qpn_cfg * ggml_cuda_qpn_find_type_cfg(const ggml_type type, const int64_t n, const int64_t k) {
    const int mask = ggml_cuda_qpn_stream_mask();
    for (const qpn_type_cfg & c : qpn_type_cfgs) {
        if (c.type == type && c.n == n && c.k == k && (mask & c.bit)) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// LLAMA_QPN_STREAM bit 4 (default on): at 1 to 4 tokens, every type on qpn_stream_loop (80 registers, 24 warps per SM)
// with the split that gives it the warps: microbenchmark on the real tensors at 4 tokens, card 1, SM clock recorded,
// us against the whole-superblock kernel. type GGML_TYPE_COUNT = any type.
struct qpn_cfg_0091 { ggml_type type; int64_t n, k; qpn_cfg cfg; };
static const qpn_cfg_0091 qpn_cfgs_0091[] = {
    // the Q3_K and Q8_0 types, the best of a sweep at 1 and 4 tokens (card 0, SM clock recorded), us against dp4a:
    { GGML_TYPE_Q3_K,  98304,  5120, { 2, 4, 1, 1, 3 } }, // the draft's head subset: 1 token 564.1 -> 329.4, 4 tokens 856.0 -> 353.0
    { GGML_TYPE_Q8_0,   1024,  5120, { 8, 1, 4, 0, 3 } }, // attn v, k: 1 token 7.4 -> 12.6, 4 tokens 10.9 -> 13.2 (32 row tiles)
    { GGML_TYPE_Q8_0,   5120,  6144, { 8, 1, 1, 0, 3 } }, // GDN out: 1 token 38.8 -> 40.7, 4 tokens 44.1 -> 38.4
    { GGML_TYPE_COUNT, 17408,  5120, { 2, 4, 1, 1, 3 } }, // ffn gate/up, staged (it still wins by 1-4%: Q5_K 73.8 against 73.3 at S = 3, IQ4_NL 63.7 against 66.1)
    { GGML_TYPE_COUNT, 10240,  5120, { 4, 1, 1, 0, 3 } }, // GDN qkv: Q6_K 62.4 -> 52.7, IQ4_XS 47.0 -> 40.5, Q5_K 54.8 -> 47.5
    { GGML_TYPE_COUNT, 12288,  5120, { 4, 2, 1, 0, 3 } }, // attn q + gate: Q4_K 59.8 -> 44.2, Q5_K 59.3 -> 54.7, IQ4_XS 55.5 -> 47.3
    { GGML_TYPE_COUNT,  5120, 17408, { 4, 1, 2, 0, 3 } }, // ffn down: Q5_K 88.4 -> 82.2, Q6_K 100.5 -> 94.0, IQ4_XS 76.8 -> 70.8, IQ4_NL 81.3 -> 71.8
    { GGML_TYPE_COUNT,  6144,  5120, { 8, 1, 1, 0, 3 } }, // GDN z: Q5_K 34.9 -> 32.1, Q4_K 28.5 -> 25.9, Q6_K 38.3 -> 34.6
    { GGML_TYPE_Q6_K,   5120,  6144, { 8, 1, 1, 0, 3 } }, // GDN out, attn out: Q6_K 39.8 -> 35.3
    { GGML_TYPE_COUNT,  5120,  6144, { 4, 1, 2, 0, 3 } }, //                    Q5_K 36.0 -> 33.4, IQ4_XS 31.2 -> 29.2
    { GGML_TYPE_COUNT, 248320, 5120, { 4, 2, 1, 0, 3 } }, // the output head: Q6_K 1220.7 -> 1209.0 (streamed at A's 1,4: 1273.9)
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_0091(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_0091) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// LLAMA_QPN_WIDE bit 2 (default on): at 5 to 8 tokens, the splits of the streaming kernel at T = 8 per shape and type
// (type GGML_TYPE_COUNT = any type), from a sweep of S, R, P, staging and blocks per SM (mode) on the real tensors at 8 tokens, checked at
// 5, card 1, SM clock recorded; us at 8 tokens against the default (the streaming kernel at 24 warps per SM at the original
// split). New summation orders.
static const qpn_cfg_0091 qpn_cfgs_0095[] = {
    { GGML_TYPE_Q3_K,   17408,  5120, { 2, 4, 1, 0, 0 } }, // ffn gate/up, Q3_K: the whole-superblock kernel, 68.6 (streamed 74.6 at 16 warps per SM, 82.4 at 24)
    { GGML_TYPE_COUNT,  17408,  5120, { 2, 4, 1, 0, 2 } }, // ffn gate/up, 16 warps per SM: Q5_K 83.6 -> 75.2, Q6_K 94.6 -> 86.8, Q4_K 70.6 -> 62.0, IQ4_NL 74.5 -> 70.2, IQ4_XS 70.8 -> 67.6
    { GGML_TYPE_COUNT,  10240,  5120, { 4, 1, 1, 0, 3 } }, // GDN qkv: Q5_K 67.5 -> 55.3, Q6_K 70.0 -> 60.6, Q4_K 59.6 -> 41.7, IQ4_NL 63.3 -> 46.2, IQ4_XS 57.9 -> 47.8
    { GGML_TYPE_IQ4_XS, 12288,  5120, { 2, 2, 1, 0, 0 } }, // attn q + gate, IQ4_XS: the whole-superblock kernel, 51.7 (streamed at best 54.4; at 5 tokens 51.4 against 54.1)
    { GGML_TYPE_COUNT,  12288,  5120, { 4, 2, 1, 0, 3 } }, //                Q5_K 69.8 -> 61.9, Q4_K 61.2 -> 47.2
    { GGML_TYPE_IQ4_XS,  5120, 17408, { 4, 1, 1, 0, 0 } }, // ffn down, IQ4_XS: the whole-superblock kernel, 80.9 (streamed at best 85.4; at 5 tokens 80.1 against 84.1)
    { GGML_TYPE_IQ4_NL,  5120, 17408, { 4, 1, 2, 0, 3 } }, //           IQ4_NL 106.3 -> 84.0
    { GGML_TYPE_Q4_K,    5120, 17408, { 8, 1, 1, 0, 3 } }, //           Q4_K 99.3 -> 74.4 (16 warps per SM: 78.1)
    { GGML_TYPE_COUNT,   5120, 17408, { 4, 1, 1, 0, 2 } }, //           Q5_K 112.4 -> 94.9, Q6_K 120.6 -> 97.7
    { GGML_TYPE_COUNT,   6144,  5120, { 4, 1, 1, 0, 2 } }, // GDN z: Q5_K 37.3 -> 32.0, Q6_K 40.0 -> 33.6, Q4_K 32.6 -> 26.7, IQ4_XS 31.8 -> 29.6
    { GGML_TYPE_Q8_0,    5120,  6144, { 4, 1, 1, 0, 0 } }, // GDN out, Q8_0: the whole-superblock kernel, 38.6 (streamed at best 39.1)
    { GGML_TYPE_IQ4_XS,  5120,  6144, { 4, 1, 1, 0, 0 } }, // GDN out, attn out, IQ4_XS: the whole-superblock kernel, 32.9 (streamed at best 33.9; at 5 tokens 32.7 against 33.8)
    { GGML_TYPE_COUNT,   5120,  6144, { 4, 1, 1, 0, 2 } }, //                    Q5_K 42.7 -> 36.3, Q6_K 44.3 -> 36.4, Q4_K 37.4 -> 30.2
    { GGML_TYPE_COUNT, 248320,  5120, { 4, 2, 1, 0, 2 } }, // the output head: Q6_K 1375.8 -> 1268.7 (at 5 tokens 1361.5 -> 1216.5)
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_0095(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_0095) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// the DFlash2 draft's shapes at 5 to 8 tokens (it runs them at 8), taken from the nearest shape of qpn_cfgs_0095 (no
// sweep): attn q 4096x5120 as GDN z 6144x5120, attn out 5120x4096 as 5120x6144, fc 5120x25600 as the Q4_K ffn down, and the grids of
// 40 row tiles or fewer (the conv projections 1280x5120, k/v 1024x5120, selector_hidden 256x5120) split along K as the Q8_0 1024-row k/v
// at 1-4 tokens; the Q6_K head subset as the output head. Only consulted when qpn_cfgs_0095 has no entry.
static const qpn_cfg_0091 qpn_cfgs_dflash[] = {
    { GGML_TYPE_COUNT,  4096,  5120, { 4, 1, 1, 0, 2 } },
    { GGML_TYPE_COUNT,  5120,  4096, { 4, 1, 1, 0, 2 } },
    { GGML_TYPE_COUNT,  5120, 25600, { 8, 1, 1, 0, 3 } },
    { GGML_TYPE_COUNT,  1280,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_COUNT,  1024,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_COUNT,   256,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_Q6_K,  98304,  5120, { 4, 2, 1, 0, 2 } },
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_dflash(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_dflash) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

static const qpn_cfg * ggml_cuda_qpn_find_cfg(const int64_t n, const int64_t k) {
    for (const qpn_shape_cfg & c : qpn_cfgs) {
        if (c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// Wide verify. LLAMA_QPN_WIDE: bitmask, default 15. 2 = the split table
// qpn_cfgs_0095 at 5 to 8 tokens (a new summation order); 4 = pairs at 5 to 8 tokens; 8 = the attention input's q + gate,
// k and v as one launch (ggml_cuda_mul_mat_qpn3) at 4 to 8 tokens. 1 = products of 5 to
// 8 tokens on qpn_stream_loop (the kernel at T = 8 with the token count at run time), at their split (outputs bit-identical
// to the whole-superblock kernel at the same split), unstaged, at QPN_WIDE_MODE blocks of 8 warps per SM. 0 = the
// whole-superblock kernel at 5 to 8 tokens.
#define QPN_WIDE_MODE 3
static int ggml_cuda_qpn_wide_mask() {
    static const int mask = [] { const char * e = getenv("LLAMA_QPN_WIDE"); return e == nullptr ? 15 : atoi(e); }();
    return mask;
}

// the block's staged activations must fit next to the static shared memory in 48 KiB
static size_t ggml_cuda_qpn_stage_bytes(const int nsb, const int P, const int T) {
    return (size_t) ((nsb + P - 1)/P)*32*T*sizeof(uint4);
}
static constexpr size_t QPN_STAGE_MAX = 40*1024;
static constexpr size_t QPN_STAGE_MAX_WIDE = 36*1024; // at 5 to 8 tokens the reduction buffer is 8 KiB, not 4

static qpn_cfg ggml_cuda_qpn_config(const ggml_type type, const int64_t n, const int64_t k, const int T) {
    const int nsb = (int) (k/QK_K);
    qpn_cfg c = { 4, 2, 1, 1 };
    const qpn_cfg * c91 = T <= 4 && (ggml_cuda_qpn_stream_mask() & 8) ? ggml_cuda_qpn_find_cfg_0091(type, n, k) : nullptr;
    const qpn_cfg * c95 = T > 4 && (ggml_cuda_qpn_wide_mask() & 2) ? ggml_cuda_qpn_find_cfg_0095(type, n, k) : nullptr;
    if (c95 == nullptr && T > 4 && (ggml_cuda_qpn_wide_mask() & 2)) {
        c95 = ggml_cuda_qpn_find_cfg_dflash(type, n, k);
    }
    if (c95) {
        c = *c95;
    } else if (c91) {
        c = *c91;
    } else if (const qpn_cfg * tc = T <= 4 ? ggml_cuda_qpn_find_type_cfg(type, n, k) : nullptr) {
        c = *tc;
    } else if (const qpn_cfg * sc = ggml_cuda_qpn_find_cfg(n, k)) {
        c = *sc;
    } else if (n >= 65536) {
        c = { 1, 4, 1, 0 };
    }
    const char * ecfg = getenv("LLAMA_MMVQ_QPN_CFG");
    if (ecfg) { // "S,R,P,stage[,mode]", for the sweep
        c.mode = 0;
        sscanf(ecfg, "%d,%d,%d,%d,%d", &c.S, &c.R, &c.P, &c.stage, &c.mode);
    }
    if (T <= 4 && c.mode == 0 && (ggml_cuda_qpn_stream_mask() & 4) && ecfg == nullptr) {
        c.mode = 3; // the streaming loop for every other key, at its split (bit-identical)
    }
    if (T > 4 && ecfg == nullptr) { // 5 to 8 tokens streamed, unstaged, at the key's split (the table's split and mode)
        if (!(ggml_cuda_qpn_wide_mask() & 1)) {
            c.mode = 0;
        } else if (c95 == nullptr) {
            c.mode  = QPN_WIDE_MODE;
            c.stage = 0;
        }
    }
    if (c.mode != 2 && c.mode != 3 && c.mode != 4 && c.mode != 5) {
        c.mode = 0;
    }
    c.P = std::max(1, std::min(c.P, nsb));
    c.S = std::max(1, std::min(c.S, QPN_MAX_WARPS));
    c.R = std::max(1, std::min(c.R, QPN_MAX_WARPS / c.S));
    if (ggml_cuda_qpn_stage_bytes(nsb, c.P, T) > (T > 4 ? QPN_STAGE_MAX_WIDE : QPN_STAGE_MAX)) {
        c.stage = 0;
    }
    if (T <= 4 && c.mode == 2 && !c.stage) {
        c.mode = 3;
    }
    return c;
}

// the single products (ggml_cuda_mul_mat_qpn) whose pass of 9 to 16 tokens runs as paired 8-token blocks instead of the
// kernel at T = 16 (LLAMA_QPN_PAIR16: unset = this table, 0 = none, 1 = every unstaged key). Paired blocks decode every weight twice, so
// they win only where the kernel at T = 16 (128 registers, 16 warps per SM) leaves a second wave or idle warps: the checker on the real
// tensors, card 0, 1380 MHz, us at 16 tokens, T = 16 -> paired, two runs each. Not routed: Q5_K, Q6_K, IQ4_XS
// ffn down (154 -> 168, 166 -> 189, 137 -> 161), IQ4_NL ffn down (174 -> 173), Q5_K, Q6_K, IQ4_XS 5120x6144 (61 -> 69, 64 -> 73, 54 -> 65)
struct qpn_pair16_key { ggml_type type; int64_t n, k; };
static const qpn_pair16_key qpn_pair16_keys[] = {
    { GGML_TYPE_Q4_K, 5120, 17408 }, // ffn down (5 layers)  172.6, 167.7 -> 161.1, 160.9
    { GGML_TYPE_Q8_0, 5120,  6144 }, // GDN out (1)          71.5, 71.5 -> 61.1, 61.1
    { GGML_TYPE_Q4_K, 5120,  6144 }, // GDN out (1)          58.3, 58.3 -> 55.8, 55.7
};
static bool ggml_cuda_qpn_pair16(const ggml_type type, const int64_t n, const int64_t k) {
    static const int mode = [] { const char * e = getenv("LLAMA_QPN_PAIR16"); return e == nullptr ? -1 : atoi(e); }();
    if (mode >= 0) {
        return mode != 0;
    }
    for (const qpn_pair16_key & e : qpn_pair16_keys) {
        if ((e.type == type || e.type == GGML_TYPE_COUNT) && e.n == n && e.k == k) {
            return true;
        }
    }
    return false;
}

// which single products run MODE 5 (qpn_stream_loop with a lookahead of 4 units, at most 168 registers, blocks of at most 4
// warps): the ffn down and the GDN / attention out projections, whose grids are 160 blocks of 4 warps (about 9 warps per SM, so the
// registers cost no occupancy) and which stalled on their loads. The checker (p6dcheck, random weights of the real types, card 1, 1380 MHz,
// two runs each, us per product, one card's shape / the --split-mode tensor slice): at 16 tokens every type gains (Q5_K down 155.1 ->
// 124.3 / 85.0 -> 69.0, Q6_K 167.3 -> 133.8 / 91.2 -> 73.6, IQ4_XS 138.6 -> 123.2 / 75.8 -> 67.5, IQ4_NL 175.6 -> 144.9 / 106.5 -> 87.3,
// Q5_K out 61.7 -> 50.6 / 35.5 -> 30.1, Q6_K out 66.6 -> 54.5 / 38.5 -> 32.9); at 8 tokens Q4_K, Q5_K and Q6_K gain (Q5_K down 102.1 ->
// 90.7 / 55.6 -> 50.3, Q6_K down 102.0 -> 97.1 / 57.5 -> 55.2, Q5_K out 40.1 -> 37.1 / 23.5 -> 22.6, Q4_K out 33.8 -> 31.9 / 20.1 -> 19.8)
// and IQ4_XS and IQ4_NL lose (85.2 -> 89.5 / 48.2 -> 50.3, 92.3 -> 100.0 / 52.0 -> 58.5). LLAMA_QPN_DEEP: unset = this rule, 0 = never,
// 1 = every eligible product.
// and IQ4_XS at exactly 8 tokens, with QPN_FULL (random weights of the type, card 1, the slice of ffn down 5120 x 8704:
// 46.2 us on today's whole-superblock kernel, 42.1 deep and full; IQ4_NL stays streamed at LA 2: 47.3 against 54.0 deep)
static bool ggml_cuda_qpn_full_on();
static bool ggml_cuda_qpn_deep(const ggml_type type, const int T) {
    static const int mode = [] { const char * e = getenv("LLAMA_QPN_DEEP"); return e == nullptr ? -1 : atoi(e); }();
    const bool eligible = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS ||
        type == GGML_TYPE_IQ4_NL; // two code chunks per two units (qpn_stream_t::CPU2 == 2)
    if (!eligible || mode == 0) {
        return false;
    }
    if (mode > 0 || T > 8) {
        return true;
    }
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || (type == GGML_TYPE_IQ4_XS && T == 8 && ggml_cuda_qpn_full_on());
}

// a pass of 9 to 16 tokens takes the 8-token key's split (S, R, P), so each column's sums are in the order of its pass
// of 8 (bit-identical to it), on qpn_stream_loop at T = 16, unstaged, at 2 blocks of 8 warps per SM (128 registers: two tiles'
// accumulators)
static qpn_cfg ggml_cuda_qpn_config16(const ggml_type type, const int64_t n, const int64_t k) {
    qpn_cfg c = ggml_cuda_qpn_config(type, n, k, 8);
    c.mode  = 2;
    c.stage = 0;
    return c;
}

// t's type is enabled and t is a plain 2-D weight the kernels can take
bool ggml_cuda_qpn_takes(const ggml_tensor * t, const int cc) {
    const int bit = ggml_cuda_qpn_type_bit(t->type);
    if (!bit || !(ggml_cuda_qpn_mask() & bit) || !volta_mma_available(cc)) {
        return false;
    }
    return !(t->view_src != nullptr || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) ||
            t->ne[0] % QK_K != 0 || t->ne[1] % 32 != 0 || (t->flags & GGML_TENSOR_FLAG_BACKEND_LAYOUT));
}

bool ggml_cuda_qpn_eligible(const ggml_tensor * t, const int cc) {
    if (!ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    return ggml_cuda_qpn_force(ggml_cuda_qpn_type_bit(t->type)) || ggml_cuda_qpn_find_route(t->type, t->ne[1], t->ne[0]) != nullptr;
}

// a draft model's weights (its MTP layer, and its LM head subset, which llama_model::set_head_subset makes as a
// tensor of its own) have their own routes, because the draft runs them at 1 token per draft step and at the verify width only
// in the catch-up. LLAMA_QPN_DRAFT=0 repacks none of them; LLAMA_QPN_DRAFT_FORCE: 1 = every enabled type, any other nonzero
// value = a bitmask of types (as LLAMA_MMVQ_QPN_FORCE's), for study. The routes: the same rule over the draft steps (1 token)
// and the catch-up (the verify width), per round at n-max 3 / 7, from the captures of qpn_routes_0096 (drift of the draft graphs'
// untouched kernels at most 2.4%), us per round on poetry, dp4a -> QPN; CSV and code within 1-3 points of it.
// Not routed: eh_proj 5120x10240 (+21% / +7%), attn k, v 1024x5120 (+126% to +137%).
static const qpn_route qpn_routes_draft[] = {
    { GGML_TYPE_Q3_K, 98304,  5120 }, // the LM head over the 98,304-token draft vocabulary  1706 -> 1030 / 3983 -> 2423
    { GGML_TYPE_Q4_K, 17408,  5120 }, // ffn gate, up (dp4a: one fused kernel)               664 ->  510 / 1459 -> 1054
    { GGML_TYPE_Q6_K,  5120,  6144 }, // attn out                                            203 ->  156 /  435 ->  321
    { GGML_TYPE_Q4_K,  5120, 17408 }, // ffn down                                            311 ->  297 /  719 ->  616
    { GGML_TYPE_Q6_K, 12288,  5120 }, // attn q + gate                                       305 ->  302 /  643 ->  613
};

bool ggml_cuda_qpn_eligible_draft(const ggml_tensor * t, const int cc) {
    static const bool on    = [] { const char * e = getenv("LLAMA_QPN_DRAFT"); return e == nullptr || atoi(e) != 0; }();
    static const int  force = [] { const char * e = getenv("LLAMA_QPN_DRAFT_FORCE"); const int v = e == nullptr ? 0 : atoi(e);
                                   return v == 1 ? GGML_CUDA_QPN_ALL : v; }();
    if (!on || !ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    if (force & ggml_cuda_qpn_type_bit(t->type)) {
        return true;
    }
    for (const qpn_route & r : qpn_routes_draft) {
        if (r.type == t->type && r.n == t->ne[1] && r.k == t->ne[0]) {
            return true;
        }
    }
    // k and v alone lose to dp4a (+126% to +137%, above); in the launch with q they win (ggml_cuda_qpn_draft3_ok)
    return t->type == GGML_TYPE_Q6_K && t->ne[1] == 1024 && t->ne[0] == 5120 && ggml_cuda_qpn_draft_kv_on();
}

// a DFlash2 draft's weights (its five layers' attn q, k, v, out, ffn gate, up, down, the two dynamic-conv projections,
// fc, selector_hidden, and its LM head subset) have their own routes, because the draft runs all of them at 8 rows (the block, and the
// injection of the verify's 8 rows) and none of its attention shapes is a target shape. Before it, the trunk went through the target's
// table, so only the shapes that equal a target key (ffn gate/up, down, and the 1024-row k/v) took QPN. LLAMA_QPN_DFLASH=0 repacks
// none of them; LLAMA_QPN_DFLASH_FORCE: 1 = every enabled type, any other nonzero value = a bitmask of types, for study. The routes:
// in-graph timing at 8 rows, node-level captures of the draft pass and the injection (poetry at T=0, generated tokens 100-220, card 1,
// the same text in every arm). Per shape, dp4a -> QPN with prep, us per round (the
// five layers): q 326 -> 176, k + v 467 -> 389 (draft pass + injection), out 346 -> 159, gate + up 2295 -> 672, down 1214 -> 487, fc 355
// -> 118, the head 1702 -> 448; the conv projections 337 -> 333 and selector_hidden 18 -> 28 are even or worse alone, and the two Q6_K
// v 44 -> 67. But the products interact in the graph: with only the winning shapes routed the Q4_K k took 28 us a layer against
// 17 with every weight on QPN, and the whole draft pass is fastest with every weight routed: 3.70 ms against 4.14
// (before the Q6_K down was added) and 7.82 on dp4a. So every weight of this draft's layers, fc and selector_hidden is routed.
static const qpn_route qpn_routes_dflash[] = {
    { GGML_TYPE_Q4_K, 17408,  5120 }, // ffn gate, up
    { GGML_TYPE_Q4_K,  5120, 17408 }, // ffn down
    { GGML_TYPE_Q6_K,  5120, 17408 }, // ffn down (Q4_K_M's Q6_K half)
    { GGML_TYPE_Q4_K,  4096,  5120 }, // attn q
    { GGML_TYPE_Q4_K,  1024,  5120 }, // attn k, v (draft pass + injection)
    { GGML_TYPE_Q6_K,  1024,  5120 }, // attn v (Q4_K_M's Q6_K half)
    { GGML_TYPE_Q4_K,  5120,  4096 }, // attn out
    { GGML_TYPE_Q4_K,  1280,  5120 }, // the dynamic-conv projections (attn, ffn)
    { GGML_TYPE_Q4_K,  5120, 25600 }, // fc (the injection)
    { GGML_TYPE_Q4_K,   256,  5120 }, // selector_hidden
    { GGML_TYPE_Q3_K, 98304,  5120 }, // the LM head subset over the 98,304-token draft vocabulary, from the MTP GGUF's head
    { GGML_TYPE_Q6_K, 98304,  5120 }, // ... or from the target's head (not captured)
};

bool ggml_cuda_qpn_eligible_dflash(const ggml_tensor * t, const int cc) {
    static const bool on    = [] { const char * e = getenv("LLAMA_QPN_DFLASH"); return e == nullptr || atoi(e) != 0; }();
    static const int  force = [] { const char * e = getenv("LLAMA_QPN_DFLASH_FORCE"); const int v = e == nullptr ? 0 : atoi(e);
                                   return v == 1 ? GGML_CUDA_QPN_ALL : v; }();
    if (!on || !ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    if (force & ggml_cuda_qpn_type_bit(t->type)) {
        return true;
    }
    for (const qpn_route & r : qpn_routes_dflash) {
        if (r.type == t->type && r.n == t->ne[1] && r.k == t->ne[0]) {
            return true;
        }
    }
    return false;
}

// per device: the counters of products split across blocks, one row per stream (kernels on one stream run in order,
// and the last block of a tile resets its counter); allocated at load, zeroed once
static unsigned * qpn_counters[GGML_CUDA_MAX_DEVICES] = { nullptr };

template <ggml_type type>
static void ggml_cuda_qpn_repack_type(ggml_tensor * t, char * tmp, const size_t tmp_size, cudaStream_t stream) {
    constexpr int TSB = qpn_tsb<type>();
    const int     nsb       = (int) (t->ne[0]/QK_K);
    const int64_t ntiles    = t->ne[1]/32;
    const size_t  tile_size = (size_t) nsb*TSB;
    const int64_t per_pass  = std::max<int64_t>(1, (int64_t) (tmp_size / tile_size));
    char * data = (char *) t->data;
    for (int64_t t0 = 0; t0 < ntiles; t0 += per_pass) {
        const int64_t nt = std::min(per_pass, ntiles - t0);
        CUDA_CHECK(cudaMemcpyAsync(tmp, data + t0*tile_size, nt*tile_size, cudaMemcpyDeviceToDevice, stream));
        qpn_repack_kernel<type><<<(unsigned) (nt*nsb), 32, 0, stream>>>(tmp, data + t0*tile_size, nsb);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_qpn_repack(ggml_tensor * t, const int device) {
    ggml_cuda_set_device(device);
    if (qpn_counters[device] == nullptr) {
        const size_t sz = (size_t) GGML_CUDA_MAX_STREAMS*QPN_MAX_TILES*sizeof(unsigned);
        CUDA_CHECK(cudaMalloc(&qpn_counters[device], sz));
        CUDA_CHECK(cudaMemset(qpn_counters[device], 0, sz));
    }
    const size_t nbytes = ggml_nbytes(t);
    const size_t tile_size = (size_t) (t->ne[0]/QK_K)*32*ggml_type_size(t->type);
    // one scratch of at most 64 MiB (whole row tiles), freed before returning
    const size_t tmp_size = std::min(nbytes, std::max(tile_size, (size_t) 64*1024*1024 / tile_size * tile_size));
    char * tmp = nullptr;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&tmp, tmp_size));
    cudaStream_t stream = 0;
    switch (t->type) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q4_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q5_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q6_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn_repack_type<GGML_TYPE_IQ4_XS>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn_repack_type<GGML_TYPE_IQ4_NL>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q3_K:   ggml_cuda_qpn_repack_type<GGML_TYPE_Q3_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q8_0:   ggml_cuda_qpn_repack_type<GGML_TYPE_Q8_0>(t, tmp, tmp_size, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
    t->flags |= GGML_TENSOR_FLAG_BACKEND_LAYOUT;
    return true;
}

// the inverse of the repack: the GGUF bytes of tiles [0, ntiles) of this pass, from the repacked W
template <ggml_type type>
static __global__ void qpn_unrepack_kernel(const char * __restrict__ W, char * __restrict__ dst, const int nsb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), BS = sizeof(block);
    __shared__ block blk[32];
    const int lane = threadIdx.x;
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    uint32_t w[NCH*4];
    uint4    tl;
    qpn_load_row<type>(W + ((int64_t) tile*nsb + sb)*TSB, lane, w, tl);
    qpn_unpack_row<type>(w, tl, &blk[lane]);
    __syncwarp();
    const uint16_t * s16 = (const uint16_t *) blk;
    for (int i = lane; i < 32*BS/2; i += 32) {
        const int r = i / (BS/2), o = i % (BS/2);
        ((uint16_t *) (dst + ((int64_t) (tile*32 + r)*nsb + sb)*BS))[o] = s16[i];
    }
}

bool ggml_cuda_qpn_unpack(const ggml_tensor * t, void * host_dst, const int device) {
    if (!ggml_cuda_qpn_is_repacked(t) || t->view_src != nullptr || ggml_cuda_qpn_type_bit(t->type) == 0) {
        return false;
    }
    ggml_cuda_set_device(device);
    const int     nsb       = (int) (t->ne[0]/QK_K);
    const int64_t ntiles    = t->ne[1]/32;
    const size_t  tile_size = (size_t) nsb*32*ggml_type_size(t->type);
    const size_t  tmp_size  = std::min(ggml_nbytes(t), std::max(tile_size, (size_t) 64*1024*1024 / tile_size * tile_size));
    const int64_t per_pass  = (int64_t) (tmp_size / tile_size);
    char * tmp = nullptr;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&tmp, tmp_size));
    cudaStream_t stream = 0;
    const char * data = (const char *) t->data;
    for (int64_t t0 = 0; t0 < ntiles; t0 += per_pass) {
        const int64_t  nt   = std::min(per_pass, ntiles - t0);
        const unsigned nblk = (unsigned) (nt*nsb);
        switch (t->type) {
            case GGML_TYPE_Q4_K: qpn_unrepack_kernel<GGML_TYPE_Q4_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q5_K: qpn_unrepack_kernel<GGML_TYPE_Q5_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q6_K: qpn_unrepack_kernel<GGML_TYPE_Q6_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_IQ4_XS: qpn_unrepack_kernel<GGML_TYPE_IQ4_XS><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_IQ4_NL: qpn_unrepack_kernel<GGML_TYPE_IQ4_NL><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q3_K:   qpn_unrepack_kernel<GGML_TYPE_Q3_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q8_0:   qpn_unrepack_kernel<GGML_TYPE_Q8_0><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            default: GGML_ABORT("fatal error");
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync((char *) host_dst + t0*tile_size, tmp, nt*tile_size, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(tmp));
    return true;
}

void ggml_cuda_qpn_to_fp16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int     nsb    = (int) (src0->ne[0]/QK_K);
    const int64_t ntiles = src0->ne[1]/32;
    const char *  W      = (const char *) src0->data;
    const unsigned nblk  = (unsigned) (ntiles*nsb);
    switch (src0->type) {
        case GGML_TYPE_Q4_K: qpn_to_fp16_kernel<GGML_TYPE_Q4_K><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q5_K: qpn_to_fp16_kernel<GGML_TYPE_Q5_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q6_K: qpn_to_fp16_kernel<GGML_TYPE_Q6_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_IQ4_XS: qpn_to_fp16_kernel<GGML_TYPE_IQ4_XS><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_IQ4_NL: qpn_to_fp16_kernel<GGML_TYPE_IQ4_NL><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q3_K:   qpn_to_fp16_kernel<GGML_TYPE_Q3_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q8_0:   qpn_to_fp16_kernel<GGML_TYPE_Q8_0><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

struct qpn_args {
    const char * W; int nsb, ntiles, S, P;
    half * xh, * xs; float * xsc;
    const float * x; int64_t stride_col_x; float * dst; int64_t stride_col_dst;
    float * ws; unsigned * cnt;
    int nh = 1; // 2 = paired 8-token blocks (qpn_mul_kernel)
    ggml_cuda_p2p_push push; // peer == nullptr, no push
};

// true if it launched the pushing kernel (a.push set, and one of the TP verify's kernels: 5 to 16 tokens, unstaged; only those
// are instantiated with the push, so every other kernel is the one-card kernel unchanged)
template <ggml_type type, int T, bool STAGE, int MODE, bool PAIR = false>
static bool ggml_cuda_qpn_launch_m(const int device, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a,
        const int nt) {
    if constexpr (STAGE) { // staged activations prefer all of the shared memory carveout; set once per device and kernel
        static bool done[GGML_CUDA_MAX_DEVICES] = { false };
        if (!done[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(qpn_mul_kernel<type, T, true, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout,
                (int) cudaSharedmemCarveoutMaxShared));
            done[device] = true;
        }
    }
    if constexpr (!STAGE && qpn_mode_m(MODE) != 4 && (T == 8 || (T == 16 && (qpn_mode_m(MODE) == 2 || qpn_mode_m(MODE) == 5)))) {
        if (a.push.peer != nullptr) {
            qpn_mul_kernel<type, T, false, MODE, true, PAIR><<<grid, block, 0, stream>>>(a.W, a.nsb, a.ntiles, a.S, a.P, a.xh, a.xs, a.xsc, a.x,
                a.stride_col_x, a.dst, a.stride_col_dst, a.ws, a.cnt, nt, a.nh, a.push);
            return true;
        }
    }
    qpn_mul_kernel<type, T, STAGE, MODE, false, PAIR><<<grid, block, STAGE ? smem : 0, stream>>>(a.W, a.nsb, a.ntiles, a.S, a.P, a.xh, a.xs, a.xsc, a.x,
        a.stride_col_x, a.dst, a.stride_col_dst, a.ws, a.cnt, nt, a.nh, a.push);
    return false;
}

// LLAMA_QPN_FULL (default on; 0 = off): a pass of exactly 8 or 16 tokens on a streaming kernel runs its QPN_FULL variant
static bool ggml_cuda_qpn_full_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_FULL"); return e == nullptr || atoi(e) != 0; }();
    return on;
}

// mode 0: the whole-superblock loop at QPN_MIN_BLOCKS blocks per SM, one kernel per T; 2, 3, 4: qpn_stream_loop at that many
// blocks of 8 warps per SM, one kernel per T at 1 to 4 tokens and
// one kernel at T = 8 for 5 to 8 tokens; at exactly 8 tokens, unstaged, its QPN_FULL variant
template <ggml_type type, int T>
static bool ggml_cuda_qpn_launch_t(const bool stage, const int mode, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    constexpr int TK = T <= 4 ? T : 8;
    if constexpr (T == 8) {
        if (!stage && ggml_cuda_qpn_full_on()) {
            switch (mode) {
                case 2: return ggml_cuda_qpn_launch_m<type, 8, false, 2 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                case 3: return ggml_cuda_qpn_launch_m<type, 8, false, 3 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                case 5: return ggml_cuda_qpn_launch_m<type, 8, false, 5 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                default: break;
            }
        }
    }
    switch (mode) {
        case 2: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 2>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 2>(device, grid, block, smem, stream, a, T);
        case 3: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 3>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 3>(device, grid, block, smem, stream, a, T);
        case 4: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 4>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 4>(device, grid, block, smem, stream, a, T);
        case 5: if constexpr (TK == 8) { GGML_ASSERT(!stage); return ggml_cuda_qpn_launch_m<type, 8, false, 5>(device, grid, block, smem, stream, a, T); } break;
        default: break;
    }
    return stage ? ggml_cuda_qpn_launch_m<type, T, true, 0>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, T, false, 0>(device, grid, block, smem, stream, a, T);
}

template <ggml_type type>
static bool ggml_cuda_qpn_launch(const int T, const bool stage, const int mode, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a) {
    switch (T) {
        case 1: return ggml_cuda_qpn_launch_t<type, 1>(stage, mode, grid, block, smem, stream, a);
        case 2: return ggml_cuda_qpn_launch_t<type, 2>(stage, mode, grid, block, smem, stream, a);
        case 3: return ggml_cuda_qpn_launch_t<type, 3>(stage, mode, grid, block, smem, stream, a);
        case 4: return ggml_cuda_qpn_launch_t<type, 4>(stage, mode, grid, block, smem, stream, a);
        case 5: return ggml_cuda_qpn_launch_t<type, 5>(stage, mode, grid, block, smem, stream, a);
        case 6: return ggml_cuda_qpn_launch_t<type, 6>(stage, mode, grid, block, smem, stream, a);
        case 7: return ggml_cuda_qpn_launch_t<type, 7>(stage, mode, grid, block, smem, stream, a);
        case 8: return ggml_cuda_qpn_launch_t<type, 8>(stage, mode, grid, block, smem, stream, a);
        default: {
            // 9 to 16 tokens, one kernel at T = 16 (two B tiles, the token count at run time), unstaged, 2 blocks of 8 warps per SM;
            // or (a.nh = 2) paired blocks of the kernel at T = 8, unstaged, at 3 blocks of 8 warps per SM
            GGML_ASSERT(T >= 9 && T <= 16 && !stage && (a.nh == 1 ? mode == 2 || mode == 5 : mode == 3));
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            const bool full = T == 16 && ggml_cuda_qpn_full_on();
            if (a.nh == 1 && mode == 5) {
                return full ? ggml_cuda_qpn_launch_m<type, 16, false, 5 + QPN_FULL>(device, grid, block, smem, stream, a, T)
                            : ggml_cuda_qpn_launch_m<type, 16, false, 5>(device, grid, block, smem, stream, a, T);
            }
            if (a.nh == 1) {
                return full ? ggml_cuda_qpn_launch_m<type, 16, false, 2 + QPN_FULL>(device, grid, block, smem, stream, a, T)
                            : ggml_cuda_qpn_launch_m<type, 16, false, 2>(device, grid, block, smem, stream, a, T);
            }
            return ggml_cuda_qpn_launch_m<type, 8, false, 3, true>(device, grid, block, smem, stream, a, T);
        }
    }
}

// the prepared input of T tokens: xh (fp16 fragments), xs (per-32 sums, used by Q4_K and Q5_K), xsc (range scales)
static void ggml_cuda_qpn_prep_sizes(const int64_t K, const int T, size_t & sz_xh, size_t & sz_xs, size_t & sz_xsc) {
    const int nsb = (int) (K/QK_K);
    sz_xh  = GGML_PAD((size_t) T*K*sizeof(half), 256);
    sz_xs  = GGML_PAD((size_t) nsb*T*8*sizeof(half), 256);
    sz_xsc = GGML_PAD((size_t) nsb*T*sizeof(float), 256);
}

size_t ggml_cuda_qpn_prep_bytes(const int64_t K, const int T) {
    size_t a, b, c;
    ggml_cuda_qpn_prep_sizes(K, T, a, b, c);
    return a + b + c;
}

void ggml_cuda_qpn_prep_ptrs(char * base, const int64_t K, const int T, half ** xh, half ** xs, float ** xsc) {
    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    *xh  = (half  *) base;
    *xs  = (half  *) (base + sz_xh);
    *xsc = (float *) (base + sz_xh + sz_xs);
}

void ggml_cuda_qpn_prep(const float * x, const int64_t stride_col_x, const int64_t K, const int T, char * base, cudaStream_t stream) {
    half * xh; half * xs; float * xsc;
    ggml_cuda_qpn_prep_ptrs(base, K, T, &xh, &xs, &xsc);
    const int nsb = (int) (K/QK_K), nwarps = nsb*T;
    qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, stride_col_x, nsb, T, xh, xs, xsc);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_qpn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_qpn_is_repacked(src0) && src0->view_src == nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    // a 3-D input whose rows are one run is ne[1]*ne[2]*ne[3] columns (ggml_cuda_qpn_flat_cols)
    GGML_ASSERT(ggml_cuda_qpn_flat_cols(src1) && ggml_cuda_qpn_flat_cols(dst) && src1->ne[0] == src0->ne[0] && src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float) && dst->ne[0] == src0->ne[1] && ggml_nrows(dst) == ggml_nrows(src1));
    GGML_ASSERT(ggml_nrows(src1) <= GGML_CUDA_QPN_MAX_TOKENS);

    const int64_t K      = src0->ne[0];
    const int     nsb    = (int) (K/QK_K);
    const int64_t N      = src0->ne[1];
    const int     ntiles = (int) (N/32);
    const int     ncols  = (int) ggml_nrows(src1);
    const bool    mins   = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K;
    cudaStream_t  stream = ctx.stream();

    const int64_t stride_col_x   = src1->nb[1]/sizeof(float);
    const int64_t stride_col_dst = dst->nb[1]/sizeof(float);

    // with --split-mode tensor, dst is the partial of the ALLREDUCE node the graph loop names, so each finished tile goes to
    // the peer as the product writes it, and that node only polls and sums (ggml_cuda_p2p_ar_push_target says whether it can)
    ggml_cuda_p2p_push push;
    // (one pass only, so the call pushes all of its partial or none of it)
    const bool pushing = ctx.p2p_push_ar != nullptr && ncols <= ggml_cuda_qpn_pass_width() && ctx.p2p_push_ar->data == dst->data && ggml_is_contiguous(dst) &&
        stride_col_dst == N && ggml_nelements(ctx.p2p_push_ar) == ggml_nelements(dst) && (uintptr_t) dst->data % 16 == 0 &&
        ggml_cuda_p2p_ar_push_target(ctx.p2p_push_ar, &push);

    // passes of 16 tokens (LLAMA_QPN_W16=0: of 8, as before), so 9 to 16 tokens read the weights once
    const int pw = ggml_cuda_qpn_pass_width();
    for (int col0 = 0; col0 < ncols; ) {
        const int T = std::min(pw, ncols - col0);
        // by the weight's key (a slice under the split takes the whole weight's split); P stays within the slice's K.
        // a pass of 9 to 16 tokens on a routed key runs as paired 8-token blocks (the 8-token kernel and split, unstaged);
        // routed by the same key, so a slice under the split pairs as its whole weight does on one card
        const int64_t kn = qpn_key_n(src0), kk = qpn_key_k(src0);
        qpn_cfg c = T > 8 ? ggml_cuda_qpn_config16(src0->type, kn, kk) : ggml_cuda_qpn_config(src0->type, kn, kk, T);
        int nh = 1;
        // not at exactly 16 tokens, where the kernel at T = 16 runs full (card 1, the slices, us: Q4_K down 85.2 paired ->
        // 73.0, Q4_K GDN out 28.2 -> 21.8)
        if (T > 8 && ggml_cuda_qpn_pair16(src0->type, kn, kk) && !(T == 16 && ggml_cuda_qpn_full_on())) {
            const qpn_cfg c8 = ggml_cuda_qpn_config(src0->type, kn, kk, 8);
            if (!c8.stage && 2*ntiles <= QPN_MAX_TILES) {
                c      = c8;
                c.mode = 3;
                nh     = 2;
            }
        }
        // the lookahead of 4 units (MODE 5) for the single products of blocks of at most 4 warps (the split is the key's, so the
        // outputs are bit-identical)
        if (T >= 5 && nh == 1 && !c.stage && c.S*c.R <= 4 && ggml_cuda_qpn_deep(src0->type, T)) {
            c.mode = 5;
        }
        c.P = std::max(1, std::min(c.P, nsb));
        const dim3 grid(nh*((ntiles + c.R - 1)/c.R), c.P), block(WARP_SIZE, c.S*c.R);
        const size_t smem = c.stage ? ggml_cuda_qpn_stage_bytes(nsb, c.P, T) : 0;
        const int NL = T > 8 ? 16 : T > 4 ? 8 : 4;

        size_t sz_xh, sz_xs, sz_xsc;
        ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
        const size_t sz_ws  = c.P > 1 ? GGML_PAD((size_t) ntiles*c.P*NL*WARP_SIZE*sizeof(float), 256) : 0;
        // a single pass may share its prepared input with the other products reading src1 (it then
        // always holds the per-32 sums, which only the K-quants with mins read)
        bool   ready  = false;
        char * shared = ncols <= pw ? ggml_cuda_qpn_share_buffer(src0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready) : nullptr;
        const size_t sz_scratch = (shared ? 0 : sz_xh + sz_xs + sz_xsc) + sz_ws;
        ggml_cuda_pool_alloc<char> scratch(ctx.pool());
        if (sz_scratch > 0) {
            scratch.alloc(sz_scratch);
        }
        char * prep = shared ? shared : scratch.get();
        qpn_args a;
        a.xh  = (half  *) prep;
        a.xs  = mins ? (half *) (prep + sz_xh) : nullptr;
        a.xsc = (float *) (prep + sz_xh + sz_xs);
        a.ws  = c.P > 1 ? (float *) (scratch.get() + (shared ? 0 : sz_xh + sz_xs + sz_xsc)) : nullptr;
        a.cnt = nullptr;
        if (c.P > 1) {
            GGML_ASSERT(qpn_counters[ctx.device] != nullptr && ntiles <= QPN_MAX_TILES && ctx.curr_stream_no < GGML_CUDA_MAX_STREAMS);
            a.cnt = qpn_counters[ctx.device] + (size_t) ctx.curr_stream_no*QPN_MAX_TILES;
        }
        a.W = (const char *) src0->data; a.nsb = nsb; a.ntiles = ntiles; a.S = c.S; a.P = c.P;
        a.x = (const float *) src1->data + col0*stride_col_x; a.stride_col_x = stride_col_x;
        a.dst = (float *) dst->data + col0*stride_col_dst;    a.stride_col_dst = stride_col_dst;
        a.nh  = nh;
        if (pushing) {
            a.push = push;
            ggml_cuda_p2p_push_advance(a.push, col0*stride_col_dst);
        }

        if (!ready) {
            const int nwarps = nsb*T;
            qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(a.x, stride_col_x, nsb, T, a.xh,
                shared ? (half *) (prep + sz_xh) : a.xs, a.xsc);
            CUDA_CHECK(cudaGetLastError());
        }
        // a check (off by default): LLAMA_QPN_FALLBACK_ALL marks every slice non-finite, so every column goes
        // through the fp32 fallback (qpn_fallback), whose dequantization is then compared with ggml's by the caller
        static const bool fallback_all = getenv("LLAMA_QPN_FALLBACK_ALL") != nullptr;
        if (fallback_all) {
            CUDA_CHECK(cudaMemsetAsync(a.xsc, 0, sz_xsc, stream));
        }

        bool pushed = false;
        switch (src0->type) {
            case GGML_TYPE_Q4_K: pushed = ggml_cuda_qpn_launch<GGML_TYPE_Q4_K>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_Q5_K: pushed = ggml_cuda_qpn_launch<GGML_TYPE_Q5_K>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_Q6_K: pushed = ggml_cuda_qpn_launch<GGML_TYPE_Q6_K>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_IQ4_XS: pushed = ggml_cuda_qpn_launch<GGML_TYPE_IQ4_XS>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_IQ4_NL: pushed = ggml_cuda_qpn_launch<GGML_TYPE_IQ4_NL>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_Q3_K:   pushed = ggml_cuda_qpn_launch<GGML_TYPE_Q3_K>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            case GGML_TYPE_Q8_0:   pushed = ggml_cuda_qpn_launch<GGML_TYPE_Q8_0>(T, c.stage, c.mode, grid, block, smem, stream, a); break;
            default: GGML_ABORT("fatal error");
        }
        if (pushed) {
            ctx.p2p_pushed = ctx.p2p_push_ar;
        }
        CUDA_CHECK(cudaGetLastError());
        col0 += T;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// two products of one input in one launch (ggml-cuda.cu plans the pairs)

static constexpr int qpn_type_idx(const ggml_type t) {
    return t == GGML_TYPE_Q4_K ? 0 : t == GGML_TYPE_Q5_K ? 1 : t == GGML_TYPE_Q6_K ? 2 : t == GGML_TYPE_IQ4_XS ? 3 : t == GGML_TYPE_IQ4_NL ? 4 :
           t == GGML_TYPE_Q3_K ? 5 : -1; // Q3_K (the target's Q3_K ffn gate/up pair with IQ4_XS); Q8_0 is in no configured pair
}

// the pair's split, shared by both products (one block shape): S, R and staging; P = 1. Per (N0 + N1, K), the best of a
// sweep at 4 tokens; LLAMA_QPN_GROUP_CFG="S,R,stage" overrides it, for the sweep (at 5 to 8 tokens "S,R,mode").
struct qpn_group_shape_cfg { int64_t n0, n1, k; int S, R, stage; };
static const qpn_group_shape_cfg qpn_group_cfgs[] = {
    { 12288,  1024, 5120, 4, 2, 0 }, // attn q + gate with k, when v is not repacked (q alone: 4,2, unstaged)
    { 10240,  6144, 5120, 3, 2, 0 }, // GDN qkv + z: Q5_K 81.7 -> 79.7 us (staged 2,4: 77.0), Q4_K 65.9 -> 61.6 (staged 63.4); no staging
    { 17408, 17408, 5120, 2, 4, 1 }, // ffn up + gate, staged as each alone (so bit-identical to two launches): Q5_K + Q4_K 137.4 -> 132.2
};                                   //   (unstaged at 3,2: 134.3, so staging still wins)

// at 5 to 8 tokens, the split of each product alone at those widths (qpn_cfgs_0095), so a pair is bit-identical to
// two launches; stage = 0 always; mode = blocks of 8 warps per SM (2: 128 registers, 3: 80), as the products alone
static const qpn_group_shape_cfg qpn_group_cfgs_0095[] = {
    { 10240,  6144, 5120, 4, 1, 2 }, // GDN qkv + z (alone: 4,1 at mode 3 and 4,1 at mode 2)
    { 17408, 17408, 5120, 2, 4, 2 }, // ffn up + gate (alone: 2,4 at mode 2)
    { 12288,  1024, 5120, 4, 2, 3 }, // attn q + gate with k, when v is not repacked (q alone: 4,2 at mode 3)
};

static bool ggml_cuda_qpn_group_config(const int64_t n0, const int64_t n1, const int64_t k, const int T, int & S, int & R, int & stage) {
    bool found = false;
    const qpn_group_shape_cfg * tab = T > 4 ? qpn_group_cfgs_0095 : qpn_group_cfgs;
    const size_t ntab = T > 4 ? sizeof(qpn_group_cfgs_0095)/sizeof(qpn_group_cfgs_0095[0]) : sizeof(qpn_group_cfgs)/sizeof(qpn_group_cfgs[0]);
    for (size_t i = 0; i < ntab; ++i) {
        const qpn_group_shape_cfg & c = tab[i];
        if (((c.n0 == n0 && c.n1 == n1) || (c.n0 == n1 && c.n1 == n0)) && c.k == k) {
            S = c.S; R = c.R; stage = c.stage; // at T > 4: stage holds the mode
            found = true;
        }
    }
    if (const char * e = getenv("LLAMA_QPN_GROUP_CFG")) {
        sscanf(e, "%d,%d,%d", &S, &R, &stage);
        found = true;
    }
    if (!found) {
        return false;
    }
    S = std::max(1, std::min(S, QPN_MAX_WARPS));
    R = std::max(1, std::min(R, QPN_MAX_WARPS / S));
    return true;
}

// the draft step's (one token) q + gate with k, or with v: Q6_K, 12,288 and 1,024 rows
static bool ggml_cuda_qpn_draft_pair_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * src1) {
    const ggml_tensor * q = qpn_key_n(a) > qpn_key_n(b) ? a : b, * kv = qpn_key_n(a) > qpn_key_n(b) ? b : a;
    return src1->ne[1] == 1 && ggml_cuda_qpn_draft_kv_on() && (ggml_cuda_qpn_wide_mask() & 8) && q->type == GGML_TYPE_Q6_K && kv->type == GGML_TYPE_Q6_K &&
        qpn_key_n(q) == 12288 && qpn_key_n(kv) == 1024;
}

bool ggml_cuda_qpn_group_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    // the GDN pair (in-proj + z) only at 5 to 8 tokens, where both products' own split is the pair's (S = 4, R = 1): at 3 and 4 tokens the
    // pair's split (3, 2) is not either product's own (4, 1 and 8, 1), so it would not be bit-identical to two launches
    if (T < 5 && ((qpn_key_n(a) == 10240 && qpn_key_n(b) == 6144) || (qpn_key_n(a) == 6144 && qpn_key_n(b) == 10240))) {
        return false;
    }
    // LLAMA_QPN_PAIR_SLICE12 (default on): under --split-mode tensor the ffn gate and up slices (68 blocks each, fewer than the
    // SMs) pair at 1 and 2 tokens too (the MTP draft step), at qpn_group_cfgs' split, which is each alone's (so bit-identical to two launches)
    static const bool slice12 = [] { const char * e = getenv("LLAMA_QPN_PAIR_SLICE12"); return e == nullptr || atoi(e) != 0; }();
    // (a draft's slices only, of one type: the pair kernel is instantiated at T 1 and 2 for pairs of one type, and the target's decodes
    // of 1 or 2 tokens stay on two launches, as before)
    const bool pair12 = slice12 && T <= 2 && a->type == b->type && qpn_is_draft_slice(a) && qpn_is_draft_slice(b) && qpn_key_n(a) == 17408 &&
        qpn_key_n(b) == 17408 && qpn_key_k(a) == 5120;
    // and 9 to 16 tokens (one pass of two B tiles), at the pair's 8-token split
    return enabled && (T == 3 || T == 4 || pair12 || (T >= 5 && T <= ggml_cuda_qpn_pass_width() && (ggml_cuda_qpn_wide_mask() & 5) == 5) || ggml_cuda_qpn_draft_pair_ok(a, b, src1)) && ggml_cuda_qpn_is_repacked(a) && ggml_cuda_qpn_is_repacked(b) &&
        a->view_src == nullptr && b->view_src == nullptr && qpn_type_idx(a->type) >= 0 && qpn_type_idx(b->type) >= 0 &&
        a->ne[0] == b->ne[0] && a->ne[0] == src1->ne[0] && qpn_key_k(a) == qpn_key_k(b) &&
        ggml_cuda_qpn_group_config(qpn_key_n(a), qpn_key_n(b), qpn_key_k(a), T, S, R, stage);
}

template <ggml_type t0, ggml_type t1, int T, bool STAGE, int MODE = 3>
static void ggml_cuda_qpn2_launch_k(const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1,
        const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    if constexpr (STAGE) {
        static bool done[GGML_CUDA_MAX_DEVICES] = { false };
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        if (!done[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(qpn_mul2_kernel<t0, t1, T, true, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout,
                (int) cudaSharedmemCarveoutMaxShared));
            done[device] = true;
        }
    }
    qpn_mul2_kernel<t0, t1, T, STAGE, MODE><<<grid, block, STAGE ? smem : 0, stream>>>(g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
}

template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn2_launch_tt(const int T, const int stage, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream,
        const qpn_seg & g0, const qpn_seg & g1, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        if (T == 1 || T == 2) {
            // the draft's Q6_K q + k or v (ggml_cuda_qpn_draft_pair_ok), unstaged; a pair of slices of one type
            // under --split-mode tensor (the ffn gate and up of the MTP draft step), at its split, staged or not
            if constexpr (t0 == t1) {
                if (T == 1) {
                    stage ? ggml_cuda_qpn2_launch_k<t0, t1, 1, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                          : ggml_cuda_qpn2_launch_k<t0, t1, 1, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
                } else {
                    stage ? ggml_cuda_qpn2_launch_k<t0, t1, 2, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                          : ggml_cuda_qpn2_launch_k<t0, t1, 2, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if (T == 3) {
            stage ? ggml_cuda_qpn2_launch_k<t0, t1, 3, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                  : ggml_cuda_qpn2_launch_k<t0, t1, 3, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
        } else if (T == 4) {
            stage ? ggml_cuda_qpn2_launch_k<t0, t1, 4, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                  : ggml_cuda_qpn2_launch_k<t0, t1, 4, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
        } else if (T <= 8) { // 5 to 8 tokens, one kernel at T = 8, unstaged, at 2 or 3 blocks of 8 warps per SM (stage holds it)
            if (T == 8 && ggml_cuda_qpn_full_on()) {
                stage == 2 ? ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 2 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                           : ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 3 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            } else {
                stage == 2 ? ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 2>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                           : ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 3>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            }
        } else { // 9 to 16 tokens, one kernel at T = 16 (two B tiles), unstaged, at 2 blocks of 8 warps per SM
            if (T == 16 && ggml_cuda_qpn_full_on()) {
                ggml_cuda_qpn2_launch_k<t0, t1, 16, false, 2 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            } else {
                ggml_cuda_qpn2_launch_k<t0, t1, 16, false, 2>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            }
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

template <ggml_type t0>
static void ggml_cuda_qpn2_launch_t(const ggml_type t1, const int T, const int stage, const dim3 grid, const dim3 block, const size_t smem,
        cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1, const int nb0, const int nsb, const int S, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q4_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q5_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q6_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_IQ4_XS>(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_IQ4_NL>(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q3_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q3_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_qpn2(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[2], const ggml_tensor * src1, ggml_tensor * const dsts[2]) {
    GGML_ASSERT(ggml_cuda_qpn_group_ok(src0s[0], src0s[1], src1));
    // the lower type index first (the instantiated order); the outputs follow their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1 }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K ||
                                                         w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage; // at T > 4, stage holds the mode (2 or 3); nothing is staged
    ggml_cuda_qpn_group_config(qpn_key_n(w0), qpn_key_n(w1), qpn_key_k(w0), T, S, R, stage); // by the weights' keys
    if (T <= 4 && ggml_cuda_qpn_stage_bytes(nsb, 1, T) > QPN_STAGE_MAX) {
        stage = 0;
    }
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R;
    const dim3 grid(nb0 + nb1, 1), block(WARP_SIZE, S*R);
    const size_t smem = T <= 4 && stage ? ggml_cuda_qpn_stage_bytes(nsb, 1, T) : 0;

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0 = { (const char *) w0->data, nt0, (float *) d0->data, (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1 = { (const char *) w1->data, nt1, (float *) d1->data, (int64_t) (d1->nb[1]/sizeof(float)) };
    const half * xsp = mins ? xs : nullptr;
    switch (w0->type) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q4_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q5_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q6_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2_launch_t<GGML_TYPE_IQ4_XS>(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2_launch_t<GGML_TYPE_IQ4_NL>(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_Q3_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q3_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------------------------
// three products of one input in one launch (the attention input q + gate, k, v; ggml-cuda.cu plans them)

// the instantiated types: product 0 (q + gate) Q4_K, Q5_K or IQ4_XS; products 1 and 2 (k, v) Q4_K, Q5_K, Q6_K or Q8_0,
// in the order of qpn3_idx
static constexpr bool qpn3_t0_ok(const ggml_type t) { return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; }
static constexpr int  qpn3_idx(const ggml_type t) {
    return t == GGML_TYPE_Q4_K ? 0 : t == GGML_TYPE_Q5_K ? 1 : t == GGML_TYPE_Q6_K ? 2 : t == GGML_TYPE_Q8_0 ? 3 : -1;
}
static constexpr bool qpn3_t1_ok(const ggml_type t) { return qpn3_idx(t) >= 0; }

// the triple's split: q's own (S, R: qpn_cfgs_0091 at 4 tokens, qpn_cfgs_0095 at 5 to 8), so q's outputs are bit-identical to its own
// launch (IQ4_XS's at 5 to 8 tokens excepted, whose own launch is the whole-superblock kernel at S = 2); k and v at split-K S12 in
// blocks of the same 8 warps; mode = blocks of 8 warps per SM
struct qpn_group3_cfg { int64_t n0, n1, n2, k; int S, R, S12, mode; };
static const qpn_group3_cfg qpn_group3_cfgs[] = {
    { 12288, 1024, 1024, 5120, 4, 2, 8, 3 }, // k, v at split-K 8, one tile a block, first: per layer, against q alone with k and v on dp4a,
};                                           // 79.5 -> 68.4 us at 4 tokens, 99.2 -> 83.5 at 8 (S12 = 4: 81.9, 89.4; 2: 76.4, 85.7)

// the draft step's (one token) q + gate, k, v, all Q6_K: q's split (4, 2), k and v at split-K 8 (qpn_group3_cfgs)
static bool ggml_cuda_qpn_draft3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    if (!enabled || src1->ne[1] != 1 || !ggml_cuda_qpn_draft_kv_on() || !(ggml_cuda_qpn_wide_mask() & 8) ||
            a->type != GGML_TYPE_Q6_K || b->type != GGML_TYPE_Q6_K || c->type != GGML_TYPE_Q6_K ||
            qpn_key_n(a) != 12288 || qpn_key_n(b) != 1024 || qpn_key_n(c) != 1024) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0] || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    return true;
}

// the GDN layer's in-proj (a), z (b) and merged alpha/beta (c, Q8_0, 96 rows), 5 to 8 tokens, on the pair's configured shape
// (5 to 16, one pass of two B tiles at 9 to 16)
static constexpr bool qpn_gdn_type_ok(const ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_IQ4_NL;
}
static bool ggml_cuda_qpn_gdn3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    if (!enabled || !ggml_cuda_qpn_gdn_ab_on() || T < 5 || T > ggml_cuda_qpn_pass_width() || (ggml_cuda_qpn_wide_mask() & 5) != 5 ||
            c->type != GGML_TYPE_Q8_0 || c->ne[1] != 96 || a->ne[1] != 10240 || b->ne[1] != 6144 ||
            !qpn_gdn_type_ok(a->type) || !qpn_gdn_type_ok(b->type)) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0]) {
            return false;
        }
    }
    return ggml_cuda_qpn_group_config(a->ne[1], b->ne[1], a->ne[0], T, S, R, stage) && stage == 2; // the kernel is instantiated at mode 2
}

// LLAMA_QPN_GDN_AB (default on; 0 = none): under --split-mode tensor alpha and beta stay upstream's two Q8_0 weights, 48 x 5,120, each
// split into a 24-row slice a device. A slice keeps its GGUF layout for every reader it had (the dp4a products, the prompt path's fp16
// expansions); at load it also gets a private copy, padded to one 32-row tile with rows 24 to 31 zero and repacked, which only the GDN
// layer's in-proj + z launch reads (ggml_cuda_mul_mat_qpn4ab). The slice's buffer holds exactly its 24 rows, so the padded tile cannot be in
// place. Copies live in 2 MiB device chunks for the process's life (as qpn_counters); written at load only.
// (the entry points ggml-cuda.cu calls, declared there and here)
bool ggml_cuda_qpn_ab_split_on();
bool ggml_cuda_qpn_ab_slice_copy(const ggml_tensor * slice, int device, int64_t n, int64_t k);
bool ggml_cuda_qpn_has_ab_copy(const ggml_tensor * t);
bool ggml_cuda_qpn_gdn4_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * d, const ggml_tensor * src1);
void ggml_cuda_mul_mat_qpn4ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], const ggml_tensor * src1, ggml_tensor * const dsts[4]);

struct qpn_ab_copy { const char * W; int nrow; };
static std::unordered_map<const void *, qpn_ab_copy> qpn_ab_copies;

bool ggml_cuda_qpn_ab_split_on() {
    return ggml_cuda_qpn_gdn_ab_on();
}

bool ggml_cuda_qpn_ab_slice_copy(const ggml_tensor * slice, const int device, const int64_t n, const int64_t k) {
    if (!ggml_cuda_qpn_gdn_ab_on() || slice->type != GGML_TYPE_Q8_0 || slice->view_src != nullptr || !ggml_is_contiguous(slice) ||
            slice->ne[1] < 1 || slice->ne[1] >= 32 || slice->ne[2] != 1 || slice->ne[3] != 1 || slice->ne[0] % QK_K != 0) {
        return false;
    }
    ggml_tensor pad = *slice; // the padded tile, as the repack sees it
    pad.ne[1]    = 32;
    pad.nb[2]    = pad.nb[3] = 32*pad.nb[1];
    pad.view_src = nullptr;
    pad.flags    = 0;
    if (!ggml_cuda_qpn_takes(&pad, ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    static char * chunk[GGML_CUDA_MAX_DEVICES] = { nullptr };
    static size_t used[GGML_CUDA_MAX_DEVICES]  = { 0 };
    const size_t chunk_size = 2*1024*1024, size = ggml_nbytes(&pad);
    GGML_ASSERT(size <= chunk_size);
    ggml_cuda_set_device(device);
    if (chunk[device] == nullptr || used[device] + size > chunk_size) {
        CUDA_CHECK(cudaMalloc((void **) &chunk[device], chunk_size));
        used[device] = 0;
    }
    pad.data = chunk[device] + used[device];
    used[device] += size;
    CUDA_CHECK(cudaMemset(pad.data, 0, size)); // Q8_0 zero blocks (d = 0): rows 24 to 31 give 0
    CUDA_CHECK(cudaMemcpy(pad.data, slice->data, ggml_nbytes(slice), cudaMemcpyDeviceToDevice));
    ggml_cuda_qpn_repack(&pad, device);
    qpn_ab_copies[slice->data] = { (const char *) pad.data, (int) slice->ne[1] };
    ggml_cuda_qpn_set_slice_key(slice, n, k); // the record of the whole weight's key, read here only (the slice itself is not repacked)
    return true;
}

bool ggml_cuda_qpn_has_ab_copy(const ggml_tensor * t) {
    return !qpn_ab_copies.empty() && t != nullptr && qpn_ab_copies.count(t->data) != 0;
}

// in-proj (a, key 10,240) and z (b, key 6,144) as the one-card triple takes them, with two Q8_0 slices that have a private copy and the
// whole-weight key 48 x 5,120 (c, d: beta and alpha), one input of 5 to 16 tokens
bool ggml_cuda_qpn_gdn4_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * d, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    if (!enabled || !ggml_cuda_qpn_gdn_ab_on() || qpn_ab_copies.empty() || T < 5 || T > ggml_cuda_qpn_pass_width() || (ggml_cuda_qpn_wide_mask() & 5) != 5 ||
            src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            qpn_key_n(a) != 10240 || qpn_key_n(b) != 6144 || !qpn_gdn_type_ok(a->type) || !qpn_gdn_type_ok(b->type)) {
        return false;
    }
    for (const ggml_tensor * w : { a, b }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0] || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    for (const ggml_tensor * w : { c, d }) {
        if (!ggml_cuda_qpn_has_ab_copy(w) || ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->type != GGML_TYPE_Q8_0 ||
                w->ne[0] != src1->ne[0] || qpn_key_n(w) != 48 || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    return ggml_cuda_qpn_group_config(qpn_key_n(a), qpn_key_n(b), qpn_key_k(a), T, S, R, stage) && stage == 2; // the kernel is instantiated at mode 2
}

bool ggml_cuda_qpn_group3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    if (ggml_cuda_qpn_gdn3_ok(a, b, c, src1) || ggml_cuda_qpn_draft3_ok(a, b, c, src1)) {
        return true;
    }
    // and 9 to 16 tokens, at the triple's 8-token split
    if (!enabled || !(ggml_cuda_qpn_wide_mask() & 8) || !(T == 4 || (T >= 5 && T <= ggml_cuda_qpn_pass_width() && (ggml_cuda_qpn_wide_mask() & 1)))) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0]) {
            return false;
        }
    }
    if (!qpn3_t0_ok(a->type) || !qpn3_t1_ok(b->type) || !qpn3_t1_ok(c->type)) {
        return false;
    }
    if (qpn_key_k(a) != qpn_key_k(b) || qpn_key_k(a) != qpn_key_k(c)) {
        return false;
    }
    for (const qpn_group3_cfg & g : qpn_group3_cfgs) { // by the weights' keys
        if (g.n0 == qpn_key_n(a) && g.n1 == qpn_key_n(b) && g.n2 == qpn_key_n(c) && g.k == qpn_key_k(a)) {
            return true;
        }
    }
    return false;
}

template <ggml_type t0, ggml_type t1, ggml_type t2>
static void ggml_cuda_qpn3_launch_ttt(const int T, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1,
        const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx) {
    if constexpr (qpn3_idx(t1) <= qpn3_idx(t2)) {
        if (T == 4) {
            qpn_mul3_kernel<t0, t1, t2, 4, 3><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
        } else if (T <= 8) {
            if (T == 8 && ggml_cuda_qpn_full_on()) {
                qpn_mul3_kernel<t0, t1, t2, 8, 3 + QPN_FULL><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            } else {
                qpn_mul3_kernel<t0, t1, t2, 8, 3><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            }
        } else { // 9 to 16 tokens, two B tiles, at 2 blocks of 8 warps per SM
            if (T == 16 && ggml_cuda_qpn_full_on()) {
                qpn_mul3_kernel<t0, t1, t2, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            } else {
                qpn_mul3_kernel<t0, t1, t2, 16, 2><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            }
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders k and v by type
    }
}

template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn3_launch_tt(const ggml_type t2, const int T, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & g0,
        const qpn_seg & g1, const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh, const half * xs,
        const float * xsc, const float * x, const int64_t sx) {
    switch (t2) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q4_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q5_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q6_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q8_0: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q8_0>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

template <ggml_type t0>
static void ggml_cuda_qpn3_launch_t(const ggml_type t1, const ggml_type t2, const int T, const dim3 grid, const dim3 block, cudaStream_t stream,
        const qpn_seg & g0, const qpn_seg & g1, const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx) {
    switch (t1) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q4_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q5_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q6_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q8_0: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q8_0>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_mul_mat_qpn2ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]);

void ggml_cuda_mul_mat_qpn3(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]) {
    GGML_ASSERT(ggml_cuda_qpn_group3_ok(src0s[0], src0s[1], src0s[2], src1));
    if (ggml_cuda_qpn_gdn3_ok(src0s[0], src0s[1], src0s[2], src1)) {
        ggml_cuda_mul_mat_qpn2ab(ctx, src0s, src1, dsts);
        return;
    }
    // products 1 and 2 in type order (the instantiated order); the outputs follow their weights
    const int i1 = qpn3_idx(src0s[1]->type) <= qpn3_idx(src0s[2]->type) ? 1 : 2;
    const ggml_tensor * w[3] = { src0s[0], src0s[i1], src0s[3 - i1] };
    ggml_tensor       * d[3] = { dsts[0],  dsts[i1],  dsts[3 - i1] };
    for (const ggml_tensor * t : d) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w[0]->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    bool mins = false;
    for (const ggml_tensor * t : w) {
        mins = mins || t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K;
    }
    cudaStream_t stream = ctx.stream();
    int S = 4, R = 2, S12 = 4;
    for (const qpn_group3_cfg & g : qpn_group3_cfgs) { // by the weights' keys
        if (g.n0 == qpn_key_n(w[0]) && g.n1 == qpn_key_n(w[1]) && g.n2 == qpn_key_n(w[2]) && g.k == qpn_key_k(w[0])) {
            S = g.S; R = g.R; S12 = g.S12;
        }
    }
    if (const char * e = getenv("LLAMA_QPN_GROUP3_CFG")) { // "S,R,S12", for the sweep
        sscanf(e, "%d,%d,%d", &S, &R, &S12);
    }
    const int R12 = std::max(1, S*R/S12);
    const int nt[3] = { (int) (w[0]->ne[1]/32), (int) (w[1]->ne[1]/32), (int) (w[2]->ne[1]/32) };
    const int nb[3] = { (nt[0] + R - 1)/R, (nt[1] + R12 - 1)/R12, (nt[2] + R12 - 1)/R12 };
    const dim3 grid(nb[0] + nb[1] + nb[2], 1), block(WARP_SIZE, S*R);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w[0], src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    qpn_seg g[3];
    for (int i = 0; i < 3; ++i) {
        g[i] = { (const char *) w[i]->data, nt[i], (float *) d[i]->data, (int64_t) (d[i]->nb[1]/sizeof(float)) };
    }
    const half * xsp = mins ? xs : nullptr;
    switch (w[0]->type) {
        case GGML_TYPE_Q6_K: // the draft step's Q6_K triple (ggml_cuda_qpn_draft3_ok), one token
            qpn_mul3_kernel<GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 1, 3><<<grid, block, 0, stream>>>(g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx, T);
            break;
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn3_launch_t<GGML_TYPE_Q4_K  >(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn3_launch_t<GGML_TYPE_Q5_K  >(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn3_launch_t<GGML_TYPE_IQ4_XS>(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}


// the GDN layer's in-proj + z + merged alpha/beta as one launch (ggml_cuda_qpn_gdn3_ok), 5 to 8 tokens (to 16), the pair's split and mode
template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn2ab_launch_tt(const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gab, const qpn_seg & g0, const qpn_seg & g1,
        const int nbab, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x,
        const int64_t sx, const int nt) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        const bool full = (nt == 8 || nt == 16) && ggml_cuda_qpn_full_on();
        if (nt <= 8) {
            full ? qpn_mul2ab_kernel<t0, t1, 8, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul2ab_kernel<t0, t1, 8, 2><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        } else { // 9 to 16 tokens, two B tiles
            full ? qpn_mul2ab_kernel<t0, t1, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul2ab_kernel<t0, t1, 16, 2><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

template <ggml_type t0>
static void ggml_cuda_qpn2ab_launch_t(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gab, const qpn_seg & g0,
        const qpn_seg & g1, const int nbab, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx, const int nt) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q4_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q5_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q6_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_IQ4_XS>(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_IQ4_NL>(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        default: GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_mul_mat_qpn2ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]) {
    // src0s = { in-proj, z, ab }; the pair in type order (the instantiated order), the outputs following their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0], * wab = src0s[2];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0],  * dab = dsts[2];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1, (const ggml_tensor *) dab }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K || w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage;
    ggml_cuda_qpn_group_config(w0->ne[1], w1->ne[1], K, T, S, R, stage);
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32), ntab = (int) (wab->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R, nbab = (ntab + R - 1)/R;
    const dim3 grid(nbab + nb0 + nb1, 1), block(WARP_SIZE, S*R);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0  = { (const char *) w0->data,  nt0, (float *) d0->data,  (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1  = { (const char *) w1->data,  nt1, (float *) d1->data,  (int64_t) (d1->nb[1]/sizeof(float)) };
    const qpn_seg gab = { (const char *) wab->data, ntab, (float *) dab->data, (int64_t) (dab->nb[1]/sizeof(float)) };
    const half * xsp = mins ? xs : nullptr;
    switch (w0->type) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q4_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q5_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q6_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2ab_launch_t<GGML_TYPE_IQ4_XS>(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2ab_launch_t<GGML_TYPE_IQ4_NL>(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

// in-proj, z, beta and alpha under --split-mode tensor as one launch (ggml_cuda_qpn_gdn4_ok), 5 to 16 tokens, the pair's split and mode
template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn4ab_launch_tt(const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gb, const qpn_seg & ga, const int nrowb,
        const int nrowa, const qpn_seg & g0, const qpn_seg & g1, const int nbb, const int nba, const int nb0, const int nsb, const int S, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        const bool full = (nt == 8 || nt == 16) && ggml_cuda_qpn_full_on();
        if (nt <= 8) {
            full ? qpn_mul4ab_kernel<t0, t1, 8, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul4ab_kernel<t0, t1, 8, 2><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        } else { // 9 to 16 tokens, two B tiles
            full ? qpn_mul4ab_kernel<t0, t1, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul4ab_kernel<t0, t1, 16, 2><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

template <ggml_type t0>
static void ggml_cuda_qpn4ab_launch_t(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gb, const qpn_seg & ga,
        const int nrowb, const int nrowa, const qpn_seg & g0, const qpn_seg & g1, const int nbb, const int nba, const int nb0, const int nsb, const int S,
        const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q4_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q5_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q6_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_IQ4_XS>(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_IQ4_NL>(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        default: GGML_ABORT("fatal error");
    }
}

// src0s = { in-proj, z, beta, alpha }: beta's and alpha's padded repacked private copies as the two one-tile Q8_0 members, each storing its
// slice's rows into its own output; in-proj and z as ggml_cuda_mul_mat_qpn2ab runs them, at the split of their whole weights' keys
void ggml_cuda_mul_mat_qpn4ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], const ggml_tensor * src1, ggml_tensor * const dsts[4]) {
    GGML_ASSERT(ggml_cuda_qpn_gdn4_ok(src0s[0], src0s[1], src0s[2], src0s[3], src1));
    // the pair in type order (the instantiated order), the outputs following their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1 }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K || w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage;
    ggml_cuda_qpn_group_config(qpn_key_n(w0), qpn_key_n(w1), qpn_key_k(w0), T, S, R, stage); // by the whole weights' keys
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R, nbm = (1 + R - 1)/R; // nbm: a one-tile member's blocks
    const dim3 grid(2*nbm + nb0 + nb1, 1), block(WARP_SIZE, S*R);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0 = { (const char *) w0->data, nt0, (float *) d0->data, (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1 = { (const char *) w1->data, nt1, (float *) d1->data, (int64_t) (d1->nb[1]/sizeof(float)) };
    qpn_seg gm[2];
    int     nrow[2];
    for (int i = 0; i < 2; ++i) { // beta, alpha
        const ggml_tensor * w = src0s[2 + i];
        ggml_tensor       * d = dsts[2 + i];
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[0] == w->ne[1] && d->ne[1] == src1->ne[1]);
        const qpn_ab_copy & c = qpn_ab_copies.at(w->data);
        gm[i]   = { c.W, 1, (float *) d->data, (int64_t) (d->nb[1]/sizeof(float)) };
        nrow[i] = c.nrow;
    }
    const half * xsp = mins ? xs : nullptr;
    switch (w0->type) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q4_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q5_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q6_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn4ab_launch_t<GGML_TYPE_IQ4_XS>(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn4ab_launch_t<GGML_TYPE_IQ4_NL>(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
