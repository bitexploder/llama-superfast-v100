#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// A dim-0 concat that builds a recurrent conv input, fused with the copies that save its tail
// columns into rollback slots: slot k receives columns [off[k], off[k] + n_cols) of every
// (channel, sequence) row, written as dst[k] + seq*nb1[k] + (channel*n_cols + col)*4.
#define GGML_CUDA_CONV_SLOTS_MAX 8

struct ggml_cuda_conv_slots {
    int     n_slots = 0;
    int64_t n_cols  = 0;
    char *  data[GGML_CUDA_CONV_SLOTS_MAX] = {};
    int64_t off[GGML_CUDA_CONV_SLOTS_MAX]  = {};
    size_t  nb1[GGML_CUDA_CONV_SLOTS_MAX]  = {};

    // LLAMA_GDN_FUSE_CONV: the depth-4 conv over each written row and its silu, as ssm_conv_f32
    // with a fused silu computes them, written to conv_y [channels, n_t, seq]; null leaves the conv to its node
    const float * conv_w     = nullptr; // [4, channels], rows conv_w_nb1 bytes apart
    size_t        conv_w_nb1 = 0;
    float *       conv_y     = nullptr;
    size_t        conv_y_nb1 = 0;       // bytes between tokens
    size_t        conv_y_nb2 = 0;       // bytes between sequences
    float         conv_b     = 0.0f;    // the bias ssm_conv adds when it has none

    // LLAMA_FOLD_CONV_GATHER: the concat's first source is a GET_ROWS of whole cache rows that was
    // not launched; sequence i2's row is read from gather_rows + ids[i2]*gather_nb1 instead, laid out as src0
    const char *    gather_rows = nullptr;
    const int32_t * gather_ids  = nullptr;
    size_t          gather_nb1  = 0;

    // false when the slot saves and the folded conv are the concat's only readers (register kernel only)
    bool write_dst = true;
};

void ggml_cuda_op_concat_conv_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_conv_slots & slots);

// whether ggml_cuda_op_concat_conv_slots runs this concat with every row in registers, one thread per channel for
// all its sequences (then a folded gather may read cache rows that another sequence's slot saves overwrite)
bool ggml_cuda_conv_slots_reg_ok(const ggml_tensor * dst, const ggml_cuda_conv_slots & slots);
