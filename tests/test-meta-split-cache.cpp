// the meta backend's split-state cache (ggml-backend-meta.cpp) over two CPU devices, without a model:
//
//   deep: a graph rebuilt in the same memory whose first difference is a node deep in it (here the scale of node 2,990 of a
//         3,000-node chain; in llama, for example, the sampling nodes at the end of a decode graph). The allocation's walk
//         initialises every tensor in order; the first changed node used to clear the buffer's whole cache, so its split state
//         was computed again by recursing through every ancestor, about 20 KB of stack a level: a few hundred levels overflow
//         8 MB. Expected: no crash, and every tensor's per-device slices as a fresh allocation makes them.
//   stale: a node whose own bytes are unchanged while its source, at the same address, became a view of a differently split
//         weight. The cache used to compare only the node's own bytes, so it kept the old split. Expected: the node's
//         per-device slices follow its source's split.
//
// test-meta-split-cache [n_chain]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

// the static weights' split: "w_split" along dim 0 in halves, everything else mirrored
static ggml_backend_meta_split_state test_split_state(const ggml_tensor * t, void * ud) {
    (void) ud;
    ggml_backend_meta_split_state ss = {};
    ss.n_segments = 1;
    ss.nr[0] = 1;
    if (strcmp(t->name, "w_split") == 0) {
        ss.axis  = GGML_BACKEND_SPLIT_AXIS_0;
        ss.ne[0] = t->ne[0]/2;
        ss.ne[1] = t->ne[0] - t->ne[0]/2;
    } else {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return ss;
}

// a graph in caller-owned memory, so a rebuild puts every tensor at the same address, as llama's graph results do
struct graph_mem {
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    explicit graph_mem(size_t n_tensors) : buf(n_tensors*ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false) + 4096) {}
    ggml_context * fresh() {
        if (ctx) {
            ggml_free(ctx);
        }
        ggml_init_params p = { buf.size(), buf.data(), /*no_alloc =*/ true };
        ctx = ggml_init(p);
        return ctx;
    }
    ~graph_mem() { if (ctx) ggml_free(ctx); }
};

// a chain of n scales on a mirrored input; node k's scale is s_k
static ggml_cgraph * build_chain(ggml_context * ctx, int n, int k, float s_k, ggml_tensor ** out) {
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 16);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * t = x;
    for (int i = 0; i < n; i++) {
        t = ggml_scale(ctx, t, i == k ? s_k : 1.0f);
        ggml_format_name(t, "s%d", i);
    }
    ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n + 16, false);
    ggml_build_forward_expand(gf, t);
    *out = t;
    return gf;
}

int main(int argc, char ** argv) {
    const int n_chain = argc > 1 ? atoi(argv[1]) : 3000;

    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }
    ggml_backend_dev_t devs[2] = { cpu, cpu };
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_split_state, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta);

    // deep: build, allocate, rebuild in the same memory with node n_chain - 10 changed, allocate again
    {
        graph_mem gm(n_chain + 64);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        ggml_tensor * out = nullptr;

        ggml_cgraph * gf = build_chain(gm.fresh(), n_chain, n_chain - 10, 1.0f, &out);
        CHECK(ggml_gallocr_alloc_graph(galloc, gf), "deep: first allocation");

        gf = build_chain(gm.fresh(), n_chain, n_chain - 10, 2.0f, &out);
        CHECK(ggml_gallocr_alloc_graph(galloc, gf), "deep: second allocation");

        // the output can be written and read through the meta buffer (a split-state query on the deepest node)
        std::vector<float> v(ggml_nelements(out), 1.5f), r(ggml_nelements(out), 0.0f);
        ggml_backend_tensor_set(out, v.data(), 0, ggml_nbytes(out));
        ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
        CHECK(memcmp(v.data(), r.data(), ggml_nbytes(out)) == 0, "deep: output round trip");
        for (int j = 0; j < 2; j++) {
            ggml_tensor * s = ggml_backend_meta_tensor_simple(out, j);
            CHECK(s != nullptr && s->ne[0] == 64 && s->ne[1] == 16, "deep: mirrored output slice on device %d", j);
        }
        // the host cost of allocating the same graph again (every entry holds): microseconds per allocation
        if (getenv("META_SPLIT_CACHE_BENCH") != nullptr) {
            const int n_rep = 50;
            const int64_t t0 = ggml_time_us();
            for (int r = 0; r < n_rep; r++) {
                gf = build_chain(gm.fresh(), n_chain, n_chain - 10, 2.0f, &out);
                ggml_gallocr_alloc_graph(galloc, gf);
            }
            printf("bench: build + allocate a %d-node graph again: %.1f us\n", n_chain, (ggml_time_us() - t0)/(double) n_rep);
        }
        ggml_gallocr_free(galloc);
        printf("deep (chain of %d, node %d changed): done\n", n_chain, n_chain - 10);
    }

    // stale: n = scale(v), v a view of the mirrored weight, then (same addresses) of the split one
    {
        ggml_init_params pw = { 4*ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
        ggml_context * ctx_w = ggml_init(pw);
        ggml_tensor * w_mir = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 64);
        ggml_set_name(w_mir, "w_mirrored");
        ggml_tensor * w_spl = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 64);
        ggml_set_name(w_spl, "w_split");
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
        CHECK(buf_w != nullptr, "stale: weights buffer");
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        graph_mem gm(64);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        for (int pass = 0; pass < 2; pass++) {
            ggml_context * ctx = gm.fresh();
            ggml_tensor * v = ggml_view_1d(ctx, pass == 0 ? w_mir : w_spl, 64, 0);
            ggml_set_name(v, "v");
            ggml_tensor * n = ggml_scale(ctx, v, 2.0f);
            ggml_set_name(n, "n");
            ggml_set_output(n);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
            ggml_build_forward_expand(gf, n);
            CHECK(ggml_gallocr_alloc_graph(galloc, gf), "stale: allocation %d", pass);
            const int64_t want0 = pass == 0 ? 64 : 32;
            for (int j = 0; j < 2; j++) {
                ggml_tensor * vs = ggml_backend_meta_tensor_simple(v, j);
                ggml_tensor * ns = ggml_backend_meta_tensor_simple(n, j);
                CHECK(vs != nullptr && vs->ne[0] == want0, "stale: pass %d device %d: view slice %lld, want %lld", pass, j,
                    vs ? (long long) vs->ne[0] : -1LL, (long long) want0);
                CHECK(ns != nullptr && ns->ne[0] == want0, "stale: pass %d device %d: node slice %lld, want %lld (its source's split)", pass, j,
                    ns ? (long long) ns->ne[0] : -1LL, (long long) want0);
            }
        }
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
        printf("stale (a view of a mirrored, then of a split weight at the same address): done\n");
    }

    printf("%s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
