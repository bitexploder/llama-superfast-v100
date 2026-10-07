#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-rtimer.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ true, // one event per simple device, when they all have them
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

// an event of the meta device: one event per simple device. Recorded on the meta backend it is recorded on each simple
// backend's stream (the meta graph compute returns once every simple backend has queued its graph), waited for on each
static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    auto * evs = new std::vector<ggml_backend_event_t>();
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_event_t e = ggml_backend_event_new(simple_dev);
        if (e == nullptr) {
            for (ggml_backend_event_t f : *evs) {
                ggml_backend_event_free(f);
            }
            delete evs;
            return nullptr;
        }
        evs->push_back(e);
    }
    return new ggml_backend_event { /* .device = */ dev, /* .context = */ evs };
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * evs = (std::vector<ggml_backend_event_t> *) event->context;
    for (ggml_backend_event_t e : *evs) {
        ggml_backend_event_free(e);
    }
    delete evs;
    delete event;
    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    for (ggml_backend_event_t e : *(std::vector<ggml_backend_event_t> *) event->context) {
        ggml_backend_event_synchronize(e);
    }
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

// the arena whose graphs are being allocated and computed (ggml_backend_meta_set_arena): a context that keeps two
// graphs allocated side by side on two schedulers (llama's MTP draft: the catch-up and the draft step) gives each its own
// arena, so a rebuild of one graph rotates only its own arena's view containers and the other graph's views stay valid
static int ggml_backend_meta_arena_cur = 0;
static constexpr int GGML_BACKEND_META_N_ARENAS = 4;

void ggml_backend_meta_set_arena(int arena) {
    GGML_ASSERT(arena >= 0 && arena < GGML_BACKEND_META_N_ARENAS);
    ggml_backend_meta_arena_cur = arena;
}

int ggml_backend_meta_get_arena(void) {
    return ggml_backend_meta_arena_cur;
}

// a number unique to each buffer context. An entry's pointer into a source's entry is read only while the source
//     is still in the buffer it was in when the entry was computed: a buffer freed and allocated again (as the graph allocator
//     does when a graph outgrows its compute buffer) gets a new number, while the KV cache's views keep their entries
static std::atomic<uint64_t> ggml_backend_meta_buffer_uid{0};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    // one rotating pair per arena (ggml_backend_meta_set_arena); arena 0's pair is made with the buffer, the
    //     others' on first use, with the same context size
    struct arena_containers {
        ggml_backend_meta_simple_tensor_container stc_compute[2];
        int stc_compute_index      = 0;
        int stc_compute_index_next = 0;
        uint64_t gen[2] = {0, 0}; // how often each container was cleared: a cached graph's views are valid while it holds
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    std::vector<arena_containers> arenas;
    size_t compute_mem_size = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    // an entry holds for the tensor's bytes and for the entries of its sources it was computed from: each entry has
    //     a stamp of its own and the stamps of its sources' entries, so a source computed again (a graph rebuilt in the same
    //     memory, whose own bytes changed) makes every entry computed from it invalid. Before, only the tensor's own bytes were
    //     compared, so a node whose source changed kept its old split; and a mismatch cleared the whole map, after which a query
    //     recursed through every uncached ancestor (about 20 KB of stack a level: a 27B decode graph overflowed 8 MB)
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    // a hash map: the tree map's searches were a large part of each shape change's host time. Entries are
    // never erased, so the pointers to them that split_state_entry keeps stay valid as the map grows
    struct split_state_key_hash {
        size_t operator()(const std::pair<const ggml_tensor *, bool> & k) const {
            return std::hash<const void *>()(k.first) ^ (size_t) k.second;
        }
    };
    struct split_state_entry {
        ggml_backend_meta_split_state split_state;
        char                          bytes[nbtc];
        uint64_t                      stamp;
        uint64_t                      src_stamp[GGML_MAX_SRC];
        const split_state_entry *     src_entry[GGML_MAX_SRC]; // the entry each src_stamp came from
        uint64_t                      src_uid[GGML_MAX_SRC];   // the uid of the buffer context src_entry points into
    };
    uint64_t uid = ++ggml_backend_meta_buffer_uid; // see ggml_backend_meta_buffer_uid
    std::unordered_map<std::pair<const ggml_tensor *, bool>, split_state_entry, split_state_key_hash> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs,
            size_t compute_mem_size)
            : stc_static(std::move(stc_static)), compute_mem_size(compute_mem_size) {
        arenas.resize(GGML_BACKEND_META_N_ARENAS);
        arenas[0].stc_compute[0] = std::move(stc_compute_0);
        arenas[0].stc_compute[1] = std::move(stc_compute_1);
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    // the current arena's pair, made on first use
    arena_containers & arena() {
        arena_containers & a = arenas[ggml_backend_meta_arena_cur];
        if (a.stc_compute[0].ctxs.empty()) {
            const ggml_init_params params = {
                /*.mem_size   =*/ compute_mem_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            const int n_simple = (int) arenas[0].stc_compute[0].ctxs.size();
            a.stc_compute[0] = ggml_backend_meta_simple_tensor_container(params, n_simple);
            a.stc_compute[1] = ggml_backend_meta_simple_tensor_container(params, n_simple);
        }
        return a;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        arena_containers & a = arena();
        return a.stc_compute[a.stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

// a per-device tensor's sources: the per-device tensors of its meta tensor's sources (itself for a self
//     reference, the source itself for one outside the meta buffers). Set when the tensor is allocated, and again each time its graph
//     is split: a split's merge rewires the sources in place to AllReduce tensors of that split's merged context (see
//     ggml_backend_meta_graph_compute), which a later split of the same allocation must not inherit
static void ggml_backend_meta_simple_set_srcs(const ggml_tensor * tensor, ggml_tensor * t_ij, size_t j) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        t_ij->src[i] = tensor->src[i];
        if (tensor->src[i] == tensor) {
            t_ij->src[i] = t_ij;
        } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
            t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
        }
    }
}

size_t ggml_backend_meta_buffer_n_simple(ggml_backend_buffer_t buf) {
    return buf != nullptr && ggml_backend_buffer_is_meta(buf) ? ggml_backend_meta_buffer_n_bufs(buf) : 0;
}

struct ggml_tensor * ggml_backend_meta_tensor_simple(const struct ggml_tensor * tensor, size_t index) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer) ||
            index >= ggml_backend_meta_buffer_n_bufs(tensor->buffer)) {
        return nullptr;
    }
    return ggml_backend_meta_buffer_simple_tensor(tensor, index);
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

// the stamp of the last split state computed (split states are computed on the thread that builds and allocates graphs)
static uint64_t ggml_backend_meta_split_stamp = 0;

// a split state that is computed from the sources' (a compute tensor or a view), not given by the device's
//     get_split_state for a statically allocated tensor
static bool ggml_backend_meta_split_state_from_srcs(const ggml_tensor * tensor) {
    return ggml_nelements(tensor) != 0 &&
        (ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE || tensor->view_src != nullptr);
}

// the cache entry of (tensor, assume_sync) if it holds: the tensor's bytes are the ones it was computed for and, for a
//     state computed from the sources, every source's entry is the one it was computed from, for the bytes that source has now.
//     In the allocator's walk every source is checked (and computed again if needed, with a new stamp) before its users, so a
//     source that changed anywhere upstream makes every entry below it computed again, each from sources that hold
static const ggml_backend_meta_buffer_context::split_state_entry * ggml_backend_meta_split_state_valid(
        const ggml_tensor * tensor, bool assume_sync) {
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const auto it = buf_ctx->split_state_cache.find(std::make_pair(tensor, assume_sync));
    if (it == buf_ctx->split_state_cache.end() || memcmp(it->second.bytes, (const char *) tensor, sizeof(it->second.bytes)) != 0) {
        return nullptr;
    }
    if (!ggml_backend_meta_split_state_from_srcs(tensor)) {
        return &it->second;
    }
    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = tensor->src[i];
        if (src == nullptr || src == tensor) {
            continue;
        }
        const ggml_backend_meta_buffer_context * src_ctx = (const ggml_backend_meta_buffer_context *) src->buffer->context;
        const ggml_backend_meta_buffer_context::split_state_entry * es = it->second.src_entry[i];
        if (es == nullptr || src_ctx->uid != it->second.src_uid[i] || es->stamp != it->second.src_stamp[i] ||
                memcmp(es->bytes, (const char *) src, sizeof(es->bytes)) != 0) {
            return nullptr;
        }
    }
    return &it->second;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // the used key count of a decode graph over the whole cache, the same on every device
        GGML_ASSERT(tensor->src[6] == nullptr || src_ss[6].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_flash_attn_ext_banded = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // rel_logits is [E, H, Q, B], so its head shard is axis 1.
        GGML_ASSERT(                             src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                GGML_ABORT("%s: unknown split state of src[%zu] %s (%s) of %s (%s)", __func__, i, tensor->src[i]->name,
                    ggml_op_name(tensor->src[i]->op), tensor->name, ggml_op_name(tensor->op));
            }
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_TOP_K_SPLIT: {
                // each device writes its own slot: a row split along dim 0 gives a partial result that the
                //     AllReduce gathers (a sum with zeros elsewhere); a mirrored row is computed whole by every device in slot 0
                if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                    GGML_ASSERT(src_ss[0].n_segments == 1);
                    split_state = {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
                } else {
                    GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    split_state = src_ss[0];
                }
            } break;
            case GGML_OP_DRAFT_PICK: {
                // every input mirrored (the merged candidates), every device picks the same
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_EXT_BANDED: {
                split_state = handle_flash_attn_ext_banded(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    // an entry that does not hold is computed again, its sources first and without recursion: every source whose
    //     entry does not hold is computed before its users, deepest first, so each computation finds all of its sources' entries
    //     (one level of calls at any graph depth). Nothing else in the map is dropped
    const std::pair key = std::make_pair(tensor, assume_sync);
    const ggml_backend_meta_buffer_context::split_state_entry * entry = ggml_backend_meta_split_state_valid(tensor, assume_sync);
    if (entry == nullptr) {
        // GGML_META_SPLIT_LOG=1: an entry whose own bytes still match and whose split differs
        //     once computed from its sources as they are now, and the deepest list of sources computed first
        static const bool split_log = getenv("GGML_META_SPLIT_LOG") != nullptr && atoi(getenv("GGML_META_SPLIT_LOG")) != 0;
        bool                          had_old = false;
        ggml_backend_meta_split_state old_ss  = {};
        size_t                        todo_max = 0;
        if (split_log) {
            const auto it_old = buf_ctx->split_state_cache.find(key);
            if (it_old != buf_ctx->split_state_cache.end() && memcmp(it_old->second.bytes, (const char *) tensor, sizeof(it_old->second.bytes)) == 0) {
                had_old = true;
                old_ss  = it_old->second.split_state;
            }
        }
        if (ggml_backend_meta_split_state_from_srcs(tensor)) {
            std::vector<const ggml_tensor *> todo;
            auto push_invalid_srcs = [&](const ggml_tensor * t) -> bool {
                bool all_valid = true;
                for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                    const ggml_tensor * src = t->src[i];
                    if (src == nullptr || src == t || ggml_backend_meta_split_state_valid(src, /*assume_sync =*/ true) != nullptr) {
                        continue;
                    }
                    todo.push_back(src);
                    all_valid = false;
                }
                return all_valid;
            };
            push_invalid_srcs(tensor);
            while (!todo.empty()) {
                GGML_ASSERT(todo.size() < ((size_t) 1 << 24) && "split state: the sources do not end (a cycle)");
                const ggml_tensor * u = todo.back();
                todo_max = std::max(todo_max, todo.size());
                if (ggml_backend_meta_split_state_valid(u, /*assume_sync =*/ true) == nullptr) {
                    if (ggml_backend_meta_split_state_from_srcs(u) && !push_invalid_srcs(u)) {
                        continue;
                    }
                    ggml_backend_meta_get_split_state(stc, u, /*assume_sync =*/ true); // all of u's sources hold: computed here
                }
                todo.pop_back();
            }
        }

        // every source's entry holds now: calculate_split_state reads them without computing anything
        const ggml_backend_meta_split_state split_state = calculate_split_state();
        if (split_log) {
            static size_t todo_logged = 32;
            if (todo_max > todo_logged) {
                todo_logged = todo_max*2;
                GGML_LOG_WARN("%s: split states of %zu sources computed first (no recursion) for %s (%s)\n", __func__, todo_max, tensor->name, ggml_op_desc(tensor));
            }
            bool same = !had_old || (old_ss.axis == split_state.axis && old_ss.n_segments == split_state.n_segments);
            for (size_t s = 0; had_old && same && s < split_state.n_segments; s++) {
                same = old_ss.nr[s] == split_state.nr[s];
                for (size_t j = 0; same && j < n_bufs; j++) {
                    same = old_ss.ne[s*n_bufs + j] == split_state.ne[s*n_bufs + j];
                }
            }
            static int n_stale = 0;
            if (!same && n_stale++ < 40) {
                GGML_LOG_WARN("%s: stale split avoided: %s (%s, sync %d): cached axis %s ne %lld/%lld, from its sources now axis %s ne %lld/%lld\n", __func__,
                    tensor->name, ggml_op_desc(tensor), (int) assume_sync, ggml_backend_meta_split_axis_name(old_ss.axis), (long long) old_ss.ne[0],
                    (long long) old_ss.ne[1], ggml_backend_meta_split_axis_name(split_state.axis), (long long) split_state.ne[0], (long long) split_state.ne[1]);
            }
        }
        ggml_backend_meta_buffer_context::split_state_entry & e = buf_ctx->split_state_cache[key];
        e.split_state = split_state;
        memcpy(e.bytes, tensor, sizeof(e.bytes));
        e.stamp = ++ggml_backend_meta_split_stamp;
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_backend_meta_buffer_context::split_state_entry * es = nullptr;
            if (tensor->src[i] != nullptr && tensor->src[i] != tensor && ggml_backend_meta_split_state_from_srcs(tensor)) {
                es = ggml_backend_meta_split_state_valid(tensor->src[i], /*assume_sync =*/ true);
                GGML_ASSERT(es != nullptr);
            }
            e.src_stamp[i] = es ? es->stamp : 0;
            e.src_entry[i] = es;
            e.src_uid[i]   = es ? ((const ggml_backend_meta_buffer_context *) tensor->src[i]->buffer->context)->uid : 0;
        }
        entry = &e;
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].split_state;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].split_state.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = entry->split_state;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        if (tensor->op == GGML_OP_TOP_K_SPLIT) {
            // this device's rank and the row index of the first element of its slice
            const ggml_backend_meta_split_state ss_src = ggml_backend_meta_get_split_state(stc, tensor->src[0], /*assume_sync =*/ true);
            if (ss_src.axis == GGML_BACKEND_SPLIT_AXIS_0) {
                GGML_ASSERT(ss_src.n_segments == 1);
                GGML_ASSERT(ggml_get_op_params_i32(tensor, 1) == (int32_t) n_simple_bufs);
                int64_t offset = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    offset += ss_src.ne[jj] * ss_src.nr[0];
                }
                ggml_set_op_params_i32(t_ij, 2, (int32_t) j);
                ggml_set_op_params_i32(t_ij, 3, (int32_t) offset);
            }
        }
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (simple_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(simple_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        ggml_backend_meta_simple_set_srcs(tensor, t_ij, j);

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    auto & a = buf_ctx->arena();
    a.stc_compute_index = a.stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs, params.mem_size);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs, params_compute.mem_size);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

// runs jobs for one simple backend on its own host thread
struct ggml_backend_meta_worker {
    std::mutex              mutex;
    std::condition_variable cv;
    std::function<void()>   job;
    bool                    busy = false;
    bool                    stop = false;
    std::thread             thread; // last: the thread starts only after the members it reads are constructed

    ggml_backend_meta_worker() : thread([this] { run(); }) {}

    ~ggml_backend_meta_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        thread.join();
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv.wait(lock, [this] { return busy || stop; });
            if (stop) {
                return;
            }
            lock.unlock();
            job();
            lock.lock();
            busy = false;
            cv.notify_all();
        }
    }

    void post(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            job  = std::move(fn);
            busy = true;
        }
        cv.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !busy; });
    }
};

typedef void * (*ggml_backend_p2p_ar_init_t)      (ggml_backend_t * backends, size_t n_backends);
typedef void   (*ggml_backend_p2p_ar_free_t)      (void * ar);
typedef bool   (*ggml_backend_p2p_ar_reserve_t)   (void * ar, size_t nbytes);
typedef void   (*ggml_backend_p2p_ar_set_params_t)(void * ar, size_t rank, struct ggml_tensor * node);
typedef bool   (*ggml_backend_graph_ready_t)      (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
typedef void   (*ggml_backend_p2p_ar_allreduce_t)(void * ar, ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);
// the copy-engine reduction in two halves (ggml_cuda_p2p_ar_ce_start / _finish)
typedef size_t  (*ggml_backend_p2p_ar_ce_min_bytes_t)(void * ar);
typedef int64_t (*ggml_backend_p2p_ar_start_t) (void * ar, ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);
typedef void    (*ggml_backend_p2p_ar_finish_t)(void * ar, ggml_backend_t backend, size_t rank, struct ggml_tensor * tensor, int64_t call);

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    // in-graph AllReduce: the whole graph runs as one graph per simple backend, each launched from its own thread
    void *                           p2p_ar            = nullptr;
    ggml_backend_p2p_ar_free_t       p2p_ar_free       = nullptr;
    ggml_backend_p2p_ar_reserve_t    p2p_ar_reserve    = nullptr;
    ggml_backend_p2p_ar_set_params_t p2p_ar_set_params = nullptr;
    ggml_backend_graph_ready_t       graph_ready       = nullptr; // chunks are launched without lockstep only if this says so
    ggml_backend_p2p_ar_allreduce_t  p2p_ar_allreduce  = nullptr; // the unmerged path's reductions, eager, same kernel
    ggml_backend_p2p_ar_ce_min_bytes_t p2p_ar_ce_min_bytes = nullptr;
    ggml_backend_p2p_ar_start_t        p2p_ar_start        = nullptr;
    ggml_backend_p2p_ar_finish_t       p2p_ar_finish       = nullptr;
    std::vector<ggml_backend_t>      simple_backends_v;            // for p2p_ar_allreduce
    ggml_context_ptr                 ctx_merged;
    std::vector<ggml_cgraph *>       cgraphs_merged; // per simple backend, empty if the graph can not be merged
    // per simple backend: the merged graph cut into chunks, launched one after the other. Launching a CUDA graph costs
    //     host time per node and the GPU starts only when the launch returns, so with chunks the GPU starts early.
    std::vector<std::vector<ggml_cgraph *>> cgraphs_chunks;
    ggml_context_ptr                        ctx_chunks;
    bool                             merge_tried = false;
    // shapes that already ran once: a new shape first runs unmerged, from one thread. First runs do blocking driver calls
    //     (pool growth, lazy kernel loads) that can deadlock while a peer GPU spins in the AllReduce.
    std::set<uint64_t>               seen_shapes;
    std::vector<std::unique_ptr<ggml_backend_meta_worker>> workers; // for simple backends 1..n-1

    // the split graphs other than the active one, by the uid of the graph they were built for: a context that
    //     alternates graphs (llama's MTP draft: its catch-up and its draft step, each kept allocated by a scheduler of its
    //     own) swaps the graph back in instead of splitting, re-merging and re-chunking it again. GGML_META_GRAPH_CACHE: how
    //     many are kept (default 3, 0: off). A graph's per-device views must stay valid while it is cached: llama gives each
    //     such scheduler a meta arena of its own (ggml_backend_meta_set_arena)
    // the view containers a split graph's per-device tensors live in, with the generation they had
    struct view_record {
        ggml_backend_meta_buffer_context * buf_ctx;
        int                                arena;
        int                                idx;
        uint64_t                           gen;
    };
    struct graph_state {
        uint64_t uid      = 0;
        uint64_t last_use = 0;
        std::vector<view_record> views;
        std::vector<std::vector<cgraph_config>> cgraphs; // per simple backend
        std::vector<std::vector<ggml_tensor *>> nodes;   // per simple backend
        ggml_context_ptr                        ctx;
        std::vector<ggml_cgraph *>              cgraphs_aux;
        std::vector<ggml_tensor *>              nodes_aux;
        int                                     max_nnodes    = 0;
        size_t                                  max_subgraphs = 0;
        size_t                                  n_subgraphs   = 0;
        ggml_context_ptr                        ctx_merged;
        std::vector<ggml_cgraph *>              cgraphs_merged;
        std::vector<std::vector<ggml_cgraph *>> cgraphs_chunks;
        ggml_context_ptr                        ctx_chunks;
        bool                                    merge_tried   = false;
    };
    std::vector<graph_state> graph_cache;
    uint64_t                 graph_clock    = 0;
    uint64_t                 last_use       = 0; // of the active graph
    std::vector<view_record> views;              // of the active graph

    // a cached graph is swapped back in only while none of its containers was cleared since it was split (another graph
    // of its arena kept alive and rebuilt meanwhile): otherwise it is split again, as before the cache
    static bool views_valid(const std::vector<view_record> & v) {
        for (const view_record & r : v) {
            if (r.buf_ctx->arenas[r.arena].gen[r.idx] != r.gen) {
                return false;
            }
        }
        return true;
    }
    size_t                   ar_bytes_max   = 0; // the largest in-graph AllReduce reserved so far

    // exchange the active graph with g
    void graph_swap(graph_state & g) {
        std::swap(uid, g.uid);
        std::swap(last_use, g.last_use);
        std::swap(views, g.views);
        g.cgraphs.resize(backend_configs.size());
        g.nodes.resize(backend_configs.size());
        for (size_t j = 0; j < backend_configs.size(); j++) {
            std::swap(backend_configs[j].cgraphs, g.cgraphs[j]);
            std::swap(backend_configs[j].nodes,   g.nodes[j]);
        }
        std::swap(ctx,            g.ctx);
        std::swap(cgraphs_aux,    g.cgraphs_aux);
        std::swap(nodes_aux,      g.nodes_aux);
        std::swap(max_nnodes,     g.max_nnodes);
        std::swap(max_subgraphs,  g.max_subgraphs);
        std::swap(n_subgraphs,    g.n_subgraphs);
        std::swap(ctx_merged,     g.ctx_merged);
        std::swap(cgraphs_merged, g.cgraphs_merged);
        std::swap(cgraphs_chunks, g.cgraphs_chunks);
        std::swap(ctx_chunks,     g.ctx_chunks);
        std::swap(merge_tried,    g.merge_tried);
    }

    // a growing in-graph AllReduce moves its staging buffers: the cached graphs were merged (and captured) for the old ones
    void ar_reserved(size_t nbytes) {
        if (nbytes > ar_bytes_max) {
            ar_bytes_max = nbytes;
            for (graph_state & g : graph_cache) {
                g.uid = 0;
            }
        }
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (n_devs > 1) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            ggml_backend_p2p_ar_init_t p2p_ar_init = (ggml_backend_p2p_ar_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_init");
            if (p2p_ar_init != nullptr) {
                p2p_ar = p2p_ar_init(simple_backends.data(), simple_backends.size());
            }
            if (p2p_ar != nullptr) {
                p2p_ar_free       = (ggml_backend_p2p_ar_free_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_free");
                p2p_ar_reserve    = (ggml_backend_p2p_ar_reserve_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_reserve");
                p2p_ar_set_params = (ggml_backend_p2p_ar_set_params_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_set_params");
                graph_ready       = (ggml_backend_graph_ready_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_ready");
                p2p_ar_allreduce  = (ggml_backend_p2p_ar_allreduce_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_allreduce");
                p2p_ar_ce_min_bytes = (ggml_backend_p2p_ar_ce_min_bytes_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_ce_min_bytes");
                p2p_ar_start        = (ggml_backend_p2p_ar_start_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_start");
                p2p_ar_finish       = (ggml_backend_p2p_ar_finish_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_finish");
                if (p2p_ar_ce_min_bytes == nullptr || p2p_ar_start == nullptr || p2p_ar_finish == nullptr) {
                    p2p_ar_ce_min_bytes = nullptr;
                    p2p_ar_start        = nullptr;
                    p2p_ar_finish       = nullptr;
                }
                simple_backends_v = simple_backends;
                GGML_ASSERT(p2p_ar_free && p2p_ar_reserve && p2p_ar_set_params);
                for (size_t i = 1; i < n_devs; i++) {
                    workers.emplace_back(new ggml_backend_meta_worker());
                }
                GGML_LOG_INFO("%s: using in-graph P2P AllReduce\n", __func__);
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        workers.clear();
        if (p2p_ar != nullptr) {
            p2p_ar_free(p2p_ar);
        }
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (size == 0) {
        return; // empty tensors have no split state
    }
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("%s: unsupported split axis %d for tensor %s", __func__, (int) split_state.axis, tensor->name);
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // accepts what ggml_backend_meta_buffer_get_tensor accepts (segments, nr > 1, row offsets), with the same byte layout;
    // each copy is queued on its simple backend's stream
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (size == 0) {
        return;
    }
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_backends, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_backends; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_backends + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_backends + j]/blck_size * tensor->nb[0];
                        if (nbytes != 0) {
                            ggml_backend_tensor_get_2d_async(ggml_backend_meta_simple_backend(backend, j), simple_tensor, (char *) data + offset_data,
                                simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                                row_count, simple_tensor->nb[1], tensor->nb[1]);
                        }
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_backends; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_backends + j] * tensor->nb[1];
                    if (nbytes != 0) {
                        ggml_backend_tensor_get_2d_async(ggml_backend_meta_simple_backend(backend, j), simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                            row_count, simple_tensor->nb[2], tensor->nb[2]);
                    }
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// the meta device's events (ggml_backend_meta_device_event_new), on each simple backend
static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    const auto & evs = *(std::vector<ggml_backend_event_t> *) event->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(evs.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_record(evs[i], ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    const auto & evs = *(std::vector<ggml_backend_event_t> *) event->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(evs.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, i), evs[i]);
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    // a graph built before another one ran is swapped back in from the cache; a new graph is built in the least
    //     recently used entry's place, and the graph that was active goes to the cache
    {
        static const size_t n_cache = [] {
            const char * env = getenv("GGML_META_GRAPH_CACHE");
            return env ? (size_t) atoll(env) : (size_t) 3;
        }();
        if (backend_ctx->graph_cache.size() != n_cache) {
            backend_ctx->graph_cache.resize(n_cache);
        }
        if (needs_rebuild && cgraph->uid != 0 && n_cache > 0) {
            ggml_backend_meta_context::graph_state * hit = nullptr;
            ggml_backend_meta_context::graph_state * lru = nullptr;
            for (auto & g : backend_ctx->graph_cache) {
                if (g.uid == cgraph->uid) {
                    hit = &g;
                }
                if (lru == nullptr || g.last_use < lru->last_use) {
                    lru = &g;
                }
            }
            if (hit != nullptr && !ggml_backend_meta_context::views_valid(hit->views)) {
                // split again in its own entry's place
                hit->uid = 0;
                backend_ctx->graph_swap(*hit);
                hit = nullptr;
            } else {
                backend_ctx->graph_swap(hit ? *hit : *lru);
            }
            needs_rebuild = hit == nullptr;
            if (hit) {
                GGML_RT_COUNT("meta.graph_cached", 1);
            } else {
                GGML_RT_COUNT("meta.graph_split", 1);
            }
        }
        backend_ctx->last_use = ++backend_ctx->graph_clock;
    }

    // GGML_META_TIMING=1: host time of rebuilds and dispatches, logged every 200 calls
    static const bool timing = getenv("GGML_META_TIMING") != nullptr;
    struct meta_timing { int64_t n = 0, n_rebuild = 0, t_rebuild = 0, t_dispatch = 0; };
    static thread_local std::map<const void *, meta_timing> timings;
    const int64_t t_start = timing ? ggml_time_us() : 0;
    int64_t t_rebuilt = t_start;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        backend_ctx->views.clear();
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            auto & a = buf_ctx->arena();
            a.stc_compute_index_next = a.stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = a.stc_compute[a.stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
            a.gen[a.stc_compute_index_next]++;
            // this graph's per-device tensors are in the arena's current container
            backend_ctx->views.push_back({ buf_ctx, ggml_backend_meta_arena_cur, a.stc_compute_index, a.gen[a.stc_compute_index] });
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // The scheduler can retain host views (state inputs or token embeddings) in this split.
                    // They are no-ops; consumers use the scheduler's per-backend copies.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
                // a graph split again without being allocated again (its scheduler kept it while its
                //     entry left the split cache) still has the last merge's sources, AllReduce tensors of a merged context that has
                //     been reset since: start from the allocation's sources, as the merge below expects
                ggml_backend_meta_simple_set_srcs(node, bcj.nodes[i], j);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // sized for the largest graph seen: a later graph may have more subgraphs or nodes without a reallocation
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    // In-graph AllReduce: append an ALLREDUCE node after each partial node and make the later nodes read it,
    //     then run the whole graph as one graph per simple backend.
    bool try_merge = false;
    if (backend_ctx->p2p_ar != nullptr) {
        bool defer = false;
        if (needs_rebuild) {
            backend_ctx->cgraphs_merged.clear();
            backend_ctx->merge_tried = false;
            uint64_t sig = 1469598103934665603ull; // FNV-1a over ops, types and shapes
            auto mix = [&](uint64_t v) { sig = (sig ^ v) * 1099511628211ull; };
            mix(cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                mix(t->op);
                mix(t->type);
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    mix(t->ne[d]);
                }
            }
            defer = backend_ctx->seen_shapes.insert(sig).second;
        }
        try_merge = !defer && !backend_ctx->merge_tried;
    }
    if (try_merge) {
        backend_ctx->merge_tried = true;
        const size_t n_sub = backend_ctx->n_subgraphs;
        bool ok = n_sub > 1;
        // GGML_META_CHUNK_NODES: nodes per chunk of a merged graph (0: no chunks)
        static const size_t chunk_nodes = [] {
            const char * env = getenv("GGML_META_CHUNK_NODES");
            return env ? (size_t) atoll(env) : (size_t) 512;
        }();
        size_t max_bytes = 0;
        for (size_t i = 0; ok && i + 1 < n_sub; i++) {
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_cgraph * g = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                const ggml_tensor * p = g->nodes[g->n_nodes - 1];
                // a partial that is not computed (zero-sized slice) is still reduced: the AllReduce adds 0 for it
                if (p->type != GGML_TYPE_F32 || !ggml_is_contiguous(p) || p->data == nullptr) {
                    // the upstream host-driven AllReduce runs this graph instead: one line per graph size that falls back
                    static std::set<int> logged;
                    if (logged.insert(cgraph->n_nodes).second) GGML_LOG_WARN("%s: no in-graph AllReduce (upstream AllReduce for this graph, %d nodes): subgraph %zu partial %s op=%s type=%s contiguous=%d compute=%d data=%p\n",
                        __func__, cgraph->n_nodes, i, p->name, ggml_op_desc(p), ggml_type_name(p->type), (int) ggml_is_contiguous(p), (int) ((p->flags & GGML_TENSOR_FLAG_COMPUTE) != 0), p->data);
                    ok = false;
                    break;
                }
                max_bytes = std::max(max_bytes, ggml_nbytes(p));
            }
        }
        // a graph with copy-engine reductions (prefill sizes) is not merged: it runs eagerly segment by segment, where
        //     each reduction's copy overlaps the compute that does not read it (see the unmerged loop). The copy engine's call index is
        //     counted on the host, which a captured graph's replays would not advance; and launches are no cost at these sizes.
        if (ok && backend_ctx->p2p_ar_ce_min_bytes != nullptr && max_bytes > backend_ctx->p2p_ar_ce_min_bytes(backend_ctx->p2p_ar)) {
            ok = false;
        }
        ok = ok && backend_ctx->p2p_ar_reserve(backend_ctx->p2p_ar, max_bytes);
        if (ok) {
            backend_ctx->ar_reserved(max_bytes);
            // same memory and allocation order on every rebuild: identical graphs get identical node addresses,
            //     so the CUDA graphs captured for them stay valid
            const size_t n_merged = cgraph->n_nodes + n_sub;
            const size_t mem_size = n_backends * (ggml_graph_overhead_custom(n_merged, false) + n_sub*ggml_tensor_overhead());
            if (!backend_ctx->ctx_merged || ggml_get_mem_size(backend_ctx->ctx_merged.get()) < mem_size) {
                const ggml_init_params params = {
                    /*.mem_size   =*/ 2*mem_size,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                backend_ctx->ctx_merged.reset(ggml_init(params));
            }
            ggml_context * ctx_m = backend_ctx->ctx_merged.get();
            ggml_reset(ctx_m);
            backend_ctx->cgraphs_chunks.assign(n_backends, {});

            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_cgraph * gm = ggml_new_graph_custom(ctx_m, n_merged, false);
                auto add = [&](ggml_tensor * t, int32_t use_count) {
                    gm->nodes[gm->n_nodes++] = t;
                    gm->use_counts[ggml_hash_insert(&gm->visited_hash_set, t)] = use_count;
                };
                for (size_t i = 0; i < n_sub; i++) {
                    const int i_start = bcj.cgraphs[i].offset;
                    const int i_stop  = i + 1 < n_sub ? bcj.cgraphs[i + 1].offset : cgraph->n_nodes;
                    for (int k = i_start; k < i_stop; k++) {
                        const int32_t use_count = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[k])];
                        if (i + 1 == n_sub || k + 1 < i_stop) {
                            add(bcj.nodes[k], use_count);
                            continue;
                        }
                        ggml_tensor * p  = bcj.nodes[k];
                        ggml_tensor * ar = ggml_new_tensor(ctx_m, p->type, GGML_MAX_DIMS, p->ne);
                        memcpy(ar->nb, p->nb, sizeof(p->nb));
                        ar->op        = GGML_OP_ALLREDUCE;
                        ar->src[0]    = p;
                        ar->view_src  = p->view_src ? p->view_src : p;
                        ar->view_offs = p->view_src ? p->view_offs : 0;
                        ar->data      = p->data;
                        ar->buffer    = p->buffer;
                        ar->flags    |= GGML_TENSOR_FLAG_COMPUTE;
                        ggml_format_name(ar, "allreduce_%zu", i);
                        backend_ctx->p2p_ar_set_params(backend_ctx->p2p_ar, j, ar);
                        add(p, 1);
                        add(ar, use_count);
                        for (int k2 = i_stop; k2 < cgraph->n_nodes; k2++) {
                            for (int s = 0; s < GGML_MAX_SRC; s++) {
                                if (bcj.nodes[k2]->src[s] == p) {
                                    bcj.nodes[k2]->src[s] = ar;
                                }
                            }
                        }
                    }
                }
                gm->uid = ggml_graph_next_uid();
                backend_ctx->cgraphs_merged.push_back(gm);

            }
        }
        if (ok) {
            // cut the merged graphs into chunks, each ending with an AllReduce: the CUDA backend fuses no pattern across
            //     an AllReduce, so no fusion reads past the end of a chunk
            std::vector<std::vector<std::pair<int, int>>> bounds(n_backends);
            size_t mem_chunks = 0;
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_cgraph * gm = backend_ctx->cgraphs_merged[j];
                int k0 = 0;
                for (int k = 0; k < gm->n_nodes; k++) {
                    const bool last = k + 1 == gm->n_nodes;
                    if (last || (chunk_nodes > 0 && k + 1 - k0 >= (int) chunk_nodes && gm->nodes[k]->op == GGML_OP_ALLREDUCE &&
                            gm->n_nodes - (k + 1) >= (int) chunk_nodes/2)) {
                        bounds[j].emplace_back(k0, k + 1);
                        mem_chunks += ggml_graph_overhead_custom(k + 1 - k0, false);
                        k0 = k + 1;
                    }
                }
            }
            if (!backend_ctx->ctx_chunks || ggml_get_mem_size(backend_ctx->ctx_chunks.get()) < mem_chunks) {
                const ggml_init_params params = {
                    /*.mem_size   =*/ 2*mem_chunks,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                backend_ctx->ctx_chunks.reset(ggml_init(params));
            }
            ggml_context * ctx_c = backend_ctx->ctx_chunks.get();
            ggml_reset(ctx_c);
            for (size_t j = 0; j < n_backends; j++) {
                ggml_cgraph * gm = backend_ctx->cgraphs_merged[j];
                if (bounds[j].size() == 1) {
                    backend_ctx->cgraphs_chunks[j].push_back(gm);
                    continue;
                }
                for (const auto & [k0, k1] : bounds[j]) {
                    ggml_cgraph * gc = ggml_new_graph_custom(ctx_c, k1 - k0, false);
                    for (int k = k0; k < k1; k++) {
                        ggml_tensor * t = gm->nodes[k];
                        gc->nodes[gc->n_nodes++] = t;
                        gc->use_counts[ggml_hash_insert(&gc->visited_hash_set, t)] = gm->use_counts[ggml_hash_find(&gm->visited_hash_set, t)];
                    }
                    gc->uid = ggml_graph_next_uid();
                    backend_ctx->cgraphs_chunks[j].push_back(gc);
                }
            }
        }
        static const bool debug = getenv("GGML_META_DEBUG") != nullptr;
        if (debug) {
            GGML_LOG_WARN("%s: graph %d nodes, %zu subgraphs: %s, %zu chunks\n", __func__, cgraph->n_nodes, n_sub, ok ? "merged" : "not merged",
                backend_ctx->cgraphs_chunks.empty() ? (size_t) 0 : backend_ctx->cgraphs_chunks[0].size());
        }
    }

    if (timing) {
        t_rebuilt = ggml_time_us();
    }
    auto log_timing = [&]() {
        if (!timing) {
            return;
        }
        meta_timing & mt = timings[backend_ctx];
        mt.n++;
        mt.n_rebuild += needs_rebuild;
        mt.t_rebuild += t_rebuilt - t_start;
        mt.t_dispatch += ggml_time_us() - t_rebuilt;
        if (mt.n % 200 == 0) {
            GGML_LOG_WARN("meta-timing ctx=%p n_nodes=%d rebuilt=%" PRId64 "/200 avg_us: rebuild=%.0f (per rebuild %.0f) dispatch=%.0f\n", (void *) backend_ctx, cgraph->n_nodes,
                mt.n_rebuild, mt.t_rebuild/200.0, mt.n_rebuild ? mt.t_rebuild/(double) mt.n_rebuild : 0.0, mt.t_dispatch/200.0);
            mt = meta_timing();
        }
    };

    if (!backend_ctx->cgraphs_merged.empty()) {
        // each backend enqueues from its own thread: a host sync in one graph must not block the peer's AllReduce
        //     (without workers the jobs of backends 1..n-1 run inline on this thread)
        std::vector<ggml_status> status(n_backends, GGML_STATUS_SUCCESS);
        auto post = [backend_ctx](size_t j, std::function<void()> fn) {
            if (backend_ctx->workers.empty()) {
                fn();
            } else {
                backend_ctx->workers[j - 1]->post(std::move(fn));
            }
        };
        auto wait = [backend_ctx](size_t j) {
            if (!backend_ctx->workers.empty()) {
                backend_ctx->workers[j - 1]->wait();
            }
        };

        // A chunk that is not a plain replay does driver work (capture, instantiation, eager launches) on the host.
        // Driver calls can block while the own GPU spins in an AllReduce of an earlier chunk, and that deadlocks.
        // So in that case run the chunks in lockstep: all GPUs finish chunk k before any thread starts chunk k+1.
        const size_t n_chunks = backend_ctx->cgraphs_chunks[0].size();
        bool lockstep = false;
        if (n_chunks > 1) {
            for (size_t j = 0; j < n_backends && !lockstep; j++) {
                for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[j]) {
                    if (backend_ctx->graph_ready == nullptr || !backend_ctx->graph_ready(backend_ctx->backend_configs[j].backend, gc)) {
                        lockstep = true;
                        break;
                    }
                }
            }
        }
        if (lockstep) {
            for (size_t k = 0; k < n_chunks; k++) {
                for (size_t j = 1; j < n_backends; j++) {
                    post(j, [backend_ctx, &status, j, k] {
                        status[j] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, backend_ctx->cgraphs_chunks[j][k]);
                    });
                }
                status[0] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[0].backend, backend_ctx->cgraphs_chunks[0][k]);
                for (size_t j = 1; j < n_backends; j++) {
                    wait(j);
                }
                for (size_t j = 0; j < n_backends; j++) {
                    if (status[j] != GGML_STATUS_SUCCESS) {
                        return status[j];
                    }
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }
            }
            log_timing();
            return GGML_STATUS_SUCCESS;
        }
        for (size_t j = 1; j < n_backends; j++) {
            post(j, [backend_ctx, &status, j] {
                for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[j]) {
                    status[j] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, gc);
                    if (status[j] != GGML_STATUS_SUCCESS) {
                        break;
                    }
                }
            });
        }
        for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[0]) {
            status[0] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[0].backend, gc);
            if (status[0] != GGML_STATUS_SUCCESS) {
                break;
            }
        }
        for (size_t j = 1; j < n_backends; j++) {
            wait(j);
        }
        log_timing();
        for (ggml_status st : status) {
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    // two-batch overlap of the copy-engine reductions (prefill sizes). Each such reduction is started right after its
    //     partial (the pre kernel, then the copy on the copy engine) and finished (the wait for the copy and the sum in place) only
    //     just before the first node that reads the partial, so the compute in between, the other half of a split ubatch (qwen35
    //     under the split builds each ubatch as two halves, interleaved layer by layer), runs while the copy engine moves the
    //     partial. Each finish is the eager call's sum, so the outputs are bitwise the same. GGML_META_AR_DEFER=0: start and finish
    //     back to back.
    static const bool ar_defer_on = [] {
        const char * e = getenv("GGML_META_AR_DEFER");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool ar_defer = ar_defer_on && n_backends == 2 && backend_ctx->p2p_ar_start != nullptr && backend_ctx->p2p_ar_allreduce != nullptr;
    struct pending_ar {
        std::vector<ggml_tensor *> t;    // the partial on each device
        int64_t                    call;
        std::vector<char>          done; // finished on each device
    };
    std::vector<pending_ar> pending; // in call order
    int n_async = 0; // GGML_META_DEBUG: copy-engine calls, and finishes placed inside a subgraph
    int n_split = 0;
    if (ar_defer && backend_ctx->n_subgraphs > 1) {
        // the landing slots move when the reservation grows, so reserve for the largest partial before any call is pending
        size_t max_bytes = 0;
        for (size_t i = 0; i + 1 < backend_ctx->n_subgraphs; i++) {
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_cgraph * g = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                const ggml_tensor * p = g->nodes[g->n_nodes - 1];
                if (p->type == GGML_TYPE_F32 && ggml_is_contiguous(p) && p->data != nullptr) {
                    max_bytes = std::max(max_bytes, ggml_nbytes(p));
                }
            }
        }
        if (max_bytes > 0 && backend_ctx->p2p_ar_reserve(backend_ctx->p2p_ar, max_bytes)) {
            backend_ctx->ar_reserved(max_bytes);
        }
    }
    // whether node reads memory of partial p (views and no-ops read nothing; their readers are found instead)
    auto reads_partial = [](const ggml_tensor * node, const ggml_tensor * p) -> bool {
        if (ggml_op_is_empty(node->op)) {
            return false;
        }
        const char * p0 = (const char *) p->data;
        const char * p1 = p0 + ggml_nbytes(p);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr || src->data == nullptr) {
                continue;
            }
            const char * s0 = (const char *) src->data;
            const char * s1 = s0 + ggml_nbytes(src);
            if (s0 < p1 && p0 < s1) {
                return true;
            }
        }
        return false;
    };
    auto finish_upto = [&](size_t j, size_t q) {
        for (size_t e = 0; e <= q && e < pending.size(); e++) {
            if (!pending[e].done[j]) {
                backend_ctx->p2p_ar_finish(backend_ctx->p2p_ar, backend_ctx->backend_configs[j].backend, j, pending[e].t[j], pending[e].call);
                pending[e].done[j] = 1;
            }
        }
    };
    auto finish_all = [&]() {
        for (size_t j = 0; j < n_backends && !pending.empty(); j++) {
            finish_upto(j, pending.size() - 1);
        }
        pending.clear();
    };

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cg = bcj.cgraphs[i].cgraph_main;
            // run the nodes before the first reader of a pending partial, finish it (and the ones started before it), go on
            int k0 = 0;
            while (!pending.empty()) {
                int    kmin = cg->n_nodes;
                size_t qmin = SIZE_MAX;
                for (size_t q = 0; q < pending.size(); q++) {
                    if (pending[q].done[j]) {
                        continue;
                    }
                    for (int k = k0; k < kmin; k++) {
                        if (reads_partial(cg->nodes[k], pending[q].t[j])) {
                            kmin = k;
                            qmin = q;
                            break;
                        }
                    }
                }
                if (qmin == SIZE_MAX) {
                    break;
                }
                if (kmin > k0) {
                    n_split += j == 0;
                    ggml_cgraph part = ggml_graph_view(cg, k0, kmin);
                    const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, &part);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
                finish_upto(j, qmin);
                k0 = kmin;
            }
            ggml_status status = GGML_STATUS_SUCCESS;
            if (k0 == 0) {
                status = ggml_backend_graph_compute_async(bcj.backend, cg);
            } else if (k0 < cg->n_nodes) {
                ggml_cgraph part = ggml_graph_view(cg, k0, cg->n_nodes);
                status = ggml_backend_graph_compute_async(bcj.backend, &part);
            }
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        while (!pending.empty()) {
            bool all_done = true;
            for (size_t j = 0; j < n_backends; j++) {
                all_done = all_done && pending.front().done[j];
            }
            if (!all_done) {
                break;
            }
            pending.erase(pending.begin());
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            // a copy-engine reduction is started here and finished before its first reader
            if (ar_defer) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                bool ok = true;
                size_t max_bytes = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_cgraph * cgraph_ij = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                    ggml_tensor * p = cgraph_ij->nodes[cgraph_ij->n_nodes-1];
                    ok = ok && p->type == GGML_TYPE_F32 && ggml_is_contiguous(p) && p->data != nullptr;
                    max_bytes = std::max(max_bytes, ggml_nbytes(p));
                    nodes.push_back(p);
                }
                if (ok && max_bytes > backend_ctx->p2p_ar_ce_min_bytes(backend_ctx->p2p_ar) &&
                        backend_ctx->p2p_ar_reserve(backend_ctx->p2p_ar, max_bytes)) {
                    // The CUDA transport has two landing slots; starting a third can deadlock on call k-2.
                    GGML_ASSERT(pending.size() < 2);
                    backend_ctx->ar_reserved(max_bytes);
                    const int64_t call = backend_ctx->p2p_ar_start(backend_ctx->p2p_ar, backend_ctx->simple_backends_v.data(), nodes.data(), n_backends);
                    if (call >= 0) {
                        n_async++;
                        pending.push_back({ nodes, call, std::vector<char>(n_backends, 0) });
                        continue;
                    }
                }
                // any other reduction runs with nothing pending, so the kernel path's calls stay in the same order on every device
                finish_all();
            }
            bool backend_allreduce_success = false;
            // with the in-graph AllReduce on, the unmerged run (a graph shape's first) reduces with the same
            //     fp32 kernel, launched eagerly on every device before the next segment, so all runs of a shape agree
            if (backend_ctx->p2p_ar_allreduce != nullptr) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                size_t max_bytes = 0;
                bool ok = true;
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    ggml_tensor * p = cgraph_ij->nodes[cgraph_ij->n_nodes-1];
                    ok = ok && p->type == GGML_TYPE_F32 && ggml_is_contiguous(p) && p->data != nullptr;
                    max_bytes = std::max(max_bytes, ggml_nbytes(p));
                    nodes.push_back(p);
                }
                if (ok && backend_ctx->p2p_ar_reserve(backend_ctx->p2p_ar, max_bytes)) {
                    backend_ctx->ar_reserved(max_bytes);
                    backend_ctx->p2p_ar_allreduce(backend_ctx->p2p_ar, backend_ctx->simple_backends_v.data(), nodes.data(), n_backends);
                    backend_allreduce_success = true;
                }
            }
            // the candidates of ggml_top_k_split carry token ids as floats: never through a bf16 wire
            const ggml_cgraph * cgraph_i0 = backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
            const bool exact_only = cgraph_i0->nodes[cgraph_i0->n_nodes-1]->op == GGML_OP_TOP_K_SPLIT;
            if (!backend_allreduce_success && backend_ctx->comm_ctx && !exact_only) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    finish_all();
    {
        static const bool debug = getenv("GGML_META_DEBUG") != nullptr;
        if (debug && n_async > 0) {
            GGML_LOG_WARN("%s: graph %d nodes, %zu subgraphs unmerged: %d copy-engine reductions, %d finished inside a later subgraph\n",
                __func__, cgraph->n_nodes, backend_ctx->n_subgraphs, n_async, n_split);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
