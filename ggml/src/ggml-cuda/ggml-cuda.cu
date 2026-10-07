#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/allreduce-p2p.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/diffusion-sampling.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-banded.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmv-hc.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/mmvq-tc.cuh"
#include "ggml-cuda/mmvq-qpn.cuh"
#include "ggml-cuda/qpn-source.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/draft-pick.cuh"
#include "ggml-cuda/kq-mask.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/qsa-select.cuh"
#include "ggml-cuda/qsa-union.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <cstdarg>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-rtimer.h"

// [TAG_ROUND_TIMERS] with LLAMA_ROUND_TIMERS=1, one single-thread kernel at the start and at the end of
// every graph evaluation (captured into the cuda graph with the rest) writes %globaltimer and a tag (the host's graph
// label << 1 | end) into a host-mapped ring; ggml_rt_round() collects them. Nothing is launched when it is off.
// one ring and counter per device, so stamps from several devices (--split-mode tensor) do not share a ring;
// the collector tags each entry with its device in bits 24 and up.
struct ggml_cuda_rt_entry { unsigned long long t; unsigned int tag; unsigned int seq; };
// 1 << 16 entries per device (was 1 << 14): a 64K prefill under --split-mode tensor writes thousands of stamps
// per device between two rounds
static constexpr unsigned int GGML_CUDA_RT_RING = 1u << 16;
static ggml_cuda_rt_entry * ggml_cuda_rt_host[GGML_CUDA_MAX_DEVICES] = {};
static ggml_cuda_rt_entry * ggml_cuda_rt_dev [GGML_CUDA_MAX_DEVICES] = {};
static unsigned int       * ggml_cuda_rt_ctr [GGML_CUDA_MAX_DEVICES] = {};
static unsigned int         ggml_cuda_rt_next[GGML_CUDA_MAX_DEVICES] = {};

static __global__ void ggml_cuda_rt_stamp_kernel(ggml_cuda_rt_entry * ring, unsigned int * ctr, unsigned int tag) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    const unsigned int i = atomicAdd(ctr, 1u);
    volatile ggml_cuda_rt_entry * e = ring + (i & (GGML_CUDA_RT_RING - 1));
    e->t = t;
    e->tag = tag;
    __threadfence_system();
    e->seq = i + 1;
}

// every device gets an equal share of max (device 0 no longer starves the others after a long prefill), and a
// ring that wrapped past the reader is resynchronized at the oldest entry it still holds, with the entry GGML_RT_TAG_LOST
// in front so the round that lost stamps is not counted (before, the reader stopped at the overwritten entry for good)
static int ggml_cuda_rt_collect(uint64_t * t, uint32_t * tag, int max) {
    int n_dev = 0;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        n_dev += ggml_cuda_rt_host[d] != nullptr;
    }
    const int share = n_dev > 0 ? max / n_dev : max;
    int n = 0;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (ggml_cuda_rt_host[d] == nullptr) {
            continue;
        }
        const int n_end = std::min(max, n + share);
        while (n < n_end) {
            volatile ggml_cuda_rt_entry * e = ggml_cuda_rt_host[d] + (ggml_cuda_rt_next[d] & (GGML_CUDA_RT_RING - 1));
            const unsigned int seq = e->seq;
            if (seq != ggml_cuda_rt_next[d] + 1) {
                if (seq > ggml_cuda_rt_next[d] + 1 && n + 1 < n_end) {
                    // overwritten: the slot now holds entry seq - 1, the oldest one left
                    t[n] = 0; tag[n] = GGML_RT_TAG_LOST | ((uint32_t) d << 24); n++;
                    ggml_cuda_rt_next[d] = seq - 1;
                    continue;
                }
                break;
            }
            t[n] = e->t; tag[n] = e->tag | ((uint32_t) d << 24); n++;
            ggml_cuda_rt_next[d]++;
        }
    }
    return n;
}

static void ggml_cuda_rt_init(int device) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (ggml_cuda_rt_host[device] != nullptr) {
        return;
    }
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaHostAlloc((void **) &ggml_cuda_rt_host[device], GGML_CUDA_RT_RING * sizeof(ggml_cuda_rt_entry), cudaHostAllocMapped));
    memset(ggml_cuda_rt_host[device], 0, GGML_CUDA_RT_RING * sizeof(ggml_cuda_rt_entry));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &ggml_cuda_rt_dev[device], ggml_cuda_rt_host[device], 0));
    CUDA_CHECK(cudaMalloc((void **) &ggml_cuda_rt_ctr[device], sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(ggml_cuda_rt_ctr[device], 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    ggml_rt_set_collector(ggml_cuda_rt_collect);
    fprintf(stderr, "%s: LLAMA_ROUND_TIMERS on: graph stamps in a %u-entry host-mapped ring (device %d)\n", __func__, GGML_CUDA_RT_RING, device);
}

static void ggml_cuda_rt_stamp(int device, cudaStream_t stream, bool end) {
    if (ggml_cuda_rt_ctr[device] == nullptr) {
        return;
    }
    const int l = ggml_rt_cur_label();
    ggml_cuda_rt_stamp_kernel<<<1, 1, 0, stream>>>(ggml_cuda_rt_dev[device], ggml_cuda_rt_ctr[device], ((unsigned int) (l < 0 ? 0 : l) << 1) | (end ? 1u : 0u));
    CUDA_CHECK(cudaGetLastError());
}

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

// Whether an environment variable is set to a value that means "on".
//
// Most ggml switches test presence only, and for a name like GGML_CUDA_NO_PINNED
// that is right: the only sensible use is to set it. GGML_CUDA_ENABLE_UNIFIED_MEMORY
// is spelled as a positive, so people reasonably write =0 to turn it off, and under a
// presence test that enabled the very path they were trying to leave. GGML_CUDA_DISABLE_FUSION
// in this file already parses its value for the same reason.
//
// Empty and the usual falsy spellings are off, anything else is on, so =1, =true,
// =on and =yes all keep working exactly as before.
static bool ggml_cuda_env_enabled(const char * name) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return false;
    }
    std::string v(val);
    for (char & c : v) {
        c = (char) std::tolower((unsigned char) c);
    }
    return !(v == "0" || v == "false" || v == "no" || v == "off");
}

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_env_enabled("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
#if defined(GGML_USE_HIP)
        // Say so once. Every report of this path going wrong has come from someone
        // who did not know a front-end had set the variable for them.
        //
        // This deliberately does NOT warn about corrupted output. An earlier draft
        // did, pointing at ggml-org#26148, but that defect is the cpy 2D fast path
        // using cudaMemcpyDeviceToDevice on managed pages, and it is fixed
        // separately. Warning about it in a tree that carries the fix would send
        // users chasing a bug their build does not have.
        //
        // What remains true is the cost. On an integrated GPU managed allocation
        // draws host RAM instead of the device carve-out rather than adding to it,
        // so it is usually slower and can lower the ceiling: a model that loads
        // without the variable may be OOM-killed with it.
        GGML_LOG_WARN_ONCE("GGML_CUDA_ENABLE_UNIFIED_MEMORY is set, allocating device memory as managed. "
                           "On integrated GPUs this draws host RAM instead of the device carve-out, which is "
                           "usually slower and can reduce the largest model that will load. Unset the variable "
                           "to disable it.\n");
#endif // defined(GGML_USE_HIP)
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    // The driver reads CUDA_SCALE_LAUNCH_QUEUES once, when CUDA initializes, so it is set here before any CUDA call.
    // Measured +24.5 % prompt reading at 64K on Flash-Next with 4x, with decode and memory unchanged.
    // A value set in the environment is used as is; LLAMA_CUDA_LAUNCH_QUEUES=0 leaves the driver's own default.
    const char * launch_queues_env = getenv("CUDA_SCALE_LAUNCH_QUEUES");
    const char * launch_queues_opt = getenv("LLAMA_CUDA_LAUNCH_QUEUES");
    const std::string launch_queues_user = launch_queues_env != nullptr ? launch_queues_env : "";
    const bool launch_queues_off = launch_queues_env == nullptr && launch_queues_opt != nullptr && strcmp(launch_queues_opt, "0") == 0;
    if (launch_queues_env == nullptr && !launch_queues_off) {
#ifdef _WIN32
        _putenv_s("CUDA_SCALE_LAUNCH_QUEUES", "4x");
#else
        setenv("CUDA_SCALE_LAUNCH_QUEUES", "4x", 0);
#endif // _WIN32
    }

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    if (launch_queues_env != nullptr) {
        GGML_LOG_WARN("CUDA launch queues: CUDA_SCALE_LAUNCH_QUEUES=%s (from the environment)\n", launch_queues_user.c_str());
    } else if (launch_queues_off) {
        GGML_LOG_WARN("CUDA launch queues: driver default (LLAMA_CUDA_LAUNCH_QUEUES=0)\n");
    } else {
        GGML_LOG_WARN("CUDA launch queues: CUDA_SCALE_LAUNCH_QUEUES=4x (built-in default; LLAMA_CUDA_LAUNCH_QUEUES=0 turns it off)\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

static void ggml_cuda_qpn_check_log(int device); // LLAMA_QPN_PREP_CHECK's totals

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (q8_share_buf != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(q8_share_buf));
    }
    if (qpn_share_buf != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(qpn_share_buf));
    }
    ggml_cuda_qpn_check_log(device);
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

// load time: repack a weight in a plain CUDA buffer into the tensor-core fragment order, if routed
static bool ggml_backend_cuda_qpn_repack_if(ggml_tensor * t, bool (*eligible)(const ggml_tensor *, int)) {
    if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_cuda(t->buffer) ||
            ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    const int device = ((ggml_backend_cuda_buffer_context *) t->buffer->context)->device;
    if (!eligible(t, ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    return ggml_cuda_qpn_repack(t, device);
}
bool ggml_backend_cuda_qpn_repack(ggml_tensor * t) {
    return ggml_backend_cuda_qpn_repack_if(t, ggml_cuda_qpn_eligible);
}
// a draft model's weight, by the draft's route table
bool ggml_backend_cuda_qpn_repack_draft(ggml_tensor * t) {
    return ggml_backend_cuda_qpn_repack_if(t, ggml_cuda_qpn_eligible_draft);
}
// a DFlash2 draft's weight, by its own route table
bool ggml_backend_cuda_qpn_repack_dflash(ggml_tensor * t) {
    return ggml_backend_cuda_qpn_repack_if(t, ggml_cuda_qpn_eligible_dflash);
}

// defined in mmvq-qpn.cu (the GDN alpha and beta slices under --split-mode tensor, LLAMA_QPN_GDN_AB)
bool ggml_cuda_qpn_ab_split_on();
bool ggml_cuda_qpn_ab_slice_copy(const ggml_tensor * slice, int device, int64_t n, int64_t k);
bool ggml_cuda_qpn_has_ab_copy(const ggml_tensor * t);
bool ggml_cuda_qpn_gdn4_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * d, const ggml_tensor * src1);
void ggml_cuda_mul_mat_qpn4ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], const ggml_tensor * src1, ggml_tensor * const dsts[4]);

// the GDN alpha and beta weights the split offered (Q8_0, 48 x 5,120, a 24-row slice a device), and the layers whose two weights got
// their private padded copies on every slice; load time only, read by ggml_cuda_qpn_ab_log
static std::set<const ggml_tensor *> ggml_cuda_ab_offered;
static std::map<int, int>            ggml_cuda_ab_layers;

static bool ggml_cuda_ab_weight(const ggml_tensor * full) {
    const char * name = ggml_get_name(full);
    return full->type == GGML_TYPE_Q8_0 && full->ne[0] == 5120 && full->ne[1] == 48 && full->ne[2] == 1 && full->ne[3] == 1 &&
        (strstr(name, ".ssm_alpha.weight") != nullptr || strstr(name, ".ssm_beta.weight") != nullptr);
}

// every slice of the weight gets its copy, or none does (a copy is never read alone)
static void ggml_cuda_ab_offer(const ggml_tensor * full) {
    if (!ggml_cuda_ab_offered.insert(full).second || !ggml_cuda_qpn_ab_split_on() || full->buffer == nullptr) {
        return;
    }
    const size_t n_simple = ggml_backend_meta_buffer_n_simple(full->buffer);
    std::vector<std::pair<const ggml_tensor *, int>> slices;
    for (size_t j = 0; j < n_simple; ++j) {
        const ggml_tensor * slice = ggml_backend_meta_tensor_simple(full, j);
        if (slice == nullptr || slice->buffer == nullptr || !ggml_backend_buffer_is_cuda(slice->buffer) ||
                ggml_backend_buffer_get_usage(slice->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS || slice->type != full->type ||
                slice->ne[0] != full->ne[0] || slice->ne[1] < 1 || slice->ne[1] >= 32) {
            return;
        }
        slices.emplace_back(slice, ((ggml_backend_cuda_buffer_context *) slice->buffer->context)->device);
    }
    if (slices.empty()) {
        return;
    }
    for (size_t j = 0; j < slices.size(); ++j) {
        if (!ggml_cuda_qpn_ab_slice_copy(slices[j].first, slices[j].second, full->ne[1], full->ne[0])) {
            GGML_ASSERT(j == 0 && "a slice of a GDN a/b weight was not copied after another was");
            return;
        }
    }
    int il = -1;
    if (sscanf(ggml_get_name(full), "blk.%d.", &il) == 1) {
        ggml_cuda_ab_layers[il] += 1;
    }
}

// one WARN line, under --split-mode tensor only (where alpha and beta were offered as slices), at the first graph
static void ggml_cuda_qpn_ab_log() {
    static std::once_flag once;
    if (ggml_cuda_ab_offered.empty()) {
        return;
    }
    std::call_once(once, [] {
        if (!ggml_cuda_qpn_ab_split_on()) {
            GGML_LOG_WARN("GDN a/b on the tensor cores under the split (LLAMA_QPN_GDN_AB): off\n");
            return;
        }
        int n = 0;
        for (const auto & [il, count] : ggml_cuda_ab_layers) {
            n += count == 2;
        }
        GGML_LOG_WARN("GDN a/b on the tensor cores under the split (LLAMA_QPN_GDN_AB): on, %d layers\n", n);
    });
}

// one device's slice of a weight that the meta device of --split-mode tensor holds across devices (in a plain CUDA
// buffer of its own). The route is the whole weight's, by its kind's table (0 the target's, 1 a draft's, 2 a DFlash2 draft's):
// the tables are keyed by the model's shapes, not the slices'. The slice must itself be one the kernels take (whole superblocks
// along K, whole 32-row tiles). check_only answers without repacking, so the caller repacks every slice of a weight or none
bool ggml_backend_cuda_qpn_repack_slice(ggml_tensor * slice, const ggml_tensor * full, int kind, bool check_only) {
    if (slice == nullptr || full == nullptr || slice->type != full->type || slice->buffer == nullptr ||
            !ggml_backend_buffer_is_cuda(slice->buffer) || ggml_backend_buffer_get_usage(slice->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    // the GDN alpha and beta weights are not repacked (the slice keeps its GGUF layout for every reader); their first offer gives
    // every slice its private padded copy (LLAMA_QPN_GDN_AB), and the answer stays no
    if (kind == 0 && ggml_cuda_ab_weight(full)) {
        ggml_cuda_ab_offer(full);
        return false;
    }
    const int device = ((ggml_backend_cuda_buffer_context *) slice->buffer->context)->device;
    const int cc     = ggml_cuda_info().devices[device].cc;
    bool (*eligible)(const ggml_tensor *, int) = kind == 2 ? ggml_cuda_qpn_eligible_dflash : kind == 1 ? ggml_cuda_qpn_eligible_draft : ggml_cuda_qpn_eligible;
    if (!eligible(full, cc) || !ggml_cuda_qpn_takes(slice, cc)) {
        return false;
    }
    if (check_only) {
        return true;
    }
    if (!ggml_cuda_qpn_repack(slice, device)) {
        return false;
    }
    ggml_cuda_qpn_set_slice_key(slice, full->ne[1], full->ne[0]); // the slice runs, pairs and groups by the whole weight's key
    if (kind == 1) {
        ggml_cuda_qpn_mark_draft_slice(slice); // the draft step's pairs at 1 and 2 tokens
    }
    return true;
}

// the GGUF bytes of a repacked weight in a plain CUDA buffer, into host memory
bool ggml_backend_cuda_qpn_unpack(const ggml_tensor * t, void * host_dst) {
    if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_cuda(t->buffer)) {
        return false;
    }
    const int device = ((ggml_backend_cuda_buffer_context *) t->buffer->context)->device;
    return ggml_cuda_qpn_unpack(t, host_dst, device);
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    GGML_RT_SCOPE("cuda.memset_sync");
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    GGML_RT_SCOPE("cuda.set_tensor_sync");
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// [TAG_UPLOAD_BATCH] the scheduler's user-input uploads without one sync each. the copy goes to the
// per-thread stream as in set_tensor; the scheduler calls ggml_backend_cuda_upload_sync once after the split's last
// upload and before it queues the split's compute, so the data is on the device and the host source is free again
// at the same point as with set_tensor. returns false when the tensor is not in a plain CUDA buffer (use set_tensor)
static bool ggml_backend_cuda_upload_nosync(ggml_tensor * tensor, const void * data, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == nullptr || !ggml_backend_buffer_is_cuda(buf)) {
        return false;
    }
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buf->context;
    GGML_RT_COUNT("cuda.upload_nosync", 1);
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    return true;
}

// [TAG_KQ_MASK_DEVICE] build a KQ mask on the backend's stream from the KV cells' device mirror (see
// llama_kv_cache::set_input_kq_mask_device). false when the tensors are not what the kernel handles (the caller then
// writes the mask on the host)
static bool ggml_backend_cuda_kq_mask(ggml_backend_t backend, ggml_tensor * mask, ggml_tensor * cells,
        const int32_t * p, const int32_t * py, const int32_t * px, int32_t n_tokens, bool use_2d,
        const int32_t * upd, int32_t upd_lo, int32_t upd_n) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend) || mask == nullptr || cells == nullptr) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || mask->view_src != nullptr || mask->buffer == nullptr ||
            !ggml_backend_buffer_is_cuda(mask->buffer) || mask->ne[1] != n_tokens || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (cells->type != GGML_TYPE_I32 || cells->buffer == nullptr || !ggml_backend_buffer_is_cuda(cells->buffer) || cells->ne[0] % 3 != 0) {
        return false;
    }
    if (n_tokens <= 0) {
        return false;
    }

    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    const int dev_mask  = ((ggml_backend_cuda_buffer_context *) mask->buffer->context)->device;
    const int dev_cells = ((ggml_backend_cuda_buffer_context *) cells->buffer->context)->device;
    if (dev_mask != ctx->device || dev_cells != ctx->device) {
        return false;
    }

    const int64_t kv_size = cells->ne[0] / 3;
    if (mask->ne[0] > kv_size) {
        return false;
    }

    GGML_RT_SCOPE("cuda.kq_mask");
    ggml_cuda_set_device(ctx->device);
    ggml_cuda_kq_mask(*ctx, (half *) mask->data, mask->ne[0], (int32_t *) cells->data, kv_size,
            p, py, px, n_tokens, use_2d, upd, upd_lo, upd_n);
    return true;
}

static void ggml_backend_cuda_upload_sync(ggml_tensor * tensor) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buf->context;
    GGML_RT_SCOPE("cuda.upload_sync");
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    GGML_RT_SCOPE("cuda.get_tensor_sync");
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_FLASH_ATTN_EXT_BANDED)
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const bool f32_pedantic = compute_type == GGML_TYPE_F32 &&
        src0->type == GGML_TYPE_F32 && ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    if (f32_pedantic) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11020
        cu_compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
#else
        // no pedantic compute enum here; ordinary F32 is the strongest available contract
        cu_compute_type = CUBLAS_COMPUTE_32F;
#endif
    }

    const auto cu_gemm_algo = f32_pedantic ?
        CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        if (f32_pedantic) {
            CUBLAS_CHECK(
                cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        alpha, src0_ptr, CUDA_R_32F, s01,
                               src1_ptr, CUDA_R_32F, s11,
                        beta,   dst_ptr, CUDA_R_32F, ne0,
                        cu_compute_type,
                        cu_gemm_algo));
        } else {
            CUBLAS_CHECK(
                cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        (const float *) alpha, (const float *) src0_ptr, s01,
                                               (const float *) src1_ptr, s11,
                        (const float *) beta,  (float       *)  dst_ptr, ne0));
        }
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    cu_gemm_algo));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                cu_gemm_algo));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                cu_gemm_algo));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // [TAG_GGML_PREC] any acc rank at least as strict as F32 (F32, F32_PEDANTIC) forces F32 compute
    const ggml_prec prec_acc = ggml_prec(dst->op_params[0]);
    if (prec_acc != GGML_PREC_UNDEFINED && prec_acc <= GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    // a scoped pedantic request overrides the process-wide compute type, for F32 weights only
    if (src0->type == GGML_TYPE_F32 && prec_acc == GGML_PREC_F32_PEDANTIC) {
        compute_type = GGML_TYPE_F32;
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// a repacked weight expanded to fp16 by its own graph node (a CPY to F16; build_tbo makes one per weight for both halves
// of a prompt ubatch). The CPY runs ggml_cuda_qpn_to_fp16, and a product reading it takes the cuBLAS call of ggml_cuda_mul_mat's
// fallback for a repacked weight, on the same values
static bool ggml_cuda_qpn_is_expansion(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_CPY && t->type == GGML_TYPE_F16 && ggml_cuda_qpn_is_repacked(t->src[0]);
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    // a repacked weight is read only by ggml_cuda_mul_mat_qpn, never by a fused kernel; nor is its expansion
    if (ggml_cuda_qpn_is_repacked(ffn_up->src[0]) || ggml_cuda_qpn_is_repacked(ffn_gate->src[0]) ||
        ggml_cuda_qpn_is_expansion(ffn_up->src[0]) || ggml_cuda_qpn_is_expansion(ffn_gate->src[0])) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    if (ggml_cuda_qpn_is_repacked(src0)) { // read only by ggml_cuda_mul_mat_qpn
        return false;
    }

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_sib_gemv_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
static bool ggml_cuda_qpn_sib_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
static bool ggml_cuda_mmvq_sib_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // a weight repacked at load into the tensor-core fragment order: the tensor-core kernel up to
    // GGML_CUDA_QPN_MAX_TOKENS tokens, else fp16 through cuBLAS exactly as a quantized weight goes on this GPU
    // (tokens over several sequences, [K, tokens, sequences] with its rows one run, are columns too)
    if (ggml_cuda_qpn_is_expansion(src0)) { // the fallback's cuBLAS call below, on the expansion node's fp16 weight
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_mmvq_sib_launch(ctx, dst)) { // with its dp4a sibling on the same input, in one launch
        return;
    }
    if (ggml_cuda_qpn_is_repacked(src0)) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        if (ne11*ne12*ne13 <= ggml_cuda_qpn_max_tokens() && ggml_cuda_qpn_flat_cols(src1) && ggml_cuda_qpn_flat_cols(dst)) {
            if (ggml_cuda_qpn_sib_launch(ctx, dst)) { // with its sibling on the same input, in one launch
                return;
            }
            ggml_cuda_mul_mat_qpn(ctx, src0, src1, dst);
        } else {
            ggml_cuda_pool_alloc<half> w16(ctx.pool(), ggml_nelements(src0));
            ggml_cuda_qpn_to_fp16(src0, w16.get(), ctx.stream());
            ggml_tensor src0_f16 = *src0;
            src0_f16.type     = GGML_TYPE_F16;
            src0_f16.data     = w16.get();
            src0_f16.flags   &= ~GGML_TENSOR_FLAG_BACKEND_LAYOUT;
            src0_f16.view_src = nullptr;
            src0_f16.nb[0]    = sizeof(half);
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16.nb[i] = src0_f16.nb[i - 1]*src0_f16.ne[i - 1];
            }
            ggml_cuda_mul_mat_cublas(ctx, &src0_f16, src1, dst);
        }
        return;
    }

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }
    if (hint == GGML_HINT_HC_PROJ && ggml_cuda_mul_mat_vec_hc(ctx, src0, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    // this product and its later siblings in one launch
    if (src0->type == GGML_TYPE_BF16 && ggml_cuda_sib_gemv_launch(ctx, dst)) {
        return;
    }
    if (src0->type == GGML_TYPE_BF16 && ggml_cuda_mul_mat_vec_bf16(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (!f32_pedantic && ggml_cuda_should_use_mmf(
            src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // 9 to 16 columns on a weight that is not repacked and takes MMVQ at 8 (the 27B's Q8_0 attention k and v, its IQ3_S
    // ffn down, the drafter's weights) run as MMVQ on two halves, [0, h) and [h, ne11) with h = ceil(ne11/2), instead of MMQ. On Volta
    // MMVQ computes a column the same way at any width from 5 to 8 (2 warps; 1 to 4 take 4), so every column is bit-identical to an
    // 8-column call, as the QPN products at 9 to 16 are (MMQ at 16 is not); at n + n tokens (the non-unified cache) the halves are the
    // two slots. The one half of fewer than 5 columns (at 9) runs padded to 5 with zero columns, in scratch. LLAMA_MMVQ_W16=0: MMQ.
    static const bool mmvq_w16 = [] { const char * e = getenv("LLAMA_MMVQ_W16"); return e == nullptr || atoi(e) != 0; }();
    if (mmvq_w16 && ne11 > MMVQ_MAX_BATCH_SIZE && ne11 <= 2*MMVQ_MAX_BATCH_SIZE && ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1 &&
            nb10 == sizeof(float) && nb0 == sizeof(float) && ggml_cuda_should_use_mmvq(src0->type, cc, MMVQ_MAX_BATCH_SIZE)) {
        constexpr int64_t c_min = MMVQ_MAX_BATCH_SIZE/2 + 1;
        const int64_t h = (ne11 + 1)/2;
        for (int64_t c0 = 0; c0 < ne11; c0 += h) {
            const int64_t nc = std::min<int64_t>(h, ne11 - c0);
            ggml_tensor src1_c = *src1;
            ggml_tensor dst_c  = *dst;
            src1_c.ne[1] = dst_c.ne[1] = std::max(nc, c_min);
            src1_c.data  = (char *) src1->data + c0*nb11;
            dst_c.data   = (char *) dst->data  + c0*nb1;
            ggml_cuda_pool_alloc<float> xpad(ctx.pool()), ypad(ctx.pool());
            if (nc < c_min) {
                cudaStream_t stream = ctx.stream();
                xpad.alloc(ne10*c_min);
                ypad.alloc(ne0*c_min);
                CUDA_CHECK(cudaMemsetAsync(xpad.get(), 0, ne10*c_min*sizeof(float), stream));
                CUDA_CHECK(cudaMemcpy2DAsync(xpad.get(), ne10*sizeof(float), src1_c.data, nb11, ne10*sizeof(float), nc,
                    cudaMemcpyDeviceToDevice, stream));
                src1_c.data  = xpad.get();
                src1_c.nb[1] = ne10*sizeof(float);
                dst_c.data   = ypad.get();
                dst_c.nb[1]  = ne0*sizeof(float);
            }
            for (int i = 2; i < GGML_MAX_DIMS; ++i) {
                src1_c.nb[i] = src1_c.nb[i - 1]*src1_c.ne[i - 1];
                dst_c.nb[i]  = dst_c.nb[i - 1]*dst_c.ne[i - 1];
            }
            ggml_cuda_mul_mat_vec_q(ctx, src0, &src1_c, nullptr, &dst_c);
            if (nc < c_min) {
                CUDA_CHECK(cudaMemcpy2DAsync((char *) dst->data + c0*nb1, nb1, ypad.get(), ne0*sizeof(float), ne0*sizeof(float), nc,
                    cudaMemcpyDeviceToDevice, ctx.stream()));
            }
        }
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // same gate as ggml_cuda_mul_mat_id: a pedantic F32 request never takes the mmf path
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (!f32_pedantic && ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (!f32_pedantic && ggml_cuda_should_use_mmf(
                src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;
        // [TAG_GGML_PREC] the slice stands in for dst: carry acc (0) and src precision (2, 3).
        // Not op_params[1], the op hint, which describes the unsliced operands.
        dst_slice.op_params[0] = dst->op_params[0];
        dst_slice.op_params[2] = dst->op_params[2];
        dst_slice.op_params[3] = dst->op_params[3];

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

// gated_delta_net launches with folded producer chains (see ggml_cuda_plan_gdn_folds)
#define GGML_CUDA_GDN_FOLD_MAX_TOKENS 8

struct ggml_cuda_gdn_fold_plan {
    std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_fold> gdn; // gated_delta_net node -> what it folds
    std::unordered_set<const ggml_tensor *>                     skip; // folded nodes, not launched
    // counts for the one log line per decode batch size
    int64_t n_tokens = 0, n_seqs = 0;
    int     n_gdn = 0, n_qknorm = 0, n_gates = 0, n_beta = 0, n_gate_kernel = 0, n_state = 0, n_conv = 0;
};

// planned at the start of every graph evaluation and read only during it, on the evaluating thread
static thread_local ggml_cuda_gdn_fold_plan ggml_cuda_gdn_folds;

// Glue folds that reach across other nodes (see ggml_cuda_plan_glue_folds): a producer is not
// launched where it stands, and the kernel of a later node that reads it computes it instead.
struct ggml_cuda_glue_plan {
    std::unordered_set<const ggml_tensor *>                     skip;        // not launched where they stand
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> conv_gather; // conv-input CONCAT -> its GET_ROWS
    std::unordered_map<const ggml_tensor *, std::pair<const ggml_tensor *, const ggml_tensor *>> norm_gate; // SIGMOID -> RMS_NORM, MUL
    struct moe_sum { const ggml_tensor * experts, * expert_scale, * weights; };
    std::unordered_map<const ggml_tensor *, moe_sum> moe_tail; // the shared expert's gate SIGMOID -> the weighted sum
    struct qsa_mask { const ggml_tensor * fill, * zeros; ggml_tensor * add; int n_skip; };
    std::unordered_map<const ggml_tensor *, qsa_mask> qsa_mask; // the mask's SET_ROWS -> its FILLs and ADD
};

// planned at the start of every graph evaluation and read only during it, on the evaluating thread
static thread_local ggml_cuda_glue_plan ggml_cuda_glue_folds;

static const ggml_cuda_gdn_fold * ggml_cuda_gdn_fold_of(const ggml_tensor * gdn) {
    const auto it = ggml_cuda_gdn_folds.gdn.find(gdn);
    return it == ggml_cuda_gdn_folds.gdn.end() ? nullptr : &it->second;
}

static __global__ void qsa_vis_rows_kernel(
        const uint32_t * vis, const int32_t * ids, const float * bias, float * out,
        int64_t n_words, int64_t width, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int32_t id = ids[i];
    assert(id >= 0 && id/32 < n_words);
    const uint32_t word = vis[(i/width)*n_words + (id >> 5)];
    out[i] = (((word >> (id & 31)) & 1) ? 0.0f : -INFINITY) + bias[i];
}

static void ggml_cuda_op_qsa_vis_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    qsa_vis_rows_kernel<<<(n + 255)/256, 256, 0, ctx.stream()>>>(
        (const uint32_t *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
        (const float *) dst->src[2]->data, (float *) dst->data, dst->src[0]->ne[0], dst->ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // a repacked weight holds no GGUF layout: only MUL_MAT may read it, as src0, and its fp16 expansion
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ggml_cuda_qpn_is_repacked(dst->src[i]) && !(dst->op == GGML_OP_MUL_MAT && i == 0) && !(i == 0 && ggml_cuda_qpn_is_expansion(dst))) {
            GGML_ABORT("%s: op %s reads %s, which is repacked for the tensor-core products (LLAMA_MMVQ_QPN=0 turns the repack off)",
                __func__, ggml_op_name(dst->op), dst->src[i]->name);
        }
    }
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            if (ggml_cuda_qpn_is_expansion(dst)) { // the kernel ggml_cuda_mul_mat's fallback runs, into the node
                GGML_ASSERT(ggml_are_same_shape(dst, dst->src[0]) && ggml_is_contiguous(dst));
                ggml_cuda_qpn_to_fp16(dst->src[0], (half *) dst->data, ctx.stream());
                break;
            }
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT_BANDED:
            ggml_cuda_flash_attn_ext_banded(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst, ggml_cuda_gdn_fold_of(dst));
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_ALLREDUCE:
            ggml_cuda_op_allreduce(ctx, dst);
            break;
        case GGML_OP_TOP_K_SPLIT:
            ggml_cuda_op_top_k_split(ctx, dst);
            break;
        case GGML_OP_DRAFT_PICK:
            ggml_cuda_op_draft_pick(ctx, dst);
            break;
        case GGML_OP_QSA_SELECT:
            ggml_cuda_op_qsa_select(ctx, dst);
            break;
        case GGML_OP_QSA_VIS_ROWS:
            ggml_cuda_op_qsa_vis_rows(ctx, dst);
            break;
        case GGML_OP_QSA_UNION:
            ggml_cuda_op_qsa_union(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    GGML_RT_COUNT("cuda.h2d_async", 1);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    GGML_RT_COUNT("cuda.d2h_async", 1);
    GGML_RT_COUNT("cuda.d2h_async_bytes", (int64_t) size);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    GGML_RT_COUNT("cuda.d2h_2d_async", 1);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    GGML_RT_SCOPE("cuda.sync");
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// a captured graph hard-codes its shapes, so with one key per split an alternating shape
// (a speculative verify batch) resets warmup forever. O(1) on purpose: walking nodes undoes the
// point of a cuda graph. A shape this fails to separate re-captures as before, so it cannot regress.
static uint64_t ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];

    auto mix = [&key](uint64_t v) {
        key = (key ^ v) * 0x100000001b3ull;
    };

    mix(cgraph->n_nodes);

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }

    return key;
}

static void ggml_cuda_rt_log_prop_change(const ggml_cuda_graph * graph, const ggml_cgraph * cgraph, int i,
        const ggml_cuda_graph::node_properties & a, const ggml_cuda_graph::node_properties & b) {
    // which property of node i changed, for the first changes after warmup (at most 40 lines per process)
    static int n_logged = 0;
    if (!graph->warmup_complete || n_logged >= 40) {
        return;
    }
    n_logged++;
    const char * what = "other (name, flags, extra, view)";
    if (a.node.data != b.node.data) what = "data";
    else if (memcmp(a.node.ne, b.node.ne, sizeof(a.node.ne))) what = "ne";
    else if (memcmp(a.node.nb, b.node.nb, sizeof(a.node.nb))) what = "nb";
    else if (memcmp(a.node.op_params, b.node.op_params, sizeof(a.node.op_params))) what = "op_params";
    else if (memcmp(a.node.src, b.node.src, sizeof(a.node.src))) what = "src tensor";
    else if (memcmp(a.node_src_data_ptrs, b.node_src_data_ptrs, sizeof(a.node_src_data_ptrs))) what = "src data";
    else if (memcmp(a.node_src_ne, b.node_src_ne, sizeof(a.node_src_ne))) what = "src ne";
    else if (memcmp(a.node_src_nb, b.node_src_nb, sizeof(a.node_src_nb))) what = "src nb";
    fprintf(stderr, "round_timers: cuda graph properties changed after warmup, label %d, node %d of %d (%s, %s): %s\n",
            ggml_rt_cur_label(), i, cgraph->n_nodes, ggml_op_desc(cgraph->nodes[i]), cgraph->nodes[i]->name, what);
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    GGML_RT_SCOPE("cg.propcheck");
    bool res = false;

    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        GGML_RT_COUNT("cg.uid_reuse", 1);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    GGML_RT_COUNT("cg.propcheck_full", 1);
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        if (ggml_rt_on() && graph->warmup_complete) {
            fprintf(stderr, "round_timers: cuda graph node count changed after warmup, label %d: %d -> %d\n",
                    ggml_rt_cur_label(), (int) graph->node_props.size(), cgraph->n_nodes);
        }
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            if (!res && ggml_rt_on()) {
                ggml_cuda_rt_log_prop_change(graph, cgraph, i, graph->node_props[i], prop);
            }
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        GGML_RT_COUNT("cg.execupdate_fail", 1);
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// A gated-residual product (GGML_HINT_HC_PROJ) followed only by its elementwise tail: the down
// projection's SCALE -> SILU, or the injection projection's SCALE -> SIGMOID -> SCALE (the combine's
// scatter weight). The product's kernel applies the tail at its output store and writes the last
// node, removing two or three launches. LLAMA_HC_EPILOGUE=0 keeps the separate nodes.
static int ggml_cuda_try_fuse_hc_epilogue(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool hc_epilogue = [] {
        const char * e = getenv("LLAMA_HC_EPILOGUE");
        return e == nullptr || atoi(e) != 0;
    }();

    ggml_tensor * node = cgraph->nodes[i];
    if (!hc_epilogue || node->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(node, 1) != GGML_HINT_HC_PROJ ||
        node->type != GGML_TYPE_F32) {
        return 0;
    }

    static const ggml_op sigmoid_ops[] = { GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE };
    static const ggml_op silu_ops[]    = { GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY };

    ggml_cuda_hc_epilogue epi;
    int n_ops = 0;
    if (ggml_can_fuse(cgraph, i, sigmoid_ops, 4) && ggml_get_unary_op(cgraph->nodes[i + 2]) == GGML_UNARY_OP_SIGMOID) {
        epi.kind = GGML_CUDA_HC_EPI_SCALE_SIGMOID_SCALE;
        n_ops    = 4;
    } else if (ggml_can_fuse(cgraph, i, silu_ops, 3) && ggml_get_unary_op(cgraph->nodes[i + 2]) == GGML_UNARY_OP_SILU) {
        epi.kind = GGML_CUDA_HC_EPI_SCALE_SILU;
        n_ops    = 3;
    } else {
        return 0;
    }

    for (int k = 1; k < n_ops; ++k) {
        const ggml_tensor * t = cgraph->nodes[i + k];
        if (t->type != GGML_TYPE_F32 || t->src[0] != cgraph->nodes[i + k - 1] || !ggml_is_contiguous(t)) {
            return 0;
        }
    }

    const ggml_tensor * scale0 = cgraph->nodes[i + 1];
    memcpy(&epi.s0, (const float *) scale0->op_params + 0, sizeof(float));
    memcpy(&epi.b0, (const float *) scale0->op_params + 1, sizeof(float));
    if (n_ops == 4) {
        const ggml_tensor * scale1 = cgraph->nodes[i + 3];
        memcpy(&epi.s1, (const float *) scale1->op_params + 0, sizeof(float));
        memcpy(&epi.b1, (const float *) scale1->op_params + 1, sizeof(float));
    }

    // the product reads its activations while other blocks write the tail's output
    const int out_idx = i + n_ops - 1;
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, &out_idx, 1)) {
        return 0;
    }

    if (!ggml_cuda_mul_mat_vec_hc(*cuda_ctx, node->src[0], node->src[1], node, &epi, cgraph->nodes[out_idx])) {
        return 0;
    }
    return n_ops - 1;
}

// A dim-0 CONCAT that builds a recurrent conv input, followed only by the rollback-slot saves of its
// tail ([TAG_RECURRENT_ROLLBACK_SPLITS] in qwen4exp's build_conv_state_at): per slot a CONT of a
// column window of the concat, then a CPY of it into the slot's row. The CONT of 12-byte rows runs
// as a 2D memcpy of ~10k rows, so one kernel writes the concat and every slot instead. The shared
// build_conv_state (qwen35 and the other delta-net models) saves each slot with a CPY straight from
// the window, which copies the same elements in the same order, so it folds the same way.
// LLAMA_GDN_CONV_SLOTS=0 keeps the separate nodes.
static int ggml_cuda_try_fuse_conv_slots(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool conv_slots = [] {
        const char * e = getenv("LLAMA_GDN_CONV_SLOTS");
        return e == nullptr || atoi(e) != 0;
    }();

    ggml_tensor * node = cgraph->nodes[i];
    if (!conv_slots || node->op != GGML_OP_CONCAT || ggml_get_op_params_i32(node, 0) != 0 ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || node->ne[3] != 1) {
        return 0;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return 0;
    }

    auto byte_range = [](const ggml_tensor * t) {
        return std::make_pair((uintptr_t) t->data, (uintptr_t) t->data + ggml_nbytes(t));
    };
    // an empty tensor (build_rs's copy of the extra states when there are none) touches no bytes, wherever it points
    auto overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const auto ra = byte_range(a), rb = byte_range(b);
        return ra.first < ra.second && rb.first < rb.second && ra.first < rb.second && rb.first < ra.second;
    };

    ggml_cuda_conv_slots slots;
    const ggml_tensor * dsts[GGML_CUDA_CONV_SLOTS_MAX];
    int last = 0;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if ((n->op != GGML_OP_CONT && n->op != GGML_OP_CPY) || slots.n_slots == GGML_CUDA_CONV_SLOTS_MAX) {
            break;
        }

        // CONT of a column window [off, off + n_cols) of every concat row, read by the CPY alone,
        // or (build_conv_state) the CPY of the window itself
        const bool          direct = n->op == GGML_OP_CPY;
        const ggml_tensor * win    = n->src[0];
        if (win->op != GGML_OP_VIEW || win->view_src != node || win->type != GGML_TYPE_F32 ||
            (!direct && (n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) || (n->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                         ggml_node_get_use_count(cgraph, j) != 1)) ||
            win->nb[0] != sizeof(float) || win->nb[1] != node->nb[1] || win->nb[2] != node->nb[2] ||
            win->ne[1] != node->ne[1] || win->ne[2] != node->ne[2] || win->ne[3] != 1 ||
            win->view_offs % sizeof(float) != 0) {
            break;
        }
        const int64_t n_cols = win->ne[0];
        const int64_t off    = win->view_offs / sizeof(float);
        if (off + n_cols > node->ne[0] || (slots.n_slots > 0 && n_cols != slots.n_cols)) {
            break;
        }

        // the CPY is the next real node: CONT -> [n_cols*channels, n_seqs] slot rows
        int k = j;
        if (!direct) {
            ++k;
            while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
                ++k;
            }
            if (k == cgraph->n_nodes) {
                break;
            }
        }
        const ggml_tensor * cpy = cgraph->nodes[k];
        const ggml_tensor * dst = cpy->src[1];
        if (cpy->op != GGML_OP_CPY || cpy->src[0] != (direct ? win : n) || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
            dst->nb[0] != sizeof(float) || dst->ne[0] != n_cols * node->ne[1] || dst->ne[1] != node->ne[2] ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
            break;
        }

        // the kernel reads the concat's sources and writes the slots in one pass
        bool clash = overlap(dst, node) || overlap(dst, src0) || overlap(dst, src1);
        for (int s = 0; s < slots.n_slots; ++s) {
            clash = clash || overlap(dst, dsts[s]);
        }
        if (clash) {
            break;
        }

        dsts[slots.n_slots]       = dst;
        slots.n_cols              = n_cols;
        slots.data[slots.n_slots] = (char *) dst->data;
        slots.off[slots.n_slots]  = off;
        slots.nb1[slots.n_slots]  = dst->nb[1];
        slots.n_slots++;
        last = k;
        j    = k;
    }

    if (slots.n_slots == 0) {
        return 0;
    }

    // LLAMA_FOLD_CONV_GATHER: the conv state's GET_ROWS was not launched; the kernel reads the cache rows
    // instead of src0, whose memory the allocator may then give to a later node such as the silu output
    const auto          gather = ggml_cuda_glue_folds.conv_gather.find(node);
    const ggml_tensor * rows   = gather != ggml_cuda_glue_folds.conv_gather.end() ? gather->second : nullptr;
    // several sequences read their cache rows only in the register kernel; otherwise the caller gathers first
    if (rows != nullptr && node->ne[2] > 1 && !ggml_cuda_conv_slots_reg_ok(node, slots)) {
        return 0;
    }

    // LLAMA_GDN_FUSE_CONV: the conv that reads this concat, SSM_CONV(concat, w) -> SILU, is computed
    // by the same threads while they hold each row, and its two nodes are not launched. It is looked for past the
    // slot saves, views, nodes folded into a gated_delta_net and other launches; since the silu output is now
    // written before those launches, none of them may read or write its bytes. The silu output may overlap the
    // transposed input only on elements of the same channel (the thread for a channel reads its elements before it
    // writes them), nothing else.
    static const bool fold_conv = [] {
        const char * e = getenv("LLAMA_GDN_FUSE_CONV");
        return e == nullptr || atoi(e) != 0;
    }();
    if (fold_conv) {
        std::vector<const ggml_tensor *> launched;
        int ic = -1;
        for (int j = last + 1; j < cgraph->n_nodes && j <= last + 16; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_SSM_CONV && n->src[0] == node) {
                ic = j;
                break;
            }
            if (!ggml_cuda_is_view_or_noop(n) && (n->flags & GGML_TENSOR_FLAG_COMPUTE) && !ggml_cuda_gdn_folds.skip.count(n)) {
                launched.push_back(n);
            }
        }
        const ggml_tensor * conv = ic >= 0 ? cgraph->nodes[ic] : nullptr;
        const ggml_tensor * silu = ic >= 0 && ic + 1 < cgraph->n_nodes ? cgraph->nodes[ic + 1] : nullptr;
        const ggml_tensor * w    = conv ? conv->src[1] : nullptr;
        const int64_t       n_t  = node->ne[0] - 3;
        bool ok = conv != nullptr && silu != nullptr &&
            silu->op == GGML_OP_UNARY && ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == conv &&
            silu->type == GGML_TYPE_F32 && conv->type == GGML_TYPE_F32 &&
            ggml_node_get_use_count(cgraph, ic) == 1 && !(conv->flags & GGML_TENSOR_FLAG_OUTPUT) &&
            w->type == GGML_TYPE_F32 && w->ne[0] == 4 && w->ne[1] == node->ne[1] && w->nb[0] == sizeof(float) &&
            n_t >= 1 && n_t <= 32 && node->ne[1] % 128 == 0 &&
            silu->ne[0] == node->ne[1] && silu->ne[1] == n_t && silu->ne[2] == node->ne[2] && silu->ne[3] == 1 &&
            silu->nb[0] == sizeof(float) &&
            // ssm_conv reads rows of exactly ne0 floats, as the concat writes them
            node->nb[1] == node->ne[0]*sizeof(float);
        if (ok) {
            // the silu output may overlap the transposed input where each of its elements lands on an input element
            // of the same channel, which the channel's thread has already read: laid out alike with rows of exactly
            // the channels, and either the same address or (one sequence) offset by whole rows
            const size_t    row_b   = silu->nb[1];
            const ptrdiff_t shift   = (const char *) silu->data - (const char *) src1->data;
            const bool alias_src1 = src1->nb[0] == row_b && src1->nb[1] == sizeof(float) &&
                                    row_b == (size_t) node->ne[1]*sizeof(float) &&
                                    ((shift == 0 && src1->nb[2] == silu->nb[2]) || (node->ne[2] == 1 && shift % (ptrdiff_t) row_b == 0));
            // with the gather folded the kernel reads the cache rows and their ids, not src0
            const bool src0_ok = rows != nullptr ? !overlap(silu, rows->src[0]) && !overlap(silu, rows->src[1]) : !overlap(silu, src0);
            ok = !overlap(silu, node) && src0_ok && !overlap(silu, w) && (alias_src1 || !overlap(silu, src1));
            for (int k = 0; k < slots.n_slots && ok; ++k) {
                ok = !overlap(silu, dsts[k]);
            }
            for (const ggml_tensor * n : launched) {
                ok = ok && !overlap(silu, n);
                for (int k = 0; k < GGML_MAX_SRC && ok; ++k) {
                    ok = n->src[k] == nullptr || !overlap(silu, n->src[k]);
                }
            }
        }
        if (ok) {
            slots.conv_w     = (const float *) w->data;
            slots.conv_w_nb1 = w->nb[1];
            slots.conv_y     = (float *) silu->data;
            slots.conv_y_nb1 = silu->nb[1];
            slots.conv_y_nb2 = silu->nb[2];
            ggml_cuda_gdn_folds.skip.insert(conv);
            ggml_cuda_gdn_folds.skip.insert(silu);
            ggml_cuda_gdn_folds.n_conv++;
        }
    }

    // with the conv folded, the slot windows and the conv may be the concat's only readers; then the register
    // kernel does not write the concat output
    if (slots.conv_y != nullptr && ggml_node_get_use_count(cgraph, i) == slots.n_slots + 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        ggml_cuda_conv_slots_reg_ok(node, slots)) {
        slots.write_dst = false;
    }

    // LLAMA_FOLD_CONV_GATHER: the conv state's GET_ROWS was not launched; read the cache rows here
    if (rows != nullptr) {
        slots.gather_rows = (const char *) rows->src[0]->data;
        slots.gather_ids  = (const int32_t *) rows->src[1]->data;
        slots.gather_nb1  = rows->src[0]->nb[1];
    }

    ggml_cuda_op_concat_conv_slots(*cuda_ctx, node, slots);
    return last - i;
}

// qwen4exp's QSA indexer keys (build_qsa_top_k): GET_ROWS of every block's r member rows from the
// indexer cache, CONT of each member slice with the ADDs between them, SCALE by 1/r, RMS_NORM, MUL by
// the norm weight and ROPE. Eleven launches that write f32 intermediates eight times over; one kernel
// reads the members once and writes the rope output (ggml_cuda_op_qsa_pool_rope).
// LLAMA_QSA_POOL_FUSE=0 keeps the separate nodes.
static int ggml_cuda_try_fuse_qsa_pool(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool qsa_pool_fuse = [] {
        const char * e = getenv("LLAMA_QSA_POOL_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();

    const ggml_tensor * gr = cgraph->nodes[i];
    if (!qsa_pool_fuse || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr)) {
        return 0;
    }
    const ggml_tensor * k     = gr->src[0];
    const ggml_tensor * cells = gr->src[1];
    if ((k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16) ||
        k->nb[0] != ggml_type_size(k->type) || k->ne[3] != 1 ||
        cells->type != GGML_TYPE_I32 || cells->nb[0] != sizeof(int32_t) || cells->ne[2] != 1 || cells->ne[3] != 1 ||
        cells->ne[1] != k->ne[2]) {
        return 0;
    }
    const int64_t ncols    = k->ne[0];
    const int64_t n_stream = k->ne[2];
    if (ncols % WARP_SIZE != 0 || (ncols / WARP_SIZE != 1 && ncols / WARP_SIZE != 2 && ncols / WARP_SIZE != 4 && ncols / WARP_SIZE != 8)) {
        return 0;
    }

    // a graph use count, and no output flag: nothing outside the chain reads the tensor
    auto only_used = [cgraph](const ggml_tensor * t, int n_uses) {
        const size_t h = ggml_hash_find(&cgraph->visited_hash_set, t);
        return ggml_bitset_get(cgraph->visited_hash_set.used, h) && cgraph->use_counts[h] == n_uses &&
               (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    };
    // a contiguous reshape of src that reshapes nothing else
    auto reshape_of = [&](const ggml_tensor * t, const ggml_tensor * src) {
        return t->op == GGML_OP_RESHAPE && t->src[0] == src && t->view_src == src && t->view_offs == 0 &&
               ggml_is_contiguous(t);
    };
    // the next node that does work, or nullptr. it must be one the backend would run.
    int j = i;
    auto next = [&]() -> ggml_tensor * {
        for (++j; j < cgraph->n_nodes; ++j) {
            ggml_tensor * n = cgraph->nodes[j];
            if (!ggml_cuda_is_view_or_noop(n)) {
                return (n->flags & GGML_TENSOR_FLAG_COMPUTE) ? n : nullptr;
            }
        }
        return nullptr;
    };

    if (!only_used(gr, 1)) {
        return 0;
    }

    // members = reshape(gr) to [ncols, r, n_blocks, n_stream]; slice m is view(members, offset m*nb1) -> CONT
    ggml_cuda_qsa_pool p;
    const ggml_tensor * members = nullptr;
    const ggml_tensor * pooled  = nullptr;
    int64_t n_blocks = 0;
    for (int m = 0; ; ++m) {
        const ggml_tensor * n = next();
        if (n == nullptr) {
            return 0;
        }
        if (m > 0 && n->op != GGML_OP_CONT) {
            if (members == nullptr || m != members->ne[1]) {
                return 0;
            }
            p.r = m;
            // the last ADD (or the only slice) goes on to SCALE
            if (n->op != GGML_OP_SCALE || n->src[0] != pooled || n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) ||
                !only_used(pooled, 1)) {
                return 0;
            }
            memcpy(&p.scale, (const float *) n->op_params + 0, sizeof(float));
            memcpy(&p.bias,  (const float *) n->op_params + 1, sizeof(float));
            pooled = n;
            break;
        }
        if (members != nullptr && m >= members->ne[1]) {
            return 0;
        }
        const ggml_tensor * view = n->src[0];
        if (n->op != GGML_OP_CONT || n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) ||
            view->op != GGML_OP_VIEW || view->view_src != gr || view->src[0] == nullptr || !only_used(view, 1) || !only_used(n, 1)) {
            return 0;
        }
        if (m == 0) {
            members = view->src[0];
            if (!reshape_of(members, gr) || members->ne[0] != ncols || members->ne[3] != n_stream ||
                members->ne[1] < 1 || members->ne[1] > 64 || !only_used(members, (int) members->ne[1])) {
                return 0;
            }
            n_blocks = members->ne[2];
            if (cells->ne[0] != members->ne[1]*n_blocks) {
                return 0;
            }
        }
        if (view->src[0] != members || view->type != GGML_TYPE_F32 ||
            view->ne[0] != ncols || view->ne[1] != n_blocks || view->ne[2] != n_stream || view->ne[3] != 1 ||
            view->nb[0] != sizeof(float) || view->nb[1] != members->nb[2] || view->nb[2] != members->nb[3] ||
            view->view_offs != (size_t) m*members->nb[1]) {
            return 0;
        }
        if (m == 0) {
            pooled = n;
            continue;
        }
        // pooled + slice, in that order
        const ggml_tensor * add = next();
        if (add == nullptr || add->op != GGML_OP_ADD || add->src[0] != pooled || add->src[1] != n ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || !ggml_are_same_shape(add, n) ||
            !only_used(pooled, 1)) {
            return 0;
        }
        pooled = add;
    }

    // RMS_NORM(reshape(scale)) -> MUL(norm, w) -> ROPE(reshape(mul), pos)
    const ggml_tensor * norm = next();
    if (norm == nullptr || norm->op != GGML_OP_RMS_NORM || norm->type != GGML_TYPE_F32 || !ggml_is_contiguous(norm) ||
        !reshape_of(norm->src[0], pooled) || norm->src[0]->ne[0] != ncols || !only_used(norm->src[0], 1) ||
        !only_used(pooled, 1) || !only_used(norm, 1)) {
        return 0;
    }
    memcpy(&p.eps, norm->op_params, sizeof(float));

    const ggml_tensor * mul = next();
    if (mul == nullptr || mul->op != GGML_OP_MUL || mul->src[0] != norm || mul->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mul, norm) || !ggml_is_contiguous(mul) || !only_used(mul, 1)) {
        return 0;
    }
    const ggml_tensor * w = mul->src[1];
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != ncols || ggml_nrows(w) != 1) {
        return 0;
    }

    // the rope may run in place over the MUL output, which the kernel never reads
    ggml_tensor * rope = next();
    if (rope == nullptr || rope->op != GGML_OP_ROPE) {
        return 0;
    }
    const ggml_tensor * rope_in = rope->src[0];
    const ggml_tensor * rope_pos = rope->src[1];
    const int mode = ggml_get_op_params_i32(rope, 2);
    if (!reshape_of(rope_in, mul) || !only_used(rope_in, 1) || rope->src[2] != nullptr ||
        rope_in->ne[0] != ncols || rope_in->ne[1] != 1 || rope_in->ne[3] != 1 ||
        rope->type != GGML_TYPE_F32 || !ggml_is_contiguous(rope) ||
        !(mode & GGML_ROPE_TYPE_MROPE) || mode == GGML_ROPE_TYPE_VISION ||
        rope_pos->type != GGML_TYPE_I32 || rope_pos->ne[0] != 4*rope_in->ne[2] || rope_in->ne[2] != n_blocks*n_stream) {
        return 0;
    }
    const int n_dims = ggml_get_op_params_i32(rope, 1);
    const int n_offs = ggml_get_op_params_i32(rope, 15);
    if (n_dims % 2 != 0 || n_offs % 2 != 0 || n_offs < 0 || n_offs + n_dims > ncols) {
        return 0;
    }

    // the kernel writes the rope output while it reads the cache, the cells, the positions and the weight
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if (overlap(rope, k) || overlap(rope, cells) || overlap(rope, rope_pos) || overlap(rope, w)) {
        return 0;
    }

    p.k     = k;
    p.cells = cells;
    p.w     = w;

    ggml_cuda_op_qsa_pool_rope(*cuda_ctx, rope, p);
    return j - i;
}

// LLAMA_GDN_FUSE_GATES: a GDN decay gate the fold plan could not put into its gated_delta_net launch,
// MUL(SOFTPLUS(ADD(alpha, dt)), a) with dt and a one value per row element, runs as one kernel instead of three.
// The ADD and SOFTPLUS outputs must be read by the next node only; the MUL output may be alpha itself.
static int ggml_cuda_try_fuse_gdn_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool fuse_gates = [] {
        const char * e = getenv("LLAMA_GDN_FUSE_GATES");
        return e == nullptr || atoi(e) != 0;
    }();
    const ggml_tensor * add = cgraph->nodes[i];
    if (!fuse_gates || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32) {
        return 0;
    }
    auto next_real = [&](int j) {
        for (++j; j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j]); ++j) {}
        return j;
    };
    const int j = next_real(i);
    const int k = j < cgraph->n_nodes ? next_real(j) : j;
    if (k >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sp  = cgraph->nodes[j];
    ggml_tensor *       mul = cgraph->nodes[k];
    const ggml_tensor * alpha = add->src[0];
    const ggml_tensor * dt    = add->src[1];
    const ggml_tensor * a     = mul->src[1];
    const int64_t       H     = add->ne[0];
    auto per_elem = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == H && ggml_nelements(t) == H;
    };
    auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const uintptr_t x0 = (uintptr_t) x->data, x1 = x0 + ggml_nbytes(x);
        const uintptr_t y0 = (uintptr_t) y->data, y1 = y0 + ggml_nbytes(y);
        return x0 < y1 && y0 < x1;
    };
    if (sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || sp->src[0] != add ||
        sp->type != GGML_TYPE_F32 || mul->op != GGML_OP_MUL || mul->src[0] != sp || mul->type != GGML_TYPE_F32 ||
        ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, j) != 1 ||
        (add->flags & GGML_TENSOR_FLAG_OUTPUT) || (sp->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !per_elem(dt) || !per_elem(a) || alpha->type != GGML_TYPE_F32 || !ggml_is_contiguous(alpha) ||
        !ggml_are_same_shape(alpha, add) || !ggml_are_same_shape(add, mul) || !ggml_is_contiguous(mul) ||
        (overlap(mul, alpha) && mul->data != alpha->data) || overlap(mul, dt) || overlap(mul, a)) {
        return 0;
    }
    ggml_cuda_op_gdn_gate(*cuda_ctx, alpha, dt, a, mul);
    ggml_cuda_gdn_folds.n_gate_kernel++;
    return k - i;
}

// Glue folds that reach across other nodes. Each is planned at the start of a graph evaluation:
// a producer is put in the skip set and computed by the kernel of the later node that reads it, which is only
// correct if nothing but the chain reads the producer (use counts, no output flag) and nothing launched in
// between overwrites what the fold now reads later than the producer did. Each fold has its own toggle.
//   LLAMA_FOLD_CONV_GATHER=0  the GDN conv state's GET_ROWS of one cache row, read by the conv-input CONCAT that
//                             ggml_cuda_try_fuse_conv_slots launches: that kernel reads the cache row itself
//   LLAMA_FOLD_NORM_GATE=0    the GDN output's gated norm, RMS_NORM -> MUL by the weight, then the z GEMV, then
//                             SIGMOID(z) -> MUL: the norm runs in the sigmoid's launch, after the GEMV (qwen35's
//                             SILU(z) -> MUL the same way)
//   LLAMA_FOLD_MOE_TAIL=0     the MoE weighted reduction (MUL, then the ADD chain over the experts) and, after the
//                             shared expert, its gated tail SIGMOID -> MUL -> ADD: the sum runs in the tail's launch
//   LLAMA_FOLD_QSA_MASK=0     qwen4exp's QSA attention mask, FILL(kq_mask, -inf) -> SET_ROWS of a FILL(0) at the
//                             selected cells -> ADD kq_mask: one launch at the SET_ROWS for all four nodes (=2: the
//                             staged kernel always, for the tests)
static void ggml_cuda_plan_glue_folds(const ggml_cgraph * cgraph) {
    static const bool fold_conv_gather = [] {
        const char * e = getenv("LLAMA_FOLD_CONV_GATHER");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_conv_gather_multi = [] {
        const char * e = getenv("LLAMA_FOLD_CONV_GATHER_MULTI");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_norm_gate = [] {
        const char * e = getenv("LLAMA_FOLD_NORM_GATE");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_moe_tail = [] {
        const char * e = getenv("LLAMA_FOLD_MOE_TAIL");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_qsa_mask = [] {
        const char * e = getenv("LLAMA_FOLD_QSA_MASK");
        return e == nullptr || atoi(e) != 0;
    }();
    // the concurrent-stream reorder can put the chain and the nodes between it on different streams
    static const bool graph_opt = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_OPT");
        return e != nullptr && atoi(e) == 1;
    }();

    ggml_cuda_glue_plan & plan = ggml_cuda_glue_folds;
    plan.skip.clear();
    plan.conv_gather.clear();
    plan.norm_gate.clear();
    plan.moe_tail.clear();
    plan.qsa_mask.clear();
    if (graph_opt || (!fold_conv_gather && !fold_norm_gate && !fold_moe_tail && !fold_qsa_mask)) {
        return;
    }

    // how many nodes read t, or -1 if t is not in this graph
    auto uses = [&](const ggml_tensor * t) {
        const size_t h = ggml_hash_find(&cgraph->visited_hash_set, t);
        return ggml_bitset_get(cgraph->visited_hash_set.used, h) ? cgraph->use_counts[h] : -1;
    };
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    // nothing launched strictly between first and last writes over t
    auto intact = [&](const ggml_tensor * t, int first, int last) {
        for (int j = first + 1; j < last; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || plan.skip.count(n) ||
                ggml_cuda_gdn_folds.skip.count(n)) {
                continue;
            }
            if (overlap(n, t)) {
                return false;
            }
        }
        return true;
    };

    // t is an F32 node read by exactly one node, and not a graph output
    auto once = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && uses(t) == 1;
    };
    // out may sit exactly in place of in (same bytes, same layout), or apart from it
    auto apart_or_same = [&](const ggml_tensor * out, const ggml_tensor * in) {
        return !overlap(out, in) || (out->data == in->data && ggml_are_same_layout(out, in));
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * norm = cgraph->nodes[i];
        if (!fold_norm_gate || norm->op != GGML_OP_RMS_NORM || !(norm->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            i + 1 >= cgraph->n_nodes || !once(norm)) {
            continue;
        }
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        if (mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm) || mul->src[0] == mul->src[1] || !once(mul)) {
            continue;
        }
        const ggml_tensor * x = norm->src[0];
        const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
            ggml_nelements(w) != x->ne[0] || w->ne[0] != x->ne[0] || !ggml_are_same_shape(mul, x)) {
            continue;
        }
        // the sigmoid of the gate and the product: the next few nodes, past the gate's GEMV. qwen35's build_norm_gated
        // gates with SILU instead; a SILU that is not this product does not end the search
        int is = -1;
        for (int j = i + 2; j < cgraph->n_nodes - 1 && j <= i + 12; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            const bool sigmoid = n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SIGMOID;
            const bool silu    = n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU;
            if (sigmoid || silu) {
                const ggml_tensor * m = cgraph->nodes[j + 1];
                if (m->op == GGML_OP_MUL && ((m->src[0] == mul && m->src[1] == n) || (m->src[1] == mul && m->src[0] == n))) {
                    is = j;
                    break;
                }
                if (sigmoid) {
                    break;
                }
            }
        }
        if (is < 0) {
            continue;
        }
        const ggml_tensor * sig = cgraph->nodes[is];
        const ggml_tensor * dst = cgraph->nodes[is + 1];
        const ggml_tensor * z   = sig->src[0];
        if (!once(sig) || !(sig->flags & GGML_TENSOR_FLAG_COMPUTE) || !(dst->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            dst->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_is_contiguous(dst) ||
            !ggml_are_same_shape(z, dst) || !ggml_are_same_shape(mul, dst) ||
            // the kernel reads x, w and z and writes dst; dst may sit in place of x or z, not over the weight
            !apart_or_same(dst, x) || !apart_or_same(dst, z) || overlap(dst, w) ||
            // x and w are now read after the nodes between the norm and the sigmoid (the gate's GEMV)
            !intact(x, i + 1, is) || !intact(w, i + 1, is)) {
            continue;
        }
        plan.skip.insert(norm);
        plan.skip.insert(mul);
        plan.norm_gate[sig] = { norm, mul };
    }

    for (int i = 0; i < cgraph->n_nodes && fold_moe_tail; ++i) {
        const ggml_tensor * first = cgraph->nodes[i];
        ggml_cuda_moe_weighted_reduction_match match;
        if (first->op != GGML_OP_MUL || !(first->flags & GGML_TENSOR_FLAG_COMPUTE) || plan.skip.count(first) ||
            !ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            continue;
        }
        const int last = i + match.node_count - 1;
        const ggml_tensor * moe = match.dst;
        if (!once(moe) || !ggml_is_contiguous(moe) || moe->ne[2] != 1 || moe->ne[3] != 1) {
            continue;
        }
        // the tail: SIGMOID(gate) -> MUL with the shared expert -> ADD to the sum, as ggml_cuda_try_fuse_glue's
        // LLAMA_FOLD_SHEXP_TAIL takes it, somewhere past the shared expert's GEMVs
        int is = -1;
        for (int j = last + 1; j + 2 < cgraph->n_nodes && j <= last + 48; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_ADD && (n->src[0] == moe || n->src[1] == moe)) {
                if (j >= last + 3) {
                    is = j - 2;
                }
                break;
            }
        }
        if (is < 0) {
            continue;
        }
        const ggml_tensor * sig = cgraph->nodes[is];
        const ggml_tensor * mul = cgraph->nodes[is + 1];
        const ggml_tensor * add = cgraph->nodes[is + 2];
        if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || !once(sig) ||
            mul->op != GGML_OP_MUL || (mul->src[0] == sig) == (mul->src[1] == sig) || !once(mul) ||
            add->op != GGML_OP_ADD || add->src[0] == add->src[1] || (add->src[0] != mul && add->src[1] != mul) ||
            !(sig->flags & GGML_TENSOR_FLAG_COMPUTE) || !(mul->flags & GGML_TENSOR_FLAG_COMPUTE) || !(add->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        const ggml_tensor * gate  = sig->src[0];
        const ggml_tensor * shexp = mul->src[0] == sig ? mul->src[1] : mul->src[0];
        const auto f32_2d = [](const ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[2] == 1 && t->ne[3] == 1;
        };
        if (!f32_2d(gate) || !f32_2d(shexp) || !f32_2d(add) || !ggml_is_contiguous(add) || gate->ne[0] != 1 ||
            gate->ne[1] != add->ne[1] || !ggml_are_same_shape(shexp, add) || !ggml_are_same_shape(moe, add) ||
            // the kernel reads the experts, the weights, the shared expert and the gate and writes the sum; the sum
            // may sit in place of the shared expert, nothing else
            overlap(add, match.experts) || overlap(add, match.weights) || overlap(add, gate) ||
            (match.expert_scale != nullptr && overlap(add, match.expert_scale)) || !apart_or_same(add, shexp) ||
            // the experts, the weights and the scale are now read after the nodes between the chain and the tail
            !intact(match.experts, last, is) || !intact(match.weights, last, is) ||
            (match.expert_scale != nullptr && !intact(match.expert_scale, last, is))) {
            continue;
        }
        for (int j = i; j <= last; ++j) {
            plan.skip.insert(cgraph->nodes[j]);
        }
        plan.moe_tail[sig] = { match.experts, match.expert_scale, match.weights };
    }

    for (int i = 0; i < cgraph->n_nodes && fold_qsa_mask; ++i) {
        const ggml_tensor * fill = cgraph->nodes[i];
        if (fill->op != GGML_OP_FILL || !(fill->flags & GGML_TENSOR_FLAG_COMPUTE) || (fill->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            fill->view_src != nullptr || uses(fill) != 1) {
            continue;
        }
        const ggml_tensor * mask = fill->src[0];
        if ((fill->type != GGML_TYPE_F16 && fill->type != GGML_TYPE_F32) || mask->type != fill->type ||
            !ggml_is_contiguous(fill) || fill->ne[2] != 1 || mask->nb[0] != ggml_type_size(mask->type) ||
            !ggml_are_same_shape(mask, fill)) {
            continue;
        }
        // [1, n_kv, n_batch, n_stream] view -> SET_ROWS -> view back to the fill's layout -> ADD mask: the next nodes
        const ggml_tensor * v1 = nullptr, * sr = nullptr, * v2 = nullptr;
        ggml_tensor * add = nullptr;
        int ia = -1, isr = -1;
        // views and the zero fill may sit between the fill and the SET_ROWS, only views between it and the ADD
        bool other = false;
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + 8 && add == nullptr && !other; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (v1 == nullptr && n->op == GGML_OP_VIEW && n->src[0] == fill) {
                v1 = n;
            } else if (v1 != nullptr && sr == nullptr && n->op == GGML_OP_SET_ROWS && n->src[2] == v1) {
                sr  = n;
                isr = j;
            } else if (sr != nullptr && v2 == nullptr && n->op == GGML_OP_VIEW && n->src[0] == sr) {
                v2 = n;
            } else if (v2 != nullptr && n->op == GGML_OP_ADD && n->src[0] == v2 && n->src[1] == mask) {
                add = cgraph->nodes[j];
                ia  = j;
            } else if (sr != nullptr || !(ggml_cuda_is_view_or_noop(n) || (n->op == GGML_OP_FILL && n->type == GGML_TYPE_F32))) {
                other = true;
            }
        }
        if (add == nullptr || other) {
            continue;
        }
        // set_rows keeps its operands in the legacy order: the rows, the cells, then the destination
        const ggml_tensor * zeros = sr->src[0];
        const ggml_tensor * idx   = sr->src[1];
        if (uses(v1) != 1 || uses(sr) != 1 || uses(v2) != 1 || (sr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (v1->flags & GGML_TENSOR_FLAG_OUTPUT) || (v2->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            zeros->op != GGML_OP_FILL || zeros->type != GGML_TYPE_F32 || zeros->view_src != nullptr ||
            (zeros->flags & GGML_TENSOR_FLAG_OUTPUT) || uses(zeros) != 1 || !(zeros->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) || !(add->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            !(sr->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            // set_rows writes rows of one cell: v1 = [1, n_kv, n_batch, n_stream] over the fill
            v1->ne[0] != 1 || v1->ne[1] != fill->ne[0] || v1->ne[2] != fill->ne[1] || v1->ne[3] != fill->ne[3] ||
            v1->data != fill->data || v1->nb[1] != fill->nb[0] || v1->nb[2] != fill->nb[1] || v1->nb[3] != fill->nb[2] ||
            zeros->ne[0] != 1 || zeros->ne[2] != fill->ne[1] || zeros->ne[3] != fill->ne[3] ||
            idx->ne[0] != zeros->ne[1] || fill->ne[1] % idx->ne[1] != 0 || fill->ne[3] % idx->ne[2] != 0 ||
            // v2 is the fill's own layout, and the ADD is elementwise over it
            v2->data != fill->data || !ggml_are_same_layout(v2, fill) || add->type != fill->type ||
            !ggml_are_same_shape(add, fill) || add->nb[0] != ggml_type_size(add->type) ||
            // the kernel reads the mask and the cells and writes the sum; the sum may not overlap the mask, and if it
            // overlaps the cells, one block stages all of them in shared memory first (at most 32 KiB)
            overlap(add, mask) || (overlap(add, idx) && ggml_nelements(idx)*ggml_type_size(idx->type) > 32768) ||
            !intact(mask, i, ia) || !intact(idx, i, ia)) {
            continue;
        }
        plan.skip.insert(zeros);
        plan.skip.insert(fill);
        plan.qsa_mask[sr] = { fill, zeros, add, ia - isr };
    }

    for (int i = 0; i < cgraph->n_nodes && fold_conv_gather; ++i) {
        const ggml_tensor * rows = cgraph->nodes[i];
        if (rows->op != GGML_OP_GET_ROWS || !(rows->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        // build_rs for one sequence: one whole F32 cache row, read only through a reshape by a dim-0 CONCAT. Or
        // for 2-4 sequences (LLAMA_FOLD_CONV_GATHER_MULTI=0: one only), where ggml_cuda_try_fuse_conv_slots launches the
        // concat only on its register kernel (one thread per channel for every sequence) and otherwise gathers first
        const ggml_tensor * cache = rows->src[0];
        const ggml_tensor * ids   = rows->src[1];
        const int64_t       n_sq  = rows->ne[1];
        if (rows->type != GGML_TYPE_F32 || cache->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            n_sq < 1 || n_sq > (fold_conv_gather_multi ? 4 : 1) || rows->ne[2] != 1 || rows->ne[3] != 1 ||
            ids->ne[0] != n_sq || ids->nb[0] != sizeof(int32_t) ||
            cache->ne[2] != 1 || cache->ne[3] != 1 || cache->nb[0] != sizeof(float) || rows->ne[0] != cache->ne[0] ||
            !ggml_is_contiguous(rows) || (rows->flags & GGML_TENSOR_FLAG_OUTPUT) || uses(rows) != 1) {
            continue;
        }
        const ggml_tensor * view = nullptr;
        int ic = -1;
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + 32 && ic < 0; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (n->src[k] == nullptr) {
                    continue;
                }
                if (view == nullptr && n->src[k] == rows) {
                    view = n;
                } else if (view != nullptr && n->src[k] == view) {
                    ic = j;
                }
            }
        }
        if (ic < 0) {
            continue;
        }
        const ggml_tensor * concat = cgraph->nodes[ic];
        if (view->op != GGML_OP_RESHAPE || view->view_src != rows || view->data != rows->data || !ggml_is_contiguous(view) ||
            (view->flags & GGML_TENSOR_FLAG_OUTPUT) || uses(view) != 1 ||
            concat->op != GGML_OP_CONCAT || concat->src[0] != view || ggml_get_op_params_i32(concat, 0) != 0 ||
            concat->ne[2] != n_sq || view->ne[2] != n_sq || view->ne[3] != 1 ||
            !intact(cache, i, ic) || !intact(ids, i, ic)) {
            continue;
        }
        plan.skip.insert(rows);
        plan.conv_gather[concat] = rows;
    }
}

// Glue folds. Each fold launches one kernel in place of a short elementwise chain and repeats
// the chain's arithmetic operation for operation, so its outputs are bit-identical. A chain folds only if its
// intermediate nodes have no reader outside the chain and no output flag, and no output overlaps an input.
//   LLAMA_FOLD_HC_NORM=0     DSV4_HC_POST (no comb) -> RMS_NORM -> MUL, the combine and the next mix's norm
//   LLAMA_FOLD_SHEXP_TAIL=0  UNARY(SIGMOID) -> MUL -> ADD, the shared expert's gate and the sum with the experts
//   LLAMA_FOLD_ADD_NORM=0    ADD -> RMS_NORM -> MUL, qwen35's residual add and the next trunk norm
// Counts the glue folds per graph and logs each distinct graph's counts once (the first fold of the next graph
// reports the previous one), so a server run at -lv 4 shows how many sites fold.
static void ggml_cuda_glue_fold_count(const ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i, int kind) {
    static const char * names[] = { "hc_norm", "shexp_tail", "conv_gather", "norm_gate", "attn_gate", "moe_tail", "norm_rope", "qsa_mask", "idx_sum",
                                    "add_norm" };
    constexpr int n_kinds = sizeof(names)/sizeof(names[0]);
    thread_local const ggml_cgraph * cur_graph = nullptr;
    thread_local int cur_i = -1, cur_nodes = 0, cur_dev = -1;
    thread_local int n[n_kinds] = {};
    if (cgraph != cur_graph || i <= cur_i || cgraph->n_nodes != cur_nodes) {
        if (cur_graph != nullptr) {
            char line[256];
            int len = snprintf(line, sizeof(line), "glue folds on CUDA%d in a graph of %d nodes:", cur_dev, cur_nodes);
            for (int k = 0; k < n_kinds && len < (int) sizeof(line); ++k) {
                len += snprintf(line + len, sizeof(line) - len, "%s %s %d", k ? "," : "", names[k], n[k]);
            }
            static std::mutex logged_mutex;
            static std::unordered_set<std::string> logged;
            std::lock_guard<std::mutex> lock(logged_mutex);
            if (logged.insert(line).second) {
                GGML_LOG_INFO("%s: %s\n", __func__, line);
            }
        }
        cur_graph = cgraph; cur_nodes = cgraph->n_nodes; cur_dev = cuda_ctx->device;
        for (int k = 0; k < n_kinds; ++k) {
            n[k] = 0;
        }
    }
    cur_i = i;
    GGML_ASSERT(kind >= 0 && kind < n_kinds);
    n[kind]++;
}

static int ggml_cuda_try_fuse_glue(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool fold_hc_norm = [] {
        const char * e = getenv("LLAMA_FOLD_HC_NORM");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_shexp_tail = [] {
        const char * e = getenv("LLAMA_FOLD_SHEXP_TAIL");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_attn_gate = [] {
        const char * e = getenv("LLAMA_FOLD_ATTN_GATE");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_norm_rope = [] {
        const char * e = getenv("LLAMA_FOLD_NORM_ROPE");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_idx_sum = [] {
        const char * e = getenv("LLAMA_FOLD_IDX_SUM");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool fold_add_norm = [] {
        const char * e = getenv("LLAMA_FOLD_ADD_NORM");
        return e == nullptr || atoi(e) != 0;
    }();

    ggml_tensor * node = cgraph->nodes[i];
    const auto glue_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const auto glue_same_view = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a->data == b->data && ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_nelements(a) == ggml_nelements(b) &&
               a->ne[0] == b->ne[0];
    };
    const auto f32_4d1 = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && t->ne[3] == 1 && t->nb[0] == sizeof(float); };

    // LLAMA_FOLD_CONV_GATHER: the conv input whose state gather was not launched
    if (node->op == GGML_OP_CONCAT && !ggml_cuda_glue_folds.conv_gather.empty()) {
        const auto it = ggml_cuda_glue_folds.conv_gather.find(node);
        if (it != ggml_cuda_glue_folds.conv_gather.end()) {
            if (const int n_skip = ggml_cuda_try_fuse_conv_slots(cuda_ctx, cgraph, i)) {
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 2);
                return n_skip;
            }
            // the slot fold refused this concat: gather now, then let the concat run as it would have
            ggml_tensor * rows = const_cast<ggml_tensor *>(it->second);
            ggml_cuda_glue_folds.conv_gather.erase(it);
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, rows));
            return 0;
        }
    }

    // LLAMA_FOLD_ADD_NORM: the residual ADD -> RMS_NORM -> MUL by the norm weight, qwen35's trunk. Both the
    // sum (read again by the next residual ADD) and the normed output are written. The sum and the normed output may
    // each sit in place of an addend of the same layout, and the normed output in place of a sum that nothing else
    // reads (the last layer's, before the output norm); nothing may overlap the weight
    if (fold_add_norm && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        ggml_tensor * norm = cgraph->nodes[i + 1];
        ggml_tensor * mul  = cgraph->nodes[i + 2];
        const auto folded = [](const ggml_tensor * t) {
            return (!ggml_cuda_glue_folds.skip.empty() && ggml_cuda_glue_folds.skip.count(t)) ||
                   (!ggml_cuda_gdn_folds.skip.empty() && ggml_cuda_gdn_folds.skip.count(t));
        };
        if (norm->op == GGML_OP_RMS_NORM && norm->src[0] == node && mul->op == GGML_OP_MUL &&
            (mul->src[0] == norm) != (mul->src[1] == norm) && ggml_node_has_n_uses(cgraph, i + 1, 1) &&
            !(norm->flags & GGML_TENSOR_FLAG_OUTPUT) && (norm->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            (mul->flags & GGML_TENSOR_FLAG_COMPUTE) && !folded(norm) && !folded(mul)) {
            const ggml_tensor * a = node->src[0];
            const ggml_tensor * b = node->src[1];
            const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
            const auto apart_or_same = [&](const ggml_tensor * out, const ggml_tensor * in) {
                return !glue_overlap(out, in) || (out->data == in->data && ggml_are_same_layout(out, in));
            };
            if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && norm->type == GGML_TYPE_F32 &&
                mul->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) &&
                node->ne[2] == 1 && node->ne[3] == 1 && ggml_are_same_shape(a, node) && ggml_are_same_shape(b, node) &&
                ggml_are_same_shape(mul, node) && ggml_is_contiguous(node) && ggml_is_contiguous(mul) &&
                ggml_is_contiguous(w) && w->ne[0] == node->ne[0] && ggml_nelements(w) == node->ne[0] &&
                apart_or_same(node, a) && apart_or_same(node, b) && apart_or_same(mul, a) && apart_or_same(mul, b) &&
                (!glue_overlap(mul, node) || (mul->data == node->data && ggml_are_same_layout(mul, node) &&
                                              ggml_node_has_n_uses(cgraph, i, 1) && !(node->flags & GGML_TENSOR_FLAG_OUTPUT))) &&
                !glue_overlap(node, w) && !glue_overlap(mul, w)) {
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 9);
                ggml_cuda_op_add_rms_norm_mul(*cuda_ctx, node, norm, mul);
                return 2;
            }
        }
    }

    // under --split-mode tensor: the AllReduce followed by the LLAMA_FOLD_ADD_NORM pattern above (its
    // ADD reading the AllReduce), in one kernel; the same conditions as that fold, plus few rows (allreduce-p2p.cuh). Only
    // the meta backend creates ALLREDUCE nodes, so one card never reaches this. Or the ADD reading a RESHAPE of
    // the AllReduce read by it alone (qwen35's GDN output, [n_embd, tokens, sequences] reshaped to [n_embd, rows]), which
    // the kernel reads as the same contiguous floats
    if (fold_add_norm && node->op == GGML_OP_ALLREDUCE && i + 3 < cgraph->n_nodes) {
        static const bool fold_reshape = [] {
            const char * e = getenv("LLAMA_FOLD_AR_RESHAPE");
            return e == nullptr || atoi(e) != 0;
        }();
        int ia = i + 1;
        const ggml_tensor * ar_in = node;
        {
            ggml_tensor * rs = cgraph->nodes[i + 1];
            if (fold_reshape && rs->op == GGML_OP_RESHAPE && rs->src[0] == node && rs->data == node->data &&
                rs->type == GGML_TYPE_F32 && ggml_is_contiguous(rs) && ggml_is_contiguous(node) &&
                ggml_nelements(rs) == ggml_nelements(node) && rs->ne[0] == node->ne[0] &&
                // the use counts themselves: ggml_node_has_n_uses refuses every view, and the reshape is one (of the product,
                // which the AllReduce alone reads)
                ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                !(rs->flags & GGML_TENSOR_FLAG_OUTPUT) && i + 4 < cgraph->n_nodes) {
                ia    = i + 2;
                ar_in = rs;
            }
        }
        ggml_tensor * add  = cgraph->nodes[ia];
        ggml_tensor * norm = cgraph->nodes[ia + 1];
        ggml_tensor * mul  = cgraph->nodes[ia + 2];
        const auto folded = [](const ggml_tensor * t) {
            return (!ggml_cuda_glue_folds.skip.empty() && ggml_cuda_glue_folds.skip.count(t)) ||
                   (!ggml_cuda_gdn_folds.skip.empty() && ggml_cuda_gdn_folds.skip.count(t));
        };
        if (add->op == GGML_OP_ADD && (add->src[0] == ar_in) != (add->src[1] == ar_in) && (add->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            norm->op == GGML_OP_RMS_NORM && norm->src[0] == add && mul->op == GGML_OP_MUL &&
            (mul->src[0] == norm) != (mul->src[1] == norm) && ggml_node_has_n_uses(cgraph, ia + 1, 1) &&
            !(norm->flags & GGML_TENSOR_FLAG_OUTPUT) && (norm->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            (mul->flags & GGML_TENSOR_FLAG_COMPUTE) && !folded(add) && !folded(norm) && !folded(mul) &&
            ggml_cuda_op_allreduce_add_rms_norm_mul_supported(ar_in, add)) {
            const ggml_tensor * a = add->src[0];
            const ggml_tensor * b = add->src[1];
            const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
            const auto apart_or_same = [&](const ggml_tensor * out, const ggml_tensor * in) {
                return !glue_overlap(out, in) || (out->data == in->data && ggml_are_same_layout(out, in));
            };
            if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && norm->type == GGML_TYPE_F32 &&
                mul->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) &&
                add->ne[2] == 1 && add->ne[3] == 1 && ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) &&
                ggml_are_same_shape(mul, add) && ggml_is_contiguous(add) && ggml_is_contiguous(mul) &&
                ggml_is_contiguous(w) && w->ne[0] == add->ne[0] && ggml_nelements(w) == add->ne[0] &&
                apart_or_same(add, a) && apart_or_same(add, b) && apart_or_same(mul, a) && apart_or_same(mul, b) &&
                (!glue_overlap(mul, add) || (mul->data == add->data && ggml_are_same_layout(mul, add) &&
                                             ggml_node_has_n_uses(cgraph, ia, 1) && !(add->flags & GGML_TENSOR_FLAG_OUTPUT))) &&
                !glue_overlap(add, w) && !glue_overlap(mul, w) && !glue_overlap(node, w) &&
                // through a reshape the normed output may also sit exactly in place of the AllReduce's floats: the
                // product's memory is free once the ADD has read it, and the kernel's block for a row reads that row of the partial
                // before it writes the row's outputs, while the AllReduce's own output has no reader but the fused ADD
                (!glue_overlap(mul, node) || mul->data == add->data ||
                 (ar_in != node && mul->data == node->data && ggml_are_same_layout(mul, ar_in)))) {
                // the next products' first weight bytes into L2 while the AllReduce waits: the QPN-repacked
                // products that read the normed output, among the next nodes (the layer's input products or the FFN's gate
                // and up), each tile's first budget / (all their tiles) bytes
                ggml_cuda_p2p_ar_pf pf;
                if (const int64_t budget = ggml_cuda_p2p_ar_pf_budget()) {
                    int64_t n_tiles = 0;
                    for (int j = ia + 3; j < cgraph->n_nodes && j <= ia + 24 && pf.n < GGML_CUDA_P2P_AR_PF_MAX; ++j) {
                        const ggml_tensor * n  = cgraph->nodes[j];
                        const ggml_tensor * wt = n->src[0];
                        if (n->op != GGML_OP_MUL_MAT || n->src[1] != mul || !ggml_cuda_qpn_is_repacked(wt) ||
                            wt->ne[1] % 32 != 0 || wt->ne[0] % QK_K != 0 || !ggml_is_contiguous(wt)) {
                            continue;
                        }
                        pf.W[pf.n]          = (const char *) wt->data;
                        pf.tile_bytes[pf.n] = (int64_t) (wt->ne[0] / QK_K) * 32 * (int64_t) ggml_row_size(wt->type, QK_K);
                        pf.ntiles[pf.n]     = (int) (wt->ne[1] / 32);
                        n_tiles += pf.ntiles[pf.n];
                        pf.n++;
                    }
                    const int64_t pre = n_tiles > 0 ? budget / n_tiles / 32 * 32 : 0;
                    for (int w = 0; w < pf.n; ++w) {
                        pf.pre_bytes[w] = std::min(pre, pf.tile_bytes[w]);
                    }
                    if (pre == 0) {
                        pf.n = 0;
                    }
                }
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, ia, 9);
                ggml_cuda_op_allreduce_add_rms_norm_mul(*cuda_ctx, node, ar_in, add, norm, mul, &pf);
                return ia + 2 - i;
            }
        }
    }

    // LLAMA_FOLD_ATTN_GATE: CONT of the gate's view -> SIGMOID -> MUL by the attention output, the
    // gate read through the view. The product may sit in place of the attention output, not over the gate's source
    if (fold_attn_gate && node->op == GGML_OP_CONT && i + 2 < cgraph->n_nodes) {
        ggml_tensor * sig = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        if (sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0] == node &&
            mul->op == GGML_OP_MUL && (mul->src[0] == sig) != (mul->src[1] == sig) &&
            ggml_node_has_n_uses(cgraph, i, 1) && ggml_node_has_n_uses(cgraph, i + 1, 1)) {
            const ggml_tensor * gate = node->src[0];
            const ggml_tensor * attn = mul->src[0] == sig ? mul->src[1] : mul->src[0];
            if (gate->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32 &&
                mul->type == GGML_TYPE_F32 && attn->type == GGML_TYPE_F32 && !(sig->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                ggml_is_contiguous(mul) && ggml_are_same_shape(node, mul) && ggml_are_same_shape(attn, mul) &&
                attn->nb[0] == sizeof(float) && ggml_is_contiguous_1(attn) && attn->ne[2] == 1 && attn->ne[3] == 1 &&
                ggml_nelements(gate) == ggml_nelements(mul) && gate->nb[0] % sizeof(float) == 0 &&
                !glue_overlap(mul, gate) && (!glue_overlap(mul, attn) || (mul->data == attn->data && ggml_are_same_layout(mul, attn)))) {
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 4);
                ggml_cuda_op_cont_sigmoid_mul(*cuda_ctx, node, sig, mul);
                return 2;
            }
        }
    }

    // LLAMA_FOLD_NORM_ROPE: RMS_NORM -> MUL by the weight -> ROPE multi, the q, k and indexer q
    // heads; the rotated output may sit in place of the norm's input (same layout), nothing else
    if (fold_norm_rope && node->op == GGML_OP_RMS_NORM && i + 2 < cgraph->n_nodes) {
        ggml_tensor * mul  = cgraph->nodes[i + 1];
        ggml_tensor * rope = cgraph->nodes[i + 2];
        const int mode = rope->op == GGML_OP_ROPE ? ggml_get_op_params_i32(rope, 2) : 0;
        if (mul->op == GGML_OP_MUL && (mul->src[0] == node) != (mul->src[1] == node) && rope->op == GGML_OP_ROPE &&
            rope->src[0] == mul && (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION &&
            ggml_node_has_n_uses(cgraph, i, 1) && ggml_node_has_n_uses(cgraph, i + 1, 1)) {
            const ggml_tensor * x   = node->src[0];
            const ggml_tensor * w   = mul->src[0] == node ? mul->src[1] : mul->src[0];
            const ggml_tensor * pos = rope->src[1];
            const ggml_tensor * ff  = rope->src[2];
            const int n_dims = ggml_get_op_params_i32(rope, 1);
            const int n_offs = ggml_get_op_params_i32(rope, 15);
            if (x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
                rope->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && pos->type == GGML_TYPE_I32 &&
                (ff == nullptr || ff->type == GGML_TYPE_F32) && x->nb[0] == sizeof(float) && rope->nb[0] == sizeof(float) &&
                ggml_is_contiguous(w) && w->ne[0] == x->ne[0] && ggml_nelements(w) == x->ne[0] &&
                ggml_are_same_shape(x, mul) && ggml_are_same_shape(mul, rope) && x->ne[0] % 2 == 0 &&
                n_dims % 2 == 0 && n_offs % 2 == 0 && n_offs >= 0 && n_offs + n_dims <= x->ne[0] &&
                (!glue_overlap(rope, x) || (rope->data == x->data && ggml_are_same_layout(rope, x))) &&
                !glue_overlap(rope, w) && !glue_overlap(rope, pos) && (ff == nullptr || !glue_overlap(rope, ff))) {
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 6);
                ggml_cuda_op_rms_norm_mul_rope_multi(*cuda_ctx, node, mul, rope);
                return 2;
            }
        }
    }

    // LLAMA_FOLD_IDX_SUM: the QSA indexer's score, RELU over [n_blocks, n_h, n_t, n_stream], then the
    // sum over the heads, CONT of head 0's view and one ADD per further head, in order. Only views and the chain
    // lie between the RELU and the last ADD, which is written once. The RELU's own output is never written, so the
    // sum may sit on it; if the sum sits on the RELU's input, one block reads all of it first, if it can hold it
    if (fold_idx_sum && node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU &&
        node->type == GGML_TYPE_F32 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && node->view_src == nullptr) {
        const ggml_tensor * x   = node->src[0];
        const int           n_h = (int) node->ne[1];
        const auto head_view = [&](const ggml_tensor * v, int h) {
            return v->op == GGML_OP_VIEW && v->src[0] == node && v->type == GGML_TYPE_F32 &&
                   v->view_offs == (size_t) h*node->nb[1] && v->ne[0] == node->ne[0] && v->ne[1] == node->ne[2] &&
                   v->ne[2] == node->ne[3] && v->ne[3] == 1 && v->nb[0] == node->nb[0] && v->nb[1] == node->nb[2] &&
                   v->nb[2] == node->nb[3];
        };
        int h = 0, last = -1;
        const ggml_tensor * acc = nullptr;
        bool ok = x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_are_same_shape(x, node) && n_h >= 2 &&
                  ggml_node_get_use_count(cgraph, i) == n_h;
        for (int j = i + 1; ok && j < cgraph->n_nodes && j <= i + 4*n_h && last < 0; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_VIEW) {
                // each head's view is checked where it is read; nothing else may read one
                ok = n->src[0] != node || ggml_node_get_use_count(cgraph, j) == 1;
                continue;
            }
            if (h == 0) {
                ok = n->op == GGML_OP_CONT && head_view(n->src[0], 0) && n->type == GGML_TYPE_F32 && ggml_is_contiguous(n);
            } else {
                ok = n->op == GGML_OP_ADD && n->src[0] == acc && head_view(n->src[1], h) && n->type == GGML_TYPE_F32 &&
                     ggml_is_contiguous(n) && ggml_are_same_shape(n, acc);
            }
            ok = ok && (n->flags & GGML_TENSOR_FLAG_COMPUTE);
            if (ok && ++h == n_h) {
                last = j;
            } else if (ok) {
                ok = ggml_node_has_n_uses(cgraph, j, 1);
            }
            acc = n;
        }
        if (ok && last > 0) {
            ggml_tensor * dst = cgraph->nodes[last];
            if (!glue_overlap(dst, x) || ggml_cuda_relu_head_sum_fits_staged(dst)) {
                ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 8);
                ggml_cuda_op_relu_head_sum(*cuda_ctx, node, dst);
                return last - i;
            }
        }
    }

    // LLAMA_FOLD_QSA_MASK: the planned QSA mask, launched at its SET_ROWS, through its ADD
    if (node->op == GGML_OP_SET_ROWS && !ggml_cuda_glue_folds.qsa_mask.empty()) {
        const auto it = ggml_cuda_glue_folds.qsa_mask.find(node);
        if (it != ggml_cuda_glue_folds.qsa_mask.end()) {
            ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 7);
            ggml_cuda_op_qsa_mask_fill_rows_add(*cuda_ctx, it->second.fill, it->second.zeros, node, it->second.add);
            return it->second.n_skip;
        }
    }

    // LLAMA_FOLD_MOE_TAIL: the planned weighted expert sum, launched with the shared expert's tail
    if (node->op == GGML_OP_UNARY && !ggml_cuda_glue_folds.moe_tail.empty()) {
        const auto it = ggml_cuda_glue_folds.moe_tail.find(node);
        if (it != ggml_cuda_glue_folds.moe_tail.end()) {
            ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 5);
            ggml_cuda_op_moe_weighted_reduction_shexp_tail(*cuda_ctx, it->second.experts, it->second.expert_scale,
                    it->second.weights, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    // LLAMA_FOLD_NORM_GATE: the planned gated norm, launched at its sigmoid
    if (node->op == GGML_OP_UNARY && !ggml_cuda_glue_folds.norm_gate.empty()) {
        const auto it = ggml_cuda_glue_folds.norm_gate.find(node);
        if (it != ggml_cuda_glue_folds.norm_gate.end()) {
            ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 3);
            ggml_cuda_op_rms_norm_mul_sigmoid_gate(*cuda_ctx, it->second.first, it->second.second, node, cgraph->nodes[i + 1]);
            return 1;
        }
    }

    if (fold_hc_norm && node->op == GGML_OP_DSV4_HC_POST && node->src[3] == nullptr && i + 2 < cgraph->n_nodes) {
        ggml_tensor * norm = cgraph->nodes[i + 1];
        ggml_tensor * mul  = cgraph->nodes[i + 2];
        if (norm->op != GGML_OP_RMS_NORM || norm->src[0] != node || mul->op != GGML_OP_MUL ||
            (mul->src[0] != norm && mul->src[1] != norm) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
            return 0;
        }
        const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (!f32_4d1(node) || !f32_4d1(norm) || !f32_4d1(mul) || !f32_4d1(w) || !f32_4d1(node->src[0]) ||
            !f32_4d1(node->src[1]) || !f32_4d1(node->src[2]) || !ggml_is_contiguous(node) || !ggml_is_contiguous(mul) ||
            !ggml_are_same_shape(mul, node) || w->ne[0] != node->ne[0] || node->src[1]->ne[1] != node->ne[1]) {
            return 0;
        }
        // the combine's output may sit in place of its residual, and the normed output in place of the residual
        // (each block reads its residual row before it stores that row); any other overlap refuses the fold
        const ggml_tensor * ins[] = { node->src[0], node->src[1], node->src[2], w };
        for (const ggml_tensor * out : { (const ggml_tensor *) node, (const ggml_tensor *) mul }) {
            for (const ggml_tensor * in : ins) {
                if (glue_overlap(out, in) && !(in == node->src[1] && glue_same_view(out, in))) {
                    return 0;
                }
            }
        }
        if (glue_overlap(node, mul)) {
            return 0;
        }
        ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 0);
        ggml_cuda_op_hc_post_rms_norm_mul(*cuda_ctx, node, norm, mul);
        return 2;
    }

    if (fold_shexp_tail && node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
        i + 2 < cgraph->n_nodes) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * add = cgraph->nodes[i + 2];
        if (mul->op != GGML_OP_MUL || (mul->src[0] != node && mul->src[1] != node) || add->op != GGML_OP_ADD ||
            (add->src[0] != mul && add->src[1] != mul) || mul->src[0] == mul->src[1] || add->src[0] == add->src[1] ||
            !ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
            return 0;
        }
        const ggml_tensor * gate  = node->src[0];
        const ggml_tensor * shexp = mul->src[0] == node ? mul->src[1] : mul->src[0];
        const ggml_tensor * moe   = add->src[0] == mul ? add->src[1] : add->src[0];
        // gate [1, T], shexp, moe and the sum [n_embd, T]
        if (!f32_4d1(gate) || !f32_4d1(shexp) || !f32_4d1(moe) || !f32_4d1(node) || !f32_4d1(mul) || !f32_4d1(add) ||
            !ggml_is_contiguous(add) || gate->ne[0] != 1 || gate->ne[2] != 1 || gate->ne[1] != add->ne[1] ||
            !ggml_are_same_shape(shexp, add) || !ggml_are_same_shape(moe, add) || !ggml_are_same_shape(mul, add) ||
            add->ne[2] != 1) {
            return 0;
        }
        // elementwise: the sum may sit in place of either operand of the same layout, not of the gate
        if ((glue_overlap(add, moe) && !glue_same_view(add, moe)) || (glue_overlap(add, shexp) && !glue_same_view(add, shexp)) ||
            glue_overlap(add, gate)) {
            return 0;
        }
        ggml_cuda_glue_fold_count(cuda_ctx, cgraph, i, 1);
        ggml_cuda_op_shexp_gate_tail(*cuda_ctx, node, mul, add);
        return 2;
    }

    return 0;
}

// try and fuse nodes and return the number of nodes to skip
// the ALLREDUCE node that reduces node i in place, if it follows it (after views of it only): the meta backend
// appends it right after its partial. Only the meta backend creates ALLREDUCE nodes, so one card always gets nullptr
static const ggml_tensor * ggml_cuda_p2p_ar_of(const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    const ggml_tensor * prev = node;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 3; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_ALLREDUCE) {
            return n->src[0] == prev && n->data == node->data && ggml_nelements(n) == ggml_nelements(node) &&
                   (n->flags & GGML_TENSOR_FLAG_COMPUTE) ? n : nullptr;
        }
        if (!ggml_cuda_is_view_or_noop(n) || n->src[0] != prev || n->data != node->data) {
            return nullptr;
        }
        prev = n;
    }
    return nullptr;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (const int n_skip = ggml_cuda_try_fuse_glue(cuda_ctx, cgraph, i)) { return n_skip; }

    if (const int n_skip = ggml_cuda_try_fuse_hc_epilogue(cuda_ctx, cgraph, i)) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_conv_slots(cuda_ctx, cgraph, i)) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_qsa_pool(cuda_ctx, cgraph, i)) {
        return n_skip;
    }

    if (const int n_skip = ggml_cuda_try_fuse_gdn_gate(cuda_ctx, cgraph, i)) {
        return n_skip;
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy, ggml_cuda_gdn_fold_of(node));
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // (LLAMA_FOLD_NORM2_CONCAT, default on) the MTP draft step's input: RMS_NORM -> MUL of the hidden row and of the
    // token embedding, then their CONCAT along dim 0, in one kernel (norm.cu's rms_norm_mul2_concat_f32, the norms' arithmetic as is)
    static const bool fold_norm2_concat = [] { const char * e = getenv("LLAMA_FOLD_NORM2_CONCAT"); return e == nullptr || atoi(e) != 0; }();
    if (fold_norm2_concat && node->op == GGML_OP_RMS_NORM && i + 4 < cgraph->n_nodes) {
        ggml_tensor * m_a = cgraph->nodes[i + 1];
        ggml_tensor * n_b = cgraph->nodes[i + 2];
        ggml_tensor * m_b = cgraph->nodes[i + 3];
        ggml_tensor * cc  = cgraph->nodes[i + 4];
        // the concat's output must not overlap either row it reads (the allocator may place it over a norm's dead input)
        const auto norm2_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        const auto norm_mul = [&](const ggml_tensor * n, const ggml_tensor * m, int i_n) {
            if (n->op != GGML_OP_RMS_NORM || m->op != GGML_OP_MUL || ((m->src[0] == n) == (m->src[1] == n))) {
                return false;
            }
            const ggml_tensor * x = n->src[0];
            const ggml_tensor * w = m->src[0] == n ? m->src[1] : m->src[0];
            return x->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
                x->nb[0] == sizeof(float) && x->ne[2] == 1 && x->ne[3] == 1 && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0] &&
                ggml_are_same_shape(m, x) && ggml_node_has_n_uses(cgraph, i_n, 1) && ggml_node_has_n_uses(cgraph, i_n + 1, 1) &&
                !(n->flags & GGML_TENSOR_FLAG_OUTPUT) && !(m->flags & GGML_TENSOR_FLAG_OUTPUT);
        };
        if (norm_mul(node, m_a, i) && norm_mul(n_b, m_b, i + 2) && cc->op == GGML_OP_CONCAT && ggml_get_op_params_i32(cc, 0) == 0 &&
                ((cc->src[0] == m_a && cc->src[1] == m_b) || (cc->src[0] == m_b && cc->src[1] == m_a)) &&
                cc->type == GGML_TYPE_F32 && ggml_is_contiguous(cc) && m_a->ne[0] == m_b->ne[0] && m_a->ne[1] == m_b->ne[1] &&
                cc->ne[0] == 2*m_a->ne[0] && cc->ne[1] == m_a->ne[1] && m_a->ne[0] <= 8*1024 &&
                !norm2_overlap(cc, node->src[0]) && !norm2_overlap(cc, n_b->src[0])) {
            const bool a_first = cc->src[0] == m_a;
            ggml_cuda_op_rms_norm_mul2_concat(*cuda_ctx, a_first ? node : n_b, a_first ? m_a : m_b, a_first ? n_b : node, a_first ? m_b : m_a, cc);
            return 4;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

// Fold the small producer chains of each gated_delta_net into its launch. The kernel reads a
// chain's raw input and repeats the chain's arithmetic operand for operand, so its results are bit-identical,
// and the chain's nodes are not launched. A chain folds only if nothing but the chain reads its nodes, and
// nothing launched between the chain and the gated_delta_net (its own output included) overwrites the raw
// input the kernel now reads later than the chain did. Each fold has its own toggle, read once, default on.
static bool ggml_cuda_gdn_fold_enabled(const char * name) {
    const char * e = getenv(name);
    return e == nullptr || atoi(e) != 0;
}

static void ggml_cuda_plan_gdn_folds(const ggml_cgraph * cgraph) {
    static const bool fold_qknorm = ggml_cuda_gdn_fold_enabled("LLAMA_GDN_FUSE_QKNORM");
    static const bool fold_gates  = ggml_cuda_gdn_fold_enabled("LLAMA_GDN_FUSE_GATES");
    static const bool fold_state  = ggml_cuda_gdn_fold_enabled("LLAMA_GDN_STATE_DIRECT");

    ggml_cuda_gdn_fold_plan & plan = ggml_cuda_gdn_folds;
    plan.gdn.clear();
    plan.skip.clear();
    plan.n_tokens = plan.n_seqs = 0;
    plan.n_gdn = plan.n_qknorm = plan.n_gates = plan.n_beta = plan.n_gate_kernel = plan.n_state = plan.n_conv = 0;

    std::unordered_map<const ggml_tensor *, int> pos;
    auto index_of = [&](const ggml_tensor * t) {
        const auto it = pos.find(t);
        return it == pos.end() ? -1 : it->second;
    };
    // t is a node of this graph read by exactly n_uses nodes, and not a graph output
    auto only_used_by = [&](const ggml_tensor * t, int n_uses) {
        const int j = index_of(t);
        return j >= 0 && ggml_node_get_use_count(cgraph, j) == n_uses && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    // nothing launched in (first, last] writes over raw; skip holds the nodes that will not be launched
    auto raw_intact = [&](const ggml_tensor * raw, int first, int last, const std::unordered_set<const ggml_tensor *> & skip) {
        for (int j = first + 1; j <= last; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || skip.count(n)) {
                continue;
            }
            if (overlap(n, raw)) {
                return false;
            }
        }
        return true;
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * gdn = cgraph->nodes[i];
        if (gdn->op != GGML_OP_GATED_DELTA_NET || !(gdn->flags & GGML_TENSOR_FLAG_COMPUTE) || gdn->type != GGML_TYPE_F32) {
            continue;
        }
        if (pos.empty()) {
            for (int j = 0; j < cgraph->n_nodes; ++j) {
                pos[cgraph->nodes[j]] = j;
            }
        }

        const ggml_tensor * v = gdn->src[2];
        const ggml_tensor * g = gdn->src[3];
        const int64_t S_v = v->ne[0];
        // the folds are written for the scalar-gate kernel with full 32-lane warps
        if (g->ne[0] != 1 || (S_v != 32 && S_v != 64 && S_v != 128)) {
            continue;
        }
        // every warp recomputes the norms and gates of each token, which the separate kernels do once per head:
        // a small cost for a decode or verify batch, not for a prefill
        const bool decode = v->ne[2] <= GGML_CUDA_GDN_FOLD_MAX_TOKENS;
        plan.n_tokens = v->ne[2];
        plan.n_seqs   = v->ne[3];
        plan.n_gdn++;
        if (!fold_qknorm && !fold_gates && !fold_state) {
            continue;
        }

        // a foldable chain: its nodes (not launched), each raw input with the position of its first folded
        // reader (after which the raw input's memory may be reused), and what it sets in the fold
        struct candidate {
            std::vector<const ggml_tensor *>                   chain;
            std::vector<std::pair<const ggml_tensor *, int>>   raws;
            std::function<void(ggml_cuda_gdn_fold &)>          apply;
        };
        std::vector<candidate> cands;

        // LLAMA_GDN_FUSE_QKNORM: q = SCALE(RMS_NORM(raw q)), k likewise (build_gdn_l2_norm), each read by the next only
        auto match_l2 = [&](const ggml_tensor * x, const ggml_tensor *& raw, float & eps, float & sc, float & bias) {
            const ggml_tensor * nrm = x->src[0];
            if (x->op != GGML_OP_SCALE || x->type != GGML_TYPE_F32 || !only_used_by(x, 1) ||
                nrm == nullptr || nrm->op != GGML_OP_RMS_NORM || nrm->type != GGML_TYPE_F32 || !only_used_by(nrm, 1)) {
                return false;
            }
            raw = nrm->src[0];
            if (raw->type != GGML_TYPE_F32 || raw->nb[0] != sizeof(float) || !ggml_are_same_shape(raw, x) ||
                raw->ne[0] != S_v || raw->ne[0] >= 1024) {
                return false;
            }
            memcpy(&eps,  (const float *) nrm->op_params + 0, sizeof(float));
            memcpy(&sc,   (const float *) x->op_params + 0, sizeof(float));
            memcpy(&bias, (const float *) x->op_params + 1, sizeof(float));
            return true;
        };
        if (fold_qknorm && decode) {
            const ggml_tensor * q_raw = nullptr;
            const ggml_tensor * k_raw = nullptr;
            ggml_cuda_gdn_fold qk;
            if (match_l2(gdn->src[0], q_raw, qk.eps_q, qk.scale_q, qk.bias_q) &&
                match_l2(gdn->src[1], k_raw, qk.eps_k, qk.scale_k, qk.bias_k) &&
                q_raw->nb[1] == k_raw->nb[1] && q_raw->nb[2] == k_raw->nb[2] && q_raw->nb[3] == k_raw->nb[3]) {
                const int first = std::min(index_of(gdn->src[0]->src[0]), index_of(gdn->src[1]->src[0]));
                cands.push_back({
                    { gdn->src[0], gdn->src[0]->src[0], gdn->src[1], gdn->src[1]->src[0] },
                    { { q_raw, first }, { k_raw, first } },
                    [=](ggml_cuda_gdn_fold & fd) {
                        fd.q          = (const float *) q_raw->data;
                        fd.k          = (const float *) k_raw->data;
                        fd.sq1        = q_raw->nb[1] / sizeof(float);
                        fd.sq2        = q_raw->nb[2] / sizeof(float);
                        fd.sq3        = q_raw->nb[3] / sizeof(float);
                        fd.norm_ncols = (int) S_v;
                        fd.eps_q = qk.eps_q; fd.scale_q = qk.scale_q; fd.bias_q = qk.bias_q;
                        fd.eps_k = qk.eps_k; fd.scale_k = qk.scale_k; fd.bias_k = qk.bias_k;
                    } });
            }
        }

        // LLAMA_GDN_FUSE_GATES: g = RESHAPE(MUL(SOFTPLUS(ADD(raw alpha, dt)), a)) and beta = SIGMOID(raw beta), with dt
        // and a one value per head and the raw inputs shaped as g and beta are, read at their own strides (qwen35's
        // merged a/b product gives two strided views of one tensor). The two fold separately: the raw
        // alpha is often reused by a later projection's output once its ADD has run, and then the three gate nodes
        // run as one kernel in place instead (ggml_cuda_try_fuse_gdn_gate)
        if (fold_gates && decode) {
            const int64_t H = v->ne[1];
            auto per_head = [&](const ggml_tensor * t) {
                return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == H && ggml_nelements(t) == H;
            };
            auto unary_is = [](const ggml_tensor * t, ggml_unary_op op) {
                return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op && t->type == GGML_TYPE_F32;
            };
            // a raw input's strides in floats over (head, token, sequence), from its dims d0 (head), d0 + 1 (token) and
            // d0 + 2 (sequence) when it is a strided view of that shape; a contiguous one of any shape is read as g and
            // beta are laid out
            const int64_t T = v->ne[2], S = v->ne[3];
            auto raw_strides = [&](const ggml_tensor * t, int d0, int64_t & s1, int64_t & s2, int64_t & s3) {
                if (t->type != GGML_TYPE_F32) {
                    return false;
                }
                if (ggml_is_contiguous(t)) {
                    s1 = 1; s2 = H; s3 = H*T;
                    return true;
                }
                if (t->nb[0] != sizeof(float) || t->ne[d0] != H || t->ne[d0 + 1] != T || (d0 + 2 < GGML_MAX_DIMS && t->ne[d0 + 2] != S) ||
                    (d0 + 2 >= GGML_MAX_DIMS && S != 1) || (d0 == 0 && t->ne[3] != 1) || (d0 == 1 && t->ne[0] != 1)) {
                    return false;
                }
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (t->nb[d] % sizeof(float) != 0) {
                        return false;
                    }
                }
                s1 = t->nb[d0] / sizeof(float);
                s2 = t->nb[d0 + 1] / sizeof(float);
                s3 = d0 + 2 < GGML_MAX_DIMS ? t->nb[d0 + 2] / sizeof(float) : 0;
                return true;
            };
            int64_t sa1 = 0, sa2 = 0, sa3 = 0, sr1 = 0, sr2 = 0, sr3 = 0;
            const ggml_tensor * g_r  = gdn->src[3];
            const ggml_tensor * mul  = g_r->op == GGML_OP_RESHAPE ? g_r->src[0] : g_r;
            const ggml_tensor * sp   = mul->op == GGML_OP_MUL ? mul->src[0] : nullptr;
            const ggml_tensor * add  = sp && unary_is(sp, GGML_UNARY_OP_SOFTPLUS) ? sp->src[0] : nullptr;
            if (add != nullptr && add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
                (g_r == mul || only_used_by(g_r, 1)) && only_used_by(mul, 1) && only_used_by(sp, 1) && only_used_by(add, 1) &&
                per_head(add->src[1]) && per_head(mul->src[1]) &&
                ggml_is_contiguous(g_r) && ggml_nelements(g_r) == ggml_nelements(add) &&
                ggml_are_same_shape(add, sp) && ggml_are_same_shape(add, mul) && add->ne[0] == H &&
                ggml_are_same_shape(add->src[0], add) && raw_strides(add->src[0], 0, sa1, sa2, sa3)) {
                const ggml_tensor * alpha = add->src[0];
                const ggml_tensor * dt    = add->src[1];
                const ggml_tensor * a     = mul->src[1];
                candidate c;
                c.chain = { mul, sp, add };
                if (g_r != mul) {
                    c.chain.push_back(g_r);
                }
                c.raws  = { { alpha, index_of(add) } };
                c.apply = [=](ggml_cuda_gdn_fold & fd) {
                    fd.alpha = (const float *) alpha->data;
                    fd.dt    = (const float *) dt->data;
                    fd.a     = (const float *) a->data;
                    fd.sa1   = sa1;
                    fd.sa2   = sa2;
                    fd.sa3   = sa3;
                };
                cands.push_back(std::move(c));
            }
            // beta = SIGMOID(raw beta), or SIGMOID(CONT(raw beta)) for qwen35's strided view of the merged a/b product
            // the copy folds too, the kernel reading the view the CONT reads
            const ggml_tensor * sig  = gdn->src[4];
            const ggml_tensor * bin  = unary_is(sig, GGML_UNARY_OP_SIGMOID) ? sig->src[0] : nullptr;
            const ggml_tensor * bcpy = bin != nullptr && bin->op == GGML_OP_CONT && bin->type == GGML_TYPE_F32 &&
                                       only_used_by(bin, 1) && ggml_are_same_shape(bin->src[0], bin) ? bin : nullptr;
            if (bcpy != nullptr) {
                bin = bcpy->src[0];
            }
            if (bin != nullptr && only_used_by(sig, 1) && ggml_is_contiguous(sig) &&
                ggml_are_same_shape(bin, sig) && ggml_nelements(sig) == ggml_nelements(g_r) &&
                raw_strides(bin, 1, sr1, sr2, sr3)) {
                const ggml_tensor * beta = bin;
                std::vector<const ggml_tensor *> chain = { sig };
                if (bcpy != nullptr) {
                    chain.push_back(bcpy);
                }
                cands.push_back({ chain, { { beta, index_of(bcpy != nullptr ? bcpy : sig) } },
                    [=](ggml_cuda_gdn_fold & fd) {
                        fd.beta = (const float *) beta->data;
                        fd.sr1  = sr1;
                        fd.sr2  = sr2;
                        fd.sr3  = sr3;
                    } });
            }
        }

        // LLAMA_GDN_STATE_DIRECT: state = RESHAPE(GET_ROWS(cache rows, ids)) (build_rs). The kernel reads the cache row
        // itself; its snapshot writes are to columns it has already read (see the kernel). For several sequences
        // too (LLAMA_GDN_STATE_DIRECT_MULTI=0: one only, as before); a launch whose sequences' rows cross gathers them first
        // (gdn_state_hazard). Decode batches only, so a prefill keeps main's kernel variant (the folding one holds more registers)
        static const bool fold_multi = ggml_cuda_gdn_fold_enabled("LLAMA_GDN_STATE_DIRECT_MULTI");
        const int64_t n_seqs = v->ne[3];
        if (fold_state && decode && (n_seqs == 1 || fold_multi)) {
            const ggml_tensor * st   = gdn->src[5];
            const ggml_tensor * rows = st->op == GGML_OP_RESHAPE ? st->src[0] : st;
            const int64_t       D    = S_v * S_v * v->ne[1];
            if (rows->op == GGML_OP_GET_ROWS && rows->type == GGML_TYPE_F32 && (st == rows || only_used_by(st, 1)) &&
                only_used_by(rows, 1) && rows->ne[0] == D && rows->ne[1] == n_seqs && ggml_nelements(rows) == D*n_seqs &&
                ggml_is_contiguous(rows) &&
                rows->src[0]->type == GGML_TYPE_F32 && rows->src[0]->ne[0] == D && rows->src[0]->nb[0] == sizeof(float) &&
                rows->src[0]->nb[1] % sizeof(float) == 0 &&
                rows->src[1]->type == GGML_TYPE_I32 && ggml_nelements(rows->src[1]) == n_seqs && ggml_is_contiguous(rows->src[1])) {
                const ggml_tensor * cache = rows->src[0];
                const ggml_tensor * ids   = rows->src[1];
                candidate c;
                c.chain = { rows };
                if (st != rows) {
                    c.chain.push_back(st);
                }
                c.raws  = { { cache, index_of(rows) }, { ids, index_of(rows) } };
                c.apply = [=](ggml_cuda_gdn_fold & fd) {
                    fd.states    = (const float *) cache->data;
                    fd.ids       = (const int32_t *) ids->data;
                    fd.state_row = cache->nb[1] / sizeof(float);
                    fd.n_ids     = (int) n_seqs;
                };
                cands.push_back(std::move(c));
            }
        }

        // drop a chain whose raw input something launched would overwrite, until the others hold
        for (bool dropped = true; dropped && !cands.empty(); ) {
            dropped = false;
            std::unordered_set<const ggml_tensor *> skip;
            for (const candidate & c : cands) {
                skip.insert(c.chain.begin(), c.chain.end());
            }
            for (size_t c = 0; c < cands.size() && !dropped; ++c) {
                for (const auto & [raw, first] : cands[c].raws) {
                    if (first < 0 || !raw_intact(raw, first, i, skip)) {
                        cands.erase(cands.begin() + c);
                        dropped = true;
                        break;
                    }
                }
            }
        }

        if (!cands.empty()) {
            ggml_cuda_gdn_fold fd;
            for (const candidate & c : cands) {
                c.apply(fd);
                plan.skip.insert(c.chain.begin(), c.chain.end());
            }
            plan.gdn[gdn] = fd;
            plan.n_qknorm += fd.q      != nullptr;
            plan.n_gates  += fd.alpha  != nullptr;
            plan.n_beta   += fd.beta   != nullptr;
            plan.n_state  += fd.states != nullptr;
        }
    }
}

// Quantize a shared input once (toggle LLAMA_Q8_SHARE, default on). Quantized mul_mats that read
// the same src1, with nothing launched in between writing over it, share one q8_1 copy of it: the first MMVQ
// launch quantizes into a small per-device slot and the later ones read that slot. Quantization is deterministic,
// so every consumer reads the bytes it would have made itself. A consumer that finds its slot not yet filled on
// its own stream (a different dispatch, another stream) quantizes as before.
#define GGML_CUDA_Q8_SHARE_SLOTS     16
#define GGML_CUDA_Q8_SHARE_SLOT_SIZE (64*1024)
// LLAMA_QPN_SHARE (default on): the same for products on repacked weights (ggml_cuda_mul_mat_qpn), whose
// input is prepared as fp16 fragments, range scales and per-32 sums (qpn_prep_kernel) instead of q8_1
#define GGML_CUDA_QPN_SHARE_MAX_BYTES (128*1024) // per 8 tokens of the pass (256 KiB at 9 to 16)
// LLAMA_QPN_PREP_AT_SOURCE (default on; qpn-source.cuh): when the kernel that produces the input can write
// its prepared form itself, the group starts at that producer's node and holds even a single product, and nobody
// launches the prep. Its slots are larger: ffn down's input, 17408 columns, takes 290 KB at 8 tokens (579 KB at 16)
#define GGML_CUDA_QPN_SHARE_SLOT_SIZE (640*1024)

struct ggml_cuda_q8_share_group {
    int          slot   = -1;
    bool         filled = false;
    cudaStream_t stream = nullptr;
    bool         qpn    = false; // a group of products on repacked weights: a slot of qpn_buf
    // prepared at source by the node src (nullptr: by its first product), K columns, T tokens; mins: a
    // product reads the per-32 sums
    const ggml_tensor * src  = nullptr;
    int64_t             K    = 0;
    int                 T    = 0;
    bool                mins = false;
};

struct ggml_cuda_q8_share_plan {
    int    device = -1;
    char * buf    = nullptr; // the graph's backend context's slots
    char * qpn_buf = nullptr;
    std::map<std::pair<const ggml_tensor *, const ggml_tensor *>, int> group_of; // (src0, src1) -> group
    std::vector<ggml_cuda_q8_share_group> groups;
    std::unordered_map<const ggml_tensor *, int> source_of; // producer node -> its prepared-at-source group
    int64_t n_tokens = 0;
    int     n_consumers = 0;
};

static thread_local ggml_cuda_q8_share_plan ggml_cuda_q8_shares;

// the q8_1 buffer for this mul_mat's src1 if it shares one (else nullptr); *ready: already quantized on this stream
char * ggml_cuda_q8_share_buffer(const ggml_tensor * src0, const ggml_tensor * src1, size_t nbytes, cudaStream_t stream, bool * ready) {
    *ready = false;
    ggml_cuda_q8_share_plan & plan = ggml_cuda_q8_shares;
    if (plan.groups.empty() || plan.buf == nullptr || nbytes > GGML_CUDA_Q8_SHARE_SLOT_SIZE || plan.device != ggml_cuda_get_device()) {
        return nullptr;
    }
    const auto it = plan.group_of.find({ src0, src1 });
    if (it == plan.group_of.end()) {
        return nullptr;
    }
    ggml_cuda_q8_share_group & g = plan.groups[it->second];
    if (g.qpn || (g.filled && g.stream != stream)) {
        return nullptr;
    }
    *ready   = g.filled;
    g.filled = true;
    g.stream = stream;
    return plan.buf + (size_t) g.slot*GGML_CUDA_Q8_SHARE_SLOT_SIZE;
}

// the prepared-input buffer for a product on a repacked weight if it shares one (else nullptr); *ready:
// already prepared on this stream
char * ggml_cuda_qpn_share_buffer(const ggml_tensor * src0, const ggml_tensor * src1, size_t nbytes, cudaStream_t stream, bool * ready) {
    *ready = false;
    ggml_cuda_q8_share_plan & plan = ggml_cuda_q8_shares;
    if (plan.groups.empty() || plan.qpn_buf == nullptr || nbytes > GGML_CUDA_QPN_SHARE_SLOT_SIZE || plan.device != ggml_cuda_get_device()) {
        return nullptr;
    }
    const auto it = plan.group_of.find({ src0, src1 });
    if (it == plan.group_of.end()) {
        return nullptr;
    }
    ggml_cuda_q8_share_group & g = plan.groups[it->second];
    if (!g.qpn || (g.filled && g.stream != stream)) {
        return nullptr;
    }
    *ready   = g.filled;
    g.filled = true;
    g.stream = stream;
    return plan.qpn_buf + (size_t) g.slot*GGML_CUDA_QPN_SHARE_SLOT_SIZE;
}

// qpn-source.cuh: a producer's prepared input, and the LLAMA_QPN_PREP_CHECK instrument
static int ggml_cuda_qpn_prep_check() {
    static const int mode = [] { const char * e = getenv("LLAMA_QPN_PREP_CHECK"); return e == nullptr ? 0 : atoi(e); }();
    return mode;
}

// the check's totals, in host-mapped memory the check kernels add to (so they count every replay of a CUDA graph)
struct ggml_cuda_qpn_check_totals {
    unsigned long long inputs;     // prepared inputs compared
    unsigned long long words;      // 32-bit words compared
    unsigned long long mismatches; // of those, differing
    unsigned long long nonfinite;  // slices the reference marked non-finite (xsc 0)
};

struct ggml_cuda_qpn_check_state {
    ggml_cuda_qpn_check_totals * host = nullptr;
    ggml_cuda_qpn_check_totals * dev  = nullptr;
    char   * ref  = nullptr; // the reference's prepared input, then its copy of the fp32 output
    char   * dry  = nullptr; // LLAMA_QPN_PREP_CHECK & 4: the prepared input of a producer with no QPN consumer
    unsigned counter = 0;    // picks each injection
};

static ggml_cuda_qpn_check_state & ggml_cuda_qpn_check_get(const int device) {
    static ggml_cuda_qpn_check_state st[GGML_CUDA_MAX_DEVICES];
    ggml_cuda_qpn_check_state & c = st[device];
    if (c.host == nullptr) {
        CUDA_CHECK(cudaHostAlloc((void **) &c.host, sizeof(ggml_cuda_qpn_check_totals), cudaHostAllocMapped));
        memset(c.host, 0, sizeof(ggml_cuda_qpn_check_totals));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &c.dev, c.host, 0));
        CUDA_CHECK(cudaMalloc((void **) &c.ref, GGML_CUDA_QPN_SHARE_SLOT_SIZE + GGML_CUDA_QPN_SOURCE_MAX_TOKENS*32768*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **) &c.dry, GGML_CUDA_QPN_SHARE_SLOT_SIZE));
    }
    return c;
}

static void ggml_cuda_qpn_check_log(const int device) {
    if (ggml_cuda_qpn_prep_check() == 0) {
        return;
    }
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaDeviceSynchronize());
    const ggml_cuda_qpn_check_totals t = *ggml_cuda_qpn_check_get(device).host;
    GGML_LOG_INFO("qpn prep at source, check (LLAMA_QPN_PREP_CHECK=%d), device %d: %llu prepared inputs compared, %llu words, "
        "%llu mismatches, %llu non-finite slices\n", ggml_cuda_qpn_prep_check(), device, t.inputs, t.words, t.mismatches, t.nonfinite);
}

static __global__ void qpn_check_inject(float * x, const int64_t i, const float v) {
    x[i] = v;
}

// words [0, n) of a against b; xsc words from nsc0 on, where the reference's 0 marks a non-finite slice
static __global__ void qpn_check_compare(const uint32_t * a, const uint32_t * b, const int64_t n, const int64_t nsc0, const int64_t nsc1,
        ggml_cuda_qpn_check_totals * tot) {
    unsigned long long bad = 0, nonfin = 0;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        bad += a[i] != b[i];
        nonfin += i >= nsc0 && i < nsc1 && b[i] == 0;
    }
    bad    = warp_reduce_sum((int) bad);
    nonfin = warp_reduce_sum((int) nonfin);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&tot->mismatches, bad);
        atomicAdd(&tot->nonfinite, nonfin);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&tot->inputs, nsc1 > 0 ? 1ull : 0ull); // one xsc comparison per input
        atomicAdd(&tot->words, (unsigned long long) n);
    }
}

bool ggml_cuda_qpn_source_begin(const ggml_tensor * out, const int64_t K, const int T, cudaStream_t stream, ggml_cuda_qpn_dst * d) {
    *d = ggml_cuda_qpn_dst();
    if (K % QK_K != 0 || T < 1 || T > std::min(GGML_CUDA_QPN_SOURCE_MAX_TOKENS, ggml_cuda_qpn_pass_width()) || ggml_nelements(out) != K*T) {
        return false;
    }
    ggml_cuda_q8_share_plan & plan = ggml_cuda_q8_shares;
    const int check = ggml_cuda_qpn_prep_check();
    char * base = nullptr;
    bool mins = true;
    const auto it = plan.source_of.empty() || plan.qpn_buf == nullptr || plan.device != ggml_cuda_get_device() ?
        plan.source_of.end() : plan.source_of.find(out);
    if (it != plan.source_of.end()) {
        ggml_cuda_q8_share_group & g = plan.groups[it->second];
        if (g.filled || g.K != K || g.T != T) {
            return false;
        }
        g.filled = true;
        g.stream = stream;
        base = plan.qpn_buf + (size_t) g.slot*GGML_CUDA_QPN_SHARE_SLOT_SIZE;
        mins = g.mins;
    } else if ((check & 4) && ggml_cuda_qpn_prep_bytes(K, T) <= GGML_CUDA_QPN_SHARE_SLOT_SIZE) {
        base = ggml_cuda_qpn_check_get(ggml_cuda_get_device()).dry;
    } else {
        return false;
    }
    ggml_cuda_qpn_prep_ptrs(base, K, T, &d->xh, &d->xs, &d->xsc);
    if (!mins) {
        d->xs = nullptr;
    }
    d->T = T;
    if (check & 2) {
        ggml_cuda_qpn_check_state & c = ggml_cuda_qpn_check_get(ggml_cuda_get_device());
        const unsigned n = c.counter++;
        const float vals[3] = { NAN, INFINITY, -INFINITY };
        d->inj_sb = (int) ((n*7) % (K/QK_K));
        d->inj_t  = (int) (n % T);
        d->inj_g  = (int) ((n*13) % WARP_SIZE);
        d->inj_i  = (int) ((n*5) % 8);
        d->inj_v  = vals[n % 3];
    }
    return true;
}

void ggml_cuda_qpn_source_end(const ggml_tensor * out, const int64_t K, cudaStream_t stream, const ggml_cuda_qpn_dst & d) {
    const int check = ggml_cuda_qpn_prep_check();
    if (check == 0 || d.xh == nullptr) {
        return;
    }
    ggml_cuda_qpn_check_state & c = ggml_cuda_qpn_check_get(ggml_cuda_get_device());
    const int T = d.T;
    const float * x = (const float *) out->data;
    char * ref = c.ref;
    GGML_ASSERT(K*T <= GGML_CUDA_QPN_SOURCE_MAX_TOKENS*32768);
    if (d.inj_sb >= 0) {
        float * xc = (float *) (c.ref + GGML_CUDA_QPN_SHARE_SLOT_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(xc, x, K*T*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        qpn_check_inject<<<1, 1, 0, stream>>>(xc, d.inj_t*K + (int64_t) d.inj_sb*QK_K + d.inj_g*8 + d.inj_i, d.inj_v);
        x = xc;
    }
    ggml_cuda_qpn_prep(x, K, K, T, ref, stream);
    half * rxh; half * rxs; float * rxsc;
    ggml_cuda_qpn_prep_ptrs(ref, K, T, &rxh, &rxs, &rxsc);
    const int64_t nsb = K/QK_K;
    // xh, then (where written) xs, then xsc, each against the reference's
    const auto cmp = [&](const void * a, const void * b, const int64_t n, const bool sc) {
        const int nb = (int) std::min<int64_t>((n + 255)/256, 256);
        qpn_check_compare<<<nb, 256, 0, stream>>>((const uint32_t *) a, (const uint32_t *) b, n, sc ? 0 : n, sc ? n : 0, c.dev);
        CUDA_CHECK(cudaGetLastError());
    };
    cmp(d.xh, rxh, K*T/2, false);
    if (d.xs != nullptr) {
        cmp(d.xs, rxs, nsb*T*8/2, false);
    }
    cmp(d.xsc, rxsc, nsb*T, true);
}

static void ggml_cuda_plan_q8_share(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_Q8_SHARE"); return e == nullptr || atoi(e) != 0; }();
    static const bool qpn_enabled = [] { const char * e = getenv("LLAMA_QPN_SHARE"); return e == nullptr || atoi(e) != 0; }();
    // needs LLAMA_QPN_SHARE's groups
    static const bool at_source = [] { const char * e = getenv("LLAMA_QPN_PREP_AT_SOURCE"); return e == nullptr || atoi(e) != 0; }();

    ggml_cuda_q8_share_plan & plan = ggml_cuda_q8_shares;
    plan.group_of.clear();
    plan.groups.clear();
    plan.source_of.clear();
    plan.device  = cuda_ctx->device;
    plan.buf     = nullptr;
    plan.qpn_buf = nullptr;
    if (!enabled && !qpn_enabled) {
        return;
    }

    // src: the node that writes src1 and can write its prepared form, at node first; -1 if none
    struct open_group { const ggml_tensor * src1; int first, last; std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> uses; bool qpn;
                        int src = -1; bool mins = false; };
    std::vector<open_group> open, done;
    auto same_input = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a->data == b->data && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);
    };
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    int64_t n_tokens = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        const ggml_tensor * src0 = n->src[0];
        const ggml_tensor * src1 = n->src[1];
        const bool q8 = enabled && (n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) && ggml_is_quantized(src0->type) &&
                src1->type == GGML_TYPE_F32 && src1->ne[1]*src1->ne[2]*src1->ne[3] <= MMVQ_MAX_BATCH_SIZE &&
                src1->ne[0] % QK8_1 == 0 && src1->data != nullptr &&
                (size_t) (src1->ne[1]*src1->ne[2]*src1->ne[3]*GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)) <= GGML_CUDA_Q8_SHARE_SLOT_SIZE &&
                !ggml_cuda_mmvq_tc_use(src0, src1, n, cuda_ctx->device) && // a tensor-core product quantizes nothing
                !ggml_cuda_qpn_is_repacked(src0);                           // nor does one on a repacked weight
        // products on repacked weights of one pass (at most 8 tokens; 16, over one sequence or several) share the prepared input
        const bool qpn = qpn_enabled && n->op == GGML_OP_MUL_MAT && ggml_cuda_qpn_is_repacked(src0) && src1->type == GGML_TYPE_F32 &&
                src1->data != nullptr && ggml_nrows(src1) <= ggml_cuda_qpn_pass_width() && ggml_cuda_qpn_flat_cols(src1) && ggml_cuda_qpn_flat_cols(n) &&
                src1->nb[0] == sizeof(float) && ggml_cuda_qpn_prep_bytes(src1->ne[0], (int) ggml_nrows(src1)) <= GGML_CUDA_QPN_SHARE_SLOT_SIZE;
        if (q8 || qpn) {
            open_group * g = nullptr;
            for (open_group & o : open) {
                if (o.qpn == qpn && same_input(o.src1, src1)) {
                    g = &o;
                    break;
                }
            }
            if (!g) {
                open.push_back({ src1, j, j, {}, qpn });
                g = &open.back();
                // the last node before this one that writes src1's memory produces it; the group can be
                // prepared there if that node's output is exactly src1 (T contiguous rows of K columns). Whether its
                // kernel can write the prepared form is its launcher's call (ggml_cuda_qpn_source_begin); if it
                // does not, the first product prepares as before
                if (qpn && at_source && src1->ne[0] % QK_K == 0 && src1->nb[1] == src1->ne[0]*sizeof(float)) {
                    for (int k = j - 1; k >= 0 && k >= j - 512; --k) {
                        const ggml_tensor * m = cgraph->nodes[k];
                        if (ggml_cuda_is_view_or_noop(m) || !(m->flags & GGML_TENSOR_FLAG_COMPUTE) || !overlap(m, src1)) {
                            continue;
                        }
                        if (m->data == src1->data && m->type == GGML_TYPE_F32 && ggml_is_contiguous(m) &&
                            ggml_nelements(m) == src1->ne[0]*ggml_nrows(src1)) {
                            g->src   = k;
                            g->first = k;
                        }
                        break;
                    }
                }
            }
            g->mins = g->mins || (qpn && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K));
            g->last = j;
            g->uses.push_back({ src0, src1 });
            n_tokens = std::max(n_tokens, src1->ne[1]*src1->ne[2]*src1->ne[3]);
        }
        // a launch that writes over a shared input ends its group
        for (size_t k = 0; k < open.size(); ) {
            if (overlap(n, open[k].src1)) {
                done.push_back(std::move(open[k]));
                open.erase(open.begin() + k);
            } else {
                ++k;
            }
        }
    }
    done.insert(done.end(), std::make_move_iterator(open.begin()), std::make_move_iterator(open.end()));
    std::sort(done.begin(), done.end(), [](const open_group & a, const open_group & b) { return a.first < b.first; });

    // a slot is reused only well after its previous group's last consumer (a fused launch may run a few nodes early)
    int slot_free_at[2][GGML_CUDA_Q8_SHARE_SLOTS];
    std::fill(&slot_free_at[0][0], &slot_free_at[0][0] + 2*GGML_CUDA_Q8_SHARE_SLOTS, -1);
    int n_consumers = 0, n_qpn_groups = 0, n_qpn_consumers = 0, n_src_groups = 0, n_src_consumers = 0;
    for (const open_group & o : done) {
        // a group prepared at source takes one product; one prepared by its first product needs two, and
        // keeps the same size limit
        const bool src = o.qpn && o.src >= 0;
        if (!src && (o.uses.size() < 2 || (o.qpn && ggml_cuda_qpn_prep_bytes(o.src1->ne[0], (int) ggml_nrows(o.src1)) >
                (size_t) GGML_CUDA_QPN_SHARE_MAX_BYTES*((ggml_nrows(o.src1) + 7)/8)))) {
            continue;
        }
        int * free_at = slot_free_at[o.qpn ? 1 : 0];
        int slot = -1;
        for (int s = 0; s < GGML_CUDA_Q8_SHARE_SLOTS; ++s) {
            if (free_at[s] < o.first) {
                slot = s;
                break;
            }
        }
        if (slot < 0) {
            continue;
        }
        free_at[slot] = o.last + 16;
        ggml_cuda_q8_share_group g;
        g.slot = slot;
        g.qpn  = o.qpn;
        if (src) {
            g.src  = cgraph->nodes[o.src];
            g.K    = o.src1->ne[0];
            g.T    = (int) ggml_nrows(o.src1);
            g.mins = o.mins;
            plan.source_of[g.src] = (int) plan.groups.size();
            n_src_groups    += 1;
            n_src_consumers += (int) o.uses.size();
        }
        plan.groups.push_back(g);
        for (const auto & u : o.uses) {
            plan.group_of[u] = (int) plan.groups.size() - 1;
        }
        if (o.qpn) {
            n_qpn_groups    += 1;
            n_qpn_consumers += (int) o.uses.size();
        } else {
            n_consumers += (int) o.uses.size();
        }
    }
    const int n_q8_groups = (int) plan.groups.size() - n_qpn_groups;
    if (n_q8_groups > 0 && cuda_ctx->q8_share_buf == nullptr) {
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaMalloc((void **) &cuda_ctx->q8_share_buf, GGML_CUDA_Q8_SHARE_SLOTS*GGML_CUDA_Q8_SHARE_SLOT_SIZE));
    }
    if (n_qpn_groups > 0 && cuda_ctx->qpn_share_buf == nullptr) {
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaMalloc((void **) &cuda_ctx->qpn_share_buf, GGML_CUDA_Q8_SHARE_SLOTS*GGML_CUDA_QPN_SHARE_SLOT_SIZE));
    }
    plan.buf     = cuda_ctx->q8_share_buf;
    plan.qpn_buf = cuda_ctx->qpn_share_buf;

    // one log line per distinct outcome
    static std::mutex log_mutex;
    static std::set<std::tuple<int, int64_t, int, int, int, int, int>> logged;
    std::lock_guard<std::mutex> lock(log_mutex);
    if (logged.insert({ plan.device, n_tokens, n_q8_groups, n_consumers, n_qpn_groups, n_qpn_consumers, n_src_groups }).second) {
        GGML_LOG_INFO("%s: device %d, %" PRId64 " tokens: %d shared q8_1 inputs, %d consumers, %d quantize launches saved; "
            "%d shared fp16 inputs of repacked products, %d consumers, %d prep launches saved; %d of those inputs, "
            "%d consumers, planned to be prepared at source\n",
            __func__, plan.device, n_tokens, n_q8_groups, n_consumers, n_consumers - n_q8_groups,
            n_qpn_groups, n_qpn_consumers, n_qpn_consumers - n_qpn_groups + n_src_groups, n_src_groups, n_src_consumers);
    }
}

// Grouped launches for the small sibling BF16 GEMVs (toggle LLAMA_BF16_GEMV_GROUP, default on).
// n_embd-wide BF16 products that read the same activations (the router and the shared-expert gate, GDN alpha and
// beta, the indexer's q and k) run as one ggml_cuda_mul_mat_vec_bf16_group launch, bit-identical to the separate
// ones. The siblings are not adjacent, so the launch happens at the first one the walk dispatches normally and
// computes the later ones early; the walk then skips a later sibling's own launch (ggml_cuda_sib_gemv_done), and
// only when nothing fused it first. A sibling joins a group only if running it early is safe: from the group's
// first node up to it, no node writes or reads memory overlapping its output, and nothing overwrites the input.
// Nothing is skipped that was not launched: if the first sibling never reaches its hook (a fusion took it, another
// stream), the next one launches the rest, and a sibling no group launch covered runs on its own as before.
#define GGML_CUDA_SIB_GEMV_MAX_ROWS  1024 // small products only
#define GGML_CUDA_SIB_GEMV_MAX_SPAN  64   // the farthest a sibling may run ahead of its own node

struct ggml_cuda_sib_gemv_plan {
    int device = -1;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> member_of; // node -> (group, position in group)
    std::vector<std::vector<ggml_tensor *>> groups;                        // in graph order
    std::unordered_set<const ggml_tensor *> done;                          // computed early this pass
    int64_t n_tokens = 0;
    int     n_members = 0, n_launches = 0, n_covered = 0;
};

static thread_local ggml_cuda_sib_gemv_plan ggml_cuda_sib_gemvs;

// the walk's skip for a sibling a group launch already computed this pass (called once no fusion took the node)
static bool ggml_cuda_sib_gemv_done(const ggml_tensor * node) {
    const ggml_cuda_sib_gemv_plan & plan = ggml_cuda_sib_gemvs;
    return !plan.done.empty() && plan.done.count(node) != 0;
}

static bool ggml_cuda_sib_gemv_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    ggml_cuda_sib_gemv_plan & plan = ggml_cuda_sib_gemvs;
    if (plan.member_of.empty() || plan.device != ctx.device || ctx.curr_stream_no != 0 || plan.done.count(dst)) {
        return false;
    }
    const auto it = plan.member_of.find(dst);
    if (it == plan.member_of.end()) {
        return false;
    }
    const std::vector<ggml_tensor *> & g = plan.groups[it->second.first];
    const ggml_tensor * src0[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    ggml_tensor       * dsts[GGML_CUDA_BF16_GEMV_GROUP_MAX];
    int n = 0;
    for (size_t k = it->second.second; k < g.size(); ++k) {
        if (plan.done.count(g[k]) == 0) {
            src0[n] = g[k]->src[0];
            dsts[n] = g[k];
            ++n;
        }
    }
    if (n < 2 || dsts[0] != dst) {
        return false;
    }
    ggml_cuda_mul_mat_vec_bf16_group(ctx, n, src0, dst->src[1], dsts);
    plan.done.insert(dsts + 1, dsts + n);
    plan.n_launches += 1;
    plan.n_covered  += n;
    return true;
}

static bool ggml_cuda_sib_gemv_enabled() {
    static const bool enabled = [] { const char * e = getenv("LLAMA_BF16_GEMV_GROUP"); return e == nullptr || atoi(e) != 0; }();
    return enabled;
}

// Before allocation (graph_optimize): move each later sibling to just before the first one of its input, so the
// siblings sit together and the allocator keeps their outputs apart. Where they were spread out, the allocator
// gave a later sibling the memory of an earlier one's consumed output, and nothing could run early. Moving a
// sibling up is always legal: it reads only a weight and the shared input, which the first sibling reads too.
// Each first sibling keeps its own consumers right behind it, so a fusion starting there still matches.
static void ggml_cuda_order_sib_gemv(const ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    if (!ggml_cuda_sib_gemv_enabled()) {
        return;
    }
    auto sibling = [&](const ggml_tensor * n) {
        if (n->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(n, 1) != 0) {
            return false;
        }
        const ggml_tensor * src0 = n->src[0];
        const ggml_tensor * src1 = n->src[1];
        return src0->op == GGML_OP_NONE && src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 &&
            src0->ne[1] <= GGML_CUDA_SIB_GEMV_MAX_ROWS && ggml_cuda_mul_mat_vec_bf16_ok(cuda_ctx->device, src0, src1, n);
    };
    struct open_group { int first; int size; };
    std::unordered_map<const ggml_tensor *, open_group> open; // shared input -> where its group starts
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (sibling(n)) {
            const auto it = open.find(n->src[1]);
            if (it != open.end() && j - it->second.first <= GGML_CUDA_SIB_GEMV_MAX_SPAN && it->second.size < GGML_CUDA_BF16_GEMV_GROUP_MAX) {
                const int p = it->second.first;
                memmove(&cgraph->nodes[p + 1], &cgraph->nodes[p], (size_t) (j - p)*sizeof(ggml_tensor *));
                cgraph->nodes[p] = n;
                for (auto & [src1, g] : open) {
                    g.first += g.first >= p && g.first < j;
                }
                it->second.first = p;
                it->second.size += 1;
            } else {
                open[n->src[1]] = { j, 1 };
            }
            continue;
        }
        // a node that writes into a shared input (in place or through a view of it) ends its group
        if (!ggml_cuda_is_view_or_noop(n) && n->view_src != nullptr) {
            for (auto g = open.begin(); g != open.end(); ) {
                const ggml_tensor * in = g->first;
                if (n->view_src == in || n->view_src == in->view_src) {
                    g = open.erase(g);
                } else {
                    ++g;
                }
            }
        }
    }
}

static void ggml_cuda_plan_sib_gemv(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    const bool enabled = ggml_cuda_sib_gemv_enabled();

    ggml_cuda_sib_gemv_plan & plan = ggml_cuda_sib_gemvs;

    // what the previous pass on this thread did, one line per distinct outcome (a warning, so the server shows it)
    if (plan.n_members > 0) {
        static std::mutex log_mutex;
        static std::set<std::tuple<int, int64_t, int, int, int, int>> logged;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (logged.insert({ plan.device, plan.n_tokens, (int) plan.groups.size(), plan.n_members, plan.n_launches, plan.n_covered }).second) {
            GGML_LOG_WARN("%s: CUDA%d at %" PRId64 " tokens: %d sibling BF16 GEMVs planned in %d groups; launched %d groups "
                "covering %d GEMVs, %d launches saved per graph\n", __func__, plan.device, plan.n_tokens, plan.n_members,
                (int) plan.groups.size(), plan.n_launches, plan.n_covered, plan.n_covered - plan.n_launches);
        }
    }

    plan.member_of.clear();
    plan.groups.clear();
    plan.done.clear();
    plan.device     = cuda_ctx->device;
    plan.n_tokens   = 0;
    plan.n_members  = 0;
    plan.n_launches = 0;
    plan.n_covered  = 0;
    if (!enabled || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return;
    }

    // the same tensor, or the same view of one: never an address match, which a reused buffer can fake
    auto same_input = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a == b || (a->view_src != nullptr && a->view_src == b->view_src && a->view_offs == b->view_offs &&
                          ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b));
    };
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data == nullptr || b->data == nullptr) {
            return false;
        }
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    // the nodes a normal dispatch sends to ggml_cuda_mul_mat_vec_bf16 (the checks ggml_cuda_mul_mat makes before it)
    auto candidate = [&](const ggml_tensor * n) {
        if (n->op != GGML_OP_MUL_MAT || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_cuda_gdn_folds.skip.count(n)) {
            return false;
        }
        const ggml_tensor * src0 = n->src[0];
        const ggml_tensor * src1 = n->src[1];
        if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32 ||
            ggml_get_op_params_i32(n, 1) != 0 || src0->ne[1] > GGML_CUDA_SIB_GEMV_MAX_ROWS || src1->data == nullptr ||
            n->data == nullptr) {
            return false;
        }
        const bool bad_padding_clear = src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
        return !bad_padding_clear && ggml_cuda_mul_mat_vec_bf16_ok(cuda_ctx->device, src0, src1, n);
    };
    // n may run at node first instead of its own node j: nothing in [first, j) writes or reads memory under its output
    auto early_safe = [&](int first, int j, const ggml_tensor * n) {
        for (int k = first; k < j; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(t)) {
                continue;
            }
            if (overlap(t, n)) {
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s] != nullptr && overlap(t->src[s], n)) {
                    return false;
                }
            }
        }
        return true;
    };

    struct open_group { const ggml_tensor * src1; int first; int gi; };
    std::vector<open_group> open;
    std::vector<std::pair<int, std::vector<ggml_tensor *>>> built; // (first node, members)
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (candidate(n)) {
            open_group * g = nullptr;
            for (open_group & o : open) {
                if (same_input(o.src1, n->src[1])) {
                    g = &o;
                    break;
                }
            }
            if (g && j - g->first <= GGML_CUDA_SIB_GEMV_MAX_SPAN && built[g->gi].second.size() < GGML_CUDA_BF16_GEMV_GROUP_MAX &&
                early_safe(g->first, j, n)) {
                built[g->gi].second.push_back(n);
            } else if (g) {
                *g = { n->src[1], j, (int) built.size() };
                built.push_back({ j, { n } });
            } else {
                open.push_back({ n->src[1], j, (int) built.size() });
                built.push_back({ j, { n } });
            }
            plan.n_tokens = std::max(plan.n_tokens, n->src[1]->ne[1]);
        }
        // a node that writes over a shared input ends its group
        for (size_t k = 0; k < open.size(); ) {
            if (overlap(n, open[k].src1)) {
                open.erase(open.begin() + k);
            } else {
                ++k;
            }
        }
    }
    for (auto & [first, members] : built) {
        if (members.size() < 2) {
            continue;
        }
        const int gi = (int) plan.groups.size();
        for (size_t k = 0; k < members.size(); ++k) {
            plan.member_of[members[k]] = { gi, (int) k };
        }
        plan.n_members += (int) members.size();
        plan.groups.push_back(std::move(members));
    }
}

// Pairs of products on repacked weights that read the same input (toggle
// LLAMA_QPN_GROUP, default on): GDN qkv with z, ffn up with gate, at the verify widths (3 to 16 tokens;
// LLAMA_QPN_WIDE bit 4, ggml_cuda_qpn_group_ok decides). They run as one
// ggml_cuda_mul_mat_qpn2 launch at the first one the walk dispatches, and the walk skips the second, exactly as the
// BF16 siblings (the same rules: the second may run early only if nothing between the two touches its output or
// overwrites the input). Each product's arithmetic and order are unchanged, so the outputs are bit-identical.
struct ggml_cuda_qpn_sib_plan {
    int device = -1;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> partner; // first of a pair -> second
    std::unordered_map<const ggml_tensor *, ggml_tensor *> third;   // first of a triple -> third (the attention input)
    std::unordered_set<const ggml_tensor *> done;                   // computed early this pass
    std::unordered_map<const ggml_tensor *, ggml_tensor *> mmvq_partner; // first of a pair of small dp4a products -> second
    std::unordered_map<const ggml_tensor *, std::pair<ggml_tensor *, ggml_tensor *>> ab; // first of a GDN pair -> beta, alpha
    std::unordered_set<const ggml_tensor *> ab_member;              // those beta and alpha (not the dp4a pairs)
    int64_t n_tokens = 0;
    int     n_pairs = 0, n_launches = 0;
};

static thread_local ggml_cuda_qpn_sib_plan ggml_cuda_qpn_sibs;

static bool ggml_cuda_qpn_sib_done(const ggml_tensor * node) {
    const ggml_cuda_qpn_sib_plan & plan = ggml_cuda_qpn_sibs;
    return !plan.done.empty() && plan.done.count(node) != 0;
}

static bool ggml_cuda_qpn_sib_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_qpn_sib_plan & plan = ggml_cuda_qpn_sibs;
    if (plan.partner.empty() || plan.device != ctx.device || ctx.curr_stream_no != 0 || plan.done.count(dst)) {
        return false;
    }
    const auto it = plan.partner.find(dst);
    if (it == plan.partner.end() || plan.done.count(it->second)) {
        return false;
    }
    const auto iab = plan.ab.find(dst); // in-proj, z, beta and alpha in one launch
    if (iab != plan.ab.end() && !plan.done.count(iab->second.first) && !plan.done.count(iab->second.second)) {
        ggml_tensor * b = iab->second.first, * a = iab->second.second;
        const ggml_tensor * src0s[4] = { dst->src[0], it->second->src[0], b->src[0], a->src[0] };
        ggml_tensor       * dsts[4]  = { dst, it->second, b, a };
        ggml_cuda_mul_mat_qpn4ab(ctx, src0s, dst->src[1], dsts);
        plan.done.insert(it->second);
        plan.done.insert(b);
        plan.done.insert(a);
        plan.n_launches += 1;
        return true;
    }
    const auto i3 = plan.third.find(dst);
    if (i3 != plan.third.end() && !plan.done.count(i3->second)) {
        const ggml_tensor * src0s[3] = { dst->src[0], it->second->src[0], i3->second->src[0] };
        ggml_tensor       * dsts[3]  = { dst, it->second, i3->second };
        ggml_cuda_mul_mat_qpn3(ctx, src0s, dst->src[1], dsts);
        plan.done.insert(it->second);
        plan.done.insert(i3->second);
        plan.n_launches += 1;
        return true;
    }
    const ggml_tensor * src0s[2] = { dst->src[0], it->second->src[0] };
    ggml_tensor       * dsts[2]  = { dst, it->second };
    ggml_cuda_mul_mat_qpn2(ctx, src0s, dst->src[1], dsts);
    plan.done.insert(it->second);
    plan.n_launches += 1;
    return true;
}

// the second product of a pair of small dp4a products on one input (ggml_cuda_mul_mat_vec_q_group: the GDN alpha and beta under
// --split-mode tensor), launched with the first, as the QPN pairs above and under the same rules
static bool ggml_cuda_mmvq_sib_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_qpn_sib_plan & plan = ggml_cuda_qpn_sibs;
    if (plan.mmvq_partner.empty() || plan.device != ctx.device || ctx.curr_stream_no != 0 || plan.done.count(dst)) {
        return false;
    }
    const auto it = plan.mmvq_partner.find(dst);
    if (it == plan.mmvq_partner.end() || plan.done.count(it->second)) {
        return false;
    }
    const ggml_tensor * src0s[2] = { dst->src[0], it->second->src[0] };
    ggml_tensor       * dsts[2]  = { dst, it->second };
    if (!ggml_cuda_mul_mat_vec_q_group_ok(src0s, 2, dst->src[1], dsts, ctx.device)) {
        return false;
    }
    ggml_cuda_mul_mat_vec_q_group(ctx, src0s, 2, dst->src[1], dsts);
    plan.done.insert(it->second);
    return true;
}

static void ggml_cuda_plan_qpn_sib(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    ggml_cuda_qpn_sib_plan & plan = ggml_cuda_qpn_sibs;
    ggml_cuda_qpn_ab_log();

    // what the previous pass on this thread did, one line per distinct outcome (a warning, so the server shows it)
    if (plan.n_pairs > 0) {
        static std::mutex log_mutex;
        static std::set<std::tuple<int, int64_t, int, int>> logged;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (logged.insert({ plan.device, plan.n_tokens, plan.n_pairs, plan.n_launches }).second) {
            GGML_LOG_WARN("%s: CUDA%d at %" PRId64 " tokens: %d pairs of repacked products on one input planned (%d with a third"
                "); %d launched together\n", __func__, plan.device, plan.n_tokens, plan.n_pairs, (int) plan.third.size(), plan.n_launches);
        }
        static std::set<std::tuple<int, int64_t, int>> logged_ab;
        if (!plan.ab.empty() && logged_ab.insert({ plan.device, plan.n_tokens, (int) plan.ab.size() }).second) {
            GGML_LOG_WARN("%s: CUDA%d at %" PRId64 " tokens: %d GDN in-proj + z pairs with their beta and alpha slices\n",
                __func__, plan.device, plan.n_tokens, (int) plan.ab.size());
        }
    }
    plan.partner.clear();
    plan.ab.clear();
    plan.ab_member.clear();
    plan.third.clear();
    plan.done.clear();
    plan.mmvq_partner.clear();
    plan.device     = cuda_ctx->device;
    plan.n_tokens   = 0;
    plan.n_pairs    = 0;
    plan.n_launches = 0;
    if (!cuda_ctx->stream_context().concurrent_events.empty()) {
        return;
    }

    auto same_input = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a == b || (a->view_src != nullptr && a->view_src == b->view_src && a->view_offs == b->view_offs &&
                          ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b));
    };
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data == nullptr || b->data == nullptr) {
            return false;
        }
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    // the nodes a normal dispatch sends to ggml_cuda_mul_mat_qpn at 1 to 8 tokens (1 and 2 only form the draft step's q, k, v groups;
    // to 16, one pass of two B tiles)
    auto candidate = [&](const ggml_tensor * n) {
        if (n->op != GGML_OP_MUL_MAT || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_cuda_gdn_folds.skip.count(n)) {
            return false;
        }
        const ggml_tensor * src0 = n->src[0];
        const ggml_tensor * src1 = n->src[1];
        return ggml_cuda_qpn_is_repacked(src0) && src0->view_src == nullptr && src1->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 &&
            src1->ne[1] >= 1 && src1->ne[1] <= ggml_cuda_qpn_pass_width() && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
            n->nb[0] == sizeof(float) && src1->data != nullptr && n->data != nullptr;
    };
    // a GDN beta or alpha slice with its private padded copy (ggml_cuda_qpn_gdn4_ok decides the rest)
    auto ab_candidate = [&](const ggml_tensor * n) {
        return n->op == GGML_OP_MUL_MAT && (n->flags & GGML_TENSOR_FLAG_COMPUTE) && !ggml_cuda_gdn_folds.skip.count(n) &&
            ggml_cuda_qpn_has_ab_copy(n->src[0]) && n->type == GGML_TYPE_F32 && n->nb[0] == sizeof(float) && ggml_is_contiguous(n) &&
            n->src[1]->type == GGML_TYPE_F32 && n->src[1]->nb[0] == sizeof(float) && n->src[1]->data != nullptr && n->data != nullptr;
    };
    // n may run at node first instead of its own node j: nothing in [first, j) writes or reads memory under its output
    auto early_safe = [&](int first, int j, const ggml_tensor * n) {
        for (int k = first; k < j; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(t)) {
                continue;
            }
            if (overlap(t, n)) {
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s] != nullptr && overlap(t->src[s], n)) {
                    return false;
                }
            }
        }
        return true;
    };

    struct open_one { ggml_tensor * n; int j; };
    std::vector<open_one> open; // a candidate still waiting for its partner
    struct open_pair { ggml_tensor * f, * s; int j; ggml_tensor * c = nullptr; }; // c: a beta waiting for its alpha
    std::vector<open_pair> open3; // a pair that a third product of the same input may still join
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        bool joined = false;
        if (candidate(n)) {
            for (size_t k = 0; k < open3.size(); ++k) {
                const open_pair & o = open3[k];
                if (same_input(o.f->src[1], n->src[1]) && j - o.j <= GGML_CUDA_SIB_GEMV_MAX_SPAN &&
                        ggml_cuda_qpn_group3_ok(o.f->src[0], o.s->src[0], n->src[0], n->src[1]) && early_safe(o.j, j, n)) {
                    plan.third[o.f] = n;
                    open3.erase(open3.begin() + k);
                    joined = true;
                    break;
                }
            }
        }
        if (candidate(n) && !joined) {
            bool paired = false;
            for (size_t k = 0; k < open.size(); ++k) {
                ggml_tensor * f = open[k].n;
                if (same_input(f->src[1], n->src[1]) && j - open[k].j <= GGML_CUDA_SIB_GEMV_MAX_SPAN &&
                        ggml_cuda_qpn_group_ok(f->src[0], n->src[0], n->src[1]) && early_safe(open[k].j, j, n)) {
                    plan.partner[f] = n;
                    plan.n_pairs += 1;
                    plan.n_tokens = std::max(plan.n_tokens, n->src[1]->ne[1]);
                    // a product of the same input waiting between the two (the attention v when it comes before k and
                    // cannot pair with q) joins them as the third, under the same rule; otherwise a later one may
                    const int fj = open[k].j;
                    bool third = false;
                    for (size_t m = 0; m < open.size(); ++m) {
                        ggml_tensor * o = open[m].n;
                        if (m != k && open[m].j > fj && same_input(f->src[1], o->src[1]) &&
                                ggml_cuda_qpn_group3_ok(f->src[0], n->src[0], o->src[0], n->src[1]) && early_safe(fj, open[m].j, o)) {
                            plan.third[f] = o;
                            open.erase(open.begin() + std::max(m, k));
                            open.erase(open.begin() + std::min(m, k));
                            third = true;
                            break;
                        }
                    }
                    if (!third) {
                        open3.push_back({ f, n, fj });
                        open.erase(open.begin() + k);
                    }
                    paired = true;
                    break;
                }
            }
            if (!paired) {
                open.push_back({ n, j });
            }
        }
        // under --split-mode tensor the GDN layer's beta and alpha (in that order) join an open in-proj + z pair as its third and
        // fourth products, under the same span and early rules; the pair then takes no other third
        if (ab_candidate(n)) {
            for (size_t k = 0; k < open3.size(); ++k) {
                open_pair & o = open3[k];
                if (!same_input(o.f->src[1], n->src[1]) || j - o.j > GGML_CUDA_SIB_GEMV_MAX_SPAN || !early_safe(o.j, j, n)) {
                    continue;
                }
                if (o.c == nullptr) {
                    if (ggml_cuda_qpn_gdn4_ok(o.f->src[0], o.s->src[0], n->src[0], n->src[0], n->src[1])) {
                        o.c = n;
                        break;
                    }
                } else if (ggml_cuda_qpn_gdn4_ok(o.f->src[0], o.s->src[0], o.c->src[0], n->src[0], n->src[1])) {
                    plan.ab[o.f] = { o.c, n };
                    plan.ab_member.insert(o.c);
                    plan.ab_member.insert(n);
                    open3.erase(open3.begin() + k);
                    break;
                }
            }
        }
        // a node that writes over a waiting candidate's input ends its wait
        for (size_t k = 0; k < open.size(); ) {
            if (open[k].n != n && overlap(n, open[k].n->src[1])) {
                open.erase(open.begin() + k);
            } else {
                ++k;
            }
        }
        for (size_t k = 0; k < open3.size(); ) {
            if (open3[k].f != n && open3[k].s != n && overlap(n, open3[k].f->src[1])) {
                open3.erase(open3.begin() + k);
            } else {
                ++k;
            }
        }
    }

    // pairs of small dp4a products on one input (LLAMA_MMVQ_GROUP, default on; 0 = off), the same walk and rules
    static const bool mmvq_group = [] { const char * e = getenv("LLAMA_MMVQ_GROUP"); return e == nullptr || atoi(e) != 0; }();
    if (!mmvq_group) {
        return;
    }
    auto candidate_q8 = [&](const ggml_tensor * n) {
        if (n->op != GGML_OP_MUL_MAT || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_cuda_gdn_folds.skip.count(n) || plan.done.count(n) ||
                plan.ab_member.count(n)) { // launched with their in-proj + z
            return false;
        }
        const ggml_tensor * src0s[1] = { n->src[0] };
        const ggml_tensor * dsts[1]  = { n };
        return !ggml_cuda_qpn_is_repacked(n->src[0]) && n->src[0]->type == GGML_TYPE_Q8_0 && n->src[1]->data != nullptr && n->data != nullptr &&
            (n->src[1]->ne[1]*n->src[1]->ne[2] == 8 || n->src[1]->ne[1]*n->src[1]->ne[2] == 16) && src0s[0] != nullptr && dsts[0] != nullptr;
    };
    std::vector<open_one> open_q8;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (candidate_q8(n)) {
            bool paired = false;
            for (size_t k = 0; k < open_q8.size(); ++k) {
                ggml_tensor * f = open_q8[k].n;
                const ggml_tensor * src0s[2] = { f->src[0], n->src[0] };
                const ggml_tensor * dsts[2]  = { f, n };
                if (same_input(f->src[1], n->src[1]) && j - open_q8[k].j <= GGML_CUDA_SIB_GEMV_MAX_SPAN &&
                        ggml_cuda_mul_mat_vec_q_group_ok(src0s, 2, f->src[1], dsts, cuda_ctx->device) && early_safe(open_q8[k].j, j, n)) {
                    plan.mmvq_partner[f] = n;
                    open_q8.erase(open_q8.begin() + k);
                    paired = true;
                    break;
                }
            }
            if (!paired) {
                open_q8.push_back({ n, j });
            }
        }
        for (size_t k = 0; k < open_q8.size(); ) {
            if (open_q8[k].n != n && overlap(n, open_q8[k].n->src[1])) {
                open_q8.erase(open_q8.begin() + k);
            } else {
                ++k;
            }
        }
    }
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, uint64_t graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            {
                GGML_RT_SCOPE("cg.plan");
                ggml_cuda_plan_gdn_folds(cgraph);
                ggml_cuda_plan_glue_folds(cgraph);
                ggml_cuda_plan_q8_share(cuda_ctx, cgraph);
                ggml_cuda_plan_sib_gemv(cuda_ctx, cgraph);
                ggml_cuda_plan_qpn_sib(cuda_ctx, cgraph);
            }

            static const int rt_nodes_id = ggml_rt_on() ? ggml_rt_id("cg.nodes") : -1;
            ggml_rt_scope rt_nodes(rt_nodes_id);
            if (rt_nodes_id >= 0) {
                ggml_cuda_rt_stamp(cuda_ctx->device, cuda_ctx->stream(), false);
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // folded into a later gated_delta_net launch
                if (!ggml_cuda_gdn_folds.skip.empty() && ggml_cuda_gdn_folds.skip.count(node)) {
                    continue;
                }
                // computed later by the kernel of a node that reads it
                if (!ggml_cuda_glue_folds.skip.empty() && ggml_cuda_glue_folds.skip.count(node)) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
                if (ggml_cuda_sib_gemv_done(node)) { continue; } // computed early by a sibling's group launch
                if (ggml_cuda_qpn_sib_done(node)) { continue; }  // computed early by its QPN sibling's launch
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                // a product whose output the next ALLREDUCE reduces may push it to the peer as it computes it
                cuda_ctx->p2p_push_ar = node->op == GGML_OP_MUL_MAT && !is_concurrent_event_active ? ggml_cuda_p2p_ar_of(cgraph, i) : nullptr;
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                cuda_ctx->p2p_push_ar = nullptr;
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            if (rt_nodes_id >= 0) {
                ggml_cuda_rt_stamp(cuda_ctx->device, cuda_ctx->stream(), true);
            }

            // what the folds did in a decode-sized graph, once per distinct outcome
            const ggml_cuda_gdn_fold_plan & folds = ggml_cuda_gdn_folds;
            if (folds.n_gdn > 0 && folds.n_tokens <= GGML_CUDA_GDN_FOLD_MAX_TOKENS) {
                char line[256];
                snprintf(line, sizeof(line), "GDN folds on CUDA%d at %" PRId64 " tokens x %" PRId64 " seqs: %d gated_delta_net, qknorm %d, "
                        "gate %d (else one gate kernel: %d), beta %d, state %d, conv %d",
                        cuda_ctx->device, folds.n_tokens, folds.n_seqs, folds.n_gdn, folds.n_qknorm, folds.n_gates, folds.n_gate_kernel,
                        folds.n_beta, folds.n_state, folds.n_conv);
                static std::mutex logged_mutex;
                static std::unordered_set<std::string> logged;
                std::lock_guard<std::mutex> lock(logged_mutex);
                if (logged.insert(line).second) {
                    GGML_LOG_INFO("%s: %s\n", __func__, line);
                }
            }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            {
                GGML_RT_SCOPE("cg.endcapture");
                CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            }
            if (ggml_rt_on()) {
                size_t n_cg = 0;
                CUDA_CHECK(cudaGraphGetNodes(graph->graph, nullptr, &n_cg));
                fprintf(stderr, "round_timers: captured a cuda graph, label %d: %d ggml nodes, %zu cuda graph nodes\n",
                        ggml_rt_cur_label(), cgraph->n_nodes, n_cg);
            }
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            GGML_RT_SCOPE("cg.instantiate");
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            GGML_RT_SCOPE("cg.execupdate");
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        GGML_RT_SCOPE("cg.launch");
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    GGML_RT_SCOPE("cuda.graph_compute");
    {
        GGML_RT_SCOPE("cg.set_device");
        ggml_cuda_set_device(cuda_ctx->device);
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    uint64_t graph_key = 0;

#ifdef USE_CUDA_GRAPH
    static const int rt_key_id = ggml_rt_on() ? ggml_rt_id("cg.key") : -1;
    const auto rt_k0 = rt_key_id >= 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (rt_key_id >= 0) {
        ggml_rt_add(rt_key_id, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - rt_k0).count());
    }
    if (graph->is_enabled()) {
        bool graph_compatible;
        {
            GGML_RT_SCOPE("cg.compat");
            graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        }
        if (graph_compatible) {
            bool properties_changed;
            {
                GGML_RT_SCOPE("cg.update_required");
                properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);
            }

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (ggml_rt_on()) {
        if (!use_cuda_graph) {
            GGML_RT_COUNT("cg.direct", 1);
        } else if (cuda_graph_update_required) {
            GGML_RT_COUNT("cg.capture", 1);
        } else {
            GGML_RT_COUNT("cg.replay", 1);
        }
    }
    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    GGML_RT_SCOPE("cg.eval");
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        ggml_cuda_order_sib_gemv(cuda_ctx, cgraph); // siblings together, for their grouped launch
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

#ifdef USE_CUDA_GRAPH
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
    int type_cached = -1; // [TAG_DEVTYPE_CACHE] ggml_backend_cuda_device_get_type's answer, once known
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = ggml_cuda_env_enabled("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    // [TAG_DEVTYPE_CACHE] cudaGetDeviceProperties costs about 120 us, and the pipelined decode asks for
    // every backend's type on each submit. the answer (prop.integrated) does not change: keep it
    // (LLAMA_CUDA_DEVTYPE_CACHE=0 asks the driver every time, as before)
    static const bool cache = [] { const char * e = getenv("LLAMA_CUDA_DEVTYPE_CACHE"); return e == nullptr || atoi(e) != 0; }();
    if (cache && ctx->type_cached >= 0) {
        return (enum ggml_backend_dev_type) ctx->type_cached;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    const enum ggml_backend_dev_type type = prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
    ctx->type_cached = (int) type;
    return type;
}

static bool ggml_backend_cuda_host_buffer_supported() {
    return getenv("GGML_CUDA_NO_PINNED") == nullptr;
}

static bool ggml_backend_cuda_device_supports_cuda_host_buft(int device) {
#if defined(GGML_USE_HIP)
    if (ggml_cuda_info().devices[device].integrated) {
        return false;
    }
#else
    GGML_UNUSED(device);
#endif

    return ggml_backend_cuda_host_buffer_supported();
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = ggml_backend_cuda_host_buffer_supported();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    if (!ggml_backend_cuda_host_buffer_supported()) {
        return nullptr;
    }

    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_XS:
                    case GGML_TYPE_IQ1_XXS:
                    case GGML_TYPE_IQ1_XXXS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                if (ggml_cuda_qpn_is_repacked(op->src[0])) { // only its fp16 expansion
                    return op->type == GGML_TYPE_F16 && ggml_are_same_shape(op, op->src[0]);
                }
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_FLASH_ATTN_EXT_BANDED:
            return ggml_cuda_flash_attn_ext_banded_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_ALLREDUCE:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_TOP_K_SPLIT:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_nrows(op->src[0]) == 1 &&
                op->src[0]->ne[0] >= ggml_get_op_params_i32(op, 0);
        case GGML_OP_DRAFT_PICK:
            return true;
        case GGML_OP_QSA_SELECT:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]);
        case GGML_OP_QSA_VIS_ROWS:
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 &&
                   op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]);
        case GGML_OP_QSA_UNION:
            return ggml_cuda_qsa_union_supported(op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) ||
        (integrated &&
         ggml_backend_buft_is_cuda_host(buft) &&
         ggml_backend_cuda_device_supports_cuda_host_buft(dev_ctx->device));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// in-graph P2P AllReduce for the meta backend (--split-mode tensor), see allreduce-p2p.cuh
static void * ggml_backend_cuda_p2p_ar_init(ggml_backend_t * backends, size_t n_backends) {
    std::vector<int> devices;
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
        devices.push_back(((ggml_backend_cuda_context *) backends[i]->context)->device);
    }
    ggml_cuda_p2p_ar * ar = ggml_cuda_p2p_ar_init(devices.data(), devices.size());
    if (ar != nullptr) {
        // cublasCreate can block while a peer GPU spins in the AllReduce: create the handles now
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backends[i]->context;
            ggml_cuda_set_device(ctx->device);
            ctx->cublas_handle();
        }
    }
    return ar;
}

// true if computing cgraph will only replay a captured CUDA graph: no capture, instantiation or eager launches
static bool ggml_backend_cuda_graph_ready(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
#ifdef USE_CUDA_GRAPH
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (cgraph->n_nodes == 0 || cgraph->uid == 0) {
        return false;
    }
    // look up without inserting: cuda_ctx->cuda_graph() creates an entry and may evict
    const auto it = cuda_ctx->cuda_graphs.find(ggml_cuda_graph_get_key(const_cast<ggml_cgraph *>(cgraph)));
    if (it == cuda_ctx->cuda_graphs.end()) {
        return false;
    }
    const ggml_cuda_graph & graph = *it->second;
    return graph.is_enabled() && graph.instance != nullptr && graph.warmup_complete && graph.uid == cgraph->uid &&
        ggml_time_us() - graph.last_used_time < 5'000'000; // not about to be evicted
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    return false;
#endif // USE_CUDA_GRAPH
}

static void ggml_backend_cuda_p2p_ar_free(void * ar) {
    ggml_cuda_p2p_ar_free((ggml_cuda_p2p_ar *) ar);
}
static bool ggml_backend_cuda_p2p_ar_reserve(void * ar, size_t nbytes) {
    return ggml_cuda_p2p_ar_reserve((ggml_cuda_p2p_ar *) ar, nbytes);
}
static void ggml_backend_cuda_p2p_ar_set_params(void * ar, size_t rank, struct ggml_tensor * node) {
    ggml_cuda_p2p_ar_set_params((ggml_cuda_p2p_ar *) ar, rank, node);
}
// one eager reduction of tensors[j] on backends[j], every rank's kernel launched before returning
static void ggml_backend_cuda_p2p_ar_allreduce(void * ar, ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends) {
    for (size_t j = 0; j < n_backends; ++j) {
        ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backends[j]->context;
        ggml_cuda_set_device(ctx->device);
        ggml_cuda_p2p_ar_launch((ggml_cuda_p2p_ar *) ar, (int) j, ctx->stream(), tensors[j]);
    }
}

// the copy-engine reduction in two halves for the meta backend's two-batch overlap: start on every rank (returns the
// call's index, or -1 if tensors[0] does not take the copy-engine path; then nothing was launched), finish later on each rank
static size_t ggml_backend_cuda_p2p_ar_ce_min_bytes(void * ar) {
    return ggml_cuda_p2p_ar_ce_min_bytes((ggml_cuda_p2p_ar *) ar);
}
static int64_t ggml_backend_cuda_p2p_ar_start(void * ar, ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends) {
    for (size_t j = 0; j < n_backends; ++j) {
        if (!ggml_cuda_p2p_ar_ce_eligible((ggml_cuda_p2p_ar *) ar, tensors[j])) {
            return -1;
        }
    }
    int64_t call = -1;
    for (size_t j = 0; j < n_backends; ++j) {
        ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backends[j]->context;
        ggml_cuda_set_device(ctx->device);
        const int64_t c = (int64_t) ggml_cuda_p2p_ar_ce_start((ggml_cuda_p2p_ar *) ar, (int) j, ctx->stream(), tensors[j]);
        GGML_ASSERT(call < 0 || c == call);
        call = c;
    }
    return call;
}
static void ggml_backend_cuda_p2p_ar_finish(void * ar, ggml_backend_t backend, size_t rank, struct ggml_tensor * tensor, int64_t call) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    ggml_cuda_p2p_ar_ce_finish((ggml_cuda_p2p_ar *) ar, (int) rank, ctx->stream(), tensor, (uint64_t) call);
}

// fattn-verify-stream.cu
void ggml_cuda_fattn_verify_stream_limits(int * min_keys, int * max_tok);

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_upload_nosync") == 0) {
        return (void *)ggml_backend_cuda_upload_nosync;
    }
    if (strcmp(name, "ggml_backend_upload_sync") == 0) {
        return (void *)ggml_backend_cuda_upload_sync;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_init") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_init;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_free") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_free;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_reserve") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_reserve;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_set_params") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_set_params;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_allreduce") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_allreduce;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_ce_min_bytes") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_ce_min_bytes;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_start") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_start;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_finish") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_finish;
    }
    if (strcmp(name, "ggml_backend_graph_ready") == 0) {
        return (void *)ggml_backend_cuda_graph_ready;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_cuda_kq_mask") == 0) {
        return (void *)ggml_backend_cuda_kq_mask;
    }
    if (strcmp(name, "ggml_backend_cuda_fattn_stream_limits") == 0) {
        return (void *)ggml_cuda_fattn_verify_stream_limits;
    }
    if (strcmp(name, "ggml_backend_cuda_diffusion_sample") == 0) {
        return (void *)ggml_cuda_diffusion_sample;
    }
    if (strcmp(name, "ggml_backend_cuda_qpn_repack_slice") == 0) {
        return (void *)ggml_backend_cuda_qpn_repack_slice;
    }
    if (strcmp(name, "ggml_backend_cuda_qpn_repack") == 0) {
        return (void *)ggml_backend_cuda_qpn_repack;
    }
    if (strcmp(name, "ggml_backend_cuda_qpn_unpack") == 0) {
        return (void *)ggml_backend_cuda_qpn_unpack;
    }
    if (strcmp(name, "ggml_backend_cuda_qpn_repack_draft") == 0) {
        return (void *)ggml_backend_cuda_qpn_repack_draft;
    }
    if (strcmp(name, "ggml_backend_cuda_qpn_repack_dflash") == 0) {
        return (void *)ggml_backend_cuda_qpn_repack_dflash;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    if (ggml_rt_on()) {
        ggml_cuda_rt_init(device);
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
