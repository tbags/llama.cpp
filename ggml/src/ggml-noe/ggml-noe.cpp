#include "ggml-noe.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "cix_noe_standard_api.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <unistd.h>

#define NOE_CHECK(call)                                                      \
    do {                                                                     \
        noe_status_t _status = (call);                                       \
        if (_status != NOE_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "[GGML-NOE] Error at %s:%d (code: 0x%X)\n",     \
                    __FILE__, __LINE__, _status);                            \
        }                                                                    \
    } while (0)

#define GGML_NOE_NAME "NOE"

// Device context
struct ggml_backend_noe_device_context {
    context_handler_t * ctx_handler = nullptr;
    char target_name[64] = {0};
    uint32_t cluster_count = 0;
    uint32_t core_count = 0;
    std::string description;
    bool initialized = false;
};

static ggml_backend_noe_device_context g_dev_ctx;
static std::mutex g_noe_mutex;

// Graph instance loaded in UMD
struct noe_graph_instance {
    uint64_t graph_id = 0;
    uint64_t job_id = 0;
    uint32_t in_count = 0;
    uint32_t out_count = 0;
    std::vector<tensor_desc_t> in_descs;
    std::vector<tensor_desc_t> out_descs;
};

// Backend context
struct ggml_backend_noe_context {
    ggml_backend_noe_device_context * dev_ctx = nullptr;
    std::unordered_map<std::string, noe_graph_instance> graph_cache;
    std::string model_dir;
};

// --------------------------------------------------------------------------
// Buffer interface (UMA: NPU shares host physical memory)
// --------------------------------------------------------------------------

static struct ggml_backend_device g_ggml_backend_noe_device;

ggml_backend_buffer_type_t ggml_backend_noe_buffer_type(void) {
    return ggml_backend_cpu_buffer_type();
}

// --------------------------------------------------------------------------
// Backend interface
// --------------------------------------------------------------------------

static noe_graph_instance * ggml_backend_noe_get_or_load_graph(
    ggml_backend_noe_context * ctx,
    const std::string & model_path) {
    auto it = ctx->graph_cache.find(model_path);
    if (it != ctx->graph_cache.end()) {
        return &it->second;
    }

    if (access(model_path.c_str(), R_OK) != 0) {
        return nullptr;
    }

    uint64_t graph_id = 0;
    noe_status_t status = noe_load_graph(ctx->dev_ctx->ctx_handler, model_path.c_str(), &graph_id);
    if (status != NOE_STATUS_SUCCESS || graph_id == 0) {
        fprintf(stderr, "[GGML-NOE] Failed to load %s (status: 0x%X)\n", model_path.c_str(), status);
        return nullptr;
    }

    uint32_t in_cnt = 0, out_cnt = 0;
    noe_get_tensor_count(ctx->dev_ctx->ctx_handler, graph_id, NOE_TENSOR_TYPE_INPUT, &in_cnt);
    noe_get_tensor_count(ctx->dev_ctx->ctx_handler, graph_id, NOE_TENSOR_TYPE_OUTPUT, &out_cnt);

    noe_graph_instance inst;
    inst.graph_id = graph_id;
    inst.in_count = in_cnt;
    inst.out_count = out_cnt;
    inst.in_descs.resize(in_cnt);
    inst.out_descs.resize(out_cnt);

    for (uint32_t i = 0; i < in_cnt; i++) {
        noe_get_tensor_descriptor(ctx->dev_ctx->ctx_handler, graph_id, NOE_TENSOR_TYPE_INPUT, i, &inst.in_descs[i]);
    }
    for (uint32_t i = 0; i < out_cnt; i++) {
        noe_get_tensor_descriptor(ctx->dev_ctx->ctx_handler, graph_id, NOE_TENSOR_TYPE_OUTPUT, i, &inst.out_descs[i]);
    }

    job_config_npu_t npu_cfg = {};
    job_config_t j_cfg = { &npu_cfg };
    status = noe_create_job(ctx->dev_ctx->ctx_handler, graph_id, &inst.job_id, &j_cfg);
    if (status != NOE_STATUS_SUCCESS || inst.job_id == 0) {
        fprintf(stderr, "[GGML-NOE] Failed to create job for %s (status: 0x%X)\n", model_path.c_str(), status);
        noe_unload_graph(ctx->dev_ctx->ctx_handler, graph_id);
        return nullptr;
    }

    ctx->graph_cache[model_path] = std::move(inst);
    return &ctx->graph_cache[model_path];
}

static const char * ggml_backend_noe_name(ggml_backend_t backend) {
    return GGML_NOE_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_noe_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_noe_context *) backend->context;
    if (ctx) {
        if (ctx->dev_ctx && ctx->dev_ctx->ctx_handler) {
            for (auto & kv : ctx->graph_cache) {
                if (kv.second.job_id) {
                    noe_clean_job(ctx->dev_ctx->ctx_handler, kv.second.job_id);
                }
                if (kv.second.graph_id) {
                    noe_unload_graph(ctx->dev_ctx->ctx_handler, kv.second.graph_id);
                }
            }
        }
        delete ctx;
    }
    delete backend;
}

static void quantize_f32_to_i8(const float * src, int8_t * dst, size_t n, float scale, int32_t zp) {
    for (size_t i = 0; i < n; i++) {
        int v = std::round(src[i] * scale) + zp;
        v = std::max(-128, std::min(127, v));
        dst[i] = (int8_t) v;
    }
}

static void dequantize_i8_to_f32(const int8_t * src, float * dst, size_t n, float scale, int32_t zp) {
    float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 1.0f;
    for (size_t i = 0; i < n; i++) {
        dst[i] = ((float) src[i] - (float) zp) * inv_scale;
    }
}

static enum ggml_status ggml_backend_noe_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_noe_context *) backend->context;
    GGML_ASSERT(ctx != nullptr && ctx->dev_ctx != nullptr);

    static const bool log_noe = (getenv("GGML_NOE_LOG") != nullptr);
    const char * env_model_dir = getenv("GGML_NOE_MODEL_DIR");
    std::string model_dir = env_model_dir ? env_model_dir : ctx->model_dir;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_MUL_MAT) {
            std::string kernel_path = model_dir + "/dynamic_matmul.cix";
            noe_graph_instance * inst = ggml_backend_noe_get_or_load_graph(ctx, kernel_path);
            if (inst != nullptr && inst->in_count >= 2) {
                const struct ggml_tensor * src0 = node->src[0];
                const struct ggml_tensor * src1 = node->src[1];

                if (src1->ne[0] == 3584 && src0->ne[1] == 18944 && inst->in_descs[0].size == 3584) {
                    std::vector<int8_t> act_i8(3584);
                    if (src1->type == GGML_TYPE_F32) {
                        quantize_f32_to_i8((const float *) src1->data, act_i8.data(), 3584,
                                           inst->in_descs[0].scale, inst->in_descs[0].zero_point);
                    } else {
                        memcpy(act_i8.data(), src1->data, 3584);
                    }

                    noe_load_tensor(ctx->dev_ctx->ctx_handler, inst->job_id, 0, act_i8.data());
                    noe_load_tensor(ctx->dev_ctx->ctx_handler, inst->job_id, 1, src0->data);
                    noe_status_t st = noe_job_infer_sync(ctx->dev_ctx->ctx_handler, inst->job_id, 5000);
                    if (st == NOE_STATUS_SUCCESS) {
                        std::vector<int8_t> out_i8(inst->out_descs[0].size);
                        noe_get_tensor(ctx->dev_ctx->ctx_handler, inst->job_id, NOE_TENSOR_TYPE_OUTPUT, 0, out_i8.data());
                        if (node->type == GGML_TYPE_F32) {
                            dequantize_i8_to_f32(out_i8.data(), (float *) node->data, node->ne[0],
                                                 inst->out_descs[0].scale, inst->out_descs[0].zero_point);
                        } else {
                            memcpy(node->data, out_i8.data(), inst->out_descs[0].size);
                        }
                        if (log_noe) {
                            fprintf(stderr, "[GGML-NOE] Executed '%s' on Zhouyi NPU\n",
                                    node->name[0] ? node->name : "matmul");
                        }
                        continue;
                    }
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i noe_backend_interface = {
    /* .get_name                = */ ggml_backend_noe_name,
    /* .free                    = */ ggml_backend_noe_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_noe_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_noe_guid(void) {
    static const char * guid_str = "CIX-NOE-NPU00001";
    return reinterpret_cast<ggml_guid_t>((void *)(uintptr_t) guid_str);
}

// --------------------------------------------------------------------------
// Device interface
// --------------------------------------------------------------------------

static const char * ggml_backend_noe_device_get_name(ggml_backend_dev_t dev) {
    return GGML_NOE_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_noe_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_noe_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_noe_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = (size_t) pages * page_size;
    *free = *total;
#else
    *free = 0;
    *total = 0;
#endif
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_noe_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_noe_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_noe_device_get_name(dev);
    props->description = ggml_backend_noe_device_get_description(dev);
    props->type = ggml_backend_noe_device_get_type(dev);
    ggml_backend_noe_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_noe_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * ctx = new ggml_backend_noe_context();
    ctx->dev_ctx = (ggml_backend_noe_device_context *) dev->context;

    if (params && params[0]) {
        if (strncmp(params, "model_dir=", 10) == 0) {
            ctx->model_dir = params + 10;
        } else if (strncmp(params, "path=", 5) == 0) {
            ctx->model_dir = params + 5;
        } else {
            ctx->model_dir = params;
        }
    }

    if (ctx->model_dir.empty()) {
        const char * env_dir = getenv("GGML_NOE_MODEL_DIR");
        if (env_dir && env_dir[0]) {
            ctx->model_dir = env_dir;
        } else if (access("./models/dynamic_matmul.cix", R_OK) == 0) {
            ctx->model_dir = "./models";
        } else if (access("/usr/share/cix/models/dynamic_matmul.cix", R_OK) == 0) {
            ctx->model_dir = "/usr/share/cix/models";
        }
    }

    auto * backend = new ggml_backend {
        /* .guid      = */ ggml_backend_noe_guid(),
        /* .iface     = */ noe_backend_interface,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_noe_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_noe_buffer_type();
    GGML_UNUSED(dev);
}

static bool ggml_backend_noe_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    static const bool enable_offload = (getenv("GGML_NOE_ENABLE") != nullptr);
    if (!enable_offload) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return true;
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_noe_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i noe_device_interface = {
    /* .get_name             = */ ggml_backend_noe_device_get_name,
    /* .get_description      = */ ggml_backend_noe_device_get_description,
    /* .get_memory           = */ ggml_backend_noe_device_get_memory,
    /* .get_type             = */ ggml_backend_noe_device_get_type,
    /* .get_props            = */ ggml_backend_noe_device_get_props,
    /* .init_backend         = */ ggml_backend_noe_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_noe_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_noe_device_supports_op,
    /* .supports_buft        = */ ggml_backend_noe_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static struct ggml_backend_reg g_ggml_backend_noe_reg;

// --------------------------------------------------------------------------
// Registry interface
// --------------------------------------------------------------------------

static const char * ggml_backend_noe_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_NOE_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_noe_reg_get_device_count(ggml_backend_reg_t reg) {
    std::lock_guard<std::mutex> lock(g_noe_mutex);
    return g_dev_ctx.initialized ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_noe_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    return &g_ggml_backend_noe_device;
    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i noe_reg_interface = {
    /* .get_name         = */ ggml_backend_noe_reg_get_name,
    /* .get_device_count = */ ggml_backend_noe_reg_get_device_count,
    /* .get_device       = */ ggml_backend_noe_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static void ggml_backend_noe_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_noe_mutex);
    if (g_dev_ctx.initialized && g_dev_ctx.ctx_handler) {
        noe_deinit_context(g_dev_ctx.ctx_handler);
        g_dev_ctx.ctx_handler = nullptr;
        g_dev_ctx.initialized = false;
    }
}

ggml_backend_reg_t ggml_backend_noe_reg(void) {
    std::lock_guard<std::mutex> lock(g_noe_mutex);
    if (!g_dev_ctx.initialized) {
        noe_status_t status = noe_init_context(&g_dev_ctx.ctx_handler);
        if (status == NOE_STATUS_SUCCESS && g_dev_ctx.ctx_handler) {
            noe_get_target(g_dev_ctx.ctx_handler, g_dev_ctx.target_name);
            noe_get_cluster_count(g_dev_ctx.ctx_handler, 0, &g_dev_ctx.cluster_count);
            noe_get_core_count(g_dev_ctx.ctx_handler, 0, 0, &g_dev_ctx.core_count);

            g_dev_ctx.description = std::string("Cix Zhouyi NPU (") +
                                   g_dev_ctx.target_name + ", " +
                                   std::to_string(g_dev_ctx.core_count) + " cores)";
            g_dev_ctx.initialized = true;

            g_ggml_backend_noe_reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ noe_reg_interface,
                /* .context     = */ nullptr,
            };

            g_ggml_backend_noe_device = {
                /* .iface   = */ noe_device_interface,
                /* .reg     = */ &g_ggml_backend_noe_reg,
                /* .context = */ &g_dev_ctx,
            };

            atexit(ggml_backend_noe_cleanup);
        } else {
            fprintf(stderr, "[GGML-NOE] Failed to initialize NOE UMD context (status: 0x%X)\n", status);
            return nullptr;
        }
    }
    return &g_ggml_backend_noe_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_noe_reg)

ggml_backend_t ggml_backend_noe_init(void) {
    if (ggml_backend_noe_reg() == nullptr) {
        return nullptr;
    }
    return ggml_backend_noe_device_init_backend(&g_ggml_backend_noe_device, nullptr);
}

bool ggml_backend_is_noe(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_noe_guid());
}
