
#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

namespace {

inline llaisys::tensor_t t(llaisysTensor_t h) { return h ? h->tensor : nullptr; }

struct Qwen2ModelImpl {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};

    // KV cache per layer: [maxseq, nkvh, dh]
    std::vector<llaisysTensor_t> k_cache;
    std::vector<llaisysTensor_t> v_cache;

    size_t cur_pos{0};

    // Buffers (all on CPU for this assignment)
    llaisysTensor_t idx = nullptr;         // [1] i64
    llaisysTensor_t pos_ids = nullptr;     // [1] i64

    llaisysTensor_t x = nullptr;           // [1, hs]
    llaisysTensor_t h = nullptr;           // [1, hs]

    llaisysTensor_t q_lin = nullptr;       // [1, hs]
    llaisysTensor_t k_lin = nullptr;       // [1, nkvh*dh]
    llaisysTensor_t v_lin = nullptr;       // [1, nkvh*dh]

    llaisysTensor_t q3 = nullptr;          // [1, nh, dh]
    llaisysTensor_t k3 = nullptr;          // [1, nkvh, dh]
    llaisysTensor_t v3 = nullptr;          // [1, nkvh, dh]

    llaisysTensor_t q_rope = nullptr;      // [1, nh, dh]
    llaisysTensor_t k_rope = nullptr;      // [1, nkvh, dh]

    llaisysTensor_t attn_val = nullptr;    // [1, nh, dh]
    llaisysTensor_t attn_val2 = nullptr;   // [1, hs]
    llaisysTensor_t attn_out = nullptr;    // [1, hs]

    llaisysTensor_t gate = nullptr;        // [1, di]
    llaisysTensor_t up = nullptr;          // [1, di]
    llaisysTensor_t ff = nullptr;          // [1, di]
    llaisysTensor_t mlp_out = nullptr;     // [1, hs]

    llaisysTensor_t y = nullptr;           // [1, hs]
    llaisysTensor_t logits = nullptr;      // [1, voc]

    llaisysTensor_t max_idx = nullptr;     // [1] i64
    llaisysTensor_t max_val = nullptr;     // [1] dtype

    float attn_scale{1.0f};

    void destroyTensor(llaisysTensor_t &p) {
        if (p) {
            tensorDestroy(p);
            p = nullptr;
        }
    }

    ~Qwen2ModelImpl() {
        // weights
        destroyTensor(weights.in_embed);
        destroyTensor(weights.out_embed);
        destroyTensor(weights.out_norm_w);
        if (weights.attn_norm_w) {
            for (size_t i = 0; i < meta.nlayer; ++i) destroyTensor(weights.attn_norm_w[i]);
            delete[] weights.attn_norm_w;
        }
        auto destroy_arr = [&](llaisysTensor_t *&arr) {
            if (!arr) return;
            for (size_t i = 0; i < meta.nlayer; ++i) destroyTensor(arr[i]);
            delete[] arr;
            arr = nullptr;
        };
        destroy_arr(weights.attn_q_w);
        destroy_arr(weights.attn_q_b);
        destroy_arr(weights.attn_k_w);
        destroy_arr(weights.attn_k_b);
        destroy_arr(weights.attn_v_w);
        destroy_arr(weights.attn_v_b);
        destroy_arr(weights.attn_o_w);
        destroy_arr(weights.mlp_norm_w);
        destroy_arr(weights.mlp_gate_w);
        destroy_arr(weights.mlp_up_w);
        destroy_arr(weights.mlp_down_w);

        for (auto &kc : k_cache) destroyTensor(kc);
        for (auto &vc : v_cache) destroyTensor(vc);

        destroyTensor(idx);
        destroyTensor(pos_ids);
        destroyTensor(x);
        destroyTensor(h);
        destroyTensor(q_lin);
        destroyTensor(k_lin);
        destroyTensor(v_lin);
        destroyTensor(q3);
        destroyTensor(k3);
        destroyTensor(v3);
        destroyTensor(q_rope);
        destroyTensor(k_rope);
        destroyTensor(attn_val);
        destroyTensor(attn_val2);
        destroyTensor(attn_out);
        destroyTensor(gate);
        destroyTensor(up);
        destroyTensor(ff);
        destroyTensor(mlp_out);
        destroyTensor(y);
        destroyTensor(logits);
        destroyTensor(max_idx);
        destroyTensor(max_val);
    }

    static llaisysTensor_t makeTensor(std::initializer_list<size_t> shape, llaisysDataType_t dtype) {
        std::vector<size_t> s(shape.begin(), shape.end());
        return tensorCreate(s.data(), s.size(), dtype, LLAISYS_DEVICE_CPU, 0);
    }

    void init(const LlaisysQwen2Meta *m) {
        meta = *m;
        if (meta.dh == 0) meta.dh = meta.hs / meta.nh;
        attn_scale = 1.0f / std::sqrt(float(meta.dh));

        // allocate weights
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer]{};
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer]{};
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer]{};
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer]{};
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer]{};
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer]{};

        // embedding weights
        weights.in_embed = makeTensor({meta.voc, meta.hs}, meta.dtype);
        weights.out_embed = makeTensor({meta.voc, meta.hs}, meta.dtype);
        weights.out_norm_w = makeTensor({meta.hs}, meta.dtype);

        // per-layer weights
        for (size_t i = 0; i < meta.nlayer; ++i) {
            weights.attn_norm_w[i] = makeTensor({meta.hs}, meta.dtype);

            weights.attn_q_w[i] = makeTensor({meta.hs, meta.hs}, meta.dtype);
            weights.attn_q_b[i] = makeTensor({meta.hs}, meta.dtype);

            weights.attn_k_w[i] = makeTensor({meta.nkvh * meta.dh, meta.hs}, meta.dtype);
            weights.attn_k_b[i] = makeTensor({meta.nkvh * meta.dh}, meta.dtype);

            weights.attn_v_w[i] = makeTensor({meta.nkvh * meta.dh, meta.hs}, meta.dtype);
            weights.attn_v_b[i] = makeTensor({meta.nkvh * meta.dh}, meta.dtype);

            weights.attn_o_w[i] = makeTensor({meta.hs, meta.hs}, meta.dtype);

            weights.mlp_norm_w[i] = makeTensor({meta.hs}, meta.dtype);
            weights.mlp_gate_w[i] = makeTensor({meta.di, meta.hs}, meta.dtype);
            weights.mlp_up_w[i] = makeTensor({meta.di, meta.hs}, meta.dtype);
            weights.mlp_down_w[i] = makeTensor({meta.hs, meta.di}, meta.dtype);
        }

        // kv cache
        k_cache.resize(meta.nlayer);
        v_cache.resize(meta.nlayer);
        for (size_t i = 0; i < meta.nlayer; ++i) {
            k_cache[i] = makeTensor({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype);
            v_cache[i] = makeTensor({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype);
        }

        // buffers
        idx = makeTensor({1}, LLAISYS_DTYPE_I64);
        pos_ids = makeTensor({1}, LLAISYS_DTYPE_I64);

        x = makeTensor({1, meta.hs}, meta.dtype);
        h = makeTensor({1, meta.hs}, meta.dtype);

        q_lin = makeTensor({1, meta.hs}, meta.dtype);
        k_lin = makeTensor({1, meta.nkvh * meta.dh}, meta.dtype);
        v_lin = makeTensor({1, meta.nkvh * meta.dh}, meta.dtype);

        q3 = makeTensor({1, meta.nh, meta.dh}, meta.dtype);
        k3 = makeTensor({1, meta.nkvh, meta.dh}, meta.dtype);
        v3 = makeTensor({1, meta.nkvh, meta.dh}, meta.dtype);

        q_rope = makeTensor({1, meta.nh, meta.dh}, meta.dtype);
        k_rope = makeTensor({1, meta.nkvh, meta.dh}, meta.dtype);

        attn_val = makeTensor({1, meta.nh, meta.dh}, meta.dtype);
        attn_val2 = makeTensor({1, meta.hs}, meta.dtype);
        attn_out = makeTensor({1, meta.hs}, meta.dtype);

        gate = makeTensor({1, meta.di}, meta.dtype);
        up = makeTensor({1, meta.di}, meta.dtype);
        ff = makeTensor({1, meta.di}, meta.dtype);
        mlp_out = makeTensor({1, meta.hs}, meta.dtype);

        y = makeTensor({1, meta.hs}, meta.dtype);
        logits = makeTensor({1, meta.voc}, meta.dtype);

        max_idx = makeTensor({1}, LLAISYS_DTYPE_I64);
        max_val = makeTensor({1}, meta.dtype);
    }

    void reset() { cur_pos = 0; }

    int64_t step(int64_t token_id) {
        // idx
        tensorLoad(idx, &token_id);
        // embedding
        llaisys::ops::embedding(t(x), t(idx), t(weights.in_embed));

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            // h = rmsnorm(x)
            llaisys::ops::rms_norm(t(h), t(x), t(weights.attn_norm_w[layer]), meta.epsilon);

            // q,k,v projections
            llaisys::ops::linear(t(q_lin), t(h), t(weights.attn_q_w[layer]), t(weights.attn_q_b[layer]));
            llaisys::ops::linear(t(k_lin), t(h), t(weights.attn_k_w[layer]), t(weights.attn_k_b[layer]));
            llaisys::ops::linear(t(v_lin), t(h), t(weights.attn_v_w[layer]), t(weights.attn_v_b[layer]));

            // views: [1, nh, dh], [1, nkvh, dh]
            {
                size_t s_q[3] = {1, meta.nh, meta.dh};
                size_t s_kv[3] = {1, meta.nkvh, meta.dh};
                // tensorView returns a new wrapper - we want to avoid leaks, so destroy after use
                llaisysTensor_t qv = tensorView(q_lin, s_q, 3);
                llaisysTensor_t kv = tensorView(k_lin, s_kv, 3);
                llaisysTensor_t vv = tensorView(v_lin, s_kv, 3);
                // copy into persistent q3/k3/v3 buffers (they are contiguous)
                tensorLoad(q3, tensorGetData(qv));
                tensorLoad(k3, tensorGetData(kv));
                tensorLoad(v3, tensorGetData(vv));
                tensorDestroy(qv);
                tensorDestroy(kv);
                tensorDestroy(vv);
            }

            // pos_ids
            int64_t p = static_cast<int64_t>(cur_pos);
            tensorLoad(pos_ids, &p);

            // ROPE on q and k
            llaisys::ops::rope(t(q_rope), t(q3), t(pos_ids), meta.theta);
            llaisys::ops::rope(t(k_rope), t(k3), t(pos_ids), meta.theta);

            // write cache at cur_pos: slice dim0 [cur_pos, cur_pos+1]
            {
                llaisysTensor_t kslot = tensorSlice(k_cache[layer], 0, cur_pos, cur_pos + 1);
                llaisysTensor_t vslot = tensorSlice(v_cache[layer], 0, cur_pos, cur_pos + 1);
                tensorLoad(kslot, tensorGetData(k_rope));
                tensorLoad(vslot, tensorGetData(v3));
                tensorDestroy(kslot);
                tensorDestroy(vslot);
            }

            // get cache slices [0, cur_pos+1]
            llaisysTensor_t k_all = tensorSlice(k_cache[layer], 0, 0, cur_pos + 1);
            llaisysTensor_t v_all = tensorSlice(v_cache[layer], 0, 0, cur_pos + 1);

            // attention: attn_val [1, nh, dh]
            llaisys::ops::self_attention(t(attn_val), t(q_rope), t(k_all), t(v_all), attn_scale);

            tensorDestroy(k_all);
            tensorDestroy(v_all);

            // attn_val2 = view attn_val as [1, hs] (copy)
            {
                size_t s2[2] = {1, meta.hs};
                llaisysTensor_t av2 = tensorView(attn_val, s2, 2);
                tensorLoad(attn_val2, tensorGetData(av2));
                tensorDestroy(av2);
            }

            // attn_out = linear(attn_val2, o_w, bias=null)
            llaisys::ops::linear(t(attn_out), t(attn_val2), t(weights.attn_o_w[layer]), nullptr);

            // x = x + attn_out
            llaisys::ops::add(t(x), t(x), t(attn_out));

            // MLP
            llaisys::ops::rms_norm(t(h), t(x), t(weights.mlp_norm_w[layer]), meta.epsilon);
            llaisys::ops::linear(t(gate), t(h), t(weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear(t(up), t(h), t(weights.mlp_up_w[layer]), nullptr);
            llaisys::ops::swiglu(t(ff), t(gate), t(up));
            llaisys::ops::linear(t(mlp_out), t(ff), t(weights.mlp_down_w[layer]), nullptr);
            llaisys::ops::add(t(x), t(x), t(mlp_out));
        }

        // final norm
        llaisys::ops::rms_norm(t(y), t(x), t(weights.out_norm_w), meta.epsilon);
        // logits = linear(y, out_embed, bias=null)
        llaisys::ops::linear(t(logits), t(y), t(weights.out_embed), nullptr);

        // argmax
        llaisys::ops::argmax(t(max_idx), t(max_val), t(logits));
        int64_t next = *reinterpret_cast<int64_t *>(tensorGetData(max_idx));

        cur_pos += 1;
        return next;
    }

    int64_t infer(int64_t *token_ids, size_t ntoken) {
        int64_t next = meta.end_token;
        for (size_t i = 0; i < ntoken; ++i) {
            next = step(token_ids[i]);
            if (cur_pos >= meta.maxseq) break;
        }
        return next;
    }
};

} // namespace

__C {

struct LlaisysQwen2Model {
    Qwen2ModelImpl *impl;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    (void)device;
    (void)device_ids;
    (void)ndevice;
    auto *m = new LlaisysQwen2Model{};
    m->impl = new Qwen2ModelImpl();
    m->impl->init(meta);
    return m;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;
    delete model->impl;
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return model ? &model->impl->weights : nullptr;
}

void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    if (!model) return;
    model->impl->reset();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) return 0;
    return model->impl->infer(token_ids, ntoken);
}

} // extern "C"
