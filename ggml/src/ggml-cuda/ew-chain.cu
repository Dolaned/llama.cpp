#include "ew-chain.cuh"
#include "unary.cuh"
#include "ggml-impl.h"

// One launch for a short chain of F32 elementwise ops (scale, silu, sigmoid, softplus, add, mul), where each op
// consumes the previous one and nothing else reads the intermediates. Every output element evaluates the chain
// from broadcast reads of its operands, so smaller operands (a per-token gate, a per-head bias) need no repeat.

#define EW_CHAIN_MAX_OPS 4

enum ew_chain_op_kind {
    EW_CHAIN_SCALE,
    EW_CHAIN_SILU,
    EW_CHAIN_SIGMOID,
    EW_CHAIN_SOFTPLUS,
    EW_CHAIN_ADD,
    EW_CHAIN_MUL,
};

struct ew_chain_operand {
    const float * data;
    int64_t       ne[4];
};

struct ew_chain_op {
    ew_chain_op_kind kind;
    float            scale;
    float            bias;
    ew_chain_operand other; // binary ops only
};

struct ew_chain_params {
    ew_chain_operand in;
    ew_chain_op      ops[EW_CHAIN_MAX_OPS];
    int              n_ops;
    int64_t          ne[4];
};

static __device__ __forceinline__ float ew_chain_load(const ew_chain_operand & t, const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) {
    return t.data[(((i3 % t.ne[3])*t.ne[2] + (i2 % t.ne[2]))*t.ne[1] + (i1 % t.ne[1]))*t.ne[0] + (i0 % t.ne[0])];
}

static __global__ void ew_chain_kernel(const ew_chain_params p, float * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t i0 = i % p.ne[0];
    const int64_t i1 = (i / p.ne[0]) % p.ne[1];
    const int64_t i2 = (i / (p.ne[0]*p.ne[1])) % p.ne[2];
    const int64_t i3 = i / (p.ne[0]*p.ne[1]*p.ne[2]);

    // the same expressions as the standalone kernels, so the result does not change
    float x = ew_chain_load(p.in, i0, i1, i2, i3);
    for (int k = 0; k < p.n_ops; ++k) {
        const ew_chain_op & op = p.ops[k];
        switch (op.kind) {
            case EW_CHAIN_SCALE:    x = op.scale * x + op.bias;                         break;
            case EW_CHAIN_SILU:     x = ggml_cuda_op_silu_single(x);                    break;
            case EW_CHAIN_SIGMOID:  x = 1.0f / (1.0f + expf(-x));                       break;
            case EW_CHAIN_SOFTPLUS: x = (x > 20.0f) ? x : log1pf(expf(x));             break;
            case EW_CHAIN_ADD:      x = x + ew_chain_load(op.other, i0, i1, i2, i3);   break;
            case EW_CHAIN_MUL:      x = x * ew_chain_load(op.other, i0, i1, i2, i3);   break;
        }
    }
    dst[i] = x;
}

static bool ew_chain_kind(const ggml_tensor * node, ew_chain_op_kind & kind) {
    switch (node->op) {
        case GGML_OP_SCALE: kind = EW_CHAIN_SCALE; return true;
        case GGML_OP_ADD:   kind = EW_CHAIN_ADD;   return true;
        case GGML_OP_MUL:   kind = EW_CHAIN_MUL;   return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_SILU:     kind = EW_CHAIN_SILU;     return true;
                case GGML_UNARY_OP_SIGMOID:  kind = EW_CHAIN_SIGMOID;  return true;
                case GGML_UNARY_OP_SOFTPLUS: kind = EW_CHAIN_SOFTPLUS; return true;
                default: return false;
            }
        default: return false;
    }
}

static bool ew_chain_operand_ok(const ggml_tensor * t, const ggml_tensor * out) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_can_repeat(t, out);
}

static void ew_chain_set_operand(ew_chain_operand & o, const ggml_tensor * t) {
    o.data = (const float *) t->data;
    for (int d = 0; d < 4; ++d) {
        o.ne[d] = t->ne[d];
    }
}

int ggml_cuda_try_fuse_ew_chain(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // grow the chain while the next node consumes the previous one, which nobody else reads
    ew_chain_op_kind kinds[EW_CHAIN_MAX_OPS];
    int n = 0;
    for (; n < EW_CHAIN_MAX_OPS && i + n < cgraph->n_nodes; ++n) {
        const ggml_tensor * node = cgraph->nodes[i + n];
        if (!ew_chain_kind(node, kinds[n]) || node->type != GGML_TYPE_F32 || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            break;
        }
        if (n > 0) {
            const ggml_tensor * prev = cgraph->nodes[i + n - 1];
            const bool binary = kinds[n] == EW_CHAIN_ADD || kinds[n] == EW_CHAIN_MUL;
            const bool consumes = node->src[0] == prev || (binary && node->src[1] == prev);
            if (!consumes || (binary && node->src[0] == node->src[1]) || !ggml_node_has_n_uses(cgraph, i + n - 1, 1)) {
                break;
            }
        }
    }
    if (n < 2) {
        return 0;
    }

    const ggml_tensor * out = cgraph->nodes[i + n - 1];
    if (!ggml_is_contiguous(out)) {
        return 0;
    }

    ew_chain_params p = {};
    p.n_ops = n;
    for (int d = 0; d < 4; ++d) {
        p.ne[d] = out->ne[d];
    }
    const ggml_tensor * operands[EW_CHAIN_MAX_OPS + 1];
    int n_operands = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * node = cgraph->nodes[i + k];
        if (!ggml_can_repeat(node, out)) {
            return 0;
        }
        ew_chain_op & op = p.ops[k];
        op.kind = kinds[k];
        if (op.kind == EW_CHAIN_SCALE) {
            op.scale = ggml_get_op_params_f32(node, 0);
            op.bias  = ggml_get_op_params_f32(node, 1);
        }
        // the chain value: src0 of the first node, the previous node after that
        const ggml_tensor * chained = k == 0 ? node->src[0] : cgraph->nodes[i + k - 1];
        if (k == 0) {
            if (!ew_chain_operand_ok(chained, out)) {
                return 0;
            }
            ew_chain_set_operand(p.in, chained);
            operands[n_operands++] = chained;
        }
        if (op.kind == EW_CHAIN_ADD || op.kind == EW_CHAIN_MUL) {
            const ggml_tensor * other = node->src[0] == chained ? node->src[1] : node->src[0];
            if (!ew_chain_operand_ok(other, out)) {
                return 0;
            }
            ew_chain_set_operand(op.other, other);
            operands[n_operands++] = other;
        }
    }

    // writing the output must not clobber operand values that other elements still read:
    // an operand may only share memory with the output when it is the same, same-shaped region
    const char * o0 = (const char *) out->data;
    const char * o1 = o0 + ggml_nbytes(out);
    for (int k = 0; k < n_operands; ++k) {
        const char * t0 = (const char *) operands[k]->data;
        const char * t1 = t0 + ggml_nbytes(operands[k]);
        if (t0 < o1 && o0 < t1 && !(t0 == o0 && ggml_are_same_shape(operands[k], out))) {
            return 0;
        }
    }

    const int64_t ne = ggml_nelements(out);
    const int block = 256;
    ew_chain_kernel<<<(ne + block - 1) / block, block, 0, ctx.stream()>>>(p, (float *) out->data, ne);
    CUDA_CHECK(cudaGetLastError());
    return n - 1;
}
