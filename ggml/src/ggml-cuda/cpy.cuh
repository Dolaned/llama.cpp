#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

#define GGML_CUDA_CPY_MULTI_MAX 8

// n f32 copies with the same shapes and strides in one launch, false if they would not take the generic scalar path
bool ggml_cuda_cpy_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0, ggml_tensor * const * src1, int n);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
