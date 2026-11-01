#include "common.cuh"

// Fuses a chain of elementwise ops starting at node i; returns how many nodes after i it covered (0 if none).
int ggml_cuda_try_fuse_ew_chain(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
