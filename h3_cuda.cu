/* h3_cuda.cu - CUDA backend for h3.c (feat/cuda).
 * Metal backend (h3_gpu.m/h3_shaders.metal) is preserved untouched;
 * this file implements the same h3_gpu.h C API against CUDA/cuBLAS.
 * I1 scaffold + I2 tensor layer + I3 elementwise/norm/activation/embedding real;
 * remaining compute ops are stubs.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include "h3_gpu.h"
#include "h3.h"
#include "h3_cuda.h"

#define H3_CUDA_ERR "CUDA backend: op not yet implemented (feat/cuda)"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define H3_CU_BLOCK 256

struct h3_gpu { void *dev_ctx; h3_gpu_stats stats; char error[512]; };
struct h3_gpu_tensor { void *device_ptr; h3_gpu_dtype dtype; size_t elements; size_t bytes; h3_gpu *owner; };

static unsigned h3_cu_grid(unsigned n) { return (n + H3_CU_BLOCK - 1) / H3_CU_BLOCK; }

/* BF16 helpers -- match Metal exactly (round-to-nearest-even). __device__ so
 * kernels can call them; CUDA intrinsics avoid memcpy in device code. */
__device__ float h3_bf16_to_f32(uint16_t v) {
    return __int_as_float(((uint32_t)v) << 16);
}
__device__ uint16_t h3_f32_to_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Row-major GEMM: C(rows x out) = A(rows x in) @ W^T, W stored (out x in) row-major.
 * cuBLAS is column-major; compute C^T = W @ A^T via GemmEx(OP_T, OP_T, out, rows, in).
 * ab = A/B data type, c = C data type, comp = compute type. */
static cublasStatus_t h3_cu_gemm(h3_gpu *g, cudaDataType ab, cudaDataType c,
                                 cublasComputeType_t comp,
                                 const void *w, const void *a, void *out,
                                 uint32_t rows, uint32_t in, uint32_t out_dim) {
    cublasHandle_t h = (cublasHandle_t)g->dev_ctx;
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_T, (int)out_dim, (int)rows, (int)in,
                        &alpha, w, ab, (int)in, a, ab, (int)rows, &beta, out, c, (int)out_dim,
                        comp, CUBLAS_GEMM_DEFAULT);
}

__global__ void h3_cu_linear_bias_f32(float *out, const float *bias,
                                      uint32_t rows, uint32_t out_dim, int has_bias) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * out_dim) {
        uint32_t col = i % out_dim;
        out[i] += has_bias ? bias[col] : 0.0f;
    }
}
/* BF16 bias: gemm already rounded to bf16; add bias in f32 and round once more
 * (Metal accumulates bias in f32 before the single final bf16 rounding). */
__global__ void h3_cu_linear_bias_bf16(uint16_t *out, const uint16_t *bias,
                                       uint32_t rows, uint32_t out_dim, int has_bias) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * out_dim) {
        uint32_t col = i % out_dim;
        float v = h3_bf16_to_f32(out[i]) + (has_bias ? h3_bf16_to_f32(bias[col]) : 0.0f);
        out[i] = h3_f32_to_bf16(v);
    }
}

/* f32 GEMM result + f32 bias -> bf16 output (patch_linear, f32 bias). */
__global__ void h3_cu_bias_f32_to_bf16(const float *gemm, const float *bias,
                                       uint16_t *out, uint32_t rows, uint32_t out_dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * out_dim) {
        uint32_t col = i % out_dim;
        out[i] = h3_f32_to_bf16(gemm[i] + (bias ? bias[col] : 0.0f));
    }
}

/* Gather rows: dst[r*in+d] = src[row_map[r]*in+d]. */
__global__ void h3_cu_copy_rows_f32(const float *src, const unsigned *row_map,
                                    float *dst, uint32_t rows, uint32_t width) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || col >= width) return;
    dst[(size_t)row * width + col] = src[(size_t)row_map[row] * width + col];
}

/* --- I5: attention --- */

template <typename T> __device__ float h3_cu_load(const T *p) { return (float)*p; }
template <> __device__ float h3_cu_load<uint16_t>(const uint16_t *p) { return h3_bf16_to_f32(*p); }
template <typename T> __device__ void h3_cu_store(T *p, float v) { *p = (T)v; }
template <> __device__ void h3_cu_store<uint16_t>(uint16_t *p, float v) { *p = h3_f32_to_bf16(v); }

/* Flash-style softmax attention. Q/K/V row-major [batch, row, head, dim].
 * Online softmax (rescaling) to keep one pass over the sequence. head_major_out
 * writes [batch, head, row, dim] (native SDPA layout) instead of row-major. */
template <typename T>
__global__ void h3_cu_sdpa(T *out, const T *q, const T *k, const T *v,
                           uint32_t batch, uint32_t seq, uint32_t heads,
                           uint32_t head_dim, float scale, int causal,
                           int head_major_out) {
    uint32_t head = blockIdx.x, row = blockIdx.y, b = blockIdx.z;
    uint32_t d = threadIdx.x;
    if (head >= heads || row >= seq || d >= head_dim || b >= batch) return;
    uint32_t bbase = b * seq * heads * head_dim;
    float qv = h3_cu_load(&q[bbase + (row * heads + head) * head_dim + d]);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    __shared__ float red[H3_CU_BLOCK];
    for (uint32_t s = 0; s < seq; s++) {
        if (causal && s > row) break;
        red[d] = qv * h3_cu_load(&k[bbase + (s * heads + head) * head_dim + d]);
        __syncthreads();
        for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
            if (d < (uint32_t)st) red[d] += red[d + st];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();
        float m_new = fmaxf(m, score);
        float es = expf(m - m_new);
        float e = expf(score - m_new);
        l = l * es + e;
        acc *= es;
        acc += e * h3_cu_load(&v[bbase + (s * heads + head) * head_dim + d]);
        m = m_new;
    }
    float o = (l > 0.0f) ? acc / l : 0.0f;
    uint32_t idx = bbase + (head_major_out ? (head * seq + row) : (row * heads + head))
        * head_dim + d;
    h3_cu_store(&out[idx], o);
}

/* DiT grouped QKV+RoPE (BF16): qkv row = [head0:Q|K|V][head1:Q|K|V]... (grouped).
 * Per (row,head): RMS-norm Q and K (rsqrt(sum/head_dim+eps)), RoPE +/-half pairs,
 * V copied raw. Output row-major [row, head, dim]. Mirrors h3_qkv_rope_bf16. */
__global__ void h3_cu_grouped_qkv_rope_bf16(
    const uint16_t *qkv, const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query, uint16_t *key, uint16_t *value,
    uint32_t seq, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    if (head >= heads || row >= seq || d >= head_dim) return;
    uint32_t inner = heads * head_dim;
    uint32_t row_base = row * inner * 3;
    uint32_t q_base = row_base + head * head_dim * 3;
    uint32_t k_base = q_base + head_dim;
    uint32_t v_base = k_base + head_dim;
    float q = h3_bf16_to_f32(qkv[q_base + d]);
    float k = h3_bf16_to_f32(qkv[k_base + d]);
    __shared__ float qs[H3_CU_BLOCK], ks[H3_CU_BLOCK];
    qs[d] = q * q; ks[d] = k * k;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (d < (uint32_t)st) { qs[d] += qs[d + st]; ks[d] += ks[d + st]; }
        __syncthreads();
    }
    float qi = rsqrtf(qs[0] / (float)head_dim + epsilon);
    float ki = rsqrtf(ks[0] / (float)head_dim + epsilon);
    float q0 = q * qi * h3_bf16_to_f32(q_weight[d]);
    float k0 = k * ki * h3_bf16_to_f32(k_weight[d]);
    if (d < rope_half) {
        uint32_t pair = d + rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * qi * h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * ki * h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * rope_half + d]);
        float s = h3_bf16_to_f32(rope_sin[row * rope_half + d]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (d < rope_half * 2) {
        uint32_t pair = d - rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * qi * h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * ki * h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t out = (row * heads + head) * head_dim + d;
    query[out] = h3_f32_to_bf16(q0);
    key[out] = h3_f32_to_bf16(k0);
    value[out] = qkv[v_base + d];
}

/* Video VAE QKV+RoPE (f32): qkv row = [head0:Q|K|V][head1:Q|K|V]..., raw rsqrt norm
 * (no learned weight). Mirrors h3_video_qkv_rope_f32. */
__global__ void h3_cu_video_qkv_rope_f32(
    const float *qkv, const float *rope_cos, const float *rope_sin,
    float *query, float *key, float *value,
    uint32_t seq, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    if (head >= heads || row >= seq || d >= head_dim) return;
    uint32_t base = (row * heads + head) * head_dim * 3;
    float q = qkv[base + d];
    float k = qkv[base + head_dim + d];
    __shared__ float qs[H3_CU_BLOCK], ks[H3_CU_BLOCK];
    qs[d] = q * q; ks[d] = k * k;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (d < (uint32_t)st) { qs[d] += qs[d + st]; ks[d] += ks[d + st]; }
        __syncthreads();
    }
    float qi = rsqrtf(qs[0] / (float)head_dim + epsilon);
    float ki = rsqrtf(ks[0] / (float)head_dim + epsilon);
    float q0 = q * qi, k0 = k * ki;
    if (d < rope_half) {
        uint32_t pair = d + rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + head_dim + pair] * ki;
        float c = rope_cos[row * rope_half + d];
        float s = rope_sin[row * rope_half + d];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (d < rope_half * 2) {
        uint32_t pair = d - rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + head_dim + pair] * ki;
        float c = rope_cos[row * rope_half + pair];
        float s = rope_sin[row * rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t out = (row * heads + head) * head_dim + d;
    query[out] = q0;
    key[out] = k0;
    value[out] = qkv[base + 2 * head_dim + d];
}

/* --- I6: DiT aux + text encoder --- */

#define H3_DIT_MAX 5376u

/* RMS inverse per row: inverse[row] = rsqrt(sum(x^2)/width + eps). F32 out. */
__global__ void h3_cu_rms_inverse_bf16(const uint16_t *input, float *inverse,
                                       uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const uint16_t *x = input + (size_t)row * width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < width; k += blockDim.x) {
        float v = h3_bf16_to_f32(x[k]);
        local_sum = fmaf(v, v, local_sum);
    }
    __shared__ float red[H3_CU_BLOCK];
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    inverse[row] = rsqrtf(red[0] / (float)width + epsilon);
}

/* AdaLN: RMS norm + modulation scale/shift. Block per row. */
__global__ void h3_cu_adaln_bf16(const uint16_t *input, const uint16_t *norm_weight,
    const uint16_t *modulation, const unsigned *row_map, uint16_t *output,
    uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const uint16_t *x = input + (size_t)row * width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < width; k += blockDim.x) {
        float v = h3_bf16_to_f32(x[k]);
        local_sum = fmaf(v, v, local_sum);
    }
    __shared__ float red[H3_CU_BLOCK];
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inverse = rsqrtf(red[0] / (float)width + epsilon);
    uint32_t base = row_map[row] * slots * width;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float normed = h3_bf16_to_f32(x[col]) * inverse * h3_bf16_to_f32(norm_weight[col]);
        float shift = h3_bf16_to_f32(modulation[base + shift_slot * width + col]);
        float scale = h3_bf16_to_f32(modulation[base + scale_slot * width + col]);
        output[(size_t)row * width + col] = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
    }
}

/* Gate (residual + branch*gate) rounded to bf16, then RMS norm + adaln.
 * Mirrors h3_gate_adaln_bf16. Width capped at H3_DIT_MAX (5376). */
__global__ void h3_cu_gate_adaln_bf16(const uint16_t *residual, const uint16_t *branch,
    const uint16_t *norm_weight, const uint16_t *gate_modulation,
    const uint16_t *norm_modulation, const unsigned *row_map,
    uint16_t *gated_residual, uint16_t *output,
    uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float red[H3_CU_BLOCK];
    __shared__ uint16_t gated[H3_DIT_MAX];
    uint32_t base = row_map[row] * slots * width;
    size_t row_off = (size_t)row * width;
    float local_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float gate = h3_bf16_to_f32(gate_modulation[base + gate_slot * width + col]);
        uint16_t g = h3_f32_to_bf16(h3_bf16_to_f32(residual[row_off + col]) +
                                    h3_bf16_to_f32(branch[row_off + col]) * gate);
        gated_residual[row_off + col] = g;
        gated[col] = g;
        float v = h3_bf16_to_f32(g);
        local_sum = fmaf(v, v, local_sum);
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inverse = rsqrtf(red[0] / (float)width + epsilon);
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float normed = h3_bf16_to_f32(gated[col]) * inverse * h3_bf16_to_f32(norm_weight[col]);
        float shift = h3_bf16_to_f32(norm_modulation[base + shift_slot * width + col]);
        float scale = h3_bf16_to_f32(norm_modulation[base + scale_slot * width + col]);
        output[row_off + col] = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
    }
}

/* Token pooling: pair.x==pair.y copies first only; else average the pair.
 * Writes original snapshot, pooled output, and baseline for mapped rows. */
__global__ void h3_cu_token_pool_bf16(const uint16_t *input, const uint2 *pairs,
    uint16_t *output, uint16_t *baseline, const unsigned *baseline_indices,
    uint16_t *original, size_t input_offset, size_t original_offset,
    size_t baseline_offset, uint32_t rows, uint32_t width) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || col >= width) return;
    uint2 pair = pairs[row];
    uint16_t first = input[input_offset + (size_t)pair.x * width + col];
    original[original_offset + (size_t)pair.x * width + col] = first;
    uint16_t pooled = first;
    if (pair.x != pair.y) {
        uint16_t second = input[input_offset + (size_t)pair.y * width + col];
        original[original_offset + (size_t)pair.y * width + col] = second;
        pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) + h3_bf16_to_f32(second)) * 0.5f);
    }
    output[(size_t)row * width + col] = pooled;
    uint32_t b_index = baseline_indices[row];
    if (b_index != 0xffffffffu)
        baseline[baseline_offset + (size_t)b_index * width + col] = pooled;
}

/* Token pooling + AdaLN (block per row). Mirrors h3_token_pool_adaln_bf16. */
__global__ void h3_cu_token_pool_adaln_bf16(const uint16_t *input, const uint2 *pairs,
    uint16_t *residual, uint16_t *baseline, const unsigned *baseline_indices,
    uint16_t *original, const uint16_t *norm_weight, const uint16_t *modulation,
    const unsigned *row_map, uint16_t *output,
    size_t input_offset, size_t original_offset, size_t baseline_offset,
    uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float red[H3_CU_BLOCK];
    __shared__ uint16_t pooled[H3_DIT_MAX];
    uint2 pair = pairs[row];
    uint32_t b_index = baseline_indices[row];
    float local_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        uint16_t first = input[input_offset + (size_t)pair.x * width + col];
        original[original_offset + (size_t)pair.x * width + col] = first;
        uint16_t p = first;
        if (pair.x != pair.y) {
            uint16_t second = input[input_offset + (size_t)pair.y * width + col];
            original[original_offset + (size_t)pair.y * width + col] = second;
            p = h3_f32_to_bf16((h3_bf16_to_f32(first) + h3_bf16_to_f32(second)) * 0.5f);
        }
        size_t dest = (size_t)row * width + col;
        residual[dest] = p;
        pooled[col] = p;
        if (b_index != 0xffffffffu)
            baseline[baseline_offset + (size_t)b_index * width + col] = p;
        float v = h3_bf16_to_f32(p);
        local_sum = fmaf(v, v, local_sum);
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inverse = rsqrtf(red[0] / (float)width + epsilon);
    uint32_t base = row_map[row] * slots * width;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float normed = h3_bf16_to_f32(pooled[col]) * inverse * h3_bf16_to_f32(norm_weight[col]);
        float shift = h3_bf16_to_f32(modulation[base + shift_slot * width + col]);
        float scale = h3_bf16_to_f32(modulation[base + scale_slot * width + col]);
        output[(size_t)row * width + col] = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
    }
}

/* Token expand (delta): exact-prefix rows copy reduced; else original + update_scale*(reduced-baseline). */
__global__ void h3_cu_token_expand_delta_bf16(const uint16_t *original, const uint16_t *reduced,
    const uint16_t *baseline, const unsigned *baseline_indices, const unsigned *parents,
    uint16_t *output, size_t original_offset, size_t baseline_offset,
    uint32_t rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || col >= width) return;
    uint32_t parent = parents[row];
    size_t dest = (size_t)row * width + col;
    size_t reduced_index = (size_t)parent * width + col;
    if (row < exact_prefix_rows) {
        output[dest] = reduced[reduced_index];
        return;
    }
    uint32_t baseline_row = baseline_indices[parent];
    if (baseline_row == 0xffffffffu) { output[dest] = reduced[reduced_index]; return; }
    size_t b_index = baseline_offset + (size_t)baseline_row * width + col;
    float update = h3_bf16_to_f32(reduced[reduced_index]) - h3_bf16_to_f32(baseline[b_index]);
    output[dest] = h3_f32_to_bf16(h3_bf16_to_f32(original[original_offset + dest]) + update_scale * update);
}

/* Token expand + AdaLN (block per row). Mirrors h3_token_expand_adaln_bf16. */
__global__ void h3_cu_token_expand_adaln_bf16(const uint16_t *original, const uint16_t *reduced,
    const uint16_t *baseline, const unsigned *baseline_indices, const unsigned *parents,
    const uint16_t *norm_weight, const uint16_t *modulation, const unsigned *row_map,
    uint16_t *residual, uint16_t *output,
    size_t original_offset, size_t baseline_offset,
    uint32_t rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float red[H3_CU_BLOCK];
    __shared__ uint16_t restored[H3_DIT_MAX];
    uint32_t parent = parents[row];
    uint32_t baseline_row = baseline_indices[parent];
    bool direct = row < exact_prefix_rows || baseline_row == 0xffffffffu;
    float local_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        size_t dest = (size_t)row * width + col;
        size_t reduced_index = (size_t)parent * width + col;
        uint16_t r = reduced[reduced_index];
        if (!direct) {
            size_t b_index = baseline_offset + (size_t)baseline_row * width + col;
            float update = h3_bf16_to_f32(r) - h3_bf16_to_f32(baseline[b_index]);
            r = h3_f32_to_bf16(h3_bf16_to_f32(original[original_offset + dest]) + update_scale * update);
        }
        restored[col] = r;
        residual[dest] = r;
        float v = h3_bf16_to_f32(r);
        local_sum = fmaf(v, v, local_sum);
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inverse = rsqrtf(red[0] / (float)width + epsilon);
    uint32_t base = row_map[row] * slots * width;
    for (uint32_t col = tid; col < width; col += blockDim.x) {
        float normed = h3_bf16_to_f32(restored[col]) * inverse * h3_bf16_to_f32(norm_weight[col]);
        float shift = h3_bf16_to_f32(modulation[base + shift_slot * width + col]);
        float scale = h3_bf16_to_f32(modulation[base + scale_slot * width + col]);
        output[(size_t)row * width + col] = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
    }
}

/* Euler sampler step: sample += delta*(ratio*(last-prev)+last). */
__global__ void h3_cu_euler_bf16(float *sample, const uint16_t *last, const uint16_t *previous,
    size_t sample_offset, uint32_t elements, float delta, float ratio) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= elements) return;
    float lv = h3_bf16_to_f32(last[gid]);
    float velocity = fmaf(ratio, lv - h3_bf16_to_f32(previous[gid]), lv);
    sample[sample_offset + gid] = fmaf(delta, velocity, sample[sample_offset + gid]);
}

/* Text RoPE, in place on query/key. Row-major [row, head, dim], pairs half_dim. */
__global__ void h3_cu_rope_text_bf16(uint16_t *query, uint16_t *key,
    const float *rope_cos, const float *rope_sin,
    uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim) {
    uint32_t head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    if (row >= sequence || d >= head_dim / 2) return;
    uint32_t half_dim = head_dim / 2;
    if (head < query_heads) {
        size_t base = ((size_t)row * query_heads + head) * head_dim;
        float first = h3_bf16_to_f32(query[base + d]);
        float second = h3_bf16_to_f32(query[base + half_dim + d]);
        float c = rope_cos[(size_t)row * half_dim + d];
        float s = rope_sin[(size_t)row * half_dim + d];
        query[base + d] = h3_f32_to_bf16(first * c - second * s);
        query[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
    }
    if (head < kv_heads) {
        size_t base = ((size_t)row * kv_heads + head) * head_dim;
        float first = h3_bf16_to_f32(key[base + d]);
        float second = h3_bf16_to_f32(key[base + half_dim + d]);
        float c = rope_cos[(size_t)row * half_dim + d];
        float s = rope_sin[(size_t)row * half_dim + d];
        key[base + d] = h3_f32_to_bf16(first * c - second * s);
        key[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
    }
}

/* Grouped multi-head causal attention (text encoder). Mirrors h3_gqa_causal_bf16:
 * Q scaled (bf16-rounded) then softmax(QK^T/scale) causal, weighted V. */
extern __shared__ float h3_cu_dyn[];
__global__ void h3_cu_gqa_causal_bf16(const uint16_t *q, const uint16_t *k, const uint16_t *v,
    uint16_t *out, uint32_t sequence, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, float scale) {
    uint32_t q_row = blockIdx.x, q_head = blockIdx.y, tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (q_row >= sequence || q_head >= q_heads) return;
    uint32_t kv_head = q_head / (q_heads / kv_heads);
    size_t q_base = ((size_t)q_row * q_heads + q_head) * head_dim;
    uint32_t key_count = q_row + 1;
    float *shared_query = h3_cu_dyn;
    float *scores = h3_cu_dyn + head_dim;
    float *red = h3_cu_dyn + head_dim + sequence;
    for (uint32_t d = tid; d < head_dim; d += threads)
        shared_query[d] = h3_bf16_to_f32(h3_f32_to_bf16(h3_bf16_to_f32(q[q_base + d]) * scale));
    __syncthreads();
    float local_max = -INFINITY;
    for (uint32_t kr = tid; kr < key_count; kr += threads) {
        size_t k_base = ((size_t)kr * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            dot = fmaf(shared_query[d], h3_bf16_to_f32(k[k_base + d]), dot);
        scores[kr] = dot;
        local_max = fmaxf(local_max, dot);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int st = (int)threads / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float maximum = red[0];
    float local_sum = 0.0f;
    for (uint32_t kr = tid; kr < key_count; kr += threads) {
        float p = expf(scores[kr] - maximum);
        scores[kr] = p;
        local_sum += p;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int st = (int)threads / 2; st > 0; st >>= 1) {
        if (tid < (uint32_t)st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];
    for (uint32_t d = tid; d < head_dim; d += threads) {
        float sum = 0.0f;
        for (uint32_t kr = 0; kr < key_count; kr++) {
            size_t v_index = ((size_t)kr * kv_heads + kv_head) * head_dim + d;
            sum = fmaf(scores[kr] * inv_sum, h3_bf16_to_f32(v[v_index]), sum);
        }
        out[q_base + d] = h3_f32_to_bf16(sum);
    }
}


static size_t h3_gpu_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
    case H3_GPU_F32: return sizeof(float);
    case H3_GPU_BF16: return sizeof(uint16_t);
    case H3_GPU_I8: return sizeof(int8_t);
    case H3_GPU_U32: return sizeof(uint32_t);
    default: return 0;
    }
}

__global__ void h3_cu_silu_f32(const float *in, float *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / (1.0f + expf(-in[i]));
}
__global__ void h3_cu_silu_bf16(const uint16_t *in, uint16_t *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = h3_bf16_to_f32(in[i]); out[i] = h3_f32_to_bf16(v / (1.0f + expf(-v))); }
}
__global__ void h3_cu_cast_f32_to_bf16(const float *in, uint16_t *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = h3_f32_to_bf16(in[i]);
}
__global__ void h3_cu_cast_bf16_to_f32(const uint16_t *in, float *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = h3_bf16_to_f32(in[i]);
}
__global__ void h3_cu_clip_f32(const float *in, float *out, unsigned n, float mn, float mx) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = in[i]; out[i] = fminf(fmaxf(v, mn), mx); }
}
__global__ void h3_cu_add_bf16(const uint16_t *a, const uint16_t *b, uint16_t *o, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = h3_f32_to_bf16(h3_bf16_to_f32(a[i]) + h3_bf16_to_f32(b[i]));
}
__global__ void h3_cu_sub_bf16(const uint16_t *a, const uint16_t *b, uint16_t *o, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = h3_f32_to_bf16(h3_bf16_to_f32(a[i]) - h3_bf16_to_f32(b[i]));
}
__global__ void h3_cu_add_scaled_f32(const float *l, const float *r, float *o, unsigned n, float ls, float rs) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = l[i] * ls + r[i] * rs;
}
__global__ void h3_cu_geglu_f32(const float *gate, const float *lin, float *o, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = gate[i]; float c = x*x*x;
        o[i] = 0.5f*x*(1.0f + tanhf(0.7978845608028654f*(x + 0.044715f*c))) * lin[i]; }
}
/* Conv1d (stride 1). Layouts: input NHWC [b,length,ic], weight OIHW
 * [oc,ic,kernel], output NHWC [b,output_length,oc]. */
__global__ void h3_cu_conv1d_f32(const float *input, const float *weight,
        const float *bias, float *output, unsigned batch, unsigned length,
        unsigned input_channels, unsigned output_channels, unsigned kernel,
        unsigned padding, unsigned dilation) {
    unsigned effective = dilation*(kernel-1)+1;
    unsigned output_length = (length + 2*padding - effective) + 1;
    size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t total = (size_t)batch*output_length*output_channels;
    if (idx >= total) return;
    unsigned oc = (unsigned)(idx % output_channels);
    unsigned t = (unsigned)((idx / output_channels) % output_length);
    unsigned b = (unsigned)(idx / (output_channels*output_length));
    float acc = bias ? bias[oc] : 0.0f;
    const float *w = weight + (size_t)oc*input_channels*kernel;
    const float *x = input + (size_t)b*length*input_channels;
    for (unsigned ic = 0; ic < input_channels; ic++) {
        const float *wi = w + (size_t)ic*kernel;
        const float *xi = x + ic;
        for (unsigned k = 0; k < kernel; k++) {
            int in_t = (int)(t + k*dilation) - (int)padding;
            if (in_t < 0 || in_t >= (int)length) continue;
            acc = fmaf(xi[(size_t)in_t*input_channels], wi[k], acc);
        }
    }
    output[idx] = acc;
}
/* ConvTranspose1d. Layouts: input NHWC [b,length,ic], weight transposed-OIHW
 * [ic,oc,kernel], output NHWC [b,output_length,oc]. */
__global__ void h3_cu_conv_transpose1d_f32(const float *input,
        const float *weight, const float *bias, float *output, unsigned batch,
        unsigned length, unsigned input_channels, unsigned output_channels,
        unsigned kernel, unsigned stride, unsigned padding) {
    unsigned output_length = (length-1)*stride + kernel - 2*padding;
    size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t total = (size_t)batch*output_length*output_channels;
    if (idx >= total) return;
    unsigned oc = (unsigned)(idx % output_channels);
    unsigned t = (unsigned)((idx / output_channels) % output_length);
    unsigned b = (unsigned)(idx / (output_channels*output_length));
    float acc = bias ? bias[oc] : 0.0f;
    const float *x = input + (size_t)b*length*input_channels;
    for (unsigned ic = 0; ic < input_channels; ic++) {
        const float *wi = weight + (size_t)ic*output_channels*kernel + oc*kernel;
        const float *xi = x + ic;
        for (unsigned k = 0; k < kernel; k++) {
            int num = (int)t + (int)padding - (int)k;
            if (num < 0 || num % (int)stride != 0) continue;
            int in_t = num / (int)stride;
            if (in_t < 0 || in_t >= (int)length) continue;
            acc = fmaf(xi[(size_t)in_t*input_channels], wi[k], acc);
        }
    }
    output[idx] = acc;
}
/* Alias-free SnakeBeta activation: 3D grid (channels, length, batch). */
__global__ void h3_cu_alias_free_snake_f32(const float *input,
        const float *alpha_log, const float *beta_log,
        const float *upsample_filter, const float *downsample_filter,
        float *output, unsigned batch, unsigned length, unsigned channels) {
    unsigned channel = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned time = blockIdx.y;
    unsigned b = blockIdx.z;
    if (channel >= channels || time >= length || b >= batch) return;
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = (int)(time*2) + down_k - 5;
        up_time = max(up_time, 0);
        up_time = min(up_time, (int)(length*2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = padded_time - 5;
            source_time = max(source_time, 0);
            source_time = min(source_time, (int)length - 1);
            const float *src = input + ((size_t)b*length + (unsigned)source_time)*channels + channel;
            upsampled = fmaf(src[0], 2.0f*upsample_filter[up_k], upsampled);
        }
        float sine = sinf(alpha*upsampled);
        float activated = upsampled + sine*sine/(beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_k], result);
    }
    output[(size_t)((size_t)b*length + time)*channels + channel] = result;
}
__global__ void h3_cu_gelu_bf16(const uint16_t *in, uint16_t *o, unsigned n, int approx) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = h3_bf16_to_f32(in[i]); float act;
        if (approx) {
            float inner = 0.7978845608028654f*(v + 0.044715f*v*v*v);
            act = inner <= -10.0f ? 0.0f : inner >= 10.0f ? v : 0.5f*v*(1.0f + tanhf(inner));
        } else {
            act = v <= -10.0f ? 0.0f : v >= 10.0f ? v : 0.5f*v*(1.0f + erf(v*0.7071067811865475f));
        }
        o[i] = h3_f32_to_bf16(act); }
}
__global__ void h3_cu_swiglu_f32(const float *fused, float *o, unsigned rows, unsigned width) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x; unsigned row = blockIdx.y;
    if (row < rows && col < width) { unsigned base = row*width*2;
        float g = fused[base+col]; float u = fused[base+width+col];
        o[row*width+col] = (g / (1.0f + expf(-g))) * u; }
}
__global__ void h3_cu_swiglu_bf16(const uint16_t *fused, uint16_t *o, unsigned rows, unsigned width) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x; unsigned row = blockIdx.y;
    if (row < rows && col < width) { unsigned base = row*width*2;
        float g = h3_bf16_to_f32(fused[base+col]); float u = h3_bf16_to_f32(fused[base+width+col]);
        o[row*width+col] = h3_f32_to_bf16((g / (1.0f + expf(-g))) * u); }
}
__global__ void h3_cu_silu_mul_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *o, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = h3_bf16_to_f32(gate[i]); float u = h3_bf16_to_f32(up[i]);
        o[i] = h3_f32_to_bf16((g / (1.0f + expf(-g))) * u); }
}
__global__ void h3_cu_scale_add_f32(const float *res, const float *branch, const float *scale, float *o, unsigned rows, unsigned width) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x; unsigned row = blockIdx.y;
    if (row < rows && col < width) { unsigned idx = row*width+col; o[idx] = res[idx] + branch[idx]*scale[col]; }
}
__global__ void h3_cu_embedding_bf16(const uint16_t *w, const unsigned *ids, uint16_t *o, unsigned tokens, unsigned width, unsigned vocab) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x; unsigned token = blockIdx.y;
    if (token < tokens && col < width) { unsigned id = ids[token];
        o[token*width+col] = id < vocab ? w[id*width+col] : (uint16_t)0; }
}
__global__ void h3_cu_gate_bf16(const uint16_t *res, const uint16_t *branch, const uint16_t *mod, const unsigned *row_map, uint16_t *o, unsigned rows, unsigned width, unsigned slots, unsigned gate_slot) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x; unsigned row = blockIdx.y;
    if (row < rows && col < width) { unsigned base = row_map[row]*slots*width;
        float g = h3_bf16_to_f32(mod[base + gate_slot*width + col]); unsigned idx = row*width+col;
        float v = h3_bf16_to_f32(res[idx]) + h3_bf16_to_f32(branch[idx]) * g;
        o[idx] = h3_f32_to_bf16(v); }
}
__global__ void h3_cu_rms_norm_f32(const float *in, const float *w, float *o, unsigned rows, unsigned width, float eps) {
    unsigned row = blockIdx.x; unsigned tid = threadIdx.x; unsigned threads = blockDim.x;
    if (row >= rows) return;
    extern __shared__ float red[];
    const float *x = in + row*width;
    float local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) { float v = x[k]; local = fmaf(v, v, local); }
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float inv = rsqrtf(red[0]/width + eps);
    for (unsigned col = tid; col < width; col += threads) o[row*width+col] = x[col]*inv*w[col];
}
__global__ void h3_cu_rms_norm_bf16(const uint16_t *in, const uint16_t *w, uint16_t *o, unsigned rows, unsigned width, float eps) {
    unsigned row = blockIdx.x; unsigned tid = threadIdx.x; unsigned threads = blockDim.x;
    if (row >= rows) return;
    extern __shared__ float red[];
    const uint16_t *x = in + row*width;
    float local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) { float v = h3_bf16_to_f32(x[k]); local = fmaf(v, v, local); }
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float inv = rsqrtf(red[0]/width + eps);
    for (unsigned col = tid; col < width; col += threads) {
        float norm = h3_bf16_to_f32(x[col])*inv;
        o[row*width+col] = h3_f32_to_bf16(norm * h3_bf16_to_f32(w[col])); }
}
__global__ void h3_cu_layer_norm_f32(const float *in, const float *w, const float *b, float *o, unsigned rows, unsigned width, float eps) {
    unsigned row = blockIdx.x; unsigned tid = threadIdx.x; unsigned threads = blockDim.x;
    if (row >= rows) return;
    extern __shared__ float red[];
    const float *x = in + row*width;
    float local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) local += x[k];
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float mean = red[0]/width; __syncthreads();
    local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) { float c = x[k]-mean; local = fmaf(c, c, local); }
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float inv = rsqrtf(red[0]/width + eps);
    for (unsigned col = tid; col < width; col += threads)
        o[row*width+col] = (x[col]-mean)*inv*w[col] + b[col];
}
__global__ void h3_cu_layer_norm_bf16(const uint16_t *in, const uint16_t *w, const uint16_t *b, uint16_t *o, unsigned rows, unsigned width, float eps) {
    unsigned row = blockIdx.x; unsigned tid = threadIdx.x; unsigned threads = blockDim.x;
    if (row >= rows) return;
    extern __shared__ float red[];
    const uint16_t *x = in + row*width;
    float local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) local += h3_bf16_to_f32(x[k]);
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float mean = red[0]/width; __syncthreads();
    local = 0.0f;
    for (unsigned k = tid; k < width; k += threads) { float c = h3_bf16_to_f32(x[k])-mean; local = fmaf(c, c, local); }
    red[tid] = local; __syncthreads();
    for (unsigned stride = threads/2; stride; stride >>= 1) { if (tid < stride) red[tid] += red[tid+stride]; __syncthreads(); }
    float inv = rsqrtf(red[0]/width + eps);
    for (unsigned col = tid; col < width; col += threads) {
        float norm = (h3_bf16_to_f32(x[col])-mean)*inv;
        o[row*width+col] = h3_f32_to_bf16(fmaf(norm, h3_bf16_to_f32(w[col]), h3_bf16_to_f32(b[col]))); }
}
__global__ void h3_cu_head_rms_norm_bf16(uint16_t *t, const uint16_t *w, unsigned seq, unsigned heads, unsigned head_dim, float eps) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row = idx / heads; unsigned head = idx % heads;
    if (row >= seq || head >= heads) return;
    unsigned base = (row*heads + head)*head_dim;
    float sum = 0.0f;
    for (unsigned d = 0; d < head_dim; d++) { float v = h3_bf16_to_f32(t[base+d]); sum = fmaf(v, v, sum); }
    float inv = rsqrtf(sum/head_dim + eps);
    for (unsigned d = 0; d < head_dim; d++) { float v = h3_bf16_to_f32(t[base+d]);
        t[base+d] = h3_f32_to_bf16(v*inv*h3_bf16_to_f32(w[d])); }
}
__global__ void h3_cu_weight_norm_f32(const float *v, const float *mag, float *o, unsigned outer, unsigned inner) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned base = row*inner;
    float ss = 0.0f;
    for (unsigned i = 0; i < inner; i++) { float val = v[base+i]; ss = fmaf(val, val, ss); }
    float scale = mag[row]*rsqrtf(ss);
    for (unsigned i = 0; i < inner; i++) o[base+i] = v[base+i]*scale;
}

/* Allocate device memory (optionally filled from host `values`). */
static h3_gpu_tensor *h3_gpu_tensor_alloc(h3_gpu *gpu, size_t elements, h3_gpu_dtype dtype, const void *values) {
    size_t bytes = elements * h3_gpu_dtype_size(dtype);
    if (bytes == 0) return NULL;
    void *dptr = NULL;
    if (cudaMalloc(&dptr, bytes) != cudaSuccess) return NULL;
    if (values) {
        if (cudaMemcpy(dptr, values, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dptr); return NULL;
        }
    }
    h3_gpu_tensor *t = (h3_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) { cudaFree(dptr); return NULL; }
    t->device_ptr = dptr; t->dtype = dtype; t->elements = elements; t->bytes = bytes; t->owner = gpu;
    if (gpu) {
        gpu->stats.allocated_bytes += bytes;
        gpu->stats.live_bytes += bytes;
        if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
            gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
        gpu->stats.tensor_allocations++;
    }
    return t;
}

/* Stream a file range into a device tensor via a small host staging buffer. */
static int h3_gpu_tensor_file_load(h3_gpu *gpu, h3_gpu_tensor *t, const char *path, uint64_t file_offset, char *error, size_t error_size) {
    if (!t || !path || !*path || file_offset > (uint64_t)INT64_MAX) return 0;
    size_t bytes = t->bytes;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (error && error_size) snprintf(error, error_size, "cannot open %s: %s", path, strerror(errno)); return 0; }
    size_t chunk = MIN(bytes, (size_t)(1 << 20));
    void *host = malloc(chunk ? chunk : 1);
    if (!host) { close(fd); return 0; }
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(chunk, bytes - completed);
        size_t got = 0;
        while (got < request) {
            ssize_t count = pread(fd, (char *)host + got, request - got, (off_t)(file_offset + completed + got));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                if (error && error_size) snprintf(error, error_size, "cannot read %s payload: %s", path, count < 0 ? strerror(errno) : "unexpected end of file");
                free(host); close(fd); return 0;
            }
            got += (size_t)count;
        }
        if (cudaMemcpy((char *)t->device_ptr + completed, host, request, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size) snprintf(error, error_size, "cudaMemcpy failed");
            free(host); close(fd); return 0;
        }
        completed += request;
    }
    free(host); close(fd);
    (void)gpu;
    return 1;
}

static int h3_gpu_tensor_read_file_bf16_mode(h3_gpu_tensor *tensor, const char *path, uint64_t file_offset, size_t elements, int uncached, char *error, size_t error_size) {
    (void)uncached;
    if (error && error_size) error[0] = '\0';
    if (!tensor || !path || !*path || tensor->dtype != H3_GPU_BF16 || elements != tensor->elements || file_offset > (uint64_t)INT64_MAX) {
        if (error && error_size) snprintf(error, error_size, "invalid BF16 file read request");
        return 0;
    }
    size_t bytes = elements * sizeof(uint16_t);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (error && error_size) snprintf(error, error_size, "cannot open %s: %s", path, strerror(errno)); return 0; }
    size_t chunk = MIN(bytes, (size_t)(1 << 20));
    void *host = malloc(chunk ? chunk : 1);
    if (!host) { close(fd); return 0; }
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(chunk, bytes - completed);
        size_t got = 0;
        while (got < request) {
            ssize_t count = pread(fd, (char *)host + got, request - got, (off_t)(file_offset + completed + got));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                if (error && error_size) snprintf(error, error_size, "cannot read BF16 payload from %s: %s", path, count < 0 ? strerror(errno) : "unexpected end of file");
                free(host); close(fd); return 0;
            }
            got += (size_t)count;
        }
        if (cudaMemcpy((char *)tensor->device_ptr + completed, host, request, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size) snprintf(error, error_size, "cudaMemcpy failed");
            free(host); close(fd); return 0;
        }
        completed += request;
    }
    free(host); close(fd);
    return 1;
}

int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size) {
    int count = 0;
    cudaError_t ce = cudaGetDeviceCount(&count);
    if (ce != cudaSuccess || count < 1) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA device available: %s", cudaGetErrorString(ce));
        return 0;
    }
    if (info) {
        memset(info, 0, sizeof(*info));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        snprintf(info->name, sizeof(info->name), "%s", prop.name);
        snprintf(info->architecture, sizeof(info->architecture), "sm_%d", prop.major * 100 + prop.minor * 10);
        info->physical_memory = (uint64_t)prop.totalGlobalMem;
        info->unified_memory = 1;
    }
    return 1;
}
h3_gpu *h3_gpu_create(const char *shader_source_path, char *error, size_t error_size) {
    (void)shader_source_path;
    h3_gpu *g = (h3_gpu *)calloc(1, sizeof(*g));
    if (!g) { if (error && error_size) snprintf(error, error_size, "oom"); return NULL; }
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "cudaSetDevice: %s", cudaGetErrorString(ce));
        free(g); return NULL;
    }
    cublasHandle_t h = NULL;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "cublasCreate failed (%d)", (int)st);
        free(g); return NULL;
    }
    g->dev_ctx = (void *)h;
    return g;
}
void h3_gpu_free(h3_gpu *gpu) {
    if (gpu) {
        if (gpu->dev_ctx) cublasDestroy((cublasHandle_t)gpu->dev_ctx);
        free(gpu);
    }
}
const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu && gpu->error[0] ? gpu->error : "no error";
}
static void h3_cuda_seterr(const h3_gpu *gpu) {
    if (gpu) snprintf(((h3_gpu *)gpu)->error, sizeof(((h3_gpu *)gpu)->error), "%s", H3_CUDA_ERR);
}

int h3_gpu_is_m5(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_nax_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_int8_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }
h3_gpu_tensor * h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_I8, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, values);
}
h3_gpu_tensor * h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, values);
}
h3_gpu_tensor * h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_U32, values);
}
h3_gpu_tensor * h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, NULL);
    if (!t) return NULL;
    if (!h3_gpu_tensor_file_load(gpu, t, path, file_offset, NULL, 0)) { h3_gpu_tensor_free(t); return NULL; }
    return t;
}
h3_gpu_tensor * h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, NULL);
    if (!t) return NULL;
    if (!h3_gpu_tensor_file_load(gpu, t, path, file_offset, NULL, 0)) { h3_gpu_tensor_free(t); return NULL; }
    return t;
}
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(tensor, path, file_offset, elements, 0, error, error_size);
}
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(tensor, path, file_offset, elements, 1, error, error_size);
}
void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->device_ptr) cudaFree(tensor->device_ptr);
    if (tensor->owner) { tensor->owner->stats.live_bytes = tensor->owner->stats.live_bytes >= tensor->bytes ? tensor->owner->stats.live_bytes - tensor->bytes : 0; }
    free(tensor);
}
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_F32 || source_offset > tensor->elements || elements > tensor->elements - source_offset) return 0;
    size_t bytes = elements * sizeof(float);
    return (cudaMemcpy(values, (char *)tensor->device_ptr + source_offset * sizeof(float), bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_BF16 || elements > tensor->elements) return 0;
    size_t bytes = elements * sizeof(uint16_t);
    return (cudaMemcpy(values, tensor->device_ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_F32 || destination_offset > tensor->elements || elements > tensor->elements - destination_offset) return 0;
    size_t bytes = elements * sizeof(float);
    return (cudaMemcpy((char *)tensor->device_ptr + destination_offset * sizeof(float), values, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_BF16 || destination_offset > tensor->elements || elements > tensor->elements - destination_offset) return 0;
    size_t bytes = elements * sizeof(uint16_t);
    return (cudaMemcpy((char *)tensor->device_ptr + destination_offset * sizeof(uint16_t), values, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_begin(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_continue(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_submit(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->stats;
    return 1;
}
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    (void)gpu; (void)label;
    /* no-op on CUDA */
}
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    (void)gpu; (void)phase;
    /* no-op on CUDA */
}
int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    if (!gpu || !output || !input || !weight || !output->device_ptr ||
        !input->device_ptr || !weight->device_ptr) return 0;
    cublasStatus_t st = h3_cu_gemm(gpu, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                                   weight->device_ptr, input->device_ptr, output->device_ptr,
                                   rows, input_dim, output_dim);
    if (st != CUBLAS_STATUS_SUCCESS) {
        snprintf(((h3_gpu *)gpu)->error, sizeof(((h3_gpu *)gpu)->error),
                 "cublasGemmEx f32 failed (%d)", (int)st);
        return 0;
    }
    if (bias && bias->device_ptr)
        h3_cu_linear_bias_f32<<<h3_cu_grid(rows * output_dim), H3_CU_BLOCK>>>(
            (float *)output->device_ptr, (const float *)bias->device_ptr,
            rows, output_dim, 1);
    return 1;
}
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    if (!gpu || !output || !input || !weight || !output->device_ptr ||
        !input->device_ptr || !weight->device_ptr) return 0;
    /* F32 input/weight -> BF16 output projection. GEMM to temp f32, then bias+round. */
    h3_gpu_tensor *tmp = h3_gpu_tensor_new_f32(gpu, (size_t)rows * output_dim);
    if (!tmp) return 0;
    const void *in = (const float *)input->device_ptr + input_offset;
    cublasStatus_t st = h3_cu_gemm(gpu, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                                   weight->device_ptr, in, tmp->device_ptr,
                                   rows, input_dim, output_dim);
    if (st != CUBLAS_STATUS_SUCCESS) {
        snprintf(((h3_gpu *)gpu)->error, sizeof(((h3_gpu *)gpu)->error),
                 "cublasGemmEx patch f32 failed (%d)", (int)st);
        h3_gpu_tensor_free(tmp);
        return 0;
    }
    uint16_t *out = (uint16_t *)output->device_ptr + output_offset;
    h3_cu_bias_f32_to_bf16<<<h3_cu_grid(rows * output_dim), H3_CU_BLOCK>>>(
        (const float *)tmp->device_ptr, bias ? (const float *)bias->device_ptr : NULL,
        out, rows, output_dim);
    h3_gpu_tensor_free(tmp);
    return 1;
}
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    if (!gpu || !output || !input || !weight || !row_map || !output->device_ptr ||
        !input->device_ptr || !weight->device_ptr || !row_map->device_ptr) return 0;
    /* Gather input rows by row_map into a temp, then patch_linear. */
    h3_gpu_tensor *gathered = h3_gpu_tensor_new_f32(gpu, (size_t)rows * input_dim);
    if (!gathered) return 0;
    /* gather kernel: gathered[r*in+d] = input[row_map[r]*in+d] */
    dim3 g(h3_cu_grid(input_dim), rows);
    h3_cu_copy_rows_f32<<<g, H3_CU_BLOCK>>>((const float *)input->device_ptr,
        (const unsigned *)row_map->device_ptr, (float *)gathered->device_ptr,
        rows, input_dim);
    int ok = h3_gpu_patch_linear_bf16_offset(
        gpu, output, 0, gathered, 0, weight, bias, rows, input_dim, output_dim);
    h3_gpu_tensor_free(gathered);
    (void)output_rows;
    return ok;
}
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                           bias, rows, input_dim, output_dim);
}
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!output || !input || output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_silu_f32<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const float *)input->device_ptr, (float *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!output || !input || output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_F32 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_cast_f32_to_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const float *)input->device_ptr, (uint16_t *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!output || !input || output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_BF16 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_cast_bf16_to_f32<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)input->device_ptr, (float *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    if (!destination || !source || destination->dtype != H3_GPU_BF16 || source->dtype != H3_GPU_BF16 || source_offset > source->elements || elements > source->elements - source_offset || destination_offset > destination->elements || elements > destination->elements - destination_offset) return 0;
    size_t b = elements * sizeof(uint16_t);
    return (cudaMemcpy((char *)destination->device_ptr + destination_offset*sizeof(uint16_t), (char *)source->device_ptr + source_offset*sizeof(uint16_t), b, cudaMemcpyDeviceToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    if (!destination || !source || destination->dtype != H3_GPU_F32 || source->dtype != H3_GPU_F32 || source_offset > source->elements || elements > source->elements - source_offset || destination_offset > destination->elements || elements > destination->elements - destination_offset) return 0;
    size_t b = elements * sizeof(float);
    return (cudaMemcpy((char *)destination->device_ptr + destination_offset*sizeof(float), (char *)source->device_ptr + source_offset*sizeof(float), b, cudaMemcpyDeviceToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    if (!output || !input || !weight || output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 || (size_t)rows*width > input->elements || width > weight->elements || (size_t)rows*width > output->elements) return 0;
    h3_cu_rms_norm_f32<<<rows, H3_CU_BLOCK, H3_CU_BLOCK*sizeof(float)>>>((const float *)input->device_ptr, (const float *)weight->device_ptr, (float *)output->device_ptr, rows, width, epsilon);
    return 1;
}
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    if (!gpu || !output || !query || !key || !value || !output->device_ptr ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, 1);
    h3_cu_sdpa<float><<<g, head_dim>>>((float *)output->device_ptr,
        (const float *)query->device_ptr, (const float *)key->device_ptr,
        (const float *)value->device_ptr, 1, sequence, heads, head_dim, scale, 0, 0);
    return 1;
}
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    if (!output || !fused || output->dtype != H3_GPU_F32 || fused->dtype != H3_GPU_F32 || (size_t)rows*width*2 > fused->elements || (size_t)rows*width > output->elements) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_swiglu_f32<<<g, H3_CU_BLOCK>>>((const float *)fused->device_ptr, (float *)output->device_ptr, rows, width);
    return 1;
}
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    if (!output || !residual || !branch || !scale || output->dtype != H3_GPU_F32 || residual->dtype != H3_GPU_F32 || branch->dtype != H3_GPU_F32 || scale->dtype != H3_GPU_F32 || (size_t)rows*width > residual->elements || (size_t)rows*width > branch->elements || width > scale->elements || (size_t)rows*width > output->elements) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_scale_add_f32<<<g, H3_CU_BLOCK>>>((const float *)residual->device_ptr, (const float *)branch->device_ptr, (const float *)scale->device_ptr, (float *)output->device_ptr, rows, width);
    return 1;
}
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    if (!output || !input || !weight || !bias || output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 || bias->dtype != H3_GPU_F32 || (size_t)rows*width > input->elements || width > weight->elements || width > bias->elements || (size_t)rows*width > output->elements) return 0;
    h3_cu_layer_norm_f32<<<rows, H3_CU_BLOCK, H3_CU_BLOCK*sizeof(float)>>>((const float *)input->device_ptr, (const float *)weight->device_ptr, (const float *)bias->device_ptr, (float *)output->device_ptr, rows, width, epsilon);
    return 1;
}
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr ||
        !qkv->device_ptr || !rope_cos->device_ptr || !rope_sin->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, 1);
    h3_cu_video_qkv_rope_f32<<<g, head_dim>>>(
        (const float *)qkv->device_ptr, (const float *)rope_cos->device_ptr,
        (const float *)rope_sin->device_ptr, (float *)query->device_ptr,
        (float *)key->device_ptr, (float *)value->device_ptr,
        sequence, heads, head_dim, rope_half, epsilon);
    return 1;
}
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !dilation)
        return 0;
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if ((uint64_t)length + 2*padding < effective) return 0;
    uint32_t output_length = (uint32_t)((length + 2*padding - effective) + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!output->device_ptr || !input->device_ptr || !weight->device_ptr ||
        output_count > output->elements || input_count > input->elements ||
        weight_count > weight->elements || output_count > UINT32_MAX)
        return 0;
    if (bias && (!bias->device_ptr || output_channels > bias->elements))
        return 0;
    if ((size_t)output_channels * input_channels * kernel > UINT32_MAX)
        return 0;
    h3_cu_conv1d_f32<<<h3_cu_grid(output_count), H3_CU_BLOCK>>>(
        (const float *)input->device_ptr, (const float *)weight->device_ptr,
        bias ? (const float *)bias->device_ptr : NULL,
        (float *)output->device_ptr, batch, length, input_channels,
        output_channels, kernel, padding, dilation);
    return 1;
}
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    (void)gpu; (void)output; (void)input; (void)weight; (void)bias;
    (void)batch; (void)length; (void)input_channels; (void)output_channels;
    (void)kernel; (void)stride; (void)padding; (void)dilation;
    return 0;
}
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!output->device_ptr || !input->device_ptr || !weight->device_ptr ||
        output_count > output->elements || input_count > input->elements ||
        weight_count > weight->elements || output_count > UINT32_MAX)
        return 0;
    if (bias && (!bias->device_ptr || output_channels > bias->elements))
        return 0;
    h3_cu_conv_transpose1d_f32<<<h3_cu_grid(output_count), H3_CU_BLOCK>>>(
        (const float *)input->device_ptr, (const float *)weight->device_ptr,
        bias ? (const float *)bias->device_ptr : NULL,
        (float *)output->device_ptr, batch, length, input_channels,
        output_channels, kernel, stride, padding);
    return 1;
}
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    if (!output || !vector || !magnitude || output->dtype != H3_GPU_F32 || vector->dtype != H3_GPU_F32 || magnitude->dtype != H3_GPU_F32 || (size_t)outer*inner > vector->elements || outer > magnitude->elements || (size_t)outer*inner > output->elements) return 0;
    h3_cu_weight_norm_f32<<<h3_cu_grid(outer), H3_CU_BLOCK>>>((const float *)vector->device_ptr, (const float *)magnitude->device_ptr, (float *)output->device_ptr, outer, inner);
    return 1;
}
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!output || !left || !right || output->dtype != H3_GPU_F32 || left->dtype != H3_GPU_F32 || right->dtype != H3_GPU_F32 || elements > left->elements || elements > right->elements || elements > output->elements) return 0;
    h3_cu_add_scaled_f32<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const float *)left->device_ptr, (const float *)right->device_ptr, (float *)output->device_ptr, elements, left_scale, right_scale);
    return 1;
}
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    if (!gpu || !output || !input || !alpha_log || !beta_log ||
        !upsample_filter || !downsample_filter || !batch || !length ||
        !channels)
        return 0;
    size_t count = (size_t)batch * length * channels;
    if (!output->device_ptr || !input->device_ptr || !alpha_log->device_ptr ||
        !beta_log->device_ptr || !upsample_filter->device_ptr ||
        !downsample_filter->device_ptr || count > output->elements ||
        count > input->elements || channels > alpha_log->elements ||
        channels > beta_log->elements || 12 > upsample_filter->elements ||
        12 > downsample_filter->elements)
        return 0;
    dim3 g(h3_cu_grid(channels), length, batch);
    h3_cu_alias_free_snake_f32<<<g, H3_CU_BLOCK>>>(
        (const float *)input->device_ptr, (const float *)alpha_log->device_ptr,
        (const float *)beta_log->device_ptr,
        (const float *)upsample_filter->device_ptr,
        (const float *)downsample_filter->device_ptr,
        (float *)output->device_ptr, batch, length, channels);
    return 1;
}
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale) {
    if (!gpu || !output || !query || !key || !value || !output->device_ptr ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, batch);
    h3_cu_sdpa<float><<<g, head_dim>>>((float *)output->device_ptr,
        (const float *)query->device_ptr, (const float *)key->device_ptr,
        (const float *)value->device_ptr, batch, sequence, heads, head_dim, scale, 1, 0);
    return 1;
}
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    if (!output || !gate || !linear || output->dtype != H3_GPU_F32 || gate->dtype != H3_GPU_F32 || linear->dtype != H3_GPU_F32 || elements > gate->elements || elements > linear->elements || elements > output->elements) return 0;
    h3_cu_geglu_f32<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const float *)gate->device_ptr, (const float *)linear->device_ptr, (float *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!output || !input || output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_clip_f32<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const float *)input->device_ptr, (float *)output->device_ptr, elements, minimum, maximum);
    return 1;
}
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    if (!gpu || !output || !input || !weight || !output->device_ptr ||
        !input->device_ptr || !weight->device_ptr) return 0;
    cublasStatus_t st = h3_cu_gemm(gpu, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
                                   weight->device_ptr, input->device_ptr, output->device_ptr,
                                   rows, input_dim, output_dim);
    if (st != CUBLAS_STATUS_SUCCESS) {
        snprintf(((h3_gpu *)gpu)->error, sizeof(((h3_gpu *)gpu)->error),
                 "cublasGemmEx bf16 failed (%d)", (int)st);
        return 0;
    }
    if (bias && bias->device_ptr)
        h3_cu_linear_bias_bf16<<<h3_cu_grid(rows * output_dim), H3_CU_BLOCK>>>(
            (uint16_t *)output->device_ptr, (const uint16_t *)bias->device_ptr,
            rows, output_dim, 1);
    return 1;
}
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    if (!gpu || !output || !input || !fc1_weight || !fc2_weight ||
        !output->device_ptr || !input->device_ptr || !fc1_weight->device_ptr ||
        !fc2_weight->device_ptr) return 0;
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *act = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim);
    if (!fc1 || !act) { if (fc1) h3_gpu_tensor_free(fc1); if (act) h3_gpu_tensor_free(act); return 0; }
    int ok = h3_gpu_linear_bf16(gpu, fc1, input, fc1_weight, NULL, rows, input_dim, hidden_dim * 2)
        && h3_gpu_swiglu_bf16(gpu, act, fc1, rows, hidden_dim)
        && h3_gpu_linear_bf16(gpu, output, act, fc2_weight, NULL, rows, hidden_dim, output_dim);
    h3_gpu_tensor_free(fc1); h3_gpu_tensor_free(act);
    return ok;
}
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!output || !input || output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_silu_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)input->device_ptr, (uint16_t *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    if (!output || !input || !weight || output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 || (size_t)rows*width > input->elements || width > weight->elements || (size_t)rows*width > output->elements) return 0;
    h3_cu_rms_norm_bf16<<<rows, H3_CU_BLOCK, H3_CU_BLOCK*sizeof(float)>>>((const uint16_t *)input->device_ptr, (const uint16_t *)weight->device_ptr, (uint16_t *)output->device_ptr, rows, width, epsilon);
    return 1;
}
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    if (!output || !input || !weight || !bias || output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 || bias->dtype != H3_GPU_BF16 || (size_t)rows*width > input->elements || width > weight->elements || width > bias->elements || (size_t)rows*width > output->elements) return 0;
    h3_cu_layer_norm_bf16<<<rows, H3_CU_BLOCK, H3_CU_BLOCK*sizeof(float)>>>((const uint16_t *)input->device_ptr, (const uint16_t *)weight->device_ptr, (const uint16_t *)bias->device_ptr, (uint16_t *)output->device_ptr, rows, width, epsilon);
    return 1;
}
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!output || !input || output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 || elements > input->elements || elements > output->elements) return 0;
    h3_cu_gelu_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)input->device_ptr, (uint16_t *)output->device_ptr, elements, approximate);
    return 1;
}
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    if (!gpu || !output || !input || !norm_weight || !modulation || !row_map ||
        !output->device_ptr || !input->device_ptr || !norm_weight->device_ptr ||
        !modulation->device_ptr || !row_map->device_ptr) return 0;
    h3_cu_adaln_bf16<<<rows, H3_CU_BLOCK>>>((const uint16_t *)input->device_ptr,
        (const uint16_t *)norm_weight->device_ptr, (const uint16_t *)modulation->device_ptr,
        (const unsigned *)row_map->device_ptr, (uint16_t *)output->device_ptr,
        rows, width, slots, shift_slot, scale_slot, epsilon);
    return 1;
}
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    if (!gpu || !output || !input || !norm_weight || !modulation || !row_map ||
        !output->device_ptr || !input->device_ptr || !norm_weight->device_ptr ||
        !modulation->device_ptr || !row_map->device_ptr) return 0;
    h3_cu_adaln_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)input->device_ptr + input_offset,
        (const uint16_t *)norm_weight->device_ptr, (const uint16_t *)modulation->device_ptr,
        (const unsigned *)row_map->device_ptr, (uint16_t *)output->device_ptr,
        rows, width, slots, shift_slot, scale_slot, epsilon);
    return 1;
}
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    if (!gpu || !output || !inverse || !input || !norm_weight || !modulation ||
        !row_map || !weight || !output->device_ptr || !input->device_ptr ||
        !norm_weight->device_ptr || !modulation->device_ptr || !row_map->device_ptr ||
        !weight->device_ptr) return 0;
    h3_cu_rms_inverse_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)input->device_ptr + input_offset, (float *)inverse->device_ptr,
        rows, width, epsilon);
    h3_gpu_tensor *tmp = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * width);
    if (!tmp) return 0;
    h3_cu_adaln_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)input->device_ptr + input_offset,
        (const uint16_t *)norm_weight->device_ptr, (const uint16_t *)modulation->device_ptr,
        (const unsigned *)row_map->device_ptr, (uint16_t *)tmp->device_ptr,
        rows, width, slots, shift_slot, scale_slot, epsilon);
    int ok = h3_gpu_linear_bf16(gpu, output, tmp, weight, bias, rows, width, output_dim);
    h3_gpu_tensor_free(tmp);
    return ok;
}
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    if (!output || !residual || !branch || !modulation || !row_map || output->dtype != H3_GPU_BF16 || residual->dtype != H3_GPU_BF16 || branch->dtype != H3_GPU_BF16 || modulation->dtype != H3_GPU_BF16 || row_map->dtype != H3_GPU_U32 || (size_t)rows*width > residual->elements || (size_t)rows*width > branch->elements || (size_t)rows*width > output->elements || rows > row_map->elements) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_gate_bf16<<<g, H3_CU_BLOCK>>>((const uint16_t *)residual->device_ptr, (const uint16_t *)branch->device_ptr, (const uint16_t *)modulation->device_ptr, (const unsigned *)row_map->device_ptr, (uint16_t *)output->device_ptr, rows, width, slots, gate_slot);
    return 1;
}
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    if (!gpu || !gated_residual || !output || !residual || !branch || !norm_weight ||
        !gate_modulation || !norm_modulation || !row_map || width > H3_DIT_MAX ||
        !gated_residual->device_ptr || !output->device_ptr || !residual->device_ptr ||
        !branch->device_ptr || !norm_weight->device_ptr || !gate_modulation->device_ptr ||
        !norm_modulation->device_ptr || !row_map->device_ptr) return 0;
    h3_cu_gate_adaln_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)residual->device_ptr, (const uint16_t *)branch->device_ptr,
        (const uint16_t *)norm_weight->device_ptr,
        (const uint16_t *)gate_modulation->device_ptr,
        (const uint16_t *)norm_modulation->device_ptr,
        (const unsigned *)row_map->device_ptr,
        (uint16_t *)gated_residual->device_ptr, (uint16_t *)output->device_ptr,
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon);
    return 1;
}
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    if (!gpu || !query || !key || !value || !qkv || !q_norm || !k_norm ||
        !rope_cos || !rope_sin || !query->device_ptr || !key->device_ptr ||
        !value->device_ptr || !qkv->device_ptr || !q_norm->device_ptr ||
        !k_norm->device_ptr || !rope_cos->device_ptr || !rope_sin->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, 1);
    h3_cu_grouped_qkv_rope_bf16<<<g, head_dim>>>(
        (const uint16_t *)qkv->device_ptr, (const uint16_t *)q_norm->device_ptr,
        (const uint16_t *)k_norm->device_ptr, (const uint16_t *)rope_cos->device_ptr,
        (const uint16_t *)rope_sin->device_ptr, (uint16_t *)query->device_ptr,
        (uint16_t *)key->device_ptr, (uint16_t *)value->device_ptr,
        sequence, heads, head_dim, rope_half, epsilon);
    return 1;
}
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    if (!h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim, inner * 3)) return 0;
    return h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm, k_norm,
        rope_cos, rope_sin, rows, heads, head_dim, rope_half, epsilon);
}
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    if (!gpu || !output || !query || !key || !value || !output->device_ptr ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, 1);
    h3_cu_sdpa<uint16_t><<<g, head_dim>>>((uint16_t *)output->device_ptr,
        (const uint16_t *)query->device_ptr, (const uint16_t *)key->device_ptr,
        (const uint16_t *)value->device_ptr, 1, sequence, heads, head_dim, scale, 0, 0);
    return 1;
}
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    if (!gpu || !output || !query || !key || !value || !output->device_ptr ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr) return 0;
    if (head_dim > H3_CU_BLOCK || head_dim == 0) return 0;
    dim3 g(heads, sequence, 1);
    h3_cu_sdpa<uint16_t><<<g, head_dim>>>((uint16_t *)output->device_ptr,
        (const uint16_t *)query->device_ptr, (const uint16_t *)key->device_ptr,
        (const uint16_t *)value->device_ptr, 1, sequence, heads, head_dim, scale, 0, 1);
    return 1;
}
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    if (!output || !fused || output->dtype != H3_GPU_BF16 || fused->dtype != H3_GPU_BF16 || (size_t)rows*width*2 > fused->elements || (size_t)rows*width > output->elements) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_swiglu_bf16<<<g, H3_CU_BLOCK>>>((const uint16_t *)fused->device_ptr, (uint16_t *)output->device_ptr, rows, width);
    return 1;
}
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    if (!output || !weight || !token_ids || output->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 || token_ids->dtype != H3_GPU_U32 || (size_t)tokens*width > output->elements || (size_t)vocab_size*width > weight->elements || tokens > token_ids->elements) return 0;
    dim3 g(h3_cu_grid(width), tokens);
    h3_cu_embedding_bf16<<<g, H3_CU_BLOCK>>>((const uint16_t *)weight->device_ptr, (const unsigned *)token_ids->device_ptr, (uint16_t *)output->device_ptr, tokens, width, vocab_size);
    return 1;
}
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    if (!tensor || !weight || tensor->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 || (size_t)sequence*heads*head_dim > tensor->elements || head_dim > weight->elements) return 0;
    h3_cu_head_rms_norm_bf16<<<h3_cu_grid((unsigned)sequence*heads), H3_CU_BLOCK>>>((uint16_t *)tensor->device_ptr, (const uint16_t *)weight->device_ptr, sequence, heads, head_dim, epsilon);
    return 1;
}
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    if (!gpu || !query || !key || !rope_cos_f32 || !rope_sin_f32 ||
        !query->device_ptr || !key->device_ptr || !rope_cos_f32->device_ptr ||
        !rope_sin_f32->device_ptr) return 0;
    uint32_t max_head = query_heads > kv_heads ? query_heads : kv_heads;
    dim3 g(max_head, sequence);
    h3_cu_rope_text_bf16<<<g, H3_CU_BLOCK>>>((uint16_t *)query->device_ptr,
        (uint16_t *)key->device_ptr, (const float *)rope_cos_f32->device_ptr,
        (const float *)rope_sin_f32->device_ptr, sequence, query_heads, kv_heads, head_dim);
    return 1;
}
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    if (!gpu || !output || !query || !key || !value || !output->device_ptr ||
        !query->device_ptr || !key->device_ptr || !value->device_ptr) return 0;
    dim3 g(sequence, query_heads);
    size_t dyn = (head_dim + sequence + H3_CU_BLOCK) * sizeof(float);
    h3_cu_gqa_causal_bf16<<<g, H3_CU_BLOCK, dyn>>>(
        (const uint16_t *)query->device_ptr, (const uint16_t *)key->device_ptr,
        (const uint16_t *)value->device_ptr, (uint16_t *)output->device_ptr,
        sequence, query_heads, kv_heads, head_dim, scale);
    return 1;
}
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!output || !left || !right || output->dtype != H3_GPU_BF16 || left->dtype != H3_GPU_BF16 || right->dtype != H3_GPU_BF16 || elements > left->elements || elements > right->elements || elements > output->elements) return 0;
    h3_cu_add_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)left->device_ptr, (const uint16_t *)right->device_ptr, (uint16_t *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!output || !left || !right || output->dtype != H3_GPU_BF16 || left->dtype != H3_GPU_BF16 || right->dtype != H3_GPU_BF16 || elements > left->elements || elements > right->elements || elements > output->elements) return 0;
    h3_cu_sub_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)left->device_ptr, (const uint16_t *)right->device_ptr, (uint16_t *)output->device_ptr, elements);
    return 1;
}
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    if (!gpu || !output || !input || !original || !baseline || !baseline_indices || !pairs ||
        !output->device_ptr || !input->device_ptr || !original->device_ptr ||
        !baseline->device_ptr || !baseline_indices->device_ptr || !pairs->device_ptr) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_token_pool_bf16<<<g, H3_CU_BLOCK>>>(
        (const uint16_t *)input->device_ptr, (const uint2 *)pairs->device_ptr,
        (uint16_t *)output->device_ptr, (uint16_t *)baseline->device_ptr,
        (const unsigned *)baseline_indices->device_ptr, (uint16_t *)original->device_ptr,
        input_offset, original_offset, baseline_offset, rows, width);
    (void)input_rows; (void)baseline_rows;
    return 1;
}
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    if (!gpu || !residual || !output || !input || !original || !baseline ||
        !baseline_indices || !pairs || !norm_weight || !modulation || !row_map ||
        width > H3_DIT_MAX || !residual->device_ptr || !output->device_ptr ||
        !input->device_ptr || !original->device_ptr || !baseline->device_ptr ||
        !baseline_indices->device_ptr || !pairs->device_ptr || !norm_weight->device_ptr ||
        !modulation->device_ptr || !row_map->device_ptr) return 0;
    h3_cu_token_pool_adaln_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)input->device_ptr, (const uint2 *)pairs->device_ptr,
        (uint16_t *)residual->device_ptr, (uint16_t *)baseline->device_ptr,
        (const unsigned *)baseline_indices->device_ptr, (uint16_t *)original->device_ptr,
        (const uint16_t *)norm_weight->device_ptr, (const uint16_t *)modulation->device_ptr,
        (const unsigned *)row_map->device_ptr, (uint16_t *)output->device_ptr,
        input_offset, original_offset, baseline_offset, rows, width, slots,
        shift_slot, scale_slot, epsilon);
    (void)input_rows; (void)baseline_rows;
    return 1;
}
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    if (!gpu || !output || !original || !reduced || !baseline || !baseline_indices || !parents ||
        !output->device_ptr || !original->device_ptr || !reduced->device_ptr ||
        !baseline->device_ptr || !baseline_indices->device_ptr || !parents->device_ptr) return 0;
    dim3 g(h3_cu_grid(width), rows);
    h3_cu_token_expand_delta_bf16<<<g, H3_CU_BLOCK>>>(
        (const uint16_t *)original->device_ptr, (const uint16_t *)reduced->device_ptr,
        (const uint16_t *)baseline->device_ptr, (const unsigned *)baseline_indices->device_ptr,
        (const unsigned *)parents->device_ptr, (uint16_t *)output->device_ptr,
        original_offset, baseline_offset, rows, width, exact_prefix_rows, update_scale);
    (void)reduced_rows; (void)baseline_rows;
    return 1;
}
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    if (!gpu || !residual || !output || !original || !reduced || !baseline ||
        !baseline_indices || !parents || !norm_weight || !modulation || !row_map ||
        width > H3_DIT_MAX || !residual->device_ptr || !output->device_ptr ||
        !original->device_ptr || !reduced->device_ptr || !baseline->device_ptr ||
        !baseline_indices->device_ptr || !parents->device_ptr || !norm_weight->device_ptr ||
        !modulation->device_ptr || !row_map->device_ptr) return 0;
    h3_cu_token_expand_adaln_bf16<<<rows, H3_CU_BLOCK>>>(
        (const uint16_t *)original->device_ptr, (const uint16_t *)reduced->device_ptr,
        (const uint16_t *)baseline->device_ptr, (const unsigned *)baseline_indices->device_ptr,
        (const unsigned *)parents->device_ptr, (const uint16_t *)norm_weight->device_ptr,
        (const uint16_t *)modulation->device_ptr, (const unsigned *)row_map->device_ptr,
        (uint16_t *)residual->device_ptr, (uint16_t *)output->device_ptr,
        original_offset, baseline_offset, rows, width, exact_prefix_rows, update_scale,
        slots, shift_slot, scale_slot, epsilon);
    (void)reduced_rows; (void)baseline_rows;
    return 1;
}
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!gpu || !sample || !last || !previous || !sample->device_ptr ||
        !last->device_ptr || !previous->device_ptr) return 0;
    h3_cu_euler_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>(
        (float *)sample->device_ptr, (const uint16_t *)last->device_ptr,
        (const uint16_t *)previous->device_ptr, sample_offset, elements, delta, ratio);
    return 1;
}
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    if (!output || !gate || !up || output->dtype != H3_GPU_BF16 || gate->dtype != H3_GPU_BF16 || up->dtype != H3_GPU_BF16 || elements > gate->elements || elements > up->elements || elements > output->elements) return 0;
    h3_cu_silu_mul_bf16<<<h3_cu_grid(elements), H3_CU_BLOCK>>>((const uint16_t *)gate->device_ptr, (const uint16_t *)up->device_ptr, (uint16_t *)output->device_ptr, elements);
    return 1;
}
