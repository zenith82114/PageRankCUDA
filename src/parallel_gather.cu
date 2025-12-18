#include<iostream>
#include<fstream>
#include<numeric>
#include<algorithm>
#include<chrono>
#include<vector>
#include<cuda_runtime.h>

#define MAX_ITER 100
#define DELTA 0.85f
#define REDUCE_GRID_SIZE 256
#define BLOCK_SIZE 256

// MACRO to check for CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void initPR(
    float *pr,
    int n
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) {
        pr[u] = 1.0f / n;
    }
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void danglingSum(
    const int *out_degrees,
    const float *pr_in,
    float *d_total_sum,
    int n
) {
    float local_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int u = tid; u < n; u += stride) {
        if (out_degrees[u] == 0) {
            local_sum += pr_in[u];
        }
    }
    local_sum = warpReduceSum(local_sum);

    static __shared__ float shared_warps[32]; // Max 1024 threads / 32 warps = 32
    int lane = threadIdx.x % warpSize;
    int local_warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_warps[local_warp_id] = local_sum;
    }
    __syncthreads();

    local_sum = (threadIdx.x < blockDim.x / warpSize) ? shared_warps[lane] : 0.0f;

    if (local_warp_id == 0) {
        local_sum = warpReduceSum(local_sum);
        if (threadIdx.x == 0) {
            atomicAdd(d_total_sum, local_sum);
        }
    }
}

__global__ void gatherPRandDamp(
    const int *col_indices,
    const int *row_ptr,
    const int *out_degrees,
    const float *pr_in,
    float *pr_out,
    float dangling_contrib,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        float sum = 0.0f;
        int start_edge = row_ptr[v];
        int end_edge = row_ptr[v + 1];

        for (int i = start_edge; i < end_edge; ++i) {
            int u = col_indices[i];
            float contribution = out_degrees[u] > 0 ? (pr_in[u] / (float)out_degrees[u]) : 0.0f;
            sum += contribution;
        }
        pr_out[v] = (1.0f - DELTA) / n + DELTA * (dangling_contrib + sum);
    }
}

int main(int argc, char **argv) {
    std::fstream fin;
    fin.open(argv[1]);

    int n, m; fin >> n >> m;
    std::vector<std::vector<int> > adj_transposed(n);
    std::vector<int> out_degrees(n);
    for (int i = 0; i < m; ++i) {
        int u, v; fin >> u >> v;
        adj_transposed[v].push_back(u);
        out_degrees[u]++;
    }
    fin.close();

    int *h_col_indices = (int*)malloc(m * sizeof(int));
    int *h_row_ptr = (int*)malloc((n + 1) * sizeof(int));
    h_row_ptr[0] = 0;
    for (int u = 0; u < n; ++u) {
        int in_deg = adj_transposed[u].size();
        memcpy(h_col_indices + h_row_ptr[u], adj_transposed[u].data(), in_deg * sizeof(int));
        h_row_ptr[u + 1] = h_row_ptr[u] + in_deg;
    }

    // Allocate device memory
    int *d_col_indices, *d_row_ptr;
    cudaCheckError(cudaMalloc(&d_col_indices, m * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    int *d_out_degrees;
    cudaCheckError(cudaMalloc(&d_out_degrees, n * sizeof(int)));
    float *d_pr_in, *d_pr_out;
    cudaCheckError(cudaMalloc(&d_pr_in, n * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_pr_out, n * sizeof(float)));
    float *d_dangling_sum;
    cudaCheckError(cudaMalloc(&d_dangling_sum, sizeof(float)));

    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices, m * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out_degrees, out_degrees.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    const auto start_time = std::chrono::high_resolution_clock::now();

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initPR<<<grid_size, BLOCK_SIZE>>>(d_pr_out, n);
    cudaCheckError(cudaPeekAtLastError());

    int iter = 0;
    for (; iter < MAX_ITER; ++iter) {
        std::swap(d_pr_in, d_pr_out);

        cudaCheckError(cudaMemset(d_dangling_sum, 0, sizeof(float)));
        danglingSum<<<REDUCE_GRID_SIZE, BLOCK_SIZE>>>(
            d_out_degrees, d_pr_in, d_dangling_sum, n);
        float h_dangling_sum;
        cudaMemcpy(&h_dangling_sum, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost);

        cudaCheckError(cudaMemset(d_pr_out, 0, n * sizeof(float)));
        gatherPRandDamp<<<grid_size, BLOCK_SIZE>>>(
            d_col_indices, d_row_ptr, d_out_degrees, d_pr_in, d_pr_out, h_dangling_sum / n, n);
    }

    // Copy result back to host
    float *h_pr = (float*)malloc(n * sizeof(float));
    cudaCheckError(cudaMemcpy(h_pr, d_pr_in, n * sizeof(float), cudaMemcpyDeviceToHost));

    const auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Wall clock time passed: "
        << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms\n";

    // Print result
    int *order = (int*)malloc(n * sizeof(int));
    std::iota(order, order + n, 0);
    std::sort(order, order + n, [&h_pr] (int u, int v) { return h_pr[u] > h_pr[v]; });

    std::fstream fout;
    fout.open(argv[2], std::ios::out | std::ios::trunc);

    for (int i = 0; i < std::min(10, n); ++i) {
        fout << order[i] << '\t' << h_pr[order[i]] << '\n';
    }
    fout.close();

    // Free memory
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_out_degrees);
    cudaFree(d_pr_in);
    cudaFree(d_pr_out);
    cudaFree(d_dangling_sum);
    free(h_col_indices);
    free(h_row_ptr);
    free(h_pr);
    free(order);

    return 0;
}
