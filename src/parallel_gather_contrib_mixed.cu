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
#define GATHER_LARGE_GRID_MAX_SIZE 1024

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
    float *dangling_sum,
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
            atomicAdd(dangling_sum, local_sum);
        }
    }
}

__global__ void computeContributions(
    const int *out_degrees,
    const float *pr_in,
    float *contrib,
    int n
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) {
        contrib[u] = out_degrees[u] > 0 ? (pr_in[u] / (float)out_degrees[u]) : 0.0f;
    }
}

__global__ void gatherAndDampByThread(
    const int *nodes,
    const int *col_indices,
    const int *row_ptr,
    const float *contrib,
    float *pr_out,
    float dangling_contrib,
    int len,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        int v = nodes[idx];
        int start_edge = row_ptr[v];
        int end_edge = row_ptr[v + 1];
        float sum = 0.0f;

        for (int i = start_edge; i < end_edge; ++i) {
            int u = col_indices[i];
            sum += contrib[u];
        }
        pr_out[v] = (1.0f - DELTA) / n + DELTA * (dangling_contrib + sum);
    }
}

__global__ void gatherAndDampByWarp(
    const int *nodes,
    const int *col_indices,
    const int *row_ptr,
    const float *contrib,
    float *pr_out,
    float dangling_contrib,
    int len,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % warpSize;
    int global_warp_id = tid / warpSize;
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    for (int idx = global_warp_id; idx < len; idx += total_warps) {
        int v = nodes[idx];
        int start_edge = row_ptr[v];
        int end_edge = row_ptr[v + 1];
        float sum = 0.0f;

        for (int i = start_edge + lane; i < end_edge; i += warpSize) {
            int u = col_indices[i];
            sum += contrib[u];
        }
        sum = warpReduceSum(sum);
        if (lane == 0) {
            pr_out[v] = (1.0f - DELTA) / n + DELTA * (dangling_contrib + sum);
        }
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

    // Classify small/large nodes
    int degree_threshold = std::atoi(argv[3]);
    std::vector<int> h_small_nodes, h_large_nodes;
    for (int u = 0; u < n; ++u) {
        int degree = h_row_ptr[u + 1] - h_row_ptr[u];
        (degree < degree_threshold ? h_small_nodes : h_large_nodes).push_back(u);
    }
    const int n_small = h_small_nodes.size();
    const int n_large = h_large_nodes.size();

    std::cout << "\tSmall nodes: " << n_small << " / " << n << " (" << ((100. * n_small) / n) << "%)\n";
    std::cout << "\tLarge nodes: " << n_large << " / " << n << " (" << ((100. * n_large) / n) << "%)\n";

    // Allocate device memory
    int *d_small_nodes, *d_large_nodes;
    cudaCheckError(cudaMalloc(&d_small_nodes, n_small * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_large_nodes, n_large * sizeof(int)));
    int *d_col_indices, *d_row_ptr;
    cudaCheckError(cudaMalloc(&d_col_indices, m * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_row_ptr,     (n + 1) * sizeof(int)));
    int *d_out_degrees;
    cudaCheckError(cudaMalloc(&d_out_degrees, n * sizeof(int)));
    float *d_pr_in, *d_pr_out;
    cudaCheckError(cudaMalloc(&d_pr_in,       n * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_pr_out,      n * sizeof(float)));
    float *d_dangling_sum;
    cudaCheckError(cudaMalloc(&d_dangling_sum, sizeof(float)));
    float *d_contrib;
    cudaCheckError(cudaMalloc(&d_contrib,     n * sizeof(float)));

    cudaStream_t stream_small, stream_large;
    cudaStreamCreate(&stream_small);
    cudaStreamCreate(&stream_large);

    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_small_nodes, h_small_nodes.data(),
                                n_small * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_large_nodes, h_large_nodes.data(),
                                n_large * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices,
                                m * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_row_ptr, h_row_ptr,
                                (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out_degrees, out_degrees.data(),
                                n * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    const auto start_time = std::chrono::high_resolution_clock::now();

    const int basic_grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int gather_small_grid_size = (n_small + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gather_large_grid_size = (n_large * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (gather_large_grid_size > GATHER_LARGE_GRID_MAX_SIZE) {
        gather_large_grid_size = GATHER_LARGE_GRID_MAX_SIZE;
    }

    initPR<<<basic_grid_size, BLOCK_SIZE>>>(d_pr_out, n);
    cudaCheckError(cudaPeekAtLastError());

    int iter = 0;
    for (; iter < MAX_ITER; ++iter) {
        std::swap(d_pr_in, d_pr_out);

        cudaCheckError(cudaMemset(d_dangling_sum, 0, sizeof(float)));
        danglingSum<<<REDUCE_GRID_SIZE, BLOCK_SIZE>>>(
            d_out_degrees, d_pr_in, d_dangling_sum, n);
        float h_dangling_sum;
        cudaMemcpy(&h_dangling_sum, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost);

        computeContributions<<<basic_grid_size, BLOCK_SIZE>>>(
            d_out_degrees, d_pr_in, d_contrib, n);

        if (n_small) {
            gatherAndDampByThread<<<gather_small_grid_size, BLOCK_SIZE, 0, stream_small>>>(
                d_small_nodes, d_col_indices, d_row_ptr, d_contrib,
                d_pr_out, h_dangling_sum / n, n_small, n);
        }
        if (n_large) {
            gatherAndDampByWarp<<<gather_large_grid_size, BLOCK_SIZE, 0, stream_large>>>(
                d_large_nodes, d_col_indices, d_row_ptr, d_contrib,
                d_pr_out, h_dangling_sum / n, n_large, n);
        }
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
    cudaFree(d_small_nodes);
    cudaFree(d_large_nodes);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_out_degrees);
    cudaFree(d_pr_in);
    cudaFree(d_pr_out);
    cudaFree(d_dangling_sum);
    cudaFree(d_contrib);
    cudaStreamDestroy(stream_small);
    cudaStreamDestroy(stream_large);
    free(h_col_indices);
    free(h_row_ptr);
    free(h_pr);
    free(order);

    return 0;
}
