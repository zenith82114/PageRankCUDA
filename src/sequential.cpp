#include<iostream>
#include<fstream>
#include<numeric>
#include<algorithm>
#include<chrono>

#define MAX_ITER 100
#define DELTA 0.85f

int main(int argc, char **argv) {
    std::fstream fin;
    fin.open(argv[1]);

    int n, m; fin >> n >> m;
    int *col_indices = (int*)malloc(m * sizeof(int));
    int *row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int last = -1;
    for (int i = 0; i < m; ++i) {
        int u; fin >> u >> col_indices[i];
        while (last != u) {
            row_ptr[++last] = i;
        }
    }
    while (last != n) {
        row_ptr[++last] = m;
    }
    fin.close();

    const auto start_time = std::chrono::high_resolution_clock::now();

    float *pr_in = (float*)malloc(n * sizeof(float));
    float *pr_out = (float*)malloc(n * sizeof(float));
    for (int u = 0; u < n; ++u) pr_out[u] = 1.0f / n;

    int iter = 0;
    for (; iter < MAX_ITER; ++iter) {
        std::swap(pr_out, pr_in);
        std::fill(pr_out, pr_out + n, 0.0f);

        float sink = 0.0f;

        for (int u = 0; u < n; ++u) {
            int out_deg = row_ptr[u + 1] - row_ptr[u];
            if (out_deg > 0) {
                float contribution = pr_in[u] / out_deg;
                for (int i = row_ptr[u]; i < row_ptr[u + 1]; ++i) {
                    int v = col_indices[i];
                    pr_out[v] += contribution;
                }
            }
            else {
                sink += pr_in[u];
            }
        }
        for (int u = 0; u < n; ++u) {
            // std::cout << pr_out[u] << ' ';
            pr_out[u] = (1.0f - DELTA) / n + DELTA * (pr_out[u] + sink / n);
        }
        // std::cout << std::endl;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Wall clock time passed: "
        << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms\n";

    int *order = (int*)malloc(n * sizeof(int));
    std::iota(order, order + n, 0);
    std::sort(order, order + n, [&pr_out] (int a, int b) { return pr_out[a] > pr_out[b]; });

    std::fstream fout;
    fout.open(argv[2], std::ios::out | std::ios::trunc);

    for (int i = 0; i < std::min(10, n); ++i) {
        fout << order[i] << '\t' << pr_out[order[i]] << '\n';
    }
    fout.close();

    free(col_indices);
    free(row_ptr);
    free(pr_in);
    free(pr_out);
    free(order);

    return 0;
}
