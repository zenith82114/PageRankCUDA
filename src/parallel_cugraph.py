import cugraph
import cudf
import time
import sys
import os

def main(input_file_path, output_file_path):
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found.")
        return

    gdf = cudf.read_csv(
        input_file_path,
        names=['src', 'dst'],
        dtype=['int32', 'int32'],
        sep=' ',
        skiprows=1,
    )

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source='src', destination='dst', store_transposed=True)

    # warm-up run to prevent JIT or lazy execution
    cugraph.pagerank(G, alpha=0.85, max_iter=100)

    start_time = time.perf_counter()

    pr_df, converged = cugraph.pagerank(
        G, alpha=0.85,
        tol=1e-100,    # force `max_iter` iterations
        max_iter=100,
        fail_on_nonconvergence=False)

    # dummy access to ensure completion
    _ = pr_df['pagerank'].iloc[0]

    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000
    print(f"Wall clock time passed: {elapsed_ms:.4f} ms")

    top_10 = pr_df.sort_values('pagerank', ascending=False).head(10)

    with open(output_file_path, 'w') as f:
        f.write(str(top_10))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parallel_cugraph.py <data.in> <result.out>")
    else:
        main(sys.argv[1], sys.argv[2])
