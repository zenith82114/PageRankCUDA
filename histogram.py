import matplotlib.pyplot as plt
import sys
import argparse


def read_graph(filepath: str):
    with open(filepath, 'r') as f:
        n, m = map(int, f.readline().strip().split())

        in_degrees = [0] * n
        out_degrees = [0] * n

        for u in range(n):
            in_degrees[u] = 0
            out_degrees[u] = 0

        for line in f:
            u, v = map(int, line.strip().split())
            out_degrees[u] += 1
            in_degrees[v] += 1

    return n, in_degrees, out_degrees

def plot_degree_histograms(filepath: str, in_degrees: list, out_degrees: list, N: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # In-degree histogram
    ax1.hist(in_degrees, bins=range(max(in_degrees) + 2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('In-Degree')
    ax1.set_title(f'In-Degree Distribution (N={N})')

    # Out-degree histogram
    ax2.hist(out_degrees, bins=range(max(out_degrees) + 2),
             edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Out-Degree')
    ax2.set_title(f'Out-Degree Distribution (N={N})')

    for ax in [ax1, ax2]:
        ax.set_ylabel('Number of Vertices')
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"{filepath.split('.')[0]}.png", dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"Graph Statistics:")
    print(f"  Number of vertices: {N}")
    print(f"  Number of edges: {sum(out_degrees)}")
    print(f"\nIn-Degree Statistics:")
    print(f"  Min: {min(in_degrees)}, Max: {max(in_degrees)}")
    print(f"  Average: {sum(in_degrees) / len(in_degrees):.2f}")
    print(f"\nOut-Degree Statistics:")
    print(f"  Min: {min(out_degrees)}, Max: {max(out_degrees)}")
    print(f"  Average: {sum(out_degrees) / len(out_degrees):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the graph file')

    args = parser.parse_args()

    try:
        n, in_degrees, out_degrees = read_graph(args.file_path)
        plot_degree_histograms(args.file_path, in_degrees, out_degrees, n)
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
