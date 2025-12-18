import json
import os


datasets = [
    ('congress_network', 0),
    ('facebook_combined.txt', 0),
    ('web-Google.txt', 0),
    ('soc-pokec-relationships.txt', 1),
    ('soc-LiveJournal1.txt', 0),
]


if __name__ == "__main__":
    DATA_DIR = './data'
    TEST_DIR = './test'
    assert os.path.exists(DATA_DIR)
    assert os.path.exists(TEST_DIR)

    for dataset, index_base in datasets:
        dataset_path = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(dataset_path):
            continue
        print(dataset_path)

        if dataset == 'congress_network':
            data = json.load(open(os.path.join(dataset_path, "congress_network_data.json")))
            out_list = data[0]['outList']
            n = len(out_list)
            m = sum(map(len, out_list))

            with open(os.path.join(TEST_DIR, 'congress_network.in'), 'w') as dst:
                dst.write(f"{n} {m}\n")
                for i, i_out in enumerate(out_list):
                    for j in i_out:
                        dst.write(f"{i} {j}\n")

        elif dataset == 'web-Google.txt':
            m = 0
            s = set()

            with open(dataset_path, 'r') as src:
                for line in src:
                    if line.startswith('#'):
                        continue
                    i, j = map(int, line.split())
                    s.add(i)
                    s.add(j)
                    m += 1

            s = sorted(s)
            n = len(s)
            g = [[] for _ in range(n)]

            from bisect import bisect_left

            with open(dataset_path, 'r') as src:
                for line in src:
                    if line.startswith('#'):
                        continue
                    i, j = map(int, line.split())
                    i = bisect_left(s, i)
                    j = bisect_left(s, j)
                    g[i].append(j)

            test_path = os.path.join(TEST_DIR, 'web-Google.in')

            with open(test_path, 'w') as dst:
                dst.write(f"{n} {m}\n")
                for i in range(n):
                    for j in g[i]:
                        dst.write(f"{i} {j}\n")
        else:
            n, m = 0, 0

            with open(dataset_path, 'r') as src:
                for line in src:
                    if line.startswith('#'):
                        continue
                    i, j = map(int, line.split())
                    n = max(n, i, j)
                    m += 1

            n += (index_base ^ 1)
            test_path = os.path.join(TEST_DIR, f"{dataset.split('.')[0]}.in")

            with open(dataset_path, 'r') as src, open(test_path, 'w') as dst:
                dst.write(f"{n} {m}\n")
                for line in src:
                    if line.startswith('#'):
                        continue
                    i, j = map(int, line.split())
                    dst.write(f"{i - index_base} {j - index_base}\n")
