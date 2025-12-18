
# PageRankCUDA

Parallel and Optimized PageRank with CUDA

Done as a class project in *M3239.005400 Computing for Data Science 2, Fall 2025*

## Setup

```sh
pip install cugraph-cu13 --extra-index-url=https://pypi.nvidia.com

git clone https://github.com/zenith82114/PageRankCUDA.git
cd PageRankCUDA
mkdir data test

cd data
wget https://snap.stanford.edu/data/congress_network.zip \
    https://snap.stanford.edu/data/facebook_combined.txt.gz \
    https://snap.stanford.edu/data/web-Google.txt.gz \
    https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz \
    https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
unzip congress_network.zip
gzip -d *.txt.gz
cd ..

python preprocess.py

make

./run.sh congress_network
```
