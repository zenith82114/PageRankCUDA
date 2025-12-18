#!/bin/bash

DATASET=$1


echo "Running sequential..."
./bin/sequential test/${DATASET}.in out/${DATASET}_seq.out
echo

echo "Running parallel_naive..."
./bin/parallel_naive test/${DATASET}.in out/${DATASET}_naive.out
echo

echo "Running parallel_gather..."
./bin/parallel_gather test/${DATASET}.in out/${DATASET}_gather.out
echo

echo "Running parallel_gather_contrib..."
./bin/parallel_gather_contrib test/${DATASET}.in out/${DATASET}_gather_contrib.out
echo

# echo "Running parallel_gather_contrib_mixed (threshold = 8)..."
# ./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 8
# echo

# echo "Running parallel_gather_contrib_mixed (threshold = 16)..."
# ./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 16
# echo

# echo "Running parallel_gather_contrib_mixed (threshold = 32)..."
# ./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 32
# echo

echo "Running parallel_gather_contrib_mixed (threshold = 64)..."
./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 64
echo

# echo "Running parallel_gather_contrib_mixed (threshold = 128)..."
# ./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 128
# echo

# echo "Running parallel_gather_contrib_mixed (threshold = 256)..."
# ./bin/parallel_gather_contrib_mixed test/${DATASET}.in out/${DATASET}_gather_contrib_mixed.out 256
# echo

echo "Running parallel_cugraph..."
python src/parallel_cugraph.py test/${DATASET}.in out/${DATASET}_cugraph.out
echo
