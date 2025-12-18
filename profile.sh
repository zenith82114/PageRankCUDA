#!/bin/bash

DATASET=$1


echo "Profiling parallel_naive..."
nsys profile --stats=true --output=naive ./bin/parallel_naive test/${DATASET}.in out/${DATASET}_naive.out
echo

echo "Profiling parallel_gather..."
nsys profile --stats=true --output=gather ./bin/parallel_gather test/${DATASET}.in out/${DATASET}_gather.out
echo
