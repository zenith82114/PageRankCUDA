SRC_DIR=src
BIN_DIR=bin
GCC=g++
NVCC=nvcc

TARGETS=sequential parallel_naive parallel_gather parallel_gather_contrib parallel_gather_contrib_mixed

all: $(TARGETS)

sequential: $(BIN_DIR)
	$(GCC) -std=c++11 $(SRC_DIR)/sequential.cpp -o $(BIN_DIR)/sequential

parallel_%: $(BIN_DIR)
	$(NVCC) -std=c++11 $(SRC_DIR)/parallel_$*.cu -o $(BIN_DIR)/parallel_$*

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
