NVCC?=/usr/local/cuda/bin/nvcc

all: assignment.exe

assignment.exe: assignment.cu
	$(NVCC) -o assignment.exe -std=c++11 -lcudart -g assignment.cu

.PHONY: clean

clean:
	rm -f assignment.exe
