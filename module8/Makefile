NVCC?=/usr/bin/env nvcc

SOURCES= \
	src/main.cu \
	src/helpers.cu \
	src/cuda_rand_kernel.cu \
	src/arg_parse.cpp

all: assignment8

assignment8: $(SOURCES)
	$(NVCC) -g -o assignment8 -lcudart -lcurand -lcublas -lcuda -std=c++11 $(SOURCES)

.PHONY: clean

clean:
	rm -f assignment8
