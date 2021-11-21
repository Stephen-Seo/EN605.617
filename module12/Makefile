NVCC ?= /usr/bin/env nvcc

SOURCES = \
	assignment_src/main.cc \
	assignment_src/arg_parse.cc

all: assignment12

assignment12: $(SOURCES)
	$(NVCC) -o assignment12 -std=c++11 -g -O0 -lOpenCL $^

.PHONY: clean

clean:
	rm -f assignment12
