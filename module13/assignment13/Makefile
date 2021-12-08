NVCC ?= /usr/bin/env nvcc

SOURCES = \
	src/main.cc \
	src/arg_parse.cc \
	src/csv_parser.cc \
	src/dependencies.cc

all: assignment13

assignment13: $(SOURCES)
	$(NVCC) -o assignment13 -std=c++11 -g -O0 -lOpenCL $^

.PHONY: clean

clean:
	rm -f assignment13
