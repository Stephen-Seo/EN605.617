NVCC ?= /usr/bin/env nvcc

SOURCES = \
	assignment_src/main.cc

all: assignment12

assignment12: $(SOURCES)
	$(NVCC) -o assignment12 -std=c++11 -g -O0 -Wall -Wextra -Wpedantic -lOpenCL $^
