NVCC?=/usr/bin/env nvcc

SOURCES = \
	assignment_src/main.cc \
	assignment_src/arg_parse.cc

all: assignment11

assignment11: $(SOURCES)
	$(NVCC) -g -O0 -std=c++11 -lOpenCL -o assignment11 $^

.PHONY: clean

clean:
	rm -f assignment11

# vim: noet
