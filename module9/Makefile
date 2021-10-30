NVCC ?=/usr/bin/env nvcc -std=c++11

THRUST_SOURCES = \
	src_thrust/arg_parse.cpp \
	src_thrust/main.cu

all: thrust_assignment

thrust_assignment: $(THRUST_SOURCES)
	$(NVCC) -o thrust_assignment -g $^

.PHONY: clean

clean:
	rm -f thrust_assignment

# vim: noet: ts=4: sw=4
