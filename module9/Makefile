NVCC ?=/usr/bin/env nvcc -std=c++11

THRUST_SOURCES = \
	src_thrust/arg_parse.cpp \
	src_thrust/main.cu \
	src_thrust/thrust_operations.cu

all: thrust_assignment9

thrust_assignment9: $(THRUST_SOURCES)
	$(NVCC) -o thrust_assignment9 -g -lcuda -lcudart $^

.PHONY: clean

clean:
	rm -f thrust_assignment9

# vim: noet: ts=4: sw=4
