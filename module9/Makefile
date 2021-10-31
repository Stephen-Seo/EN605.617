NVCC ?=/usr/bin/env nvcc -std=c++11

THRUST_SOURCES = \
	src_thrust/arg_parse.cc \
	src_thrust/main.cu \
	src_thrust/thrust_operations.cu

NPP_SOURCES = \
	src_npp/main.cu \
	src_npp/arg_parse.cc \
	src_npp/pgm_rw.cc

all: thrust_assignment9 npp_assignment9

thrust_assignment9: $(THRUST_SOURCES)
	$(NVCC) -o thrust_assignment9 -g -lcuda -lcudart $^

npp_assignment9: $(NPP_SOURCES)
	$(NVCC) -o npp_assignment9 -g -lcuda -lcudart $^

.PHONY: clean

clean:
	rm -f thrust_assignment9
	rm -f npp_assignment9

# vim: noet: ts=4: sw=4
