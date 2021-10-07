NVCC?=/usr/bin/env nvcc

SOURCES= \
	src/mathexpressions.cu \
	src/helpers.cu \
	src/arg_parse.cpp \
	src/assignment.cu

all: assignment6

assignment6: $(SOURCES)
	$(NVCC) -g -std=c++11 -o assignment6 $^

.PHONY: clean

clean:
	rm -f assignment6

# vim: noet: tabstop=4
