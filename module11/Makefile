NVCC?=/usr/bin/env nvcc

SOURCES = \
	assignment_src/main.cc

all: assignment11

assignment11: $(SOURCES)
	$(NVCC) -g -O0 -lOpenCL -o assignment11 $^

.PHONY: clean

clean:
	rm -f assignment11

# vim: noet
