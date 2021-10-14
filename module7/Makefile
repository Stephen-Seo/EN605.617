NVCC?=/usr/bin/env nvcc

SOURCES = \
	src/arg_parse.cpp \
	src/main.cu \
	src/kernel.cu \
	src/helpers.cu

all: assignment7

assignment7: $(SOURCES)
	$(NVCC) -g -o assignment7 -std=c++11 $(SOURCES)

.PHONY: clean

clean:
	rm -f assignment7

# vim: noet: tabstop=4: shiftwidth=4
