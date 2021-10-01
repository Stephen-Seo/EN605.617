NVCC?=/usr/bin/env nvcc

SOURCES = \
	assignment_src/assignment.cu \
	assignment_src/mathexpressions.cu

all: assignment.exe

assignment.exe: $(SOURCES)
	$(NVCC) -g -std=c++11 -o assignment.exe $^

.PHONY: clean

clean:
	rm -f assignment.exe

# vim: noet
