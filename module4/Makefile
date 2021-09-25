NVCC?=/usr/bin/env nvcc

all: assignment.exe

assignment.exe: assignment_src/constants.c assignment_src/helpers.cu assignment_src/mathexpressions.cu assignment_src/assignment.cu
	$(NVCC) -o assignment.exe $^

.PHONY: clean

clean:
	rm -f assignment.exe
