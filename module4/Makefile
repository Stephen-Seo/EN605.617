NVCC?=/usr/bin/env nvcc

SOURCES= \
	assignment_src/constants.c \
	assignment_src/helpers.cu \
	assignment_src/mathexpressions.cu \
	assignment_src/cypher.cu \
	assignment_src/assignment.cu

all: assignment.exe

assignment.exe: $(SOURCES)
	$(NVCC) -g -o assignment.exe $^

.PHONY: clean

clean:
	rm -f assignment.exe
