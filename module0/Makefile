NVCC?=/usr/local/cuda/bin/nvcc
CC?=/usr/bin/gcc
CXX?=/usr/bin/g++

CPP_SOURCES=helloworld.cpp
CPP_OBJECTS=$(addsuffix .obj,$(CPP_SOURCES))

CUDA_SOURCES=hello-world.cu
CUDA_OBJECTS=$(addsuffix .obj,$(CUDA_SOURCES))

OPENCL_SOURCES=hello_world_cl.c
OPENCL_OBJECTS=$(addsuffix .obj,$(OPENCL_SOURCES))

all: helloworld helloworld_cuda helloworld_opencl

helloworld: $(CPP_OBJECTS)
	$(CXX) -o helloworld $^

helloworld_cuda: $(CUDA_OBJECTS)
	$(NVCC) -o helloworld_cuda $<

helloworld_opencl: $(OPENCL_OBJECTS)
	$(CXX) -o helloworld_opencl $^ -lOpenCL

%.cu.obj: %.cu
	$(NVCC) -c -o $@ $<

%.c.obj: %.c
	$(CC) -c -o $@ $^

%.cpp.obj: %.cpp
	$(CXX) -c -o $@ $^

.PHONY: run.sh clean

run.sh:
	./run.sh -a

clean:
	rm -f *.obj
	rm -f helloworld
	rm -f helloworld_cuda
	rm -f helloworld_opencl
