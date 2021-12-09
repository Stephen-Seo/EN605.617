__kernel void Kernel(__global const unsigned int *input,
                     volatile __global unsigned int *shared) {
  atomic_add(shared, *input);
}
