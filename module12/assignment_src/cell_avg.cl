__kernel void cell_avg(__global const int *input_buffer,
                       __global float *output_buffer,
                       const int in_size,
                       const int idx) {
  int sum = 0;

  for (int i = 0; i < in_size; ++i) {
    sum += input_buffer[i];
  }

  output_buffer[idx] = sum / (float)in_size;
}
