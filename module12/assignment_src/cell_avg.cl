__kernel void cell_avg(__global const int *input_buffer,
                       __global float *output_buffer,
                       const int sub_size) {
  size_t id = get_global_id(0);

  int sum = 0;

  for (int i = 0; i < sub_size; ++i) {
    sum += input_buffer[id + i];
  }

  output_buffer[id] = sum / (float)sub_size;
}
