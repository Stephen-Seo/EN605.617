__kernel void cell_avg(__global const int *input_buffer,
                       __global float *output_buffer,
                       const int width,
                       const int size) {
  size_t id = get_global_id(0);
  size_t x = id % width;
  size_t y = id / width;

  int sum = 0;

  /*
   *  There doesn't appear to be a way to create sub-buffers for a rectangular
   *  region nor per thread of excecution. Instead, this kernel function will
   *  just sum a 2x2 region starting at the current index with wrap-around.
   */

  // first row
  sum += input_buffer[x + y * width];
  if (x + 1 >= width) {
    // wrap-around to beginning of row
    sum += input_buffer[y * width];
  } else {
    // next row item
    sum += input_buffer[x + 1 + y * width];
  }

  // second row
  if ((y + 1) * width >= size) {
    // wrap-around to first row
    sum += input_buffer[x];
    if (x + 1 >= width) {
      // wrap-around to beginning of row
      sum += input_buffer[0];
    } else {
      // next item in row
      sum += input_buffer[x + 1];
    }
  } else {
    // next row is within range
    sum += input_buffer[x + (y + 1) * width];
    if (x + 1 >= width) {
      // wrap-around to beginning of row
      sum += input_buffer[(y + 1) * width];
    } else {
      // next item in row
      sum += input_buffer[x + 1 + (y + 1) * width];
    }
  }

  output_buffer[id] = sum / 4.0F;
}
