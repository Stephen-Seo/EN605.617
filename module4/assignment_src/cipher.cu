#include "cipher.h"

__global__
void caesar_cipher(char *in, char *out, int offset) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int temp = in[thread_idx];
    if (temp >= 65 && temp <= 90) {
        temp += offset - 'A';
        while (temp < 0) {
            temp += 26;
        }
        temp = (temp % 26) + 'A';
    } else {
        temp += offset - 'a';
        while (temp < 0) {
            temp += 26;
        }
        temp = (temp % 26) + 'a';
    }
    out[thread_idx] = temp;
}

// vim: cindent: ts=4: sw=4: et
