#ifndef IGPUP_MODULE_6_MATH_EXPRESSIONS_H
#define IGPUP_MODULE_6_MATH_EXPRESSIONS_H

__global__
void mathexpressions_registers(const int *x, const int *y, int *out);

__device__
void copy_to_shared(const int *x,
                    const int *y,
                    int *x_shared,
                    int *y_shared,
                    const unsigned thread_idx);

__device__
void copy_from_shared_out(int *out,
                          const int *out_shared,
                          const unsigned int thread_idx);

__global__
void mathexpressions_shared(const int *x, const int *y, int *out);

#endif
