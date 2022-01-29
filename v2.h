#ifndef V2_H
#define V2_H

__global__ void update_model_v2(int *pad_in_matrix, int *out_matrix. int size);

void ising_model_v2(int *in_matrix, int *out_matrix, int size, int bsize,
                    int num_iterations);

#endif
