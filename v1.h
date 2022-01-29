#ifndef V1_H
#define V1_H

__global__ void add_halo_kernel(int *matrix, int size, int *pad_matrix);

__global__ void update_model_v1(int *pad_in_matrix, int *out_matrix, int size);

int *ising_model_v1(int *in_matrix, int model_size, int num_iterations);

#endif
