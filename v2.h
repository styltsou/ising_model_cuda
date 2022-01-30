#ifndef V2_H
#define V2_H

__device__ int calculate_moment_v2(int *matrix, int size, int i, int j);

__global__ void add_halo_v2(int *in_matrix, int size, int tile_width,
                            int *pad_in_matrix);

__global__ void update_model_v2(int *pad_in_matrix, int *out_matrix, int size);

int *ising_model_v2(int *in_matrix, int size, int tile_width, int num_iterations);

#endif
