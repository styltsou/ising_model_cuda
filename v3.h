#ifndef V3_H
#define V3_H

__device__ int calculate_moment_v3(int *matrix, int size, int i, int j);

__global__ void add_halo_v3(int *in_matrix, int size, int tile_width,
                            int *pad_in_matrix);

__global__ void update_model_v3(int *pad_in_matrix, int *out_matrix, int size);

int *ising_model_v3(int *in_matrix, int size, int tile_width,
                    int num_iterations);

#endif