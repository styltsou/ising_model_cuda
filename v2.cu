#include <stdio.h>
#include <stdlib.h>

#include "v2.h"
#include "utils.h"

__global__ void update_model_v2(int *pad_in_matrix, int *out_matrix,
                            int size) {
}

// Here we implement the model where 1 threads calculates multiple moments
int *ising_model_v2(int *in_matrix, int size, int tile_width,
                    int num_iterations) {
  int *out_matrix = (int *)malloc(size * size, sizeof(int));

  // Allocate memory for device copies
  int matrix_bytes = size * model_size * sizeof(int);
  int pad_matrix_bytes = (size + 2) * (size + 2) * sizeof(int);

  int *in_matrix_d;
  int *pad_in_matrix_d;
  int *out_matrix_d;

  cudaMalloc((void **)&in_matrix_d, matrix_bytes);
  cudaMalloc((void **)&pad_in_matrix_d, pad_matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);

  // Copy data to device
  cudaMemcpy(in_matrix_d, in_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Calculate block and grid dimensions
  // Find way to calc depending on the tile_width and matrix size
  dim3 block_dim(1, 1);
  dim3 grid_dim(1, 1);

  int k = 0;
  while (k < num_iterations) {
    // Add halo to matrix
      // (Dont know how to launch the previous kernel now)
    update_model_v2<<<grid_dim, block_dim>>>(pad_in_matrix_d, out_matrix_d, size);

    swap_matrices(&in_matrix_d, &out_matrix_d);
    k++;
  }

  cudaMemcpy(out_matrix, in_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

  cudaFree(in_matrix_d);
  cudaFree(pad_in_matrix_d);
  cudaFree(out_matrix_d);

  return out_matrix;
}
