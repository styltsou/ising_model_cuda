#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v3.h"

__device__ void print_model(int *matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) printf("%2d ", matrix[i * size + j]);
    printf("\n");
  }

  printf("\n");
}

__device__ int calculate_moment_v3(int *matrix, int size, int i, int j) {
  int sign = matrix[(i - 1) * size + j] + matrix[(i + 1) * size + j] +
             matrix[i * size + j] + matrix[i * size + (j - 1)] +
             matrix[i * size + (j + 1)];

  return sign > 0 ? 1 : -1;
}

__global__ void add_halo_v3(int *matrix, int size, int tile_width,
                            int *pad_matrix) {
  int row_start = blockIdx.y * tile_width;
  int row_end = row_start + tile_width;
  int col_start = blockIdx.x * tile_width;
  int col_end = col_start + tile_width;

  for (int i = row_start; i < row_end; i++) {
    for (int j = col_start; j < col_end; j++) {
      if (i < size && j < size) {
        // Copy elements from matrix to padded matrix
        pad_matrix[(i + 1) * (size + 2) + (j + 1)] = matrix[i * size + j];

        if (i == 0)
          pad_matrix[(size + 2) * (size + 1) + (j + 1)] = matrix[size * i + j];
        if (i == size - 1) pad_matrix[j + 1] = matrix[size * i + j];
        if (j == 0)
          pad_matrix[(size + 2) * (i + 1) + (size + 1)] = matrix[size * i + j];
        if (j == size - 1)
          pad_matrix[(size + 2) * (i + 1)] = matrix[size * i + j];
      }
    }
  }
}

__global__ void update_model_v3(int *pad_in_matrix, int *out_matrix, int size,
                                int tile_width) {
  extern __shared__ int shared_mem[];

  int row_start = blockIdx.y * tile_width;
  int row_end = row_start + tile_width;
  int col_start = blockIdx.x * tile_width;
  int col_end = col_start + tile_width;

  // find a way to map global to local indexes

  // Fill the shared memory here
  for (int i = 0; i < tile_width + 2; i++) {
    for (int j = 0; j < tile_width + 2; j++) {
      if (row_start + i < size + 2 && col_start + j < size + 2) {
        shared_mem[i * (tile_width + 2) + j] =
            pad_in_matrix[(row_start + i) * (size + 2) + (col_start + j)];
      }
    }
  }

  // Calculate moments using data from shared memory
  // FInd the correct indexes in the function call
  // for (int i = row_start; i < row_end; i++)
  //   for (int j = col_start; j < col_end; j++)
  //     if (i < size && j < size)
  //       out_matrix[i * size + j] =
  //           calculate_moment_v2(shared_mem, tile_width + 2, i + 1, j + 1);

  if (blockIdx.y == 0 && blockIdx.x == 0)
    print_model(shared_mem, tile_width + 2);
}

// A thread calculates a tile of moments
int *ising_model_v3(int *in_matrix, int size, int tile_width,
                    int num_iterations) {
  int *out_matrix = (int *)malloc(size * size * sizeof(int));

  // Allocate memory for device copies
  int matrix_bytes = size * size * sizeof(int);
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
  dim3 block_dim(1, 1);

  int GRID_SIZE = (size + tile_width - 1) / tile_width;
  dim3 grid_dim(GRID_SIZE, GRID_SIZE);

  int shared_mem_bytes = (tile_width + 2) * (tile_width + 2) * sizeof(int);

  int k = 0;
  while (k < num_iterations) {
    add_halo_v3<<<grid_dim, block_dim>>>(in_matrix_d, size, tile_width,
                                         pad_in_matrix_d);

    update_model_v3<<<grid_dim, block_dim, shared_mem_bytes>>>(
        pad_in_matrix_d, out_matrix_d, size, tile_width);

    swap_matrices(&in_matrix_d, &out_matrix_d);
    k++;
  }

  cudaMemcpy(out_matrix, in_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

  cudaFree(in_matrix_d);
  cudaFree(pad_in_matrix_d);
  cudaFree(out_matrix_d);

  return out_matrix;
}
