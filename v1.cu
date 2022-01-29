#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v1.h"

__device__ int calculate_moment_v1(int *matrix, int size, int i, int j) {
  int sign = matrix[(i - 1) * size + j] + matrix[(i + 1) * size + j] +
             matrix[i * size + j] + matrix[i * size + (j - 1)] +
             matrix[i * size + (j + 1)];

  return sign > 0 ? 1 : -1;
}

// Kernel to add padding in a given matrix (for handling boundaries conditions)
__global__ void add_halo_v1(int *matrix, int size, int *pad_matrix) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size && j < size) {
    // Copy elements from matrix to padded matrix
    pad_matrix[(i + 1) * (size + 2) + (j + 1)] = matrix[i * size + j];

    //  Prevent kernel for assigning the padding multiple times
    if (j == 0) {
      // Top padding
      pad_matrix[i + 1] = matrix[(size - 1) * size + i];
      // Right padding
      pad_matrix[(i + 1) * (size + 2) + (i + 1)] = matrix[i * size];
      // Bottom padding
      pad_matrix[(size + 1) * (size + 2) + (i + 1)] = matrix[i];
      // Left padding
      pad_matrix[(i + 1) * (size + 2)] = matrix[i * size + (size - 1)];
    }
  }
}

// Define the kernel to calculate a moment per thread
__global__ void update_model_v1(int *pad_in_matrix, int *out_matrix, int size) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for index out of bounds
  if (i < size && j < size) {
    out_matrix[i * size + j] =
        calculate_moment_v1(pad_in_matrix, size + 2, i + 1, j + 1);
  }
}

int *ising_model_v1(int *in_matrix, int size, int num_iterations) {
  int matrix_bytes = size * _size * sizeof(int);

  // Allocate memory for output matrix
  int *out_matrix = (int *)malloc(matrix_bytes);

  // Allocate memory for device copies
  int *in_matrix_d;
  int *pad_in_matrix_d;
  int *out_matrix_d;

  int pad_matrix_bytes = (size + 2) * (size + 2) * sizeof(int);

  cudaMalloc((void **)&in_matrix_d, matrix_bytes);
  cudaMalloc((void **)&pad_in_matrix_d, pad_matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);

  // Copy data to device
  cudaMemcpy(in_matrix_d, in_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Calculate grid and block  dimensions
  int BLOCK_SIZE = 32;
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

  int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid_dim(GRID_SIZE, GRID_SIZE);

  int k = 0;
  while (k < num_iterations) {
    add_halo_v1<<<grid_dim, block_dim>>>(in_matrix_d, size, pad_in_matrix_d);

    update_model_v1<<<grid_dim, block_dim>>>(pad_in_matrix_d, out_matrix_d,
                                             size);

    swap_matrices(&in_matrix_d, &out_matrix_d);
    k++;
  }

  // After the swap, the in_matrix_d contains the actual output
  cudaMemcpy(out_matrix, in_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

  // Device cleanup
  cudaFree(in_matrix_d);
  cudaFree(pad_in_matrix_d);
  cudaFree(out_matrix_d);

  return out_matrix;
}
