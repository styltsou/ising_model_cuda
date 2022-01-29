#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v1.h"

// Kernel to add padding in a given matrix (for handling boundaries conditions)
__global__ void add_halo(int *matrix, int size, int *pad_matrix) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size && j < size) {
    // Copy elements from matrix to padded matrix
    pad_matrix[(i + 1) * (size + 2) + (j + 1)] = matrix[i * size + j];

    // TODO:L This may not be a best practice.It might be wise to assign more
    // job to every thread.
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

// Device function to calculate the new moment of a lattice
__device__ int calculate_moment_d(int *matrix, int size, int i, int j) {
  int sign = matrix[(i - 1) * size + j] +
    matrix[(i + 1) * size + j] +
    matrix[i * size + j] +
    matrix[i * size + (j - 1)] +
    matrix[i * size + (j + 1)];

  return sing > 0 ? 1 : -1;
}

// Define the kernel to calculate a moment per thread
__global__ void update_model(int *pad_in_matrix, int *out_matrix,
                             int size) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Computation must not be performed on the border elements
  if (i < size && j < size) {
    // calcualte moment and update out matrix
    out_matrix[i * size + j] =
        calculate_moment_d(pad_in_matrix, size + 2, i + 1, j + 1);
  }
}

int *ising_model_v1(int *in_matrix, int size,
                    int num_iterations) {

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
  int BLOCK_SIZE = 32;  // So a block contains 1024 threads
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

  int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid_dim(GRID_SIZE, GRID_SIZE);

  int k = 0;
  while (k < num_iterations) {
    // 1. Launch kernel to pad the matrix
    add_halo<<<grid_dim, block_dim>>>(in_matrix_d, size, pad_in_matrix_d);
    // 2. Now that we have the padded matrix, launch kernel to calc moments
    update_model<<<grid_dim, block_dim>>>(pad_in_matrix_d, out_matrix_d,
                                          size);
    // 3. Swap in and out matrices (device copies)
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
