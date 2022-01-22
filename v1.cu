#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

// Define the kernel to
__global__ void calc_moment(int *in_matrix, int *out_matrix, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < size && j < size) {
    // calcualte moment and update out matrix
  }
}

void ising_model_parallel(int *in_matrix, int out_matrix, int size,
                          int num_iterations) {
  // First allocate memory for device copies
  int matrix_bytes = size * size * sizeof(int);

  int *in_matrix_d;
  int *out_matrix_d;

  // Padded matrix allocation might be needed before
  cudaMalloc((void **)&in_matrix_d, matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);

  // Copy data to device
  cudaMemcpy(in_matrix_d, in_matrix, matrix_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out_matrix_d, out_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Calculate grid dimensions
  int BLOCK_SIZE = 32;  // So a block contains 1024 threads
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

  int GRID_SIZE = ceil(size / BLOCK_SIZE);
  dim grid_dim(GRID_SIZE, GRID_SIZE);

  // THIS WILL BE KIND OF TRICKY (MAYBE theres no need for swap)
  // Matrix pad may be needed before launcing the kernel
  // for number of iterations,
  // Check weather in or out matrix contains the result and swap if needed

  // Copy data back from the device
  cudaMemcpy(out_matrix, out_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(in_matrix_d);
  cudaFree(out_matrix_d);
}