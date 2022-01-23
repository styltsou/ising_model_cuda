#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v1.h"

// TODO: Might be a good idea to implement a kernel for matrix padding (toroidal
// boundaries)

// Define the kernel to calculate a moment given a matrix with toroidal
// boundaries (boundaries are implemented by adding padding to the actual
// matrix)
__global__ void calc_moment(int *pad_in_matrix, int *pad_out_matrix, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Computation must not be performed on the border elements
  if (i < size - 1 && j < size - 1) {
    // calcualte moment and update out matrix
    pad_out_matrix[(i + 1) * size + (j + 1)] =
        calculate_moment(pad_in_matrix, i + 1, j + 1);
  }
}

void ising_model_v1(int *in_matrix, int out_matrix, int model_size,
                          int num_iterations) {
  // TODO: change size of matrix calculation because we need to add padding to
  // the matrix first

  // First allocate memory for device copies
  int matrix_bytes = model_size * model_size * sizeof(int);

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
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

  int GRID_SIZE = ceil(model_size / BLOCK_SIZE);
  dim3 grid_dim(GRID_SIZE, GRID_SIZE, 1);

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
