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

  // Define the kernel to calculate a moment per thread
__global__ void calc_moments(int *pad_in_matrix, int *out_matrix,
                             int model_size) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Computation must not be performed on the border elements
  if (i < model_size && j < model_size) {
    // calcualte moment and update out matrix
    //out_matrix[i * model_size + j] =
        //calculate_moment(pad_in_matrix, model_size + 2, i + 1, j + 1);
  }
}

void ising_model_v1(int *in_matrix, int *out_matrix, int model_size,
                    int num_iterations) {
  // Allocate memory for device copies
  int matrix_bytes = model_size * model_size * sizeof(int);
  int pad_matrix_bytes = (model_size + 1) * (model_size + 1) * sizeof(int);

  int *in_matrix_d;
  int *pad_in_matrix_d;
  int *out_matrix_d;

  cudaMalloc((void **)&in_matrix_d, matrix_bytes);
  cudaMalloc((void **)&pad_in_matrix_d, pad_matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);

  // Copy data to device
  cudaMemcpy(in_matrix_d, in_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Calculate grid dimensions
  int BLOCK_SIZE = 32;  // So a block contains 1024 threads
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

  int GRID_SIZE = (model_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid_dim(GRID_SIZE, GRID_SIZE);

  int k = 0;
  while (k < num_iterations) {
    // 1. Launch kernel to pad the matrix
    add_halo<<<grid_dim, block_dim>>>(in_matrix_d, model_size, pad_in_matrix_d);
    // 2. Now that we have the padded matrix, launch kernel to calc moments
    calc_moments<<<grid_dim, block_dim>>>(pad_in_matrix_d, out_matrix_d,
                                          model_size);
    // 3. Swap in and out matrices (device copies)
    swap_matrices(&in_matrix_d, &out_matrix_d);
    k++;
  }

  // if number of iteration is even, then in_matrix_d contains the actual output
  if (num_iterations % 2 == 0) {
    cudaMemcpy(out_matrix, in_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(out_matrix, out_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);
  }

  // Device cleanup
  cudaFree(in_matrix_d);
  cudaFree(pad_in_matrix_d);
  cudaFree(out_matrix_d);
}
