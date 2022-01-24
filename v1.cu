#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v1.h"
 
/*
  TODO: Might be a good idea to implement a kernel for matrix padding
  instead of performing in on host
*/

__global__ void padd_matrix(int *matrix, int size, int *pad_matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Ensur the indexes stay within bounds
  if (i < size && j < size) {
    // Copy elements from matrix to padded matrix
    pad_matrix[(i + 1) * (size + 2) + (j + 1)] = matrix[i * size + j];

    // Add top padding
    pad_matrix[i + 1] = matrix[(size - 1) * size + i];
    // Add right padding
    pad_matrix[(i + 1) * (size + 2) + (i + 1)] = matrix[i * size];
    // Bottom padding
    pad_mat[(size + 1) * (size + 2) + (i + 1)] = matrix[i];
    // Add left padding
    pad_mat[(i + 1) * (size + 2)] = matrix[i * size + (size - 1)];
  }
}

// Define the kernel to calculate a moment per thread
__global__ void calc_moment(int *pad_in_matrix, int *out_matrix, int msize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Computation must not be performed on the border elements
  if (i < msize - 1 && j < msize - 1) {
    // calcualte moment and update out matrix
    pad_out_matrix[(i + 1) * msize + (j + 1)] =
        calculate_moment(pad_in_matrix, i + 1, j + 1);
  }
}

void update_ising_model(int *matrix, int *out_matrix, int model_size) {
  // function that performs the update of the model
  // pass in matrix, out matrix returns with the updated spins
  // the function also calls the kernel 
  // and handles memory copies maybe
}

void ising_model_v1(int *in_matrix, int *out_matrix, int model_size,
                          int num_iterations) {
  // Add appropriate padding to matrix to avoid checks on boundries
  int *pad_in_matrix = pad_matrix(in_matrix, model_size);
  int *out_matrix = (int *)calloc(model_size * model_size, sizeof(int));

  // Allocate memory for device copies
  int matrix_bytes = model_size * model_size * sizeof(int);
  int pad_matrix_bytes = (model_size + 1) * (model_size + 1) * sizeof(int);

  int *pad_in_matrix_d;
  int *out_matrix_d;

  cudaMalloc((void **)&pad_in_matrix_d, pad_matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);

  // Copy data to device
  cudaMemcpy(pad_in_matrix_d, pad_in_matrix, pad_matrix_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out_matrix_d, out_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Calculate grid dimensions
  int BLOCK_SIZE = 32;  // So a block contains 1024 threads
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

  int GRID_SIZE = ceil(model_size / BLOCK_SIZE);
  dim3 grid_dim(GRID_SIZE, GRID_SIZE, 1);

  int k = 0;
  while (k < num_iterations) {
    // START UPDATE ISING

    // EMD UPDATE ISING

    // SWAP HERE
  
    k++;
  }

  // Check weather in or out matrix contains the result and swap if needed

  // Copy data back from the device
  cudaMemcpy(out_matrix, out_matrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

  // Device cleanup
  cudaFree(pad_in_matrix_d);
  cudaFree(out_matrix_d);
}
