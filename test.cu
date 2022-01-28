#include <stdio.h>
#include <stdlib.h>

#include "utils.h"


// Kernel to add padding in a given matrix (for handling boundaries conditions)
__global__ void add_halo(int *matrix, int size, int *pad_matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < size && j < size) {
    // Copy elements from matrix to padded matrix
    pad_matrix[(i + 1) * (size + 2) + (j + 1)] = matrix[i * size + j];
  
    // This may not be a best practice.
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

int main(int argc, char **argv) {
  printf("Test program\n");

  int model_size = atoi(argv[1]);

  int *in_matrixi = initialize_model(model_size);
  int *out_matrix = (int *)malloc(model_size * model_size * sizeof(int));

  printf("Initial matrix\n");
  print_model_state(in_matrix, model_size);

  int *in_matrix_d;
  int *pad_in_matrix_d;
  int *out_matrix_d;

  int matrix_bytes = model_size * model_size * sizeof(int);
  int pad_matrix_bytes = (model_size + 1) * (model_size + 1) * sizeof(int);

  int *pad_in_matrix = (int *)malloc(pad_matrix_bytes);
  int *pad_in_matrix_host = add_halo(in_matrix, model_size);

  printf("Matrix with halo from host\n");
  print_model_state(pad_in_matrix, model_size + 2);

  // Allocate memory for device
  cudaMalloc((void **)&in_matrix_d, matrix_bytes);
  cudaMalloc((void **)&out_matrix_d, matrix_bytes);
  cudaMalloc((void **)&pad_in_matrix_d, pad_matrix_bytes);

  // Copy data to device
  cudaMemcpy(in_matrix_d, in_matrix, matrix_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out_matrix_d, out_matrix, matrix_bytes, cudaMemcpyHostToDevice);

  // Block size etc
  int BLOCK_SIZE = 4; 
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

  int GRID_SIZE = (model_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid_dim(GRID_SIZE, GRID_SIZE);

  // Launch kernel
  add_halo<<<grid_dim, block_dim>>>(in_matrix_d, model_size, pad_in_matrix_d);
  cudaMemcpy(pad_in_matrix, pad_in_matrix_d, pad_matrix_bytes, cudaMemcpyDeviceToHost);

  printf("Padded matrix via kernel launch\n");
  print_model_state(pad_in_matrix, model_size + 2);

  return 0;
}
