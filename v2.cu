#include <stdio.h>
#include <stdlib.h>

#include "v2.h"
#include "utils.h"

__global__ void update_model_v2(int *pad_in_matrix, int *out_matrix,
                            int size) {
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  // grid-stride loop should so a thread calcs bxb momements
  for (int i = row_idx; i < size; i += row_stride) {
    for (int j = col_idx; j < size; j += col_stride) {
      out_matrix[(i * size + j] =
        calculate_moment(pad_in_matrix, size + 2, i + 1, j + 1);
    }
  }
}

// Here we implement the model where 1 threads calculates multiple moments
void ising_model_v2(int *in_matrix, int *out_matrix, int size, int bsize,
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

  // Max number of moments that each threat will compute
  int moments_per_thread = bsize * bsize;

  // Find mim number of threads so every thread computes bsize x bsize moments

  int BLOCK_SIZE =
      (size + bsize - 1) / bsize;  // ceil(model_size / bsize)

  dim3 = block_dim(BLOCK_SIZE, BLOCK_SIZE);
}
