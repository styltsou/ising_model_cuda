#include <stdio.h>
#include <stdlib.h>

// in and out matrices are padded to avoid control statements for boundaries
__global__ void calc_moment(int *pad_in_matrix, int *out_matrix,
                            int model_size) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  // grid-stride loop should so a thread calcs bxb momements
  for (int i = row_idx; i < model_size; i += stride_x) {
    for (int j = col_idx; j < model_size; j += stride_y) {
      // Calc moment here
      out_matrix[(i * model_size + j] =
        calculate_moment(pad_in_matrix, model_size + 2, i + 1, j + 1);
    }
  }
}

// Here we implement the model where 1 threads calculates multiple moments
void ising_model_v2(int *in_matrix, int *out_matrix, int model_size, int bsize,
                    int num_iterations) {
  int *out_matrix = (int *)malloc(model_size * model_size, sizeof(int));

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

  // Max number of moments that each threat will compute
  int moments_per_thread = bsize * bsize;

  // Find mim number of threads so every thread computes bsize x bsize moments

  int BLOCK_SIZE =
      (model_size + bsize - 1) / bsize;  // ceil(model_size / bsize)

  dim3 = block_dim(BLOCK_SIZE, BLOCK_SIZE);
}
