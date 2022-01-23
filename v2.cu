#include <stdio.h>
#include <stdlib.h>

// in and out matrices are padded to avoid control statements for boundaries
__global__ void calc_moment(int *pad_in_matrix, int *pad_out_matrix, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // a for loop should go here so a threads calc bxb momements
}


void ising_model_v1(int *in_matrix, int *out_matrix, int model_size, int block_size, int num_iterations) {
  // Here we implement the model where 1 threads calculates multiple moments
}
