#include <stdio.h>
#include <stdlib.h>

// in and out matrices are padded to avoid control statements for boundaries
__global__ void calc_moment(int *pad_in_matrix, int *pad_out_matrix, int size) {}


void ising_model_v1(int *in_matrix, int *out_matrix, int model_size, int block_size) {
  // Here we implement the model where 1 threads calculates multiple moments
}
