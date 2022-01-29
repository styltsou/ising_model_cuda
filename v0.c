#include "v0.h"

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

void update_model_v0(int *in_matrix, int *out_matrix, int size) {
  // Add padding to input matrix
  int *pad_in_matrix = add_halo_host(in_matrix, size);

  // Don't calculate moments for padded matrix boundaries
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      out_matrix[i * size + j] =
          calculate_moment(pad_in_matrix, size + 2, i + 1, j + 1);

  free(pad_in_matrix);
}

int *ising_model_v0(int *in_matrix, int size, int num_iterations) {
  int *out_matrix = (int *)malloc(size * size * sizeof(int));

  int k = 0;

  while (k < num_iterations) {
    update_model_v0(in_matrix, out_matrix, size);
    swap_matrices(&in_matrix, &out_matrix);
    k++;
  }

  // After every swap, in_matrix contains the actual output
  return in_matrix;
}
