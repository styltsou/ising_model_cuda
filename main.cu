#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v1.h"

int main(int argc, char **argv) {
  // Get model size n and number of iterations  from argv
  if (argc != 3) {
    fprintf(stderr, "Invalid number of arguments");
    exit(1);
  }

  int model_size = atoi(argv[1]);
  int num_iterations = atoi(argv[2]);

  // Create an initital state matrix with a uniform distribution
  int *in_matrix = init_ising_model(model_size);

  int *out_matrix_v0 = (int *)malloc(model_size * model_size * sizeof(int));
  int *out_matrix_v1 = (int *)malloc(model_size * model_size * sizeof(int));

  ising_model(in_matrix, out_matrix_v0, model_size, num_iterations);
  ising_model_v1(in_matrix, out_matrix_v1, model_size, num_iterations);

  printf("Model state after %d iterations\n", num_iterations);
  print_model_state(out_matrix_v0, model_size);

  if (compare_matrices(out_matrix_v0, out_matrix_v1, model_size)) {
    printf("\nV1 is correct\n");
  } else {
    printf("\nV1 has a problem\n");
  }

  // Clean up
  free(in_matrix);
  free(out_matrix_v0);
  free(out_matrix_v1);

  return 0;
}
