#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "v0.h"
#include "v1.h"
#include "v2.h"
#include "v3.h"

int main(int argc, char **argv) {
  // Get model size n and number of iterations  from argv
  if (argc != 4) {
    fprintf(stderr, "Invalid number of arguments");
    exit(1);
  }

  int model_size = atoi(argv[1]);
  int tile_width = atoi(argv[2]);
  int num_iterations = atoi(argv[3]);

  // Create an initital state matrix with a uniform distribution
  int *in_matrix_v0 = init_ising_model(model_size);
  int *in_matrix_v1 = deepcopy_matrix(in_matrix_v0, model_size);
  int *in_matrix_v2 = deepcopy_matrix(in_matrix_v0, model_size);
  int *in_matrix_v3 = deepcopy_matrix(in_matrix_v0, model_size);

  print_model_state(in_matrix_v0, model_size);
  printf("\n");

  int *out_matrix_v0 = ising_model_v0(in_matrix_v0, model_size, num_iterations);
  int *out_matrix_v1 = ising_model_v1(in_matrix_v1, model_size, num_iterations);
  int *out_matrix_v2 =
      ising_model_v2(in_matrix_v2, model_size, tile_width, num_iterations);

  int *out_matrix_v3 =
      ising_model_v3(in_matrix_v3, model_size, tile_width, num_iterations);

  if (compare_matrices(out_matrix_v0, out_matrix_v1, model_size)) {
    printf("\nV0 == V1\n");
  } else {
    printf("\nV0 != V1\n");
  }

  if (compare_matrices(out_matrix_v0, out_matrix_v2, model_size)) {
    printf("V0 == V2\n");
  } else {
    printf("V0 != V2\n");
  }

  if (compare_matrices(out_matrix_v0, out_matrix_v3, model_size)) {
    printf("V0 == V3\n");
  } else {
    ("V0 != V3\n");
  }

  // Clean up
  // free(in_matrix_v0);
  // free(out_matrix_v0);
  // free(in_matrix_v1);
  // free(out_matrix_v1);
  // free(in_matrix_v2);
  // free(out_matrix_v2);

  return 0;
}
