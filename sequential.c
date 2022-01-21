#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

int main(int argc, char **argv) {
  // Get model size n and number of iterations k from argv
  if (argc != 3) {
    fprintf(stderr, "Invalid number of arguments");
    exit(1);
  }

  int model_size = atoi(argv[1]);
  int num_iterations = atoi(argv[2]);

  // Create an initital matrix with a uniform distribution
  int **in_matrix = init_ising_model(model_size);
  // print_model_state(in_matrix, model_size);

  int **out_matrix = (int **)malloc(model_size * sizeof(int *));
  for (int i = 0; i < model_size; i++)
    out_matrix[i] = (int *)calloc(model_size, sizeof(int));

  int count;
  for (int k = 0; k < num_iterations; k++) {
    update_ising_model(in_matrix, out_matrix, model_size);
    swap_matrices(&in_matrix, &out_matrix);
    count = matrix_total_sum(in_matrix, model_size);
    system("clear");
    printf("Evolution : [%d/%d] | count : %d\n", k, num_iterations, count);
  }

  // If number of iterations is odd then in_matrix contains the actual output
  if (num_iterations % 2 == 0) swap_matrices(&in_matrix, &out_matrix);

  return 0;
}