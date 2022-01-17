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
  print_model_state(in_matrix, model_size);

  // create a new matrix that will hold the output
  // might need to dill it with zeros?

  // For loop calcs the energy and assigns result to output matrix. This shit
  // happens in a loop for k times. in the  emd of every iteration, the in and
  // out matrix pointers, swap

  return 0;
}