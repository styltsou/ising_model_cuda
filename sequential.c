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
  // int **in_matrix = init_ising_model(model_size);
  // print_model_state(in_matrix, model_size);

  // int **A = (int **)malloc(model_size * sizeof(int *));
  // for (int i = 0; i < model_size; i++)
  //   A[i] = (int *)calloc(model_size, sizeof(int));

  // int **B = (int **)malloc(model_size * sizeof(int *));
  // for (int i = 0; i < model_size; i++)
  //   B[i] = (int *)malloc(model_size * sizeof(int));

  // for (int i = 0; i < model_size; i++)
  //   for (int j = 0; j < model_size; j++) B[i][j] = 1;

  int *A = (int *)calloc(model_size, sizeof(int));
  int *B = (int *)malloc(model_size * sizeof(int));
  for (int i = 0; i < model_size; i++) B[i] = 1;

  printf("Array A\n");
  // print_matrix(A, model_size);
  print_array(A, model_size);
  printf("\nArray B\n");
  // print_matrix(B, model_size);
  print_array(B, model_size);

  // swap_matrices(A, B);
  swap_arrays(&A, &B);
  printf("\nMatrices swapped\n");

  printf("Array A\n");
  // print_matrix(A, model_size);
  print_array(A, model_size);
  printf("\nArray B\n");
  // print_matrix(B, model_size);
  print_array(B, model_size);

  // create a new matrix that will hold the output
  // might need to dill it with zeros?

  // For loop calcs the energy and assigns result to output matrix. This shit
  // happens in a loop for k times. in the  emd of every iteration, the in and
  // out matrix pointers, swap

  return 0;
}