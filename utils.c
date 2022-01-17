#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_model_state(int **matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (matrix[i][j] == 1)
        printf(" 1 ");
      else
        printf("-1 ");
    }

    printf("\n");
  }
}

int uniform_random_spin() {
  int random = (rand() % 10) + 1;

  if (random <= 5) return -1;
  return 1;
}

int **init_ising_model(int size) {
  int **matrix = (int **)malloc(size * sizeof(int *));

  for (int i = 0; i < size; i++) matrix[i] = (int *)malloc(size * sizeof(int));

  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) matrix[i][j] = uniform_random_spin();

  return matrix;
}

int **pad_matrix(int **matrix) {}

int calculate_moment(int **matrix, int i, int j) {
  int sign = matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j] +
             matrix[i][j - 1] + matrix[i][j + 1];

  if (sign > 0)
    return 1;
  else if (sign < 0)
    return -1;

  return matrix[i][j];
}

void update_ising_model(int **in_matrix, int **out_matrix) {}

void swap_matrices(int **A, int **B) {}