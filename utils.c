#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_array(int *arr, int size) {
  for (int i = 0; i < size; i++) printf("%d ", arr[i]);
}

void print_model_state(int **matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) printf("%2d ", matrix[i][j]);
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

int **pad_matrix(int **matrix, int size) {
  int **pad_mat = (int **)malloc((size + 2) * sizeof(int *));

  for (int i = 0; i < size + 2; i++) {
    pad_mat[i] = (int *)calloc(size + 2, sizeof(int));
  }

  // Copy elements to pad_mat
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) pad_mat[i + 1][j + 1] = matrix[i][j];

  // Top padding
  for (int i = 0; i < size; i++) pad_mat[0][i + 1] = matrix[size - 1][i];

  // Right padding
  for (int i = 0; i < size; i++) pad_mat[i + 1][size + 1] = matrix[i][0];

  // Bottom padding
  for (int i = 0; i < size; i++) pad_mat[size + 1][i + 1] = matrix[0][i];

  // Left padding
  for (int i = 0; i < size; i++) pad_mat[i + 1][0] = matrix[i][size - 1];

  return pad_mat;
}

int calculate_moment(int **matrix, int i, int j) {
  int sign = matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j] +
             matrix[i][j - 1] + matrix[i][j + 1];

  if (sign > 0)
    return 1;
  else if (sign < 0)
    return -1;

  return matrix[i][j];
}

void update_ising_model(int **in_matrix, int **out_matrix, int size) {
  // Add padding to input matrix
  int **padded_in_matrix = pad_matrix(in_matrix, size);

  // The matrix is padded now (dont calc moments for currnet borders)
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      out_matrix[i][j] = calculate_moment(padded_in_matrix, i + 1, j + 1);

  free(padded_in_matrix);
}

void swap_matrices(int ***A, int ***B) {
  int **tmp = *A;
  *A = *B;
  *B = tmp;
}

void ising_model(int **in_matrix, int **out_matrix, int size,
                 int num_iterations) {
  int k = 0;

  while (k < num_iterations) {
    update_ising_model(in_matrix, out_matrix, size);
    swap_matrices(&in_matrix, &out_matrix);
    k++;
  }

  if (num_iterations % 2 == 0) swap_matrices(&in_matrix, &out_matrix);
}

int matrix_total_sum(int **matrix, int size) {
  int count = 0;
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) count += matrix[i][j];
  return count;
}