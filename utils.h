#ifndef UTILS_H
#define UTILS_H

void print_array(int *arr, int size);

void print_matrix(int **matrix, int size);

void print_model_state(int **matrix, int size);

int uniform_random_spin();

int **init_ising_model(int size);

int **pad_matrix(int **matrix, int size);

int calculate_moment(int **matrix, int i, int j);

void update_ising_model(int **in_matrix, int **out_matrix, int size);

void swap_matrices(int ***A, int ***B);

int matrix_total_sum(int **matrix, int size);

#endif