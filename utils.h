#ifndef UTILS_H
#define UTILS_H

void print_model_state(int *matrix, int size);

int uniform_random_spin();

int *init_ising_model(int size);

int *add_halo(int *matrix, int size);

int calculate_moment(int *matrix, int size, int i, int j);

void update_ising_model(int *in_matrix, int *out_matrix, int size);

void swap_matrices(int **A, int **B);

void ising_model(int *in_matrix, int *out_matrix, int size, int num_iterations);

int compare_matrices(int *A, int *B, int size);

#endif
