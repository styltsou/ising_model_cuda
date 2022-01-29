#ifndef UTILS_H
#define UTILS_H

void print_model_state(int *matrix, int size);

int uniform_random_spin();

int *init_ising_model(int size);

int *deepcopy_matrix(int *matrix, int size);

int *add_halo(int *matrix, int size);

int calculate_moment(int *matrix, int size, int i, int j);

void swap_matrices(int **A, int **B);

int compare_matrices(int *A, int *B, int size);

#endif
