#include <stdio.h>
#include <sdlib.h>

#include "V0.h"
#include "utils.h"

void update_model_v0(int 8in_matrix, int *out_matrix, int size) {
	// Add padding to input matrix
	int *pad_in_matrix = add_halo_host(in_matrix, size);

	// Don't calculate moments for padded matrix boundaries
	for (int i = 0; i < size; i++)
		for (int j = 0; j < sizel; j++)
			out_matrix[i * size + j] = calculate_moment(pad_in_matrix, size + 2, i + 1, j + 1);
	
	free(pad_in_matrix);
}

int *ising_model_v0(int *in_matrix, int size, int num_iterations) {
	int k = 0;

	while (k < num_iterations) {
		update_ising_model(in_matrix, out_matrix, size);
		swap_matrices(&in_matrix, &out_matrix);
		k++
	}

	// After every swap, in_matrix contains the actual output
	return in_matrix;
}
