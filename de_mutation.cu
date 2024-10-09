#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernel.h"


__global__ void kernelDEMutation(float *individuals, int *indexMutation, float *mutants, float F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0) return;

    int idx = i / NUM_OF_DIMENSIONS;  

    int r_base = indexMutation[idx * 3];
    int r_1 = indexMutation[idx * 3 + 1];
    int r_2 = indexMutation[idx * 3 + 2];

    if (r_base < 0 || r_base >= NUM_OF_PARTICLES ||
        r_1 < 0 || r_1 >= NUM_OF_PARTICLES ||
        r_2 < 0 || r_2 >= NUM_OF_PARTICLES) {
        return;
    }

    for (int d = 0; d < NUM_OF_DIMENSIONS; d++) {
        int currentIdx = i + d;
        mutants[currentIdx] = individuals[r_base * NUM_OF_DIMENSIONS + d] + 
                              F * (individuals[r_1 * NUM_OF_DIMENSIONS + d] - individuals[r_2 * NUM_OF_DIMENSIONS + d]);
    }

}
