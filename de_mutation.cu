#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernel.h"

__global__ void kernelDEMutation(float *individuals, int *indexMutation, float *mutants, float F) {
    extern __shared__ float sharedMem[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_PARTICLES) return;
    
    int r_base = indexMutation[i * 3];
    int r_1 = indexMutation[i * 3 + 1];
    int r_2 = indexMutation[i * 3 + 2];
    
    float *base = &sharedMem[threadIdx.x * NUM_OF_DIMENSIONS * 3];
    float *x_r1 = &base[NUM_OF_DIMENSIONS];
    float *x_r2 = &x_r1[NUM_OF_DIMENSIONS];
    
    for (int d = 0; d < NUM_OF_DIMENSIONS; d++) {
        base[d] = individuals[r_base * NUM_OF_DIMENSIONS + d];
        x_r1[d] = individuals[r_1 * NUM_OF_DIMENSIONS + d];
        x_r2[d] = individuals[r_2 * NUM_OF_DIMENSIONS + d];
    }
    
    __syncthreads();
    
    for (int d = 0; d < NUM_OF_DIMENSIONS; d++) {
        mutants[i * NUM_OF_DIMENSIONS + d] = base[d] + F * (x_r1[d] - x_r2[d]);
    }
}