#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernel.h"


__global__ void kernelDEMutation(float *individuals, float *indexMutation, float *mutants, float F, int individualsSize, int dimensions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= individualsSize * dimensions || i % dimensions != 0) return;

    int idx = i / dimensions;  // Indice de l'individu

    int r_base = (int)indexMutation[idx * 3];
    int r_1 = (int)indexMutation[idx * 3 + 1];
    int r_2 = (int)indexMutation[idx * 3 + 2];

    if (r_base < 0 || r_base >= individualsSize ||
        r_1 < 0 || r_1 >= individualsSize ||
        r_2 < 0 || r_2 >= individualsSize) {
        return;
    }

    for (int d = 0; d < dimensions; d++) {
        int currentIdx = i + d;
        mutants[currentIdx] = individuals[r_base * dimensions + d] + 
                              F * (individuals[r_1 * dimensions + d] - individuals[r_2 * dimensions + d]);
    }
}

// Fonction wrapper pour lancer le kernel
extern "C" void performDEMutation(float *d_individuals, float *d_indexMutation, float *d_mutants, int individualsSize, int dimensions, float F, cudaStream_t stream)
{
    dim3 blockSize(256);  // a voir
    dim3 gridSize((individualsSize + blockSize.x - 1) / blockSize.x);
    
    int sharedMemSize = 3 * dimensions * sizeof(float);

    kernelDEMutation<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_individuals, d_indexMutation, d_mutants, F, individualsSize, dimensions);
}