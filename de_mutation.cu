#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernel.h"


__global__ void kernelDEMutation(float *individuals, float *indexMutation, float *mutants, float F, int populationSize, int dimensions)
{
    extern __shared__ float sharedMem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= populationSize) return;

    // Chargement indices en mémoire partagée
    __shared__ int sharedIndices[3];
    if (tid < 3) {
        sharedIndices[tid] = (int)indexMutation[blockIdx.x * 3 + tid];
    }
    __syncthreads();

    int r_base = sharedIndices[0];
    int r_1 = sharedIndices[1];
    int r_2 = sharedIndices[2];

    if (r_base < 0 || r_base >= populationSize ||
        r_1 < 0 || r_1 >= populationSize ||
        r_2 < 0 || r_2 >= populationSize) {
        return;
    }

    // Chargement mémoire partagée
    float *sharedBase = sharedMem;
    float *shared1 = &sharedMem[dimensions];
    float *shared2 = &sharedMem[2 * dimensions];

    for (int d = tid; d < dimensions; d += blockDim.x) {
        sharedBase[d] = individuals[r_base * dimensions + d];
        shared1[d] = individuals[r_1 * dimensions + d];
        shared2[d] = individuals[r_2 * dimensions + d];
    }
    __syncthreads();

    // Mutation DE
    for (int d = tid; d < dimensions; d += blockDim.x) {
        int currentIdx = idx * dimensions + d;
        mutants[currentIdx] = sharedBase[d] + F * (shared1[d] - shared2[d]);
    }
}

// Fonction wrapper pour lancer le kernel
extern "C" void performDEMutation(float *d_individuals, float *d_indexMutation, float *d_mutants, int populationSize, int dimensions, float F, cudaStream_t stream)
{
    dim3 blockSize(256);  // a voir
    dim3 gridSize((populationSize + blockSize.x - 1) / blockSize.x);
    
    int sharedMemSize = 3 * dimensions * sizeof(float);

    kernelDEMutation<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_individuals, d_indexMutation, d_mutants, F, populationSize, dimensions);
}