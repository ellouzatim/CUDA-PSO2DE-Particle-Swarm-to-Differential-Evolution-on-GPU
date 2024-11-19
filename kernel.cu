#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"

//constantes sur le gpu
__device__ __constant__ float d_F;
__device__ __constant__ float d_CR;
__device__ __constant__ float d_ranges[2];

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
*/

__device__ float device_fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC) {
        case 0: {
            float y1 = 1 + (x[0] - 1) / 4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;
            res += pow(sin(phi * y1), 2);
            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float y = 1 + (x[i] - 1) / 4;
                float yp = 1 + (x[i + 1] - 1) / 4;
                res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10 * cos(2 * phi * zi) + 10;
            }
            res -= 330;
            break;
        }
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i + 1] - 0 + 1;
                res += 100 * (pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2) / 4000;
                produit *= cos(zi / pow(i + 1, 0.5));
            }
            res = somme - produit + 1 - 180;
            break;
        case 4:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }
    return res;
}

__global__ void setupCurand(curandState *states, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_OF_POPULATION) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void kernelInitializePopulation(float* population, float* fitness, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION) return;

    curandState localState = states[idx];
    float individual[NUM_OF_DIMENSIONS];

    
    for(int d = 0; d < NUM_OF_DIMENSIONS; d++) {
        float random_value = d_ranges[0] + curand_uniform(&localState) * (d_ranges[1] - d_ranges[0]);
        population[idx * NUM_OF_DIMENSIONS + d] = random_value;
        individual[d] = random_value;
    }

    fitness[idx] = device_fitness_function(individual);
    states[idx] = localState;
}

__global__ void kernelPrepareMutation(int* indexMutation, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION) return;
    
    curandState localState = states[idx];
    bool used[NUM_OF_POPULATION] = {false};
    used[idx] = true;
    
    for(int i = 0; i < 3; i++) {
        int randIdx;
        do {
            randIdx = (int)(curand_uniform(&localState) * NUM_OF_POPULATION);
        } while(used[randIdx]);
        
        indexMutation[idx * 3 + i] = randIdx;
        used[randIdx] = true;
    }
    
    states[idx] = localState;
}

__global__ void kernelDEMutation(float* __restrict__ population, 
                                const int* __restrict__ indexMutation, 
                                float* __restrict__ mutants) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS) return;
    
    int individual = idx / NUM_OF_DIMENSIONS;
    int dimension = idx % NUM_OF_DIMENSIONS;
    
    int r1 = indexMutation[individual * 3];
    int r2 = indexMutation[individual * 3 + 1];
    int r3 = indexMutation[individual * 3 + 2];
    
    float x_r1 = population[r1 * NUM_OF_DIMENSIONS + dimension];
    float x_r2 = population[r2 * NUM_OF_DIMENSIONS + dimension];
    float x_r3 = population[r3 * NUM_OF_DIMENSIONS + dimension];
    
    float mutant = x_r1 + d_F * (x_r2 - x_r3);
    mutants[idx] = fmin(fmax(mutant, d_ranges[0]), d_ranges[1]);
}

__global__ void kernelCrossoverAndSelection(
    float* population,  
    float* mutants,
    float* fitness,
    int jrand,
    curandState* states) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION) return;
    
    float trial[NUM_OF_DIMENSIONS];
    curandState localState = states[idx];
    
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        float randj = curand_uniform(&localState);
        trial[j] = (randj <= d_CR || j == jrand) ? mutants[idx * NUM_OF_DIMENSIONS + j] : population[idx * NUM_OF_DIMENSIONS + j];
    }
    
    float trial_fitness = device_fitness_function(trial);
    float current_fitness = fitness[idx];
    
    if (trial_fitness < current_fitness) {
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            population[idx * NUM_OF_DIMENSIONS + j] = trial[j];
        }
        fitness[idx] = trial_fitness;
    }
    
    states[idx] = localState;
}

extern "C" void cuda_de(float *population, float *gBest) {
    //constantes sur cpu
    float h_ranges[2] = {START_RANGE_MIN, START_RANGE_MAX};
    float h_F = F;
    float h_CR = CR;
    //cpy cpu vers gpu
    cudaMemcpyToSymbol(d_ranges, h_ranges, sizeof(float) * 2);
    cudaMemcpyToSymbol(d_F, &h_F, sizeof(float));
    cudaMemcpyToSymbol(d_CR, &h_CR, sizeof(float));
    
    size_t size = NUM_OF_POPULATION * NUM_OF_DIMENSIONS;
    float *d_population, *d_mutants, *d_fitness;
    int *d_indexMutation;
    curandState *d_states;
    
    cudaMalloc(&d_population, size * sizeof(float));
    cudaMalloc(&d_mutants, size * sizeof(float));
    cudaMalloc(&d_fitness, NUM_OF_POPULATION * sizeof(float));
    cudaMalloc(&d_indexMutation, NUM_OF_POPULATION * 3 * sizeof(int));
    cudaMalloc(&d_states, NUM_OF_POPULATION * sizeof(curandState));
    
    int threadsNum = 256;
    int blockPop = (size + threadsNum - 1) / threadsNum;
    int blockInd = (NUM_OF_POPULATION + threadsNum - 1) / threadsNum;

    // Initialisation
    setupCurand<<<blockInd, threadsNum>>>(d_states, time(NULL));
    kernelInitializePopulation<<<blockInd, threadsNum>>>(d_population, d_fitness, d_states);
    cudaMemcpy(gBest, d_population, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    
    float *h_temp = new float[size];
    float best_fitness = INFINITY;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
      int jrand = rand() % NUM_OF_DIMENSIONS;
      
      kernelPrepareMutation<<<blockInd, threadsNum>>>(d_indexMutation, d_states);
      kernelDEMutation<<<blockPop, threadsNum>>>(d_population, d_indexMutation, d_mutants);
      kernelCrossoverAndSelection<<<blockInd, threadsNum>>>(d_population, d_mutants, d_fitness, jrand, d_states);
    
      cudaMemcpy(h_temp, d_population, size * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < NUM_OF_POPULATION; i++) {
          float fitness = host_fitness_function(&h_temp[i * NUM_OF_DIMENSIONS]);
          if (fitness < best_fitness) {
              best_fitness = fitness;
              memcpy(gBest, &h_temp[i * NUM_OF_DIMENSIONS], NUM_OF_DIMENSIONS * sizeof(float));
          }
      }
    }
    
    delete[] h_temp;
    cudaMemcpy(population, d_population, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_population);
    cudaFree(d_mutants);
    cudaFree(d_fitness);
    cudaFree(d_indexMutation);
    cudaFree(d_states);
}