#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"

__device__ float tempParticle[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

__device__ float tempParticleOld[NUM_OF_DIMENSIONS];
__device__ float tempParticleMutation[NUM_OF_DIMENSIONS];

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

// Kernel pour la fonction fitness sur GPU
__device__ float device_fitness_function(float x[]) {
    float res = 0;
    
    switch (SELECTED_OBJ_FUNC) {
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*M_PI*zi) + 10;
            }
            res -= 330;
            break;
        }
    }
    return res;
}

// Initialisation des états random
__global__ void setupCurand(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_OF_POPULATION) {
        curand_init(seed + idx, 0, 0, &state[idx]);
    }
}

// Initialisation de la population

__global__ void kernelInitializePopulation(float *population, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
        return;

    int individual_idx = idx / NUM_OF_DIMENSIONS;
    curandState localState = states[individual_idx];
   
    // Générer une valeur aléatoire
    float random_value = START_RANGE_MIN +
        curand_uniform(&localState) * (START_RANGE_MAX - START_RANGE_MIN);
   
    // Sauvegarder l'état mis à jour
    states[individual_idx] = localState;
   
    // Sauvegarder la valeur générée
    population[idx] = random_value;
}

// Préparation des indices pour la mutation
__global__ void kernelPrepareMutation(int *indexMutation, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION)
        return;
        
    curandState localState = states[idx];

    // Sélectionner 3 indices distincts différents de idx
    int indices[3];
    int count = 0;
    
    while(count < 3) {
        int randIdx = curand(&localState) % NUM_OF_POPULATION;
        if(randIdx != idx) {
            bool isDuplicate = false;
            for(int j = 0; j < count; j++) {
                if(indices[j] == randIdx) {
                    isDuplicate = true;
                    break;
                }
            }
            if(!isDuplicate) {
                indices[count] = randIdx;
                count++;
            }
        }
    }
    
    indexMutation[idx * 3] = indices[0];
    indexMutation[idx * 3 + 1] = indices[1];
    indexMutation[idx * 3 + 2] = indices[2];
    
    states[idx] = localState;
}

// Mutation DE
__global__ void kernelDEMutation(float *population, int *indexMutation, float *mutants) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION)
        return;
        
    int r1 = indexMutation[idx * 3];
    int r2 = indexMutation[idx * 3 + 1];
    int r3 = indexMutation[idx * 3 + 2];
    
    for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        float base = population[r1 * NUM_OF_DIMENSIONS + j];
        float diff = population[r2 * NUM_OF_DIMENSIONS + j] - 
                    population[r3 * NUM_OF_DIMENSIONS + j];
        
        float mutated = base + F * diff;
        
        // Borner les valeurs
        if(mutated > START_RANGE_MAX) mutated = START_RANGE_MAX;
        if(mutated < START_RANGE_MIN) mutated = START_RANGE_MIN;
        
        mutants[idx * NUM_OF_DIMENSIONS + j] = mutated;
    }

    // if (idx == 0) {
    // printf("\n=== Mutation de l'individu 0 ===\n");
    // printf("r1=%d, r2=%d, r3=%d\n", r1, r2, r3);
    // printf("Position originale: ");
    // for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
    //     printf("%.2f ", population[idx * NUM_OF_DIMENSIONS + j]);
    // }
    // printf("\nPosition après mutation: ");
    // for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
    //     printf("%.2f ", mutants[idx * NUM_OF_DIMENSIONS + j]);
    // }
    // printf("\n");
// }
}

// Croisement DE
__global__ void kernelCrossoverDE(float *population, float *mutants, 
                                 int jrand, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_POPULATION)
        return;

    curandState localState = states[idx];
    
    for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        float randj = curand_uniform(&localState);
        if(randj < CR || j == jrand) {
            population[idx * NUM_OF_DIMENSIONS + j] = 
                mutants[idx * NUM_OF_DIMENSIONS + j]; 
        }
    }
    // if (idx == 0) {
    // printf("\n=== Croisement de l'individu 0 ===\n");
    // printf("jrand=%d\n", jrand);
    // printf("Position après croisement: ");
    // for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
    //     printf("%.2f ", population[idx * NUM_OF_DIMENSIONS + j]);
    // }
    // printf("\n");
// }
    
    states[idx] = localState;
}

extern "C" void cuda_de(float *population, float *gBest) {
   // Allocation mémoire sur le device
   float *devPopulation, *devMutants;
   float *devEval;
   int *devIndexMutation;
   curandState *devStatesInit;
   curandState *devStatesMutation;
   curandState *devStatesCrossover;
   
   int size = NUM_OF_POPULATION * NUM_OF_DIMENSIONS;
   
   // Allocations CUDA
   cudaMalloc((void**)&devPopulation, sizeof(float) * size);
   cudaMalloc((void**)&devMutants, sizeof(float) * size);
   cudaMalloc((void**)&devEval, sizeof(float) * NUM_OF_POPULATION);
   cudaMalloc((void**)&devIndexMutation, sizeof(int) * NUM_OF_POPULATION * 3);
   cudaMalloc((void**)&devStatesInit, sizeof(curandState) * NUM_OF_POPULATION);
   cudaMalloc((void**)&devStatesMutation, sizeof(curandState) * NUM_OF_POPULATION);
   cudaMalloc((void**)&devStatesCrossover, sizeof(curandState) * NUM_OF_POPULATION);

   // Configuration des kernels
   int threadsPerBlock = 256;
   int blocksPerGrid = (NUM_OF_POPULATION + threadsPerBlock - 1) / threadsPerBlock;

   // Initialisation de la population
   setupCurand<<<blocksPerGrid, threadsPerBlock>>>(devStatesInit, time(NULL));
   kernelInitializePopulation<<<blocksPerGrid, threadsPerBlock>>>(devPopulation, devStatesInit);
   cudaDeviceSynchronize();

   // Copier la population initiale vers l'hôte pour initialiser gBest
   cudaMemcpy(population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);
   
   // Initialiser gBest avec le premier individu
   float bestFitness = host_fitness_function(population);
   memcpy(gBest, population, NUM_OF_DIMENSIONS * sizeof(float));
   
   printf("Fitness initiale: %f\n", bestFitness);

   // Boucle principale DE
    for(int iter = 0; iter < MAX_ITER; iter++) {
        // Génération des indices pour la mutation
        setupCurand<<<blocksPerGrid, threadsPerBlock>>>(devStatesMutation, time(NULL) + iter);
        kernelPrepareMutation<<<blocksPerGrid, threadsPerBlock>>>(devIndexMutation, devStatesMutation);
        
        // Mutation
        kernelDEMutation<<<blocksPerGrid, threadsPerBlock>>>(devPopulation, devIndexMutation, devMutants);
        
        // Croisement
        int jrand = rand() % NUM_OF_DIMENSIONS;
        setupCurand<<<blocksPerGrid, threadsPerBlock>>>(devStatesCrossover, time(NULL) + iter);
        kernelCrossoverDE<<<blocksPerGrid, threadsPerBlock>>>(devPopulation, devMutants, jrand, devStatesCrossover);
        
        // Évaluation et mise à jour de gBest
        cudaMemcpy(population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);
        
        // Vérifier chaque individu
        for(int i = 0; i < NUM_OF_POPULATION; i++) {
            float* currentIndividual = &population[i * NUM_OF_DIMENSIONS];
            float currentFitness = host_fitness_function(currentIndividual);
            
            if(currentFitness < bestFitness) {
                    bestFitness = currentFitness;
                    memcpy(gBest, currentIndividual, NUM_OF_DIMENSIONS * sizeof(float));
                    printf("Iteration %d: Nouvelle meilleure fitness = %f\n", iter, bestFitness);
                    printf("Nouvelle meilleure position: [");
                    for(int d = 0; d < NUM_OF_DIMENSIONS; d++) {
                        printf("%.6f", gBest[d]);
                        if(d < NUM_OF_DIMENSIONS - 1) printf(", ");
                    }
                    printf("]\n");
            }
        }
        
        // Affichage périodique
        if(iter % 1000 == 0) {
            printf("Iteration %d/%d (%.2f%%)\n", iter, MAX_ITER, (float)iter/MAX_ITER * 100);
            printf("Meilleure fitness actuelle: %f\n", bestFitness);
        }
    }

   // Nettoyage
   cudaFree(devPopulation);
   cudaFree(devMutants);
   cudaFree(devEval);
   cudaFree(devIndexMutation);
   cudaFree(devStatesInit);
   cudaFree(devStatesMutation);
   cudaFree(devStatesCrossover);

   printf("\n=== Résultats finaux ===\n");
   printf("Meilleure fitness trouvée: %f\n", bestFitness);
   printf("\n");
}