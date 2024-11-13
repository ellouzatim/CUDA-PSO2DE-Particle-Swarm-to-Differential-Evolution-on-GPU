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

__global__ void kernelInitializePopulation(float *population, curandState *states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // avoid an out of bound for the array
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
        return;

    curandState localState = states[i / NUM_OF_DIMENSIONS];
    // random entre
    population[i] = START_RANGE_MIN + curand_uniform(&localState) * (START_RANGE_MAX - START_RANGE_MIN);
}

__global__ void kernelEvaluerPopulationInitiale(float *population, float *evaluation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0) // /!\i >= NUM_OF_POPULATION  est trop petit
        return;

    float tempParticle[NUM_OF_DIMENSIONS];
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        tempParticle[j] = population[i + j];
    }
    evaluation[i / NUM_OF_DIMENSIONS] = device_fitness_function(tempParticle);
}

/**
 * Initialize a curandState
 *
 * @param states Array to store curandState objects
 * @param seed Seed for random number generation
 */

__global__ void setupCurand(curandState *states, unsigned long long seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < NUM_OF_POPULATION || i % NUM_OF_DIMENSIONS != 0) {
        curand_init(seed, i, 0, &states[i]);
    }
}

/**
 * Randomly create an array of indices for mutation.
 * For each individual, generate 3 distinct indices that are different from i, within the range 0 to NUM_OF_POPULATION.
 *
 * @param indexMutation Integer array to store the indices
 * @param states State array for random generation
 */

__global__ void kernelPrepareMutation(int *indexMutation, curandState *states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_OF_POPULATION || i % NUM_OF_DIMENSIONS != 0)
        return;

    int offsetIndividu = i / NUM_OF_DIMENSIONS;
    int offsetIndexMutation = offsetIndividu * 3;

    curandState localState = states[offsetIndividu];
    int used[NUM_OF_POPULATION] = {0};

    int count = 0;
    int attempts = 0;
    while (count < 3 && attempts < NUM_OF_POPULATION * 2) {
        unsigned int randomIdx = curand(&localState) % NUM_OF_POPULATION;
        if (randomIdx != offsetIndividu && !used[randomIdx]) {
            indexMutation[offsetIndexMutation + count] = randomIdx;
            used[randomIdx] = 1;
            count++;
            attempts++;
        }
    }
}

__global__ void kernelDEMutation(float *individuals, int *indexMutation, float *mutants, float F) {
    extern __shared__ float sharedMem[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0) return;

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

/**
 * Crossover DE
 * Update the values of the param mutated_individuals with the crossover
 *
 * Params :
 *  - previous_individuals : current population
 *  - mutated_individuals : population with mutation
 *  - k : random [0, D-1], D = dimension (generated each iteartion)
 */
__global__ void kernelCrossoverDE(
    float *previous_individuals,
    float *mutated_individuals,
    int k,
    curandState *states
) {
    // id du processus
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // avoid an out of bound for the array
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
        return;

    // individual : ceil(i / NUM_OF_DIMENSIONS), not useful to compute here

    // j : current index the individual
    int j = i % NUM_OF_DIMENSIONS;
    curandState localState = states[i / NUM_OF_DIMENSIONS];
    float randj = curand_uniform(&localState);

    // cf. crossover, equation (2) in the paper
    if (!(randj <= CR || j == k)) {
        // <=> vector U(i,j) in the paper
        mutated_individuals[i] = previous_individuals[i];
    }
}

__global__ void kernelEvaluerPopulation(float *oldPopulation, float *mutatedPopulation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // avoid an out of bound for the array
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        tempParticleOld[j] = oldPopulation[i + j];
        tempParticleMutation[j] = mutatedPopulation[i + j];
    }

    if (!(device_fitness_function(tempParticleOld) > device_fitness_function(tempParticleMutation))) {
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            oldPopulation[i + j] = tempParticleMutation[j];
        }
    }
}

extern "C" void cuda_de(float *population, float *gBest) {
    int size = NUM_OF_POPULATION * NUM_OF_DIMENSIONS;

    float *devPopulation;
    float *devEval;
    //float *devGBest; //not used
    float *devMutants;
    int *devIndexMutation;
    float evaluation[NUM_OF_POPULATION];  // float evaluation[NUM_OF_POPULATION * NUM_OF_DIMENSIONS]; /!\incohérence avec devEval qui est seulement de taille NUM_OF_POPULATION
    curandState *dstatesInitPop;
    curandState *dstatesPrepareMutation;
    curandState *dstatesCrossover;


    cudaMalloc((void**)&devPopulation, sizeof(float) * size);
    cudaMalloc((void**)&devEval, sizeof(float) * NUM_OF_POPULATION);
    cudaMalloc((void**)&devMutants, sizeof(float) * size);
    cudaMalloc((void**)&devIndexMutation, sizeof(int) * NUM_OF_POPULATION * 3);
    cudaMalloc((void**)&dstatesInitPop, sizeof(curandState) * NUM_OF_POPULATION);
    cudaMalloc((void**)&dstatesPrepareMutation, sizeof(curandState) * NUM_OF_POPULATION);
    cudaMalloc((void**)&dstatesCrossover, sizeof(curandState) * size);
    // cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);

    int threadsNum = 256;
    int blocksNum = (NUM_OF_POPULATION + threadsNum - 1) / threadsNum;

    size_t sharedMemSize = threadsNum * NUM_OF_DIMENSIONS * 3 * sizeof(float);

    cudaMemcpy(devPopulation, population, sizeof(float) * size, cudaMemcpyHostToDevice);
    //cudaMemcpy(devEval, evaluation, sizeof(float) * NUM_OF_POPULATION, cudaMemcpyHostToDevice); // /!\copie de données non initialisées vers le gpu 


    // Initialisation
    setupCurand<<<blocksNum, threadsNum>>>(dstatesInitPop, time(NULL));
    kernelInitializePopulation<<<blocksNum, threadsNum>>>(devPopulation, dstatesInitPop);
    kernelEvaluerPopulationInitiale<<<blocksNum, threadsNum>>>(devPopulation, devEval); // /!\ ne sert à rien pour l'instant

    for (int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        gBest[i] = population[i];


    for (int iter = 0; iter < MAX_ITER; iter++) {
        float *tempPopulation = new float[size];  //tableau temporaire pour toute la population
        float tempIndividual[NUM_OF_DIMENSIONS]; //pour un individu

        setupCurand<<<blocksNum, threadsNum>>>(dstatesPrepareMutation, time(NULL));
        
        kernelPrepareMutation<<<blocksNum, threadsNum>>>(devIndexMutation, dstatesPrepareMutation);
        
        kernelDEMutation<<<blocksNum, threadsNum, sharedMemSize>>>(devPopulation, devIndexMutation, devMutants, F);
        
        int r = getRandom(0, NUM_OF_DIMENSIONS - 1);
        setupCurand<<<blocksNum, threadsNum>>>(dstatesCrossover, time(NULL));
        kernelCrossoverDE<<<blocksNum, threadsNum>>>(devPopulation, devMutants, r, dstatesCrossover);

        // Ajoutez ici le kernel de sélection si nécessaire
        kernelEvaluerPopulation<<<blocksNum, threadsNum>>>(devPopulation, devMutants);
        
        cudaMemcpy(tempPopulation, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);
        
        // compute current global best
        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS) {
            for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
                tempIndividual[j] = tempPopulation[i + j]; // nouvelle version qui utilise un tableau temporaire tempPopulation pour stocker devPopulation depuis le gpu car on ne peut pas accéder à devPopulation depuis le cpu
            }
            if (host_fitness_function(tempIndividual) < host_fitness_function(gBest)) {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = tempIndividual[k];
            }
        }
        delete[] tempPopulation;
    }

    cudaMemcpy(population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(devPopulation);
    cudaFree(devEval);
    cudaFree(devMutants);
    cudaFree(devIndexMutation);
    cudaFree(dstatesPrepareMutation);
    cudaFree(dstatesCrossover);
    //cudaFree(devGBest);
}
