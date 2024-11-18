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

// __global__ void kernelInitializePopulation(float *population, curandState *states) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // avoid an out of bound for the array
//     if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
//         return;

//     curandState localState = states[i / NUM_OF_DIMENSIONS];
//     // random entre
//     population[i] = START_RANGE_MIN + curand_uniform(&localState) * (START_RANGE_MAX - START_RANGE_MIN);
// }

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
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_OF_POPULATION) {  // On initialise un état par individu
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * Randomly create an array of indices for mutation.
 * For each individual, generate 3 distinct indices that are different from i, within the range 0 to NUM_OF_POPULATION.
 *
 * @param indexMutation Integer array to store the indices
 * @param states State array for random generation
 */
/*
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
            used[randomIdx] = randomIdx;
            count++;
            attempts++;
        }
    }
}
*/

__global__ void kernelPrepareMutation(int *indexMutation, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_OF_POPULATION) {
        curandState state = states[idx];
        indexMutation[idx * 3] = curand(&state) % NUM_OF_POPULATION;
        indexMutation[idx * 3 + 1] = curand(&state) % NUM_OF_POPULATION;
        indexMutation[idx * 3 + 2] = curand(&state) % NUM_OF_POPULATION;
        states[idx] = state;
    }
}




__global__ void kernelDEMutation(float *population, int *indexMutation, float *mutants, float F) {
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
        base[d] = population[r_base * NUM_OF_DIMENSIONS + d];
        x_r1[d] = population[r_1 * NUM_OF_DIMENSIONS + d];
        x_r2[d] = population[r_2 * NUM_OF_DIMENSIONS + d];
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
 *  - population : current population
 *  - mutatedPopulation : population with mutation
 *  - k : random [0, D-1], D = dimension (generated each iteartion)
 */
__global__ void kernelCrossoverDE(
    float *population,
    float *mutatedPopulation,
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
        mutatedPopulation[i] = population[i]; // trials vector == mutatedPopulation
    }
}

__global__ void kernelEvaluerPopulation(float *population, float *mutatedPopulation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // avoid an out of bound for the array
    if (i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        tempParticle[j] = population[i + j];
        tempParticleMutation[j] = mutatedPopulation[i + j]; // trials vectors
    }

    if (!(device_fitness_function(tempParticle) > device_fitness_function(tempParticleMutation))) {
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            population[i + j] = tempParticleMutation[j];
        }
    }
}

extern "C" void cuda_de(float *population, float *gBest) {

    // Debug
    float* h_population = new float[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    float* h_init_population = new float[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    float* h_mutants = new float[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    int* h_indexMutation = new int[NUM_OF_POPULATION * 3];
    char filename[100];
    // Fin Debug

    int size = NUM_OF_POPULATION * NUM_OF_DIMENSIONS;

    float *devPopulation;
    float *devEval;
    //float *devGBest; //not used
    float *devMutants;
    int *devIndexMutation;
    float evaluation[NUM_OF_POPULATION];  // float evaluation[NUM_OF_POPULATION * NUM_OF_DIMENSIONS]; /!\incohérence avec devEval qui est seulement de taille NUM_OF_POPULATION
    curandState *devStatesInitPop;
    curandState *devStatesPrepareMutation;
    curandState *devStatesCrossover;


    cudaMalloc((void**)&devPopulation, sizeof(float) * size);
    cudaMalloc((void**)&devEval, sizeof(float) * NUM_OF_POPULATION);
    cudaMalloc((void**)&devMutants, sizeof(float) * size);
    cudaMalloc((void**)&devIndexMutation, sizeof(int) * NUM_OF_POPULATION * 3);
    cudaMalloc((void**)&devStatesInitPop, sizeof(curandState) * NUM_OF_POPULATION);
    cudaMalloc((void**)&devStatesPrepareMutation, sizeof(curandState) * NUM_OF_POPULATION);
    cudaMalloc((void**)&devStatesCrossover, sizeof(curandState) * size);
    // cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);

    int threadsPerBlock = 256;
    //int blocksNum = (NUM_OF_POPULATION + threadsNum - 1) / threadsNum;
    int blocksForPop = (NUM_OF_POPULATION * NUM_OF_DIMENSIONS + threadsPerBlock - 1) / threadsPerBlock;
    int blocksForStates = (NUM_OF_POPULATION + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedMemSize = threadsPerBlock * NUM_OF_DIMENSIONS * 3 * sizeof(float);

    //cudaMemcpy(devPopulation, population, sizeof(float) * size, cudaMemcpyHostToDevice);
    //cudaMemcpy(devEval, evaluation, sizeof(float) * NUM_OF_POPULATION, cudaMemcpyHostToDevice); // /!\copie de données non initialisées vers le gpu 


    // Initialisation
    setupCurand<<<blocksForStates, threadsPerBlock>>>(devStatesInitPop, time(NULL));
    cudaDeviceSynchronize();
    kernelInitializePopulation<<<blocksForPop, threadsPerBlock>>>(devPopulation, devStatesInitPop);
    cudaDeviceSynchronize();
    
    // Debug

    // Vérification des erreurs
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        // Gérer l'erreur...
    }
    cudaMemcpy(h_init_population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);
    sprintf(filename, "init_population.csv");
    write2DArrayToFile(h_init_population, NUM_OF_POPULATION, NUM_OF_DIMENSIONS, filename);
    // Fin Debug
    kernelEvaluerPopulationInitiale<<<blocksForPop, threadsPerBlock>>>(devPopulation, devEval); // /!\ ne sert à rien pour l'instant

    for (int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        gBest[i] = population[i];


    for (int iter = 0; iter < MAX_ITER; iter++) {
        float *tempPopulation = new float[size];  //tableau temporaire pour toute la population
        float tempIndividual[NUM_OF_DIMENSIONS]; //pour un individu

        setupCurand<<<blocksForStates, threadsPerBlock>>>(devStatesPrepareMutation, time(NULL));
        cudaDeviceSynchronize();

        kernelPrepareMutation<<<blocksForPop, threadsPerBlock>>>(devIndexMutation, devStatesPrepareMutation);
        
        kernelDEMutation<<<blocksForPop, threadsPerBlock, sharedMemSize>>>(devPopulation, devIndexMutation, devMutants, F);
        
        int r = getRandom(0, NUM_OF_DIMENSIONS - 1);
        setupCurand<<<blocksForStates, threadsPerBlock>>>(devStatesCrossover, time(NULL));
        cudaDeviceSynchronize();
        kernelCrossoverDE<<<blocksForPop, threadsPerBlock>>>(devPopulation, devMutants, r, devStatesCrossover);

        // Ajoutez ici le kernel de sélection si nécessaire
        kernelEvaluerPopulation<<<blocksForPop, threadsPerBlock>>>(devPopulation, devMutants);
        
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

        // Debug
        if (iter % 100 == 0) {
            
            
            // population
            sprintf(filename, "population_gen_%d.csv", iter);
            write2DArrayToFile(tempPopulation, NUM_OF_POPULATION, NUM_OF_DIMENSIONS, filename);
            
            // Copie et sauvegarde des indices de mutation
            cudaMemcpy(h_indexMutation, devIndexMutation,
                      NUM_OF_POPULATION * 3 * sizeof(int),
                      cudaMemcpyDeviceToHost);
            sprintf(filename, "mutation_indices_gen_%d.csv", iter);
            writeArrayToFile(h_indexMutation, NUM_OF_POPULATION * 3, filename);
            
            // Copie et sauvegarde des mutants
            cudaMemcpy(h_mutants, devMutants,
                      NUM_OF_POPULATION * NUM_OF_DIMENSIONS * sizeof(float),
                      cudaMemcpyDeviceToHost);
            sprintf(filename, "mutants_gen_%d.csv", iter);
            write2DArrayToFile(h_mutants, NUM_OF_POPULATION, NUM_OF_DIMENSIONS, filename);
        }
        //Fin Debug

        delete[] tempPopulation;
    }

    cudaMemcpy(population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(devPopulation);
    cudaFree(devEval);
    cudaFree(devMutants);
    cudaFree(devIndexMutation);
    cudaFree(devStatesInitPop);
    cudaFree(devStatesPrepareMutation);
    cudaFree(devStatesCrossover);
    //cudaFree(devGBest);

    // Debug
    delete[] h_init_population;
    delete[] h_population;
    delete[] h_mutants;
    delete[] h_indexMutation;
    // Fin Debug
}
