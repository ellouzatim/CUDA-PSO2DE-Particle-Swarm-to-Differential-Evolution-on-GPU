#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include "kernel.h"




__device__ float tempParticle[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];


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


__global__ void kernelInitializePopulation(float *population)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // avoid an out of bound for the array
    if(i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
        return;
    population[i] = getRandom(START_RANGE_MIN,START_RANGE_MAX);
}

__global__ void kernelEvaluerPopulation(float*population,float *evaluation)
{    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // avoid an out of bound for the array
        if(i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle[j] = population[i + j];
    }
    evaluation[i] = fitness_function(tempParticle);
}


__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;


    switch (SELECTED_OBJ_FUNC)  {
        case 0:
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;


            res += pow(sin(phi*y1), 2);


            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        case 1:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
       
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;  
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180;
            break;
        case 4:
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
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
__global__ void kernelCrossoverDE (
    float *previous_individuals, 
    float *mutated_individuals, 
    int k
    )
{
    // id du processus
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // avoid an out of bound for the array 
    if(i >= NUM_OF_POPULATION * NUM_OF_DIMENSIONS)
        return;

    // individual : ceil(i / NUM_OF_DIMENSIONS), not useful to compute here

    // j : current index the individual 
    int j = i % NUM_OF_DIMENSIONS; 
    float randj = getRandomClamped(); // random [0, 1], cf. kernel.cpp
    
    // cf. crossover, equation (2) in the paper
    if (! (randj <= CR || j == k))
    {
        // <=> vector U(i,j) in the paper
        mutated_individuals[i] = previous_individuals[i];
    }
}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
    }

    if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
    {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}
__global__ void kernelDEMutation(float *individuals, int *indexMutation, float *mutants, float F) {
    extern __shared__ float sharedMem[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_POPULATION) return;
    
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
  
extern "C" void cuda_de(float *population, float* evaluation)
{
    float *devPopulation;
    float *devEval;
    float *devMutants;
    int *devIndexMutation;
    float temp[NUM_OF_DIMENSIONS];
       
    cudaMalloc((void**)&devPopulation, sizeof(float) * size);
    cudaMalloc((void**)&devEval, sizeof(float) * NUM_OF_POPULATION);
    cudaMalloc((void**)&devMutants, sizeof(float) * size);
    cudaMalloc((void**)&devIndexMutation, sizeof(int) * NUM_OF_POPULATION * 3);

    int threadsNum = 256;
    int blocksNum = (NUM_OF_POPULATION + threadsNum - 1) / threadsNum;
    int sharedMemSize = threadsNum * NUM_OF_DIMENSIONS * 3 * sizeof(float);
   
    cudaMemcpy(devPopulation, population, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devEval, evaluation, sizeof(float) * NUM_OF_POPULATION, cudaMemcpyHostToDevice);
  
    for (int iter = 0; iter < MAX_ITER; iter++)
    {    
        kernelInitializePopulation<<<blocksNum, threadsNum>>>(devPopulation);  

        kernelEvaluerPopulation<<<blocksNum, threadsNum>>>(devPopulation, devEval);

        // Génération des indices de mutation (à implémenter)
        // kernelGenerateIndices<<<blocksNum, threadsNum>>>(devIndexMutation);

        kernelDEMutation<<<blocksNum, threadsNum, sharedMemSize>>>(devPopulation, devIndexMutation, devMutants, F);

        int k = getRandomClamped(0, NUM_OF_DIMENSIONS - 1);
        kernelCrossoverDE<<<blocksNum, threadsNum>>>(devPopulation, devMutants, k);
        
        // Ajoutez ici le kernel de sélection si nécessaire
    }

    cudaMemcpy(population, devPopulation, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(evaluation, devEval, sizeof(float) * NUM_OF_POPULATION, cudaMemcpyDeviceToHost);
   
    cudaFree(devPopulation);
    cudaFree(devEval);
    cudaFree(devMutants);
    cudaFree(devIndexMutation);
}
       