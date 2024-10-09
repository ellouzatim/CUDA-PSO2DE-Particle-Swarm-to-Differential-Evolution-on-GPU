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


    return res;
}

extern "C" void cuda_de(float *population, float* evaluation)
{
    int size = NUM_OF_POPULATION * NUM_OF_DIMENSIONS;
   
    // declare all the arrays on the device
    float *devPopulation;
    float *devEval;
    float temp[NUM_OF_DIMENSIONS];
       
    // Memory allocation
    cudaMalloc((void**)&devPopulation, sizeof(float) * size);

    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);
   
    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     * */
    cudaMemcpy(devPopulation, population, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devEval, evaluation, sizeof(float) * size,cudaMemcpyHostToDevice);
    // PSO main function
    // MAX_ITER = 30000;
  

    for (int iter = 0; iter < MAX_ITER; iter++)
    {    
        kernelInitializePopulation<<<blocksNum, threadsNum>>>(devPopulation);  
       

        cudaMemcpy(population, devPopulation,sizeof(float) * size,cudaMemcpyDeviceToHost);
        cudaMemcpy(devPopulation, population, sizeof(float) * size, cudaMemcpyHostToDevice);

        kernelEvaluerPopulation<<<blocksNum, threadsNum>>>(devPopulation,devEval);         
        cudaMemcpy(evaluation, devEval,sizeof(float) * size,cudaMemcpyDeviceToHost);
        
   
    cudaFree(devPopulation);
    cudaFree(devEval);

}









