#include "kernel.h"


int main() {
    float *population = new float[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    float *gBest = new float[NUM_OF_DIMENSIONS];

    srand(time(NULL));
    
    printf("=== Configuration ===\n");
    printf("Dimensions: %d\n", NUM_OF_DIMENSIONS);
    printf("Population: %d\n", NUM_OF_POPULATION);
    printf("CR: %f\n", CR);
    printf("F: %f\n", F);
    printf("Iterations max: %d\n\n", MAX_ITER);

    clock_t begin = clock();
    
    // Exécution de l'algorithme
    cuda_de(population, gBest);
    
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
    printf("\nTemps d'exécution: %f secondes\n", time_spent);
    printf("Fitness finale: %f\n", host_fitness_function(gBest));
    // for(int i=0; i<NUM_OF_DIMENSIONS; i++){
    //     printf("gBest[%d]: %f\n", i, gBest[i]);
    // }

    delete[] population;
    delete[] gBest;
    
    return 0;
}