#include "kernel.h"

int main(int argc, char** argv) {

    if(argc == 3){
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }
    

    float positions[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    float velocities[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    float pBests[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    float gBest[NUM_OF_DIMENSIONS];
    
    printf("Type \t Time \t  \t Minimum\n");
    
        // Initialisation du random
        srand((unsigned) time(NULL));

        for (int i = 0; i < NUM_OF_PARTICLES*NUM_OF_DIMENSIONS; i++) {
            positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
            pBests[i] = positions[i];
            velocities[i] = 0;
        }

        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            gBest[k] = pBests[k];

        clock_t begin = clock();
        cuda_pso(positions, velocities, pBests, gBest);    
        clock_t end = clock();
        printf("GPU \t ");
        printf("%10.3lf \t", (double)(end - begin)/CLOCKS_PER_SEC);
        
        printf(" %f\n", host_fitness_function(gBest));

    return 0;
}
