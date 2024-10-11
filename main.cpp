# include "kernel.h"

int main(int argc, char** argv) {

    if(argc == 3){
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }

    float population[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    // float evaluation[NUM_OF_POPULATION * NUM_OF_DIMENSIONS];
    float gBest[NUM_OF_DIMENSIONS];

    printf("Type \t Time \t \t Minimum\n");

    int tt = 0;

    srand((unsigned) time(NULL));
    clock_t begin = clock();
    cuda_de(population, gBest, &tt);    
    clock_t end = clock();
    printf("GPU \t ");
    printf("%10.3lf \t", (double)(end - begin) / CLOCKS_PER_SEC);
    printf("%d\n", tt);
    printf(" %f\n", host_fitness_function(gBest));

    return 0;
}
