#include "kernel.h"

int main(int argc, char** argv) {

    if(argc == 3){
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }
    

    float population[NUM_OF_POPULATION*NUM_OF_DIMENSIONS];
    float evaluation[NUM_OF_POPULATION*NUM_OF_DIMENSIONS];

    
    printf("Type \t Time \t  \t Minimum\n");
    
        // Initialisation du random
        srand((unsigned) time(NULL));
        clock_t begin = clock();
        cuda_de(population,evaluation);    
        clock_t end = clock();
        printf("GPU \t ");
        printf("%10.3lf \t", (double)(end - begin)/CLOCKS_PER_SEC);
        
    return 0;
}
