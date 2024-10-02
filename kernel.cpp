#include "kernel.h"

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
// parametre 1 individu avec ses positions
float host_fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: {
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        }
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

// Obtenir un random entre low et high
float getRandom(float low, float high) {
    return low + float(((high - low) + 1)*rand()/(RAND_MAX + 1.0));
}
// Obtenir un random entre 0.0f and 1.0f inclusif
float getRandomClamped() {
    return (float) rand()/(float) RAND_MAX;
}

