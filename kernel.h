#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

#define DEBUG_PRINT 1  // Mettre à 0 pour désactiver le débogage

// Constantes
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
// const int SELECTED_OBJ_FUNC = 1;


// Configuration DE
const int NUM_OF_DIMENSIONS = 10;
const int NUM_OF_POPULATION = 512;
const float CR = 0.3f;  // Taux de croisement
const float F = 0.5f;   // Facteur de mutation
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);
// const float START_RANGE_MIN = -600.0f;
// const float START_RANGE_MAX = 600.0f;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float phi = 3.14f;

// Sélection de la fonction objective (1 pour Shifted Rastigrin)
const int SELECTED_OBJ_FUNC = 1;

// Fonctions utiles
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

extern "C" void cuda_de(float *population, float* gBest);