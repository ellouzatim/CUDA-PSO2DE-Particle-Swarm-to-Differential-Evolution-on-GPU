/**
 * @file kernel.h
 * @brief Fichier d'en-tête pour l'implémentation de l'algorithme d'Évolution Différentielle sur GPU
 *
 * Ce fichier contient les déclarations des constantes, variables et fonctions nécessaires
 * pour l'exécution de l'algorithme d'Évolution Différentielle (DE) sur GPU utilisant CUDA.
 * Il définit les paramètres de configuration de l'algorithme ainsi que les interfaces
 * des fonctions principales.
 *
 * Paramètres de l'algorithme :
 * - Fonction objectif sélectionnée parmi plusieurs benchmarks
 * - Taille de la population
 * - Nombre de dimensions
 * - Paramètres DE (CR, F)
 * - Limites de l'espace de recherche
 *
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

/**
 * @brief Identifiant de la fonction objectif sélectionnée
 * Les valeurs possibles sont :
 * 0: Levy
 * 1: Shifted Rastigrin
 * 2: Shifted Rosenbrock
 * 3: Shifted Griewank
 * 4: Shifted Sphere
 */
const int SELECTED_OBJ_FUNC = 1;

/** @brief Taux de croisement pour l'évolution différentielle */
const float CR = 0.5;

/** @brief Taille de la population */
const int NUM_OF_POPULATION = 512;

/** @brief Nombre de dimensions du problème */
const int NUM_OF_DIMENSIONS = 10;

/** @brief Nombre maximum d'itérations (calculé en fonction du nombre de dimensions) */
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);

/** @brief Limite inférieure de l'espace de recherche */
const float START_RANGE_MIN = -5.12f;

/** @brief Limite supérieure de l'espace de recherche */
const float START_RANGE_MAX = 5.12f;

/** @brief Facteur de mutation pour l'évolution différentielle */
const float F = 0.5f;

/** @brief Constante mathématique Pi */
const float phi = 3.14159265358979323846f;

/**
 * @brief Génère un nombre aléatoire dans l'intervalle [low, high]
 * @param low Borne inférieure
 * @param high Borne supérieure
 * @return Nombre aléatoire généré
 */
float getRandom(float low, float high);

/**
 * @brief Génère un nombre aléatoire normalisé entre 0 et 1
 * @return Nombre aléatoire entre 0 et 1
 */
float getRandomClamped();

/**
 * @brief Calcule la valeur de fitness d'un individu sur CPU
 * @param x Tableau des coordonnées de l'individu
 * @return Valeur de fitness calculée
 */
float host_fitness_function(float x[]);

/**
 * @brief Fonction principale de l'algorithme DE sur GPU
 * @param population Pointeur vers le tableau de la population
 * @param evaluation Pointeur vers le tableau des meilleures solutions
 */
extern "C" void cuda_de(float *population, float* evaluation);

#endif