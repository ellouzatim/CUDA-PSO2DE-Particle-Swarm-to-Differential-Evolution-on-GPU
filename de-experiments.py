import subprocess
import pandas as pd
import time
import os
from datetime import datetime
"""
Fichier d'expérimentation pour l'algorithme d'Évolution Différentielle (DE) sur GPU

Ce fichier définit un framework complet pour exécuter et analyser des expériences
d'optimisation utilisant l'algorithme d'Évolution Différentielle implémenté en CUDA.

Fonctionnalités :
- Configuration automatique des paramètres d'expérimentation
- Compilation automatisée du code CUDA
- Exécution de tests sur différentes fonctions de benchmark
- Tests avec différentes tailles de population
- Répétitions multiples pour assurer la fiabilité statistique
- Analyse et sauvegarde automatique des résultats

Les expériences sont menées surles 4 fonctions de test standard.
Chaque expérience est répétée plusieurs 10 avec différentes tailles de population (50, 100, 500) 
pour analyser l'impact de la taille de la population sur les performanceset une dimension qu'on définit à 10 pour ce cas.

Les résultats sont automatiquement enregistrés dans des fichiers CSV, incluant :
- Les résultats détaillés de chaque exécution
- Les statistiques agrégées (moyenne, écart-type, min, max)
- Les temps d'exécution
- Les valeurs minimales trouvées
"""

class DEExperiments:
    def __init__(self):
        self.dimensions = 10
        self.populations = [50, 100, 500]
        self.repetitions = 10
        self.test_functions = {
            1: {"name": "Shifted Rastigrin", "range": (-5.0, 5.0)},
            2: {"name": "Shifted Rosenbrock", "range": (-100.0, 100.0)},
            3: {"name": "Shifted Griewank", "range": (-600.0, 600.0)},
            4: {"name": "Shifted Sphere", "range": (-100.0, 100.0)}
        }
        self.results = []
        
    def create_modified_header(self, pop_size, func_id):
        func_config = self.test_functions[func_id]
        header_content = f"""#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

const int SELECTED_OBJ_FUNC = {func_id};
const float CR = 0.5;
const int NUM_OF_POPULATION = {pop_size};
const int NUM_OF_DIMENSIONS = 10;
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);
const float START_RANGE_MIN = {func_config['range'][0]}f;
const float START_RANGE_MAX = {func_config['range'][1]}f;
const float F = 0.5f;
const float phi = 3.14159265358979323846f;

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

extern "C" void cuda_de(float *population, float* evaluation);

#endif
"""
        with open('kernel.h', 'w') as f:
            f.write(header_content)
            
    def compile_de(self, pop_size, func_id):
        try:
            if os.path.exists("programde"):
                os.remove("programde")
            
            self.create_modified_header(pop_size, func_id)
            
            compile_command = "nvcc -o programde main.cpp kernel.cpp kernel.cu"
            result = subprocess.run(compile_command.split(), 
                                 check=True, 
                                 capture_output=True,
                                 text=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Erreur de compilation: {e.stderr}")
            return False

    def run_single_experiment(self):
        try:
            result = subprocess.run(["./programde"], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.startswith('GPU'):
                    parts = line.split()
                    return float(parts[1]), float(parts[2])
        except subprocess.CalledProcessError as e:
            print(f"Erreur d'exécution: {e}")
            return None, None
            
    def run_all_experiments(self):
        for func_id, func_info in self.test_functions.items():
            print(f"\nFonction de test: {func_info['name']} (ID: {func_id})")
            print(f"Plage de valeurs: [{func_info['range'][0]}, {func_info['range'][1]}]")
            
            func_results = []
            for pop_size in self.populations:
                print(f"\nPopulation = {pop_size}")
                
                if not self.compile_de(pop_size, func_id):
                    continue
                    
                pop_results = []
                for rep in range(self.repetitions):
                    print(f"  Répétition {rep + 1}/{self.repetitions}", end='\r')
                    time_taken, minimum = self.run_single_experiment()
                    if time_taken is not None:
                        pop_results.append({
                            'function_id': func_id,
                            'function_name': func_info['name'],
                            'population': pop_size,
                            'repetition': rep + 1,
                            'time': time_taken,
                            'minimum': minimum
                        })
                
                if pop_results:
                    df_temp = pd.DataFrame(pop_results)
                    print(f"\nMoyenne: temps={df_temp['time'].mean():.4f}s, "
                          f"minimum={df_temp['minimum'].mean():.4f}")
                
                func_results.extend(pop_results)
            
            #  résumé pour cette fonction
            if func_results:
                df_func = pd.DataFrame(func_results)
                print(f"\nRésumé pour {func_info['name']}:")
                summary = df_func.groupby('population').agg({
                    'time': ['mean', 'std'],
                    'minimum': ['mean', 'std']
                }).round(4)
                print(summary)
            
            self.results.extend(func_results)
            
        self.analyze_results()
        
    def analyze_results(self):
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Sauvegarde des résultats bruts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"de_results_{timestamp}.csv", index=False)
        
        # Calcul et sauvegarder les statistiques
        stats = df.groupby(['function_name', 'population']).agg({
            'time': ['mean', 'std', 'min', 'max'],
            'minimum': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print("\nRésultats finaux par fonction et taille de population:")
        print(stats)
        stats.to_csv(f"de_stats_{timestamp}.csv")

if __name__ == "__main__":
    experiments = DEExperiments()
    experiments.run_all_experiments()
