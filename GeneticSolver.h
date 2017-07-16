#ifndef GENETICSOLVE_H
#define GENETICSOLVE_H

#include "utils.h"
#include "Solver.h"
#include "CuFunctionsManager.h"
#include "CufunctionAdapter.h"

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <algorithm>

using namespace std;
using namespace CudaFunction;

class GeneticSolver : public Solver
{
public:
    GeneticSolver();
    virtual ~GeneticSolver();

    void RunStage();
    int* GetLeader();
    int GetBestScore();

    void Prepare();

private:
    void CreateFirstPopulation();
    void Crossover();
    void SetSortedPermutation(CUdeviceptr& dest, int permutation_no, int permutation_size);
    void RandomShuffle(CUdeviceptr& array, int permutation_no, int permutation_size);
    void SortArrayByRanks(CUdeviceptr& array, CUdeviceptr& ranks, int permutation_no, int permutation_size);
    void SetValue(CUdeviceptr& array, int size, int value);
    void CrossoverTakeGensFromFirstParent(int begin_point);
    void CalculatePrefixSum(CUdeviceptr& array, int permutation_no, int permutation_size);
    void INTERNAL__CalculateBlocksPrefixSum(CUdeviceptr& cu_array, CUdeviceptr& cu_addition, int blocks_no, int blocks_per_permutation);
    void FindMissingGens();
    void CrossoverTakeGensFromSecondParent();
    void ApplyMutation();
    void JudgeFunction(CUdeviceptr& cu_array, CUdeviceptr& cu_ranks, int permutation_no, int permutation_size); // UWAGA modyficuje cu_co2_ind_
    void SwapCyclesToDistanceArray(CUdeviceptr& cu_array, CUdeviceptr& cu_dist, int permutation_no, int permutation_size);
    void FillArrayWithLastElementValue(CUdeviceptr& cu_result, CUdeviceptr& cu_from, int permutation_no, int permutation_size);
    void JudgePopulation();
    void InitMastersStructure();
    void CreatePopulationParents();
    void CopyFromTO(CUdeviceptr& from, int from_start_index, CUdeviceptr& to, int to_start_index, int permutation_no, int permutation_size);
    int GetValueAt(CUdeviceptr& from, int pos);

    inline void DEBUG_CheckCreatureCorrection(CUdeviceptr& creatures, int creatures_no, int creature_size);

    int population_size_;
    int crossover_creature_no_;
    int best_ever_size_;
    int random_creature_size_;
    int best_local_size_;
    int single_mutation_probability_;

    CUdeviceptr cu_population_;
    CUdeviceptr cu_left_;
    CUdeviceptr cu_co2_ind_; // co2 = crossoverpart 2 -> ind of remaining elements
    CUdeviceptr cu_individual_score_;
    CUdeviceptr cu_crossover_creature_;

    CUdeviceptr cu_masters_;
    CUdeviceptr cu_masters_score_;
    CUdeviceptr cu_population_ranking_;

};

#endif    /* GENETICSOLVE_H */
