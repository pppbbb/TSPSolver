#include "GeneticSolver.h"
#include "Application.h"

GeneticSolver::GeneticSolver()
{
    population_size_ = 1024;
    crossover_creature_no_ = sqrt(population_size_);
    best_ever_size_ = 4;
    random_creature_size_ = 2;
    best_local_size_ = crossover_creature_no_ - best_ever_size_ - random_creature_size_;
    single_mutation_probability_ = 30;
}

GeneticSolver::~GeneticSolver()
{

}

void GeneticSolver::Prepare()
{
    CuFunctionsManager::GetInstance().LoadModule("GeneticSolver");
    CreateFirstPopulation();
    InitMastersStructure();
}

void GeneticSolver::RunStage()
{
    Crossover();
    ApplyMutation();
    JudgePopulation();
    CreatePopulationParents();
}

void GeneticSolver::SetSortedPermutation(CUdeviceptr& dest, int permutation_no, int permutation_size)
{
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = permutation_no * permutation_size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    void* args[] = {&dest, &permutation_no, &permutation_size};

    CufunctionAdapter create_creatures;
    create_creatures.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "CreateCreatures"));
    create_creatures.SetArgs(args);
    assert
    (
        create_creatures.SetBlockNoX(blocks_no)
    );
    create_creatures.Run();

    #ifdef DEBUG_MODE
        int* initial_creatures = new int[elements_no];

        cuMemcpyDtoH((void*)initial_creatures, dest, elements_no * sizeof (int));

        REP(y, permutation_no)
        {
            REP(x, permutation_size)
            {
                assert
                (
                   initial_creatures[y * permutation_size + x] == x
                );
            }
        }
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Initialization of crossovercreatures ended successfull\n";
        #endif
        delete [] initial_creatures;
    #endif
}

void GeneticSolver::SortArrayByRanks(CUdeviceptr& array, CUdeviceptr& ranks, int permutation_no, int permutation_size)
{
    #ifdef DEBUG_MODE
        int* darray = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray, array, permutation_no * permutation_size * sizeof (int));

        int* dranks = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)dranks, ranks, permutation_no * permutation_size * sizeof (int));
    #endif

    CUdeviceptr cu_to_reverse;
    assert
    (
        CuAllocIntArray(cu_to_reverse, permutation_no)
    );
    SetValue(cu_to_reverse, permutation_no, 0);

    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = permutation_no * (permutation_size);
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

    CufunctionAdapter sort_ranks;
    sort_ranks.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "BitonicSortStep"));
    assert
    (
        sort_ranks.SetBlockNoX(blocks_no)
    );

    for (int k = 2; k <= permutation_size; k <<= 1)
      for (int j = k >> 1; j > 0; j >>=1)
      {
            void* args[] = {&array, &ranks, &j, &k};
            sort_ranks.SetArgs(args);
            sort_ranks.Run();
      }

    CufunctionAdapter fill_to_reverse;
    fill_to_reverse.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "WhichReverse"));
    assert
    (
        fill_to_reverse.SetBlockNoX(blocks_no)
    );
    {
        void* args[] = {&ranks, &permutation_size, &permutation_no, &cu_to_reverse};
        fill_to_reverse.SetArgs(args);
        fill_to_reverse.Run();
    }

    CufunctionAdapter bitonic_to_ascending_order;
    bitonic_to_ascending_order.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "BitonicToAscendingOrder"));
    assert
    (
        bitonic_to_ascending_order.SetBlockNoX(blocks_no)
    );
    {
        void* args[] = {&array, &ranks,  &cu_to_reverse, &permutation_size, &permutation_no};
        bitonic_to_ascending_order.SetArgs(args);
        bitonic_to_ascending_order.Run();
    }

    #ifdef DEBUG_MODE
        vector <pair <int, int> > v;
        REP(i, permutation_no)
        {
            v.clear();
            REP(j, permutation_size)
            {
                int index = i * permutation_size + j;
                v.push_back(pair <int, int> (dranks[index], darray[index]));
            }

            sort(v.begin(), v.end());
            REP(j, permutation_size)
            {
                int index = i * permutation_size + j;
                darray[index] = v[j].second;
                dranks[index] = v[j].first;
            }
        }

        int* darray2 = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray2, array, permutation_no * permutation_size * sizeof (int));

        int* dranks2 = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)dranks2, ranks, permutation_no * permutation_size * sizeof (int));

        cuCtxSynchronize();
        REP(i, permutation_size * permutation_no)
            assert(dranks[i] == dranks2[i]);

        delete[] darray;
        delete[] darray2;
        delete[] dranks;
        delete[] dranks2;

        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "Checking Correction after sort by ranks: ";
        #endif
        DEBUG_CheckCreatureCorrection(array, permutation_no, permutation_size);
    #endif
}

void GeneticSolver::RandomShuffle(CUdeviceptr& array, int permutation_no, int permutation_size)
{
    CUdeviceptr ranks;
    assert
    (
         cuMemAlloc(&ranks, graph_->size * crossover_creature_no_ * sizeof (*(graph_->matrix))) == CUDA_SUCCESS
    );

    CufunctionAdapter set_ranks;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = crossover_creature_no_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    int seed =  rand();
    void* args[] = {&seed, &ranks, &elements_no};

    set_ranks.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "SetRandomRanks"));
    set_ranks.SetArgs(args);
    assert
    (
        set_ranks.SetBlockNoX(blocks_no)
    );

    set_ranks.Run();

    #ifdef DEBUG_MODE
        int* host_ranks = new int[elements_no];
        cuMemcpyDtoH((void*)host_ranks, ranks, elements_no * sizeof (*(graph_->matrix)));

        #ifdef DEBUG_PRINT_ARRAYS
            REP(y, crossover_creature_no_)
            {
                REP(x, graph_->size)
                    cerr << host_ranks[y * graph_->size + x] << " ";
                cerr << "\n";
            }
        #endif

        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Ranks generating ended successfull\n";
        #endif
        delete [] host_ranks;
    #endif

    SortArrayByRanks(array, ranks, permutation_no, permutation_size);
    cuMemFree(ranks);
 }

void GeneticSolver::CreateFirstPopulation()
{
    assert
    (
        CuAllocIntArray(cu_population_, graph_->size * population_size_)
    );

    assert
    (
        CuAllocIntArray(cu_left_, graph_->size * population_size_)
    );

    assert
    (
        CuAllocIntArray(cu_co2_ind_, graph_->size * population_size_)
    );

    assert
    (
        CuAllocIntArray(cu_crossover_creature_, graph_->size * crossover_creature_no_)
    );

    assert
    (
        CuAllocIntArray(cu_individual_score_, population_size_)
    );

   SetSortedPermutation(cu_crossover_creature_, crossover_creature_no_, graph_->size);
   RandomShuffle(cu_crossover_creature_, crossover_creature_no_, graph_->size);
}

void GeneticSolver::InitMastersStructure()
{
    assert
    (
        CuAllocIntArray(cu_masters_, best_ever_size_ * graph_->size)
    );

    assert
    (
        CuAllocIntArray(cu_masters_score_, best_ever_size_)
    );

    assert
    (
        CuAllocIntArray(cu_population_ranking_, population_size_)
    );

    SetValue(cu_masters_score_, best_ever_size_, INT_MAX);
}

void GeneticSolver::SetValue(CUdeviceptr& array, int size, int value)
{
    CufunctionAdapter set_ones;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

    void* args[] = {&array, &size, &value};

    set_ones.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "SetValue"));
    set_ones.SetArgs(args);
    assert
    (
        set_ones.SetBlockNoX(blocks_no)
    );

    set_ones.Run();

    #ifdef DEBUG_MODE
        int* a_d = new int[size];
        cuMemcpyDtoH((void*)a_d, array, size * sizeof (*(graph_->matrix)));

        REP(i, size)
            assert(a_d[i] == value);

        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "Zeros filled corrected\n";
        #endif

        delete[] a_d;
    #endif
}

inline void GeneticSolver::DEBUG_CheckCreatureCorrection(CUdeviceptr& creatures, int creatures_no, int creature_size)
{
    #ifdef DEBUG_MODE
        int elements_no = creature_size * creatures_no;
        int* initial_creatures = new int[elements_no];

        cuMemcpyDtoH((void*)initial_creatures, creatures, elements_no * sizeof (*(graph_->matrix)));

        REP(y,creatures_no)
        {
            vector <bool> vis(creature_size, false);
            REP(x, creature_size)
            {
                #ifdef DEBUG_PRINT_ARRAYS
                    cerr << initial_creatures[y * creature_size+ x] << ' ';
                #endif
                assert(vis[initial_creatures[y * creature_size + x]] == false);
                vis[initial_creatures[y * creature_size + x]] = true;
            }
            #ifdef DEBUG_PRINT_ARRAYS
                cerr << "\n";
            #endif
        }

        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Creatures Correction Check ended successfull\n";
        #endif
        delete [] initial_creatures;
    #endif
}

void GeneticSolver::CrossoverTakeGensFromFirstParent(int begin_point)
{
    // Take first half of creature starting from begin point
    CufunctionAdapter assign_half_population;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = population_size_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    void* args[] = {&cu_crossover_creature_, &cu_population_, &crossover_creature_no_, &graph_->size, &begin_point, &cu_left_};

    assign_half_population.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "AssignHalfPopulation"));
    assign_half_population.SetArgs(args);
    assert
    (
        assign_half_population.SetBlockNoX(blocks_no)
    );

    assign_half_population.Run();

    #ifdef DEBUG_MODE
        int* dpopulation = new int[population_size_ * graph_->size];
        cuMemcpyDtoH((void*)dpopulation, cu_population_, graph_->size * population_size_ * sizeof (*(graph_->matrix)));

        int* dcrossover = new int[crossover_creature_no_ * graph_->size];
        cuMemcpyDtoH((void*)dcrossover, cu_crossover_creature_, crossover_creature_no_ * graph_->size * sizeof (*(graph_->matrix)));

        int no = 0;
        int act_from_pop = 0;

        REP(creature_no, population_size_)
        {
            REP(x, graph_->size / 2)
                assert(dpopulation[creature_no * graph_->size + x] == dcrossover[act_from_pop * graph_->size + begin_point + x]);

            ++ no;
            if(no == crossover_creature_no_)
            {
                no = 0;
                ++act_from_pop;
            }
        }
        delete [] dpopulation;
        delete [] dcrossover;

        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "CROSSOVER: Gens from first parent inheried properly\n";
        #endif
    #endif
}

void GeneticSolver::INTERNAL__CalculateBlocksPrefixSum(CUdeviceptr& cu_array, CUdeviceptr& cu_addition, int blocks_no, int blocks_per_permutation)
{
    CufunctionAdapter blocks_prefixsum;
    void* args[] = {&cu_array, &cu_addition, &blocks_per_permutation};
    blocks_prefixsum.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "CalculateBlocksPrefixSum"));
    blocks_prefixsum.SetArgs(args);
    assert
    (
        blocks_prefixsum.SetBlockNoX(blocks_no)
    );
    blocks_prefixsum.Run();
}

void GeneticSolver::CalculatePrefixSum(CUdeviceptr& cu_array, int permutation_no, int permutation_size)
{
    #ifdef DEBUG_MODE
        int* darray = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray, cu_array, permutation_no * permutation_size * sizeof (*(graph_->matrix)));

        REP(y, permutation_no)
            for(int x = 1; x < permutation_size; ++x)
                darray[y * permutation_size + x] += darray[y * permutation_size + x - 1];
    #endif

    CufunctionAdapter prefixsum_precalculation;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int block_per_permutation = (permutation_size / threads_no) + ((permutation_size % threads_no == 0)? 0 : 1);
    int blocks_no = block_per_permutation * permutation_no;

    CUdeviceptr cu_addition;
    assert
    (
        CuAllocIntArray(cu_addition, blocks_no)
    );

    void* args[] = {&cu_array, &cu_addition, &permutation_size, &block_per_permutation};
    prefixsum_precalculation.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "PrefixSumPrecalculate"));
    prefixsum_precalculation.SetArgs(args);
    assert
    (
        prefixsum_precalculation.SetBlockNoX(blocks_no)
    );
    prefixsum_precalculation.Run();

    INTERNAL__CalculateBlocksPrefixSum(cu_array, cu_addition, permutation_no, block_per_permutation);

    prefixsum_precalculation.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "CalculateFinalPrefixSum"));
    prefixsum_precalculation.Run();

    cuMemFree(cu_addition);

    #ifdef DEBUG_MODE
        int* darray2 = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray2, cu_array, permutation_no * permutation_size * sizeof (*(graph_->matrix)));

         REP(y, permutation_no)
         {
            REP(x, permutation_size)
            {
                assert
                (
                    darray[y * permutation_size + x] == darray2[y * permutation_size + x]
                );
            }
         }

        delete[] darray;
        delete[] darray2;
        #ifdef DEBUG_PRINT_COMMUNICATE
        cerr << "PrefixSum calculated correctly \n";
        #endif
    #endif
}

void GeneticSolver::FindMissingGens()
{
    CufunctionAdapter find_missing_gens;

    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = population_size_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    void* args[] = {&cu_crossover_creature_, &cu_left_, &cu_co2_ind_, &crossover_creature_no_, &graph_->size};

    find_missing_gens.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "FindMissingGens"));
    find_missing_gens.SetArgs(args);
    assert
    (
        find_missing_gens.SetBlockNoX(blocks_no)
    );

    find_missing_gens.Run();
}

void GeneticSolver::CrossoverTakeGensFromSecondParent()
{
    CufunctionAdapter fill_rest_population;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = population_size_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    int start_filling = (graph_->size / 2) - 1;
    void* args[] = {&cu_crossover_creature_, &cu_population_, &cu_co2_ind_,  &crossover_creature_no_, &graph_->size, &start_filling};

    fill_rest_population.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "FillRestPopulation"));
    fill_rest_population.SetArgs(args);
    assert
    (
        fill_rest_population.SetBlockNoX(blocks_no)
    );

    fill_rest_population.Run();
}

void GeneticSolver::Crossover()
{
    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "Checking crossover_creatures before creating population: ";
        #endif
    DEBUG_CheckCreatureCorrection(cu_crossover_creature_, crossover_creature_no_, graph_->size);
    #endif

    SetValue(cu_left_, population_size_ * graph_->size, 1);

    int begin_point = rand() % ((graph_->size / 2) - 1);
    assert(begin_point + (graph_->size / 2) < graph_-> size);
    CrossoverTakeGensFromFirstParent(begin_point);

    FindMissingGens();
    CalculatePrefixSum(cu_co2_ind_, population_size_, graph_->size);
    CrossoverTakeGensFromSecondParent();

    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Checking population correction after CrossoverPhrase: ";
        #endif
        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Crossover executed properly\n";
        #endif
    #endif
}

void GeneticSolver::ApplyMutation()
{
    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Checking population before mutation: ";
        #endif
        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);
    #endif
    int standard_rank_difference = 1800000000 / graph_-> size;
    int seed = rand();

    CufunctionAdapter set_mutation_ranks;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = population_size_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    void* args[] = {&cu_co2_ind_, &seed, &single_mutation_probability_, &standard_rank_difference, &graph_->size, &population_size_};

    set_mutation_ranks.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "SetMutationRanks"));
    set_mutation_ranks.SetArgs(args);
    assert
    (
        set_mutation_ranks.SetBlockNoX(blocks_no)
    );
    set_mutation_ranks.Run();

    SortArrayByRanks(cu_population_, cu_co2_ind_, population_size_, graph_->size);

    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "Checking correction after mutation: ";
        #endif

        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);

        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "Mutation Done succesfully\n";
        #endif
    #endif
}

void GeneticSolver::SwapCyclesToDistanceArray(CUdeviceptr& cu_array, CUdeviceptr& cu_dist, int permutation_no, int permutation_size)
{
    #ifdef DEBUG_MODE
        int* darray = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray, cu_array, permutation_no * permutation_size * sizeof (*(graph_->matrix)));
    #endif

    CufunctionAdapter fill_dist_array;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = permutation_no * permutation_size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);
    void* args[] = {&cu_array, &cu_dist, &cu_graph_, &permutation_no, &permutation_size};

    fill_dist_array.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "SetPathWeight"));
    fill_dist_array.SetArgs(args);
    assert
    (
        fill_dist_array.SetBlockNoX(blocks_no)
    );

    fill_dist_array.Run();

     #ifdef DEBUG_MODE
        int* darray2 = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)darray2, cu_dist, permutation_no * permutation_size * sizeof (*(graph_->matrix)));
        int* dist = new int[permutation_size * permutation_size];
        cuMemcpyDtoH((void*)dist, cu_graph_, permutation_size * permutation_size * sizeof (*(graph_->matrix)));
        cuCtxSynchronize();

        int pos = 0;
        REP(y, permutation_no)
        {
            REP(x, permutation_size - 1)
                assert
                (
                   dist[darray[pos + x] * permutation_size + darray[pos + x + 1]] == darray2[pos + x]
                );

            assert
            (
                dist[darray[pos + permutation_size - 1] * permutation_size + darray[pos]] == darray2[pos + permutation_size - 1]
            );

            pos += permutation_size;
         }

        delete[] darray;
        delete[] darray2;
        delete[] dist;

        #ifdef DEBUG_PRINT_COMMUNICATE
            cerr << "PrefixSum calculated correctly \n";
        #endif
    #endif
}

void GeneticSolver::FillArrayWithLastElementValue(CUdeviceptr& cu_result, CUdeviceptr& cu_from, int permutation_no, int permutation_size)
{
    CufunctionAdapter fill_with_last_element;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int blocks_no = (permutation_no / threads_no) + ((permutation_no % threads_no == 0)? 0 : 1);
    void* args[] = {&cu_from, &cu_result, &permutation_no, &permutation_size};

    fill_with_last_element.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "FillRanksFromDistance"));
    fill_with_last_element.SetArgs(args);
    assert
    (
        fill_with_last_element.SetBlockNoX(blocks_no)
    );

    fill_with_last_element.Run();

    #ifdef DEBUG_MODE
        cuCtxSynchronize();
        int* dfrom = new int[permutation_no * permutation_size];
        cuMemcpyDtoH((void*)dfrom, cu_from, permutation_no * permutation_size * sizeof (*(graph_->matrix)));
        int* dresult = new int[permutation_no];
        cuMemcpyDtoH((void*)dresult, cu_result, permutation_no * sizeof (*(graph_->matrix)));

        REP(i, permutation_no)
            assert
            (
                dresult[i] == dfrom[(i + 1) * permutation_size - 1]
            );

        delete[] dfrom;
        delete[] dresult;
    #endif
}

void GeneticSolver::JudgeFunction(CUdeviceptr& cu_array, CUdeviceptr& cu_ranks, int permutation_no, int permutation_size)
{
    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Checking population before JudgeFunction: ";
        #endif
        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);
    #endif

    SwapCyclesToDistanceArray(cu_array, cu_co2_ind_, permutation_no, permutation_size);
    CalculatePrefixSum(cu_co2_ind_, permutation_no, permutation_size);
    FillArrayWithLastElementValue(cu_ranks, cu_co2_ind_, permutation_no, permutation_size);

    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Checking population after JudgeFunction: ";
        #endif
        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);
    #endif
}

void GeneticSolver::JudgePopulation()
{
#ifdef DEBUG_PRINT_COMMUNICATE
    cerr << "CallingJudgeFunction\n";
#endif
    JudgeFunction(cu_population_, cu_individual_score_, population_size_, graph_->size);
}

void GeneticSolver::CreatePopulationParents()
{
    #ifdef DEBUG_MODE
        #ifdef DEBUG_PRINT_COMMUNICATE
                cerr << "Checking population before creating new crossover creatures: ";
        #endif
        DEBUG_CheckCreatureCorrection(cu_population_, population_size_, graph_->size);
    #endif

    /*
        UpdateMasters <best of all time>
    */
    SetSortedPermutation(cu_population_ranking_, 1, population_size_);
    SortArrayByRanks(cu_population_ranking_, cu_individual_score_, 1, population_size_);

    #ifdef DEBUG_MODE
    {
        int* darray = new int[graph_->size];
        cuMemcpyDtoH((void*)darray, cu_individual_score_, graph_->size * sizeof (int));

        for(int x = 1; x < graph_->size; ++x)
            assert(darray[x] >= darray[x - 1]);
        delete[] darray;
    }
    #endif

    {
        CufunctionAdapter who_become_master;
        int threads_no = CFLimits::MAX_DEMENSION_XY;
        int elements_no = best_ever_size_;
        int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

        void* args[] = {&cu_co2_ind_, &cu_individual_score_, &cu_masters_score_, &best_ever_size_};

        who_become_master.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "SetWhoBecomeMaster"));
        who_become_master.SetArgs(args);
        assert
        (
            who_become_master.SetBlockNoX(blocks_no)
        );

        who_become_master.Run();
    }

    CalculatePrefixSum(cu_co2_ind_, 1, best_ever_size_);

    int new_masters_no = GetValueAt(cu_co2_ind_, best_ever_size_ - 1);
    print(new_masters_no);

    {
        CufunctionAdapter add_new_masters;
        int threads_no = CFLimits::MAX_DEMENSION_XY;
        int elements_no = new_masters_no * graph_->size;
        int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

        void* args[] = {&new_masters_no, &cu_population_, &cu_masters_, &cu_individual_score_, &cu_population_ranking_,
                        &cu_masters_score_, &best_ever_size_, &graph_->size};

        add_new_masters.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "AddNewMasters"));
        add_new_masters.SetArgs(args);
        assert
        (
            add_new_masters.SetBlockNoX(blocks_no)
        );

        add_new_masters.Run();
    }

    /*
        Create new population, step 1: copy masters
    */

    int masters_start = 0;
    int crossover_start = 0;
    CopyFromTO(cu_masters_, masters_start, cu_crossover_creature_, crossover_start, best_ever_size_, graph_->size);

    /*
        Create new population, step 2: copy best in generation
    */
    {
        CufunctionAdapter copy_local_best_to_crossover;
        int threads_no = CFLimits::MAX_DEMENSION_XY;
        int elements_no = best_local_size_ * graph_->size;
        int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

        void* args[] = {&cu_population_, &cu_population_ranking_, &best_local_size_, &new_masters_no,
                        &cu_crossover_creature_, &best_ever_size_, &graph_->size};

        copy_local_best_to_crossover.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "CopyLocalBestsToCrossover"));
        copy_local_best_to_crossover.SetArgs(args);
        assert
        (
            copy_local_best_to_crossover.SetBlockNoX(blocks_no)
        );
        copy_local_best_to_crossover.Run();
    }

    /*
        Create new population, step 3: copy random creatures to ensure genetic diversity
    */
    {
        SetSortedPermutation(cu_co2_ind_, random_creature_size_, graph_->size);
        RandomShuffle(cu_co2_ind_, random_creature_size_, graph_->size);

        int crossover_start = (best_ever_size_ + best_local_size_) * graph_->size;
        int randoms_start = 0;
        CopyFromTO(cu_co2_ind_, randoms_start, cu_crossover_creature_, crossover_start, random_creature_size_, graph_->size);
    }


    /*
        Repair masters order to ascending
    */
    {
        int* darray = new int[best_ever_size_ * graph_->size];
        int* darray2 = new int[best_ever_size_];
        int* tmp_j = new int [graph_->size];

        cuMemcpyDtoH((void*)darray, cu_masters_, best_ever_size_ * graph_->size * sizeof (int));
        cuMemcpyDtoH((void*)darray2, cu_masters_score_, best_ever_size_ * sizeof (int));

        for(int i = best_ever_size_ - new_masters_no; i < best_ever_size_; ++i)
        {
            assert(i >= 0);
            int rank = darray2[i];

            REP(z, graph_->size)
                tmp_j[z] = darray[i * graph_->size + z];

            int j = i - 1;
            while(j >= 0 && rank < darray2[j])
            {
                darray2[j + 1] = darray2[j];
                REP(z, graph_->size)
                    darray[ (j+1) * graph_->size + z] = darray[j * graph_->size + z];
                --j;
            }

            darray2[++j] = rank;
            REP(z, graph_->size)
               darray[ (j) * graph_->size + z] = tmp_j[z];

        }

        cuMemcpyHtoD(cu_masters_, darray,  best_ever_size_ * graph_->size * sizeof (int));
        cuMemcpyHtoD(cu_masters_score_, darray2,  best_ever_size_ * sizeof (int));

        delete[] darray;
        delete[] darray2;
        delete[] tmp_j;
    }

}

void GeneticSolver::CopyFromTO(CUdeviceptr& from, int from_start_index, CUdeviceptr& to, int to_start_index, int permutation_no, int permutation_size)
{
    CufunctionAdapter set_ones;
    int threads_no = CFLimits::MAX_DEMENSION_XY;
    int elements_no = best_local_size_ * graph_->size;
    int blocks_no = (elements_no / threads_no) + ((elements_no % threads_no == 0)? 0 : 1);

    void* args[] = {&from, &from_start_index, &to, &to_start_index,
                    &permutation_no, &permutation_size};

    set_ones.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "CopyFromTo"));
    set_ones.SetArgs(args);
    assert
    (
            set_ones.SetBlockNoX(blocks_no)
    );

    set_ones.Run();
}

int GeneticSolver::GetValueAt(CUdeviceptr& from, int pos)
{
    #ifdef DEBUG_PRINT_COMMUNICATE
        cerr << "Getting value from position: " << pos << "\n";
    #endif

    CUdeviceptr resultD;
    assert
    (
        CuAllocIntArray(resultD, 1)
    );

    int threads_no = 1;
    int blocks_no = 1;
    void* args[] = {&from, &resultD, &pos};

    CufunctionAdapter get_value_at;
    get_value_at.SetCufunction(CuFunctionsManager::GetInstance().GetCufunction("GeneticSolver", "GetValue"));
    get_value_at.SetArgs(args);
    assert
    (
            get_value_at.SetBlockNoX(blocks_no)
    );

    get_value_at.Run();

    int* result_h = new int[1];
    cuMemcpyDtoH((void*)result_h, resultD, sizeof (int));
    int real_result = result_h[0];
    cuMemFree(resultD);
    delete[] result_h;

    return real_result;
}

int* GeneticSolver::GetLeader()
{
    int* result = new int[graph_->size];
    cuMemcpyDtoH((void*)result, cu_masters_, graph_->size * sizeof (int));
    return result;
}

int GeneticSolver::GetBestScore()
{
    JudgeFunction(cu_masters_, cu_left_, 1, graph_->size);
    return GetValueAt(cu_left_, 0);
}
