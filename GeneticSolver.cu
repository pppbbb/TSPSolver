#include <cstdio>

extern "C"
{
    __global__
    void CreateCreatures(int* creatures, int creatures_no, int creature_size)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < creatures_no * creature_size)
            creatures[pos] = (pos % creature_size);
    }

    __global__
    void BitonicSortStep(int* creatures, int* ranks, int j, int k)
    {
        unsigned int i, ixj;
        i = threadIdx.x + blockDim.x * blockIdx.x;
        ixj = i^j;

        if ((ixj) > i)
        {
            if((i & k) == 0)
                if(ranks[i] > ranks[ixj]) // swap
                {
                    int temp = ranks[i];
                    ranks[i] = ranks[ixj];
                    ranks[ixj] = temp;

                    temp = creatures[i];
                    creatures[i] = creatures[ixj];
                    creatures[ixj] = temp;
                }

            if((i&k)!=0)
                if(ranks[i]<ranks[ixj]) // swap
                {
                    int temp = ranks[i];
                    ranks[i] = ranks[ixj];
                    ranks[ixj] = temp;

                    temp = creatures[i];
                    creatures[i] = creatures[ixj];
                    creatures[ixj] = temp;
                }
          }
    }

    __global__
    void WhichReverse(int* array, int permutation_size, int permutation_no, int* to_reverse)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < permutation_size * permutation_no)
        {
            if(pos % permutation_size != 0)
            {
                if(array[pos - 1] > array[pos])
                {
                    to_reverse[pos / permutation_size] = 1;
                }
            }
        }
    }


    __global__
    void BitonicToAscendingOrder(int* array, int* ranks, int* to_reverse, int permutation_size, int permutation_no)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);

        if(pos < permutation_no * permutation_size && to_reverse[pos/permutation_size] == 1)
        {
            if(pos % permutation_size < (permutation_size / 2))
            {
                int second_pos = pos - (pos% permutation_size) * 2 + permutation_size - 1;

                int tmp = array[second_pos];
                array[second_pos] = array[pos];
                array[pos] = tmp;

                tmp = ranks[second_pos];
                ranks[second_pos] = ranks[pos];
                ranks[pos] = tmp;
            }
        }
    }

    __global__
    void SetRandomRanks(int seed, int* ranks, int size)
    {
        int thid = ((blockIdx.x * blockDim.x) + threadIdx.x ) + 1;
        if(thid - 1 < size)
            ranks[thid - 1] = (  (((( (long long)seed * thid) % 24837) + ((long long)seed * thid % 21447))) % 21474)% 21474;
    }

    __global__
    void SetMutationRanks(int* ranks, int seed, int probability, int rank_difference, int permutation_size, int permutation_no)
    {
        int thid = ((blockIdx.x * blockDim.x) + threadIdx.x ) + 1;

        if(thid - 1 < permutation_size * permutation_no)
        {
            int pro_rand = (( (long long) seed * thid ) % 2000001557) *  ( ((long long) seed * thid) % 2000001557) % 1000000000;

            if(pro_rand < probability)
               ranks[thid - 1] = (  (((( (long long)seed * thid) % 2147483647) + ((long long)seed * thid % 2147483647)) * (long long)seed ) % 2147483647)% 2147483647;
            else
                ranks[thid - 1] = ((thid - 1) % permutation_size) * rank_difference;
        }
    }

    __global__
    void SetValue(int* array, int size, int value)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < size)
            array[pos] = value;
    }

    __global__
    void AssignHalfPopulation(int* crossover_array, int* population, int crossover_no, int creature_size, int begin_point, int* left)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if((pos < crossover_no * crossover_no * creature_size) && ((pos % creature_size) < creature_size / 2))
        {
            population[pos] = crossover_array[(pos / (creature_size * crossover_no)) * creature_size + (pos % creature_size) + begin_point];
            left[pos - (pos % creature_size) + population[pos]] = 0;
        }
    }

    __global__
    void FindMissingGens(int* crossover_array, int* left, int* co2_ind, int crossover_no, int creature_size)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);

        if(pos < crossover_no * crossover_no * creature_size)
        {
            if(left[pos - (pos % creature_size) + crossover_array[pos % (crossover_no * creature_size)]] == 1)
                co2_ind[pos] = 1;
            else
                co2_ind[pos] = 0;
        }
    }

    __global__
    void FillRestPopulation(int* crossover_array, int* population, int* prefix_sum, int crossover_no, int creature_size, int begin_point)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        int to_fill = pos - (pos % creature_size) + begin_point;
        if(pos < crossover_no * crossover_no * creature_size)
        {
            if(pos % creature_size == 0)
            {
                if(prefix_sum[pos] == 1)
                    population[to_fill + 1] = crossover_array[pos % (crossover_no * creature_size)];

            }
            else
            {
                if(prefix_sum[pos - 1] < prefix_sum[pos])
                    population[to_fill + prefix_sum[pos]] = crossover_array[pos % (crossover_no * creature_size)];
            }

        }
    }

    __global__
    void PrefixSumPrecalculate(int* array, int* blocks_sum, int permutation_size, int blocks_per_permutation)
    {
        int my_permutation = blockIdx.x / blocks_per_permutation;
        int perm_pos = ((blockIdx.x % blocks_per_permutation) * blockDim.x) + threadIdx.x;
        int a_pos = (my_permutation * permutation_size) + perm_pos;

        __shared__ int sum[1024];

        if(perm_pos < permutation_size)
            sum[threadIdx.x] = array[a_pos];
        else
            sum[threadIdx.x] = 0;

        int tmp_sum = 0;

        for(int i = 1; i < 1024; i*= 2)
        {
            syncthreads();

            if(threadIdx.x >= i)
                tmp_sum = sum[threadIdx.x - i];
            else
                tmp_sum = 0;

            syncthreads();

            sum[threadIdx.x] += tmp_sum;
        }

        syncthreads();

        if(threadIdx.x == 0)
            blocks_sum[blockIdx.x] = sum[1023];
    }

    __global__
    void CalculateBlocksPrefixSum(int* blocks_sum, int* addition, int blocks_per_permutation)
    {
        if(threadIdx.x == 0)
        {
            int end = (blockIdx.x + 1) * blocks_per_permutation;
            addition[blockIdx.x * blocks_per_permutation] = 0;

            for(int i = (blockIdx.x * blocks_per_permutation) + 1; i < end; ++i)
                addition[i] = addition[i - 1] + blocks_sum[i - 1];
        }
    }

    __global__
    void CalculateFinalPrefixSum(int* array, int* addition, int permutation_size, int blocks_per_permutation)
    {
        int my_permutation = blockIdx.x / blocks_per_permutation;
        int perm_pos = ((blockIdx.x % blocks_per_permutation) * blockDim.x) + threadIdx.x;
        int a_pos = (my_permutation * permutation_size) + perm_pos;

        __shared__ int sum[1024];

        if(perm_pos < permutation_size)
            sum[threadIdx.x] = array[a_pos];
        else
            sum[threadIdx.x] = 0;

        if(threadIdx.x == 0)
            sum[0] += addition[blockIdx.x];

        int tmp_sum = 0;

        for(int i = 1; i < 1024; i*= 2)
        {
            syncthreads();

            if(threadIdx.x >= i)
                tmp_sum = sum[threadIdx.x - i];
            else
                tmp_sum = 0;

            syncthreads();

            sum[threadIdx.x] += tmp_sum;
        }

        syncthreads();

        if(perm_pos < permutation_size)
            array[a_pos] = sum[perm_pos];
    }

     __global__
    void SetPathWeight(int* population, int* weight, int* graph, int creatures_no, int creature_size)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);

        if(pos < creature_size * creatures_no)
        {
            int from = population[pos];
            int to = population[pos + 1 - (((pos % creature_size) == (creature_size - 1))? creature_size: 0)];
            weight[pos] = graph[from * creature_size + to];
        }
    }

    __global__
    void FillRanksFromDistance(int* distance, int* ranks, int creatures_no, int creature_size)
    {
        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < creatures_no)
            ranks[pos] = distance[((pos + 1) * creature_size) - 1];
    }

    __global__
    void SetWhoBecomeMaster(int* results, int* population_scores, int* bests_scores, int bests_size) {

        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);

        if(pos < bests_size)
            if(population_scores[pos] < bests_scores[bests_size - 1 - pos])
                results[pos] = 1;
            else
                results[pos] = 0;
    }

    __global__
    void AddNewMasters(int new_masters_no, int* population, int* masters, int* population_scores, int* population_ranking,
                       int* bests_scores, int bests_size, int individual_size)
    {

        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos <(new_masters_no  * individual_size ))
        {
            int which = pos/individual_size;
            int index = population_ranking[which];
            int elem = pos % individual_size;
            int dest_index = bests_size - new_masters_no + which;
            masters[dest_index * individual_size + elem] = population[index * individual_size + elem];
            bests_scores[dest_index] = population_scores[which];
        }
    }

    __global__
    void CopyFromTo(int* from, int from_start_index, int* to, int to_start_index, int permutation_no, int permutation_size){

        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < (permutation_no * permutation_size))
        {
            to[pos + to_start_index] = from[pos + from_start_index];
        }
    }

    __global__
    void CopyLocalBestsToCrossover(int* population, int* population_ranking, int best_population_size, int new_bests_no, int* crossover, int bests_size, int individual_size)
    {

        int pos = ((blockIdx.x * blockDim.x) + threadIdx.x);
        if(pos < (best_population_size  * individual_size ))
        {
            int which = pos/individual_size + new_bests_no;
            int index = population_ranking[which];
            int elem = pos % individual_size;
            int dest_index = bests_size + (pos/individual_size);

            crossover[dest_index * individual_size + elem] = population[index * individual_size + elem];
        }
    }

    __global__
    void GetValue(int* from, int* to, int at)
    {
        if(threadIdx.x == 0)
            to[0] = from[at];
    }
}
