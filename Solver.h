#ifndef SOLVE_H
#define	SOLVE_H

#include "Graph.h"
#include "utils.h"
#include <cuda.h>

using namespace std;

class Solver 
{
public:
    Solver();
    virtual ~Solver();
    
    virtual void RunStage() = 0;
    virtual int* GetLeader() = 0;
    virtual int GetBestScore() = 0;
    virtual void Prepare() = 0;

    void SetGraph(Graph* graph);

protected:
    Graph* graph_;
    CUdeviceptr cu_graph_;
};

#endif	/* SOLVE_H */

