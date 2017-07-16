#include "Solver.h"
#include "CuFunctionsManager.h"

Solver::Solver()
{
    
}

Solver::~Solver()
{
    cuMemFree(cu_graph_);
}

void Solver::SetGraph(Graph* graph)
{
    graph_ = graph;
    
    CuFunctionsManager::GetInstance();
    
    int matrix_size = sizeof (*graph_->matrix) * graph_->size * graph_->size;
    cuMemHostRegister(graph_->matrix, matrix_size, CU_MEMHOSTREGISTER_PORTABLE);
    cuMemAlloc(&cu_graph_, matrix_size);
    cuMemcpyHtoD(cu_graph_, graph_->matrix, matrix_size);
    cuMemHostUnregister(graph_->matrix);
}
