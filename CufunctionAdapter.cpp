#include "CufunctionAdapter.h"

CufunctionAdapter::CufunctionAdapter()
{
    thread_no_x_ = CFLimits::MAX_DEMENSION_XY;
    thread_no_y_ = 1;
    thread_no_z_ = 1;

    block_no_x_ = CFLimits::MAX_BLOCKS;
    block_no_y_ = 1;
    block_no_z_ = 1;
}

CufunctionAdapter::~CufunctionAdapter()
{

}

void CufunctionAdapter::SetCufunction(CUfunction& cu_function)
{
    cu_function_ = cu_function;
}

void CufunctionAdapter::SetThreadNoX(int thread_no_x)
{
    thread_no_x_ = thread_no_x;
}

void CufunctionAdapter::SetThreadNoY(int thread_no_y)
{
    thread_no_y_ = thread_no_y;
}

void CufunctionAdapter::SetThreadNoZ(int thread_no_z)
{
    thread_no_z_ = thread_no_z;
}

bool CufunctionAdapter::SetBlockNoX(int block_no_x)
{
    block_no_x_ = block_no_x;
    return block_no_x < CFLimits::MAX_BLOCKS;
}

void CufunctionAdapter::SetBlockNoY(int block_no_y)
{
    block_no_y_ = block_no_y;
}

void CufunctionAdapter::SetBlockNoZ(int block_no_z)
{
    block_no_z_ = block_no_z;
}

void CufunctionAdapter::SetArgs(void** args)
{
    args_ = args;
}

void CufunctionAdapter::Run()
{
    cuCtxSynchronize();
    cuLaunchKernel(cu_function_, block_no_x_, block_no_y_, block_no_z_, thread_no_x_, thread_no_y_, thread_no_z_, 0, 0, args_, 0);
}

bool CudaFunction::CuAllocIntArray(CUdeviceptr& dest, int size)
{
   return (cuMemAlloc(&dest, size * sizeof (int))  == CUDA_SUCCESS);
}
