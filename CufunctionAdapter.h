#ifndef CUFUNCTIONADAPER_H
#define	CUFUNCTIONADAPER_H

#include "utils.h"
#include <cuda.h>

enum CFLimits
{
    MAX_BLOCKS = 65535,
    MAX_DEMENSION_XY = 1024,
    MAX_DEMENSION_Z = 64
};

namespace CudaFunction
{
    bool CuAllocIntArray(CUdeviceptr& dest, int size);
};

class CufunctionAdapter
{
public:
    CufunctionAdapter();
    virtual ~CufunctionAdapter();

    void SetCufunction(CUfunction& cu_function);
    void SetThreadNoX(int thread_no_x);
    bool SetBlockNoX(int block_no_x);
    void SetThreadNoY(int thread_no_y);
    void SetBlockNoY(int block_no_y);
    void SetThreadNoZ(int thread_no_z);
    void SetBlockNoZ(int block_no_z);
    void SetArgs(void** args);

    void Run();
private:
    CUfunction cu_function_;

    int thread_no_x_;
    int thread_no_y_;
    int thread_no_z_;

    int block_no_x_;
    int block_no_y_;
    int block_no_z_;

    void** args_;
};

#endif  /* CUFUNCTIONADAPER_H */

