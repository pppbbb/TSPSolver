#ifndef CUFUNCTIONSMANAGER_H
#define	CUFUNCTIONSMANAGER_H

#include "utils.h"

#include <cuda.h>
#include <assert.h>
#include <string>
#include <map>
#include <iostream>

class CuFunctionsManager
{
public:
    static CuFunctionsManager& GetInstance();
    virtual ~CuFunctionsManager();
    void LoadModule(string name);
    CUfunction& GetCufunction(const string& module_name, const string& function_name);
    CUcontext cu_context_;
private:
    CuFunctionsManager();

    map <string, CUmodule> modules_;
    map <string, CUfunction> functions_;
    CUfunction cu_function;
};

#endif	/* CUFUNCTIONSMANAGER_H */
