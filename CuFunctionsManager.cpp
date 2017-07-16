#include "CuFunctionsManager.h"

CuFunctionsManager::CuFunctionsManager()
{
    CUdevice cu_device;
    CUresult res;

    cuInit(0);

    res = cuDeviceGet(&cu_device, 0);
    assert (res == CUDA_SUCCESS && "Can't load device");

    res = cuCtxCreate(&cu_context_, 0, cu_device);
    assert (res == CUDA_SUCCESS && "Can't create context");
}

CuFunctionsManager& CuFunctionsManager::GetInstance()
{
    static CuFunctionsManager instance;
    return instance;
}

#define VALUE second

void CuFunctionsManager::LoadModule(string name)
{
    name += ".ptx";
    CUmodule cu_module = (CUmodule)0;

    assert
    (
        cuModuleLoad(&cu_module, name.c_str() ) == CUDA_SUCCESS
    );

    name = name.substr(0, name.length() - 4);
    modules_[name] = cu_module;
}

CUfunction& CuFunctionsManager::GetCufunction(const string& module_name, const string& function_name)
{
    string function_key = module_name + '@' + function_name;

    auto fun_it = functions_.find(function_key);
    if(fun_it == functions_.end())
    {
        auto mod_it = modules_.find(module_name);
        assert
        (
            mod_it != modules_.end()
        );

        CUfunction cu_function;
        assert
        (
            cuModuleGetFunction(&cu_function, mod_it->VALUE, function_name.c_str()) == CUDA_SUCCESS
        );

        fun_it = functions_.insert(make_pair(function_key, cu_function)).first;
    }

    return fun_it-> VALUE;
}

#undef VALUE

CuFunctionsManager::~CuFunctionsManager()
{
    cuCtxDestroy(cu_context_);
}


