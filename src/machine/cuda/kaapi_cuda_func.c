#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda.h>
#include "kaapi_cuda_func.h"


/* debugging */

#define CONFIG_CUDA_DEBUG 0

#if CONFIG_CUDA_DEBUG
static void print_cuda_error(const char* s, CUresult e)
{
  printf("[!] %s: %u\n", s, e);
}
#else
#define print_cuda_error(__a, __b)
#endif


/* exported */

int kaapi_cuda_func_init(kaapi_cuda_func_t* fn)
{
  fn->off = 0;
  return 0;
}


int kaapi_cuda_func_load_ptx
(kaapi_cuda_func_t* fn, const char* mpath, const char* fname)
{
  CUresult res;

  res = cuModuleLoad(&fn->mod, mpath);
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuModuleLoad", res);
    goto on_error0;
  }

  res = cuModuleGetFunction(&fn->fu, fn->mod, fname);
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuModuleGetFunction", res);
    goto on_error1;
  }

  return 0;

 on_error1:
  cuModuleUnload(fn->mod);
 on_error0:
  return -1;
}


int kaapi_cuda_func_unload_ptx(kaapi_cuda_func_t* fn)
{
  cuModuleUnload(fn->mod);
  return 0;
}


int kaapi_cuda_func_push_uint(kaapi_cuda_func_t* fn, unsigned int value)
{
  CUresult res;

#define ALIGN_UP(__offset, __alignment) \
  __offset = ((__offset) + ((__alignment) - 1)) & ~((__alignment) - 1)

  ALIGN_UP(fn->off, __alignof(value));

  res = cuParamSeti(fn->fu, fn->off, value);
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuParamSeti", res);
    return -1;
  }

  fn->off += sizeof(unsigned int);

  return 0;
}


int kaapi_cuda_func_push_ptr(kaapi_cuda_func_t* fn, CUdeviceptr devptr)
{
  CUresult res;

  ALIGN_UP(fn->off, __alignof(devptr));

  res = cuParamSetv(fn->fu, fn->off, &devptr, sizeof(devptr));
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuParamSetv", res);
    return -1;
  }

  fn->off += sizeof(void*);

  return 0;
}


int kaapi_cuda_func_call_async
(
 kaapi_cuda_func_t* fn,
 CUstream stream,
 const kaapi_cuda_dim2_t* bdim,
 const kaapi_cuda_dim3_t* tdim
)
{
  CUresult res;

  /* set the number of threads */
  res = cuFuncSetBlockShape(fn->fu, tdim->x, tdim->y, tdim->z);
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuFuncSetBlockShape", res);
    goto on_error;
  }

  /* async kernel launch */
  res = cuLaunchGridAsync(fn->fu, bdim->x, bdim->y, stream);
  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuLaunchGridAsync", res);
    goto on_error;
  }

  return 0;

 on_error:
  return -1;
}


int kaapi_cuda_func_wait(kaapi_cuda_func_t* fn, CUstream stream)
{
  const CUresult res = cuStreamSynchronize(stream);

  if (res != CUDA_SUCCESS)
  {
    print_cuda_error("cuStreamSynchronize", res);
    return -1;
  }

  return 0;

}
