#ifndef KAAPI_CUDA_FUNC_H_INCLUDED
# define KAAPI_CUDA_FUNC_H_INCLUDED


#include <cuda.h>


typedef struct kaapi_cuda_func
{
  CUmodule mod;
  CUfunction fu;
  int off;
} kaapi_cuda_func_t;


typedef struct kaapi_cuda_dim2
{
  unsigned int x;
  unsigned int y;
} kaapi_cuda_dim2_t;


typedef struct kaapi_cuda_dim3
{
  unsigned int x;
  unsigned int y;
  unsigned int z;
} kaapi_cuda_dim3_t;


int kaapi_cuda_func_init(kaapi_cuda_func_t*);
int kaapi_cuda_func_load_ptx(kaapi_cuda_func_t*, const char*, const char*);
int kaapi_cuda_func_unload_ptx(kaapi_cuda_func_t*);
int kaapi_cuda_func_push_ptr(kaapi_cuda_func_t*, CUdeviceptr);
int kaapi_cuda_func_push_uint(kaapi_cuda_func_t*, unsigned int);
int kaapi_cuda_func_call_async
(kaapi_cuda_func_t*, CUstream, const kaapi_cuda_dim2_t*, const kaapi_cuda_dim3_t*);
int kaapi_cuda_func_wait(kaapi_cuda_func_t*, CUstream);


#endif /* ! KAAPI_CUDA_FUNC_H_INCLUDED */
