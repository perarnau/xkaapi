
#ifndef KAAPI_CUDA_CUBLAS_H_INCLUDED
#define KAAPI_CUDA_CUBLAS_H_INCLUDED

#include "../../kaapi.h"
#include "kaapi_cuda_proc.h"
#include "cublas_v2.h"

int kaapi_cuda_cublas_init( kaapi_cuda_proc_t *proc );

void kaapi_cuda_cublas_set_stream( void );

void kaapi_cuda_cublas_finalize( kaapi_cuda_proc_t *proc );

cublasHandle_t kaapi_cuda_cublas_handle( void );

#endif /* ! KAAPI_CUDA_CUBLAS_H_INCLUDED */
