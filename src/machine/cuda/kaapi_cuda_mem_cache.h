
#ifndef KAAPI_CUDA_MEM_CACHE_H_INCLUDED
#define KAAPI_CUDA_MEM_CACHE_H_INCLUDED

#include "kaapi_impl.h"

int kaapi_cuda_mem_cache_init(kaapi_cuda_proc_t* proc);

void kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t* proc);

#endif				/* ! KAAPI_CUDA_MEM_CACHE_H_INCLUDED */
