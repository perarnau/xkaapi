
#ifndef KAAPI_CUDA_MEM_CACHE_H_INCLUDED
#define KAAPI_CUDA_MEM_CACHE_H_INCLUDED

#include "kaapi_impl.h"

int kaapi_cuda_mem_cache_insert(kaapi_processor_t * proc,
                          uintptr_t ptr,
                          size_t size, kaapi_access_mode_t m);

void *kaapi_cuda_mem_cache_remove(kaapi_processor_t * proc,
                                       const size_t size);

int kaapi_cuda_mem_cache_inc_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view,
    const kaapi_access_mode_t m);

int kaapi_cuda_mem_cache_dec_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m);

int kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t * proc);

#endif				/* ! KAAPI_CUDA_MEM_CACHE_H_INCLUDED */
