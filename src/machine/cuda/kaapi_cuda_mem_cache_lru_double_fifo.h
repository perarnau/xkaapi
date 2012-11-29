
#ifndef KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED
#define KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED

#include "kaapi_impl.h"

int kaapi_cuda_mem_cache_lru_double_fifo_init(void** data);

int kaapi_cuda_mem_cache_lru_double_fifo_insert(void* data,
                          uintptr_t ptr,
                          size_t size, kaapi_access_mode_t m);

void *kaapi_cuda_mem_cache_lru_double_fifo_remove(void* data, const size_t size);

int kaapi_cuda_mem_cache_lru_double_fifo_is_full(void* data, const size_t size);

int kaapi_cuda_mem_cache_lru_double_fifo_inc_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view,
                                                 const kaapi_access_mode_t m);

int kaapi_cuda_mem_cache_lru_double_fifo_dec_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m);

void kaapi_cuda_mem_cache_lru_double_fifo_destroy(void * proc);

#endif				/* ! KAAPI_CUDA_MEM_CACHE_LRU_DOUBLE_FIFO_H_INCLUDED */
