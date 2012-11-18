
#include "kaapi_impl.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache.h"
#include "kaapi_cuda_mem_cache_lru_double_fifo.h"

int kaapi_cuda_mem_cache_init(kaapi_cuda_proc_t* proc)
{
  proc->cache.init = &kaapi_cuda_mem_cache_lru_double_fifo_init;
  proc->cache.destroy = &kaapi_cuda_mem_cache_lru_double_fifo_destroy;
  proc->cache.insert = &kaapi_cuda_mem_cache_lru_double_fifo_insert;
  proc->cache.remove = &kaapi_cuda_mem_cache_lru_double_fifo_remove;
  proc->cache.is_full = &kaapi_cuda_mem_cache_lru_double_fifo_is_full;
  proc->cache.inc_use = &kaapi_cuda_mem_cache_lru_double_fifo_inc_use;
  proc->cache.dec_use = &kaapi_cuda_mem_cache_lru_double_fifo_dec_use;
  
  proc->cache.init( &proc->cache.data );
  
  return 0;
}

void kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t* proc)
{
  proc->cache.destroy( proc->cache.data );
}
