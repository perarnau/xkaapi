
#include "kaapi_impl.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache.h"
#include "kaapi_cuda_mem_cache_lru_fifo.h"
#include "kaapi_cuda_mem_cache_lru_double_fifo.h"

static inline void kaapi_cuda_mem_cache_lru_fifo_config(kaapi_cuda_proc_t* proc)
{
  proc->cache.init = &kaapi_cuda_mem_cache_lru_fifo_init;
  proc->cache.destroy = &kaapi_cuda_mem_cache_lru_fifo_destroy;
  proc->cache.insert = &kaapi_cuda_mem_cache_lru_fifo_insert;
  proc->cache.remove = &kaapi_cuda_mem_cache_lru_fifo_remove;
  proc->cache.is_full = &kaapi_cuda_mem_cache_lru_fifo_is_full;
  proc->cache.inc_use = &kaapi_cuda_mem_cache_lru_fifo_inc_use;
  proc->cache.dec_use = &kaapi_cuda_mem_cache_lru_fifo_dec_use;
}

static inline void kaapi_cuda_mem_cache_lru_double_fifo_config(kaapi_cuda_proc_t* proc)
{
  proc->cache.init = &kaapi_cuda_mem_cache_lru_double_fifo_init;
  proc->cache.destroy = &kaapi_cuda_mem_cache_lru_double_fifo_destroy;
  proc->cache.insert = &kaapi_cuda_mem_cache_lru_double_fifo_insert;
  proc->cache.remove = &kaapi_cuda_mem_cache_lru_double_fifo_remove;
  proc->cache.is_full = &kaapi_cuda_mem_cache_lru_double_fifo_is_full;
  proc->cache.inc_use = &kaapi_cuda_mem_cache_lru_double_fifo_inc_use;
  proc->cache.dec_use = &kaapi_cuda_mem_cache_lru_double_fifo_dec_use;
}

int kaapi_cuda_mem_cache_init(kaapi_cuda_proc_t* proc)
{
  const char* gpucache = getenv("KAAPI_GPU_CACHE");
  if( gpucache != 0 )
  {
    if (strcmp(gpucache, "lru") == 0)
      kaapi_cuda_mem_cache_lru_fifo_config(proc);
    else if (strcmp(gpucache, "lru_double") == 0)
      kaapi_cuda_mem_cache_lru_double_fifo_config(proc);
    else {
      fprintf(stdout, "%s:%d:%s: *** Kaapi: bad value for 'KAAPI_GPU_CACHE': '%s'\n",
              __FILE__, __LINE__, __FUNCTION__, gpucache );
      fflush(stdout);
      abort();
      return EINVAL;
    }
  }
  else
    kaapi_cuda_mem_cache_lru_double_fifo_config(proc);
  
  proc->cache.init( &proc->cache.data );
  
  return 0;
}

void kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t* proc)
{
  proc->cache.destroy( proc->cache.data );
}
