/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
 **
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 **
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 **
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 **
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 **
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 **
 */

#include "kaapi_impl.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache_lru_fifo.h"

struct kaapi_cuda_mem_cache_blk_t;

typedef struct kaapi_cuda_mem_cache_blk_t {
  uintptr_t ptr;
  kaapi_access_mode_t m;
  size_t size;
  struct {
    uint64_t count;		/*  number of tasks accessing it */
  } u;
  struct kaapi_cuda_mem_cache_blk_t *next;
  struct kaapi_cuda_mem_cache_blk_t *prev;
} kaapi_cuda_mem_cache_blk_t;

typedef struct kaapi_cuda_mem_cache_lru_fifo_t {
  size_t total;
  size_t used;
  struct {
    struct kaapi_cuda_mem_cache_blk_t *beg;
    struct kaapi_cuda_mem_cache_blk_t *end;
  } fifo;
  
  /* all GPU allocated pointers */
  kaapi_big_hashmap_t kmem;
  
} kaapi_cuda_mem_cache_lru_fifo_t;

static inline int
kaapi_cuda_mem_cache_lru_fifo_has_access(kaapi_cuda_mem_cache_blk_t * blk)
{
  return (blk->u.count > 0);
}


int kaapi_cuda_mem_cache_lru_fifo_init(void** data)
{
  kaapi_cuda_mem_cache_lru_fifo_t* cache;
  
  cache = (kaapi_cuda_mem_cache_lru_fifo_t*)malloc(sizeof(kaapi_cuda_mem_cache_lru_fifo_t));
  /* 80% of total memory */
  cache->total = 0.8 * kaapi_processor_get_cudaproc(kaapi_get_current_processor())->deviceProp.totalGlobalMem;
  cache->used = 0;
  cache->fifo.beg = cache->fifo.end = 0;
  kaapi_big_hashmap_init(&cache->kmem, 0);
  *data = (void*)cache;
  
  return 0;
}

static inline void
kaapi_cuda_mem_cache_lru_fifo_insertlist(kaapi_cuda_mem_cache_lru_fifo_t * mem,
                                               kaapi_cuda_mem_cache_blk_t * blk)
{
  if (mem->fifo.beg == 0) {
    mem->fifo.beg = blk;
    blk->prev = 0;
  } else {
    blk->prev = mem->fifo.end;
    mem->fifo.end->next = blk;
  }
  mem->fifo.end = blk;
  blk->next = 0;
}

int kaapi_cuda_mem_cache_lru_fifo_insert(void* data,
                          uintptr_t ptr,
                          size_t size, kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  kaapi_cuda_mem_cache_blk_t *blk =
  (kaapi_cuda_mem_cache_blk_t *)malloc(sizeof(kaapi_cuda_mem_cache_blk_t));
  if (blk == NULL)
    return -1;
  
  blk->ptr = ptr;
  blk->m = m;
  blk->size = size;
  blk->prev = blk->next = 0;
  blk->u.count = 1;  
  kaapi_cuda_mem_cache_lru_fifo_insertlist(cache, blk);
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)ptr);
  entry->u.block = (kaapi_cuda_mem_cache_blk_t *) blk;
  cache->used += size;
  
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG_SYS(kaapi_get_current_processor(),
                     KAAPI_PERF_ID_CACHE_MISS) += 1;
#endif
  
  return 0;
}

static inline void kaapi_cuda_mem_cache_lru_fifo_check_host(uintptr_t ptr, size_t size)
{
  kaapi_memory_evict_pointer(ptr, size);
  cudaStreamSynchronize(kaapi_cuda_DtoH_stream());
}

/* remove from the current position  */
static inline void kaapi_cuda_mem_cache_lru_fifo_removefromlist(
                                              kaapi_cuda_mem_cache_lru_fifo_t* cache,
                                              kaapi_cuda_mem_cache_blk_t* blk
                                              )
{
  if (0 == blk->prev){
    cache->fifo.beg = blk->next;
    if(blk->next != 0)
      blk->next->prev = 0;
  } else
    blk->prev->next = blk->next;
  
  if(0 == blk->next){
    cache->fifo.end = blk->prev;
    if(blk->prev != 0)
      blk->prev->next = 0;
  } else
    blk->next->prev = blk->prev;
}

static inline void *kaapi_cuda_mem_cache_lru_fifo_remove_ro_and_rw(
                                                                   kaapi_cuda_mem_cache_lru_fifo_t *cache,
                                                                   const size_t size)
{
  uintptr_t ptr;
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  kaapi_cuda_mem_cache_blk_t *blk_next;
  size_t mem_free = 0;
  size_t ptr_size;
  void *devptr = NULL;
  kaapi_access_mode_t m;
  
  if (cache->fifo.beg == NULL)
    return NULL;
  
  blk = cache->fifo.beg;
  while (NULL != blk) {
    if(kaapi_cuda_mem_cache_lru_fifo_has_access(blk)) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] head in use ptr=%p (count=%lu)\n",
              __FUNCTION__, (void*)blk->ptr, blk->u.count);
      fflush(stdout);
#endif
      blk = blk->next;
      continue;
    }
    kaapi_cuda_mem_cache_lru_fifo_removefromlist(cache, blk);
    
    ptr = blk->ptr;
    ptr_size = blk->size;
    blk_next = blk->next;
    m = blk->m;
    free(blk);
    entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)ptr);
    entry->u.block = 0;

    if (KAAPI_ACCESS_IS_WRITE(m))
      kaapi_cuda_mem_cache_lru_fifo_check_host(ptr, ptr_size);
    
    if (ptr_size >= size) {
      devptr = (void*)ptr;
    } else
      kaapi_cuda_mem_free_((void*)ptr);
    mem_free += ptr_size;
    if (mem_free >= (size * KAAPI_CUDA_MEM_FREE_FACTOR))
      break;
    
    blk = blk_next;
  }
  if (cache->used < mem_free)
    cache->used = 0;
  else
    cache->used -= mem_free;
  
  return devptr;
}

void *kaapi_cuda_mem_cache_lru_fifo_remove(void* data, const size_t size)
{
  void *devptr = NULL;
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  
  devptr = kaapi_cuda_mem_cache_lru_fifo_remove_ro_and_rw(cache, size);
  
  return devptr;
}

int kaapi_cuda_mem_cache_lru_fifo_is_full(void* data, const size_t size)
{
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  if ((cache->used + size) >= (cache->total))
    return 1;
  else
    return 0;
}

int kaapi_cuda_mem_cache_lru_fifo_inc_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view,
                                                 const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  void *devptr = (void*)ptr;
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_cache_blk_t *) entry->u.block;
  
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG_SYS(kaapi_get_current_processor(),
                     KAAPI_PERF_ID_CACHE_HIT) += 1;
#endif
  
  blk->u.count++;
  kaapi_cuda_mem_cache_lru_fifo_removefromlist(cache, blk);
  kaapi_cuda_mem_cache_lru_fifo_insertlist(cache, blk);
  
  return 0;
}

int kaapi_cuda_mem_cache_lru_fifo_dec_use(void* data,
                                                 uintptr_t ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  void *devptr = (void*)ptr;
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_cache_blk_t *) entry->u.block;
    
#if defined(KAAPI_DEBUG)
  if (blk->u.count == 0) {
    fprintf(stdout, "%s:%d:%s: (kid=%lu) ERROR double free ptr=%p (count=%lu)\n",
            __FILE__, __LINE__, __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void*)blk->ptr, blk->u.count);
    fflush(stdout);
    abort();
  }
#endif				/* KAAPI_DEBUG */
  return (--blk->u.count);
}

void kaapi_cuda_mem_cache_lru_fifo_destroy(void * data)
{
  kaapi_cuda_mem_cache_blk_t *blk, *p;
  kaapi_cuda_mem_cache_lru_fifo_t *cache = (kaapi_cuda_mem_cache_lru_fifo_t*)data;
  
  blk = cache->fifo.beg;
  while (blk != NULL) {
    if (blk->ptr != 0)
      kaapi_cuda_mem_free_((void*)blk->ptr);
    p = blk;
    blk = blk->next;
    free(p);
  }
  kaapi_big_hashmap_destroy( &cache->kmem );
  cache->fifo.beg = cache->fifo.end = NULL;
}

