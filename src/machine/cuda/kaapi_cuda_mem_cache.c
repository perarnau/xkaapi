
#include "kaapi_impl.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache.h"

typedef struct kaapi_cuda_mem_cache_blk_t {
  uintptr_t ptr;
  kaapi_access_mode_t m;
  size_t size;
  struct {
    uint64_t wc;		/* RW number of write tasks on the GPU (not executed yet) */
    uint64_t rc;		/* RO number of read tasks on the GPU (not executed yet) */
  } u;
  struct kaapi_cuda_mem_cache_blk_t *next;
  struct kaapi_cuda_mem_cache_blk_t *prev;
} kaapi_cuda_mem_cache_blk_t;

static inline int
kaapi_cuda_mem_cache_has_access(kaapi_cuda_mem_cache_blk_t * blk)
{
  return ((blk->u.rc > 0) || (blk->u.wc > 0));
}

static inline void
kaapi_cuda_mem_cache_insertlist_ro(kaapi_cuda_mem_cache_t * mem,
                             kaapi_cuda_mem_cache_blk_t * blk)
{
  if (mem->ro.beg == 0) {
    mem->ro.beg = blk;
    blk->prev = 0;
  } else {
    blk->prev = mem->ro.end;
    mem->ro.end->next = blk;
  }
  mem->ro.end = blk;
  blk->next = 0;
}

static inline void
kaapi_cuda_mem_cache_insertlist_rw(kaapi_cuda_mem_cache_t * mem,
                             kaapi_cuda_mem_cache_blk_t * blk)
{
  if (mem->rw.beg == 0) {
    mem->rw.beg = blk;
    blk->prev = 0;
  } else {
    blk->prev = mem->rw.end;
    mem->rw.end->next = blk;
  }
  mem->rw.end = blk;
  blk->next = 0;
}

static inline void
kaapi_cuda_mem_cache_insert_ro(kaapi_cuda_mem_cache_t * mem,
                             kaapi_cuda_mem_cache_blk_t * blk)
{
  kaapi_cuda_mem_cache_insertlist_ro(mem, blk);
  blk->u.rc = 1;
}

static inline void
kaapi_cuda_mem_cache_insert_rw(kaapi_cuda_mem_cache_t * mem,
                             kaapi_cuda_mem_cache_blk_t * blk)
{
  kaapi_cuda_mem_cache_insertlist_rw(mem, blk);
  blk->u.wc = 1;
}

int
kaapi_cuda_mem_cache_insert(kaapi_processor_t * proc,
                          uintptr_t ptr,
                          size_t size, kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_t *cache = &proc->cuda_proc.cache;
  kaapi_cuda_mem_cache_blk_t *blk = (kaapi_cuda_mem_cache_blk_t *) calloc(1, sizeof(kaapi_cuda_mem_cache_blk_t));
  if (blk == NULL)
    return -1;
  
  blk->ptr = ptr;
  blk->m = m;
  blk->size = size;
  blk->prev = blk->next = 0;
  if (KAAPI_ACCESS_IS_WRITE(m))
    kaapi_cuda_mem_cache_insert_rw(cache, blk);
  else
    kaapi_cuda_mem_cache_insert_ro(cache, blk);
  
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)ptr);
  entry->u.block = blk;
  cache->used += size;
  
  return 0;
}

/* remove from the current position  */
static inline void kaapi_cuda_mem_cache_removefromlist_ro(
      kaapi_cuda_mem_cache_t* cache,
      kaapi_cuda_mem_cache_blk_t* blk
    )
{
  if (0 == blk->prev){
    cache->ro.beg = blk->next;
    if(blk->next != 0)
      blk->next->prev = 0;
  } else
    blk->prev->next = blk->next;

  if(0 == blk->next){
    cache->ro.end = blk->prev;
    if(blk->prev != 0)
      blk->prev->next = 0;
  } else
    blk->next->prev = blk->prev;
}

/* remove from the current position  */
static inline void kaapi_cuda_mem_cache_removefromlist_rw(
      kaapi_cuda_mem_cache_t* cache,
      kaapi_cuda_mem_cache_blk_t* blk
    )
{
  if (0 == blk->prev){
    cache->rw.beg = blk->next;
    if(blk->next != 0)
      blk->next->prev = 0;
  } else
    blk->prev->next = blk->next;

  if(0 == blk->next){
    cache->rw.end = blk->prev;
    if(blk->prev != 0)
      blk->prev->next = 0;
  } else
    blk->next->prev = blk->prev;
}

static inline void *kaapi_cuda_mem_cache_remove_ro(kaapi_processor_t * proc,
                                                 const size_t size)
{
  uintptr_t ptr;
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  kaapi_cuda_mem_cache_t *cache = &proc->cuda_proc.cache;
  size_t mem_free = 0;
  size_t ptr_size;
  void *devptr = 0;
  
  if (cache->ro.beg == 0)
    return 0;
  
  blk = cache->ro.beg;
  while (0 != blk) {
    if(kaapi_cuda_mem_cache_has_access(blk)) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] head in use ptr=%p (rc=%lu)\n",
              __FUNCTION__, (void*)blk->ptr, blk->u.rc);
      fflush(stdout);
#endif
      blk = blk->next;
      continue;
    }
    kaapi_cuda_mem_cache_removefromlist_ro(cache, blk);
    ptr = blk->ptr;
    ptr_size = blk->size;
    free(blk);
    entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)ptr);
    entry->u.block = NULL;
    if (ptr_size >= size) {
      devptr = (void*)ptr;
    } else
      kaapi_cuda_mem_free_((void*)ptr);
    mem_free += ptr_size;
    if (mem_free >= (size * KAAPI_CUDA_MEM_FREE_FACTOR))
      break;
  }
  if (cache->used < mem_free)
    cache->used = 0;
  else
    cache->used -= mem_free;
  
  return devptr;
}

static inline void kaapi_cuda_mem_cache_check_host(uintptr_t ptr, size_t size)
{
  const kaapi_mem_host_map_t *host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
  kaapi_mem_data_t *kmd;
  
  kaapi_mem_host_map_find_or_insert(cuda_map,
                                    kaapi_mem_host_map_generate_id((void*)ptr, size),
				    &kmd
				  );
  
  /* valid on host ? */
  if (kaapi_mem_data_has_addr(kmd, host_asid) &&
      kaapi_mem_data_is_dirty(kmd, host_asid)) {
    /* valid on this GPU */
    if (kaapi_mem_data_has_addr(kmd, cuda_asid) &&
        !kaapi_mem_data_is_dirty(kmd, cuda_asid)) {
      kaapi_data_t *src = (kaapi_data_t *) kaapi_mem_data_get_addr(kmd,
                                                                   cuda_asid);
      kaapi_data_t *dest = (kaapi_data_t *) kaapi_mem_data_get_addr(kmd,
                                                                    host_asid);
      /* TODO: optimize cudaSynchronize here */
      kaapi_cuda_mem_copy_dtoh(dest->ptr, &dest->view,
                               src->ptr, &src->view);
      cudaStreamSynchronize(kaapi_cuda_DtoH_stream());
      kaapi_mem_data_clear_dirty(kmd, host_asid);
    }
  }
  kaapi_mem_data_clear_addr(kmd, cuda_asid);
}

static inline void *kaapi_cuda_mem_cache_remove_rw(kaapi_processor_t * proc,
                                                 const size_t size)
{
  uintptr_t ptr;
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  kaapi_cuda_mem_cache_t *cache = &proc->cuda_proc.cache;
  size_t mem_free = 0;
  size_t ptr_size;
  void *devptr = NULL;
  
  if (cache->rw.beg == NULL)
    return NULL;
  
  blk = cache->rw.beg;
  while (NULL != blk) {
    if(kaapi_cuda_mem_cache_has_access(blk)) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] head in use ptr=%p (wc=%lu)\n",
              __FUNCTION__, (void*)blk->ptr, blk->u.wc);
      fflush(stdout);
#endif
      blk = blk->next;
      continue;
    }
    kaapi_cuda_mem_cache_removefromlist_rw(cache, blk);

    ptr = blk->ptr;
    ptr_size = blk->size;
    free(blk);
    entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)ptr);
    entry->u.block = NULL;
    
    kaapi_cuda_mem_cache_check_host(ptr, ptr_size);
    
    if (ptr_size >= size) {
      devptr = (void*)ptr;
    } else
      kaapi_cuda_mem_free_((void*)ptr);
    mem_free += ptr_size;
    if (mem_free >= (size * KAAPI_CUDA_MEM_FREE_FACTOR))
      break;
  }
  if (cache->used < mem_free)
    cache->used = 0;
  else
    cache->used -= mem_free;
  
  return devptr;
}

void *kaapi_cuda_mem_cache_remove(kaapi_processor_t * proc,
                                       const size_t size)
{
  void *devptr = NULL;
  
  devptr = kaapi_cuda_mem_cache_remove_ro(proc, size);
  if (devptr == NULL)
    devptr = kaapi_cuda_mem_cache_remove_rw(proc, size);
  
  return devptr;
}

int kaapi_cuda_mem_mgmt_check(kaapi_processor_t * proc)
{
#if 0
  kaapi_cuda_mem_cache_blk_t *blk;
  kaapi_cuda_mem_cache_t *cache = &proc->cuda_proc.cache;
  kaapi_hashentries_t *entry;
  
  
  if ((cache->beg == NULL) && (cache->end == NULL))
    return 0;
  
  if ((cache->beg == NULL) && (cache->end != NULL)) {
    fprintf(stdout, "%s: kid=%lu ERROR beg != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) cache->beg, (void *) cache->end);
    fflush(stdout);
    return 1;
  }
  
  if ((cache->beg != NULL) && (cache->end == NULL)) {
    fprintf(stdout, "%s: kid=%lu ERROR beg != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) cache->beg, (void *) cache->end);
    fflush(stdout);
    return 1;
  }
  
  /* first check: beg to end */
  blk = cache->beg;
  while (blk->next != NULL)
    blk = blk->next;
  if (blk != cache->end) {	/* ERROR */
    fprintf(stdout, "%s: kid=%lu ERROR blk != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) blk, (void *) cache->end);
    fflush(stdout);
    return 1;
  }
  
  /* second check: end to beg */
  blk = cache->end;
  while (blk->prev != NULL)
    blk = blk->prev;
  if (blk != cache->beg) {
    fprintf(stdout, "%s: kid=%lu ERROR blk != beg (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) blk, (void *) cache->beg);
    fflush(stdout);
    return 1;
  }
  
  /* third check: hashmap */
  blk = cache->beg;
  while (blk != NULL) {
    entry = kaapi_big_hashmap_findinsert(&cache->kmem,
                                         __kaapi_pointer2void(blk->ptr));
    if (entry->u.block != blk) {
      fprintf(stdout,
              "%s: kid=%lu ERROR hashmap diff from list (%p != %p)\n",
              __FUNCTION__, (long unsigned int) kaapi_get_current_kid(),
              (void *) blk, (void *) entry->u.block);
      return 1;
    }
    blk = blk->next;
  }
  
#endif
  return 0;
}

static inline int
kaapi_cuda_mem_cache_inc_use_ro(kaapi_cuda_mem_cache_t * mem,
                          kaapi_cuda_mem_cache_blk_t * blk,
			  const kaapi_access_mode_t m)
{
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] kid=%lu ptr=%p (rc=%lu)\n",
          __FUNCTION__,
          (long unsigned int) kaapi_get_current_kid(),
          (void*)blk->ptr, blk->u.rc + 1);
  fflush(stdout);
#endif
  
  blk->u.rc++;
  
  if( blk->m != m ){
    kaapi_cuda_mem_cache_removefromlist_rw(mem, blk);
    kaapi_cuda_mem_cache_insertlist_ro(mem, blk);
    blk->m = m;
  } else {
    if (NULL == blk->next)
      return 0;
    kaapi_cuda_mem_cache_removefromlist_ro(mem, blk);
    kaapi_cuda_mem_cache_insertlist_ro(mem, blk);
  }
  
  return 0;
}

static inline int
kaapi_cuda_mem_cache_inc_use_rw(kaapi_cuda_mem_cache_t * mem,
                          kaapi_cuda_mem_cache_blk_t * blk,
			  const kaapi_access_mode_t m)
{
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] kid=%lu ptr=%p (wc=%lu)\n",
          __FUNCTION__,
          (long unsigned int) kaapi_get_current_kid(),
          (void*)blk->ptr, blk->u.wc + 1);
  fflush(stdout);
#endif
  
  blk->u.wc++;

  if( blk->m != m ){
    kaapi_cuda_mem_cache_removefromlist_ro(mem, blk);
    kaapi_cuda_mem_cache_insertlist_rw(mem, blk);
    blk->m = m;
  } else {
    if (NULL == blk->next)
      return 0;
    kaapi_cuda_mem_cache_removefromlist_rw(mem, blk);
    kaapi_cuda_mem_cache_insertlist_rw(mem, blk);
  }
  
  return 0;
}

int kaapi_cuda_mem_cache_inc_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view,
    const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  void *devptr = __kaapi_pointer2void(*ptr);
  kaapi_cuda_mem_cache_t *cache =
  &kaapi_get_current_processor()->cuda_proc.cache;
  
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_cache_blk_t *) entry->u.block;
  
  if (KAAPI_ACCESS_IS_WRITE(m))
    return kaapi_cuda_mem_cache_inc_use_rw(cache, blk, m);
  else
    return kaapi_cuda_mem_cache_inc_use_ro(cache, blk, m);
}

static inline int
kaapi_cuda_mem_cache_dec_use_rw(kaapi_cuda_mem_cache_t * mem,
                          kaapi_cuda_mem_cache_blk_t * blk,
			  const kaapi_access_mode_t m)
{
#if defined(KAAPI_DEBUG)
  if (blk->u.wc == 0) {
    fprintf(stdout, "[%s] kid=%lu ERROR double free ptr=%p (wc=%lu,rc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void*)blk->ptr, blk->u.wc, blk->u.rc);
    fflush(stdout);
    abort();
  }
#if defined(KAAPI_VERBOSE)
  else {
    fprintf(stdout, "[%s] kid=%lu ptr=%p (wc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void*)blk->ptr, (unsigned long int)(blk->u.wc - 1));
    fflush(stdout);
  }
#endif				/* KAAPI_VERBOSE */
#endif				/* KAAPI_DEBUG */
  return (--blk->u.wc);
}

static inline int
kaapi_cuda_mem_cache_dec_use_ro(kaapi_cuda_mem_cache_t * mem,
                          kaapi_cuda_mem_cache_blk_t * blk,
			  const kaapi_access_mode_t m)
{
#if defined(KAAPI_DEBUG)
  if (blk->u.rc == 0) {
    fprintf(stdout, "[%s] kid=%lu ERROR double free ptr=%p (rc=%lu,wc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void*)blk->ptr, blk->u.rc, blk->u.wc);
    fflush(stdout);
    abort();
  }
#if defined(KAAPI_VERBOSE)
  else {
    fprintf(stdout, "[%s] kid=%lu ptr=%p (rc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void*)blk->ptr, (unsigned long int)(blk->u.rc - 1));
    fflush(stdout);
  }
#endif
#endif
  return (--blk->u.rc);
}

int
kaapi_cuda_mem_cache_dec_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_cache_blk_t *blk;
  void *devptr = __kaapi_pointer2void(*ptr);
  kaapi_cuda_mem_cache_t *cache =
  &kaapi_get_current_processor()->cuda_proc.cache;
  
  entry = kaapi_big_hashmap_findinsert(&cache->kmem, (const void*)devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_cache_blk_t *) entry->u.block;
  
  if (KAAPI_ACCESS_IS_WRITE(m))
    return kaapi_cuda_mem_cache_dec_use_rw(cache, blk, m);
  else
    return kaapi_cuda_mem_cache_dec_use_ro(cache, blk, m);
}

int kaapi_cuda_mem_cache_destroy(kaapi_cuda_proc_t * proc)
{
  kaapi_cuda_mem_cache_blk_t *blk, *p;
  kaapi_cuda_mem_cache_t *cache = &proc->cache;
  
  blk = cache->ro.beg;
  while (blk != NULL) {
    if (blk->ptr != 0)
      kaapi_cuda_mem_free_((void*)blk->ptr);
    p = blk;
    blk = blk->next;
    free(p);
  }
  blk = cache->rw.beg;
  while (blk != NULL) {
    if (blk->ptr != 0)
      kaapi_cuda_mem_free_((void*)blk->ptr);
    p = blk;
    blk = blk->next;
    free(p);
  }
  kaapi_big_hashmap_destroy( &cache->kmem );  
  cache->ro.beg = cache->ro.end = NULL;
  cache->rw.beg = cache->rw.end = NULL;
  
  return 0;
}

