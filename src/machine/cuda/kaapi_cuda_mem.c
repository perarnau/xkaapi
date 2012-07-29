
#include <stdio.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "machine/mt/kaapi_mt_machine.h"
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_ctx.h"


typedef struct kaapi_cuda_mem_blk_t {
  kaapi_pointer_t ptr;
  size_t size;
  union {
    uint64_t wc;		/* RW number of write tasks on the GPU (not executed yet) */
    uint64_t rc;		/* RO number of read tasks on the GPU (not executed yet) */
  } u;
  struct kaapi_cuda_mem_blk_t *next;
  struct kaapi_cuda_mem_blk_t *prev;
} kaapi_cuda_mem_blk_t;

static inline void
kaapi_cuda_mem_blk_insert_ro(kaapi_cuda_mem_t * mem,
                             kaapi_cuda_mem_blk_t * blk)
{
  if (mem->ro.beg == NULL) {
    mem->ro.beg = blk;
  } else {
    blk->prev = mem->ro.end;
    mem->ro.end->next = blk;
  }
  mem->ro.end = blk;
  blk->u.rc = 1;
}

static inline void
kaapi_cuda_mem_blk_insert_rw(kaapi_cuda_mem_t * mem,
                             kaapi_cuda_mem_blk_t * blk)
{
  if (mem->rw.beg == NULL) {
    mem->rw.beg = blk;
  } else {
    blk->prev = mem->rw.end;
    mem->rw.end->next = blk;
  }
  mem->rw.end = blk;
  blk->u.wc = 1;
}

static int
kaapi_cuda_mem_blk_insert(kaapi_processor_t * proc,
                          kaapi_pointer_t * ptr,
                          size_t size, kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_t *cuda_mem = &proc->cuda_proc.memory;
  kaapi_cuda_mem_blk_t *blk =
  (kaapi_cuda_mem_blk_t *) malloc(sizeof(kaapi_cuda_mem_blk_t));
  if (blk == NULL)
    return -1;
  
  blk->ptr = *ptr;
  blk->size = size;
  blk->prev = blk->next = NULL;
  if (KAAPI_ACCESS_IS_WRITE(m))
    kaapi_cuda_mem_blk_insert_rw(cuda_mem, blk);
  else
    kaapi_cuda_mem_blk_insert_ro(cuda_mem, blk);
  
  entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem,
                                       __kaapi_pointer2void(*ptr));
  entry->u.block = blk;
  cuda_mem->used += size;
  
  return 0;
}

static inline void *kaapi_cuda_mem_blk_remove_ro(kaapi_processor_t * proc,
                                                 const size_t size)
{
  kaapi_pointer_t ptr;
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_blk_t *blk;
  kaapi_cuda_mem_t *cuda_mem = &proc->cuda_proc.memory;
  size_t mem_free = 0;
  size_t ptr_size;
  void *devptr = NULL;
  
  if (cuda_mem->ro.beg == NULL)
    return NULL;
  
  blk = cuda_mem->ro.beg;
  while (NULL != blk) {
    if (blk->u.rc > 0) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] head in use ptr=%p (rc=%lu)\n",
              __FUNCTION__, __kaapi_pointer2void(blk->ptr), blk->u.rc);
      fflush(stdout);
#endif
      blk = blk->next;
      continue;
    }
    if (NULL == blk->prev)
      cuda_mem->ro.beg = blk->next;
    else
      blk->prev->next = blk->next;
    if (NULL != blk->next)
      blk->next->prev = blk->prev;
    
    ptr = blk->ptr;
    ptr_size = blk->size;
    free(blk);
    entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem,
                                         __kaapi_pointer2void(ptr));
    entry->u.block = NULL;
    if (ptr_size >= size) {
      devptr = __kaapi_pointer2void(ptr);
    } else
      kaapi_cuda_mem_free(&ptr);
    mem_free += ptr_size;
    if (mem_free >= (size * KAAPI_CUDA_MEM_FREE_FACTOR))
      break;
  }
  if (cuda_mem->used < mem_free)
    cuda_mem->used = 0;
  else
    cuda_mem->used -= mem_free;
  
  return devptr;
}

static inline void kaapi_cuda_mem_blk_check_host(kaapi_pointer_t ptr)
{
  const kaapi_mem_host_map_t *host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
  kaapi_mem_data_t *kmd;
  
  kaapi_mem_host_map_find_or_insert(cuda_map, (kaapi_mem_addr_t)
                                    __kaapi_pointer2void(ptr), &kmd);
  
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

static inline void *kaapi_cuda_mem_blk_remove_rw(kaapi_processor_t * proc,
                                                 const size_t size)
{
  kaapi_pointer_t ptr;
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_blk_t *blk;
  kaapi_cuda_mem_t *cuda_mem = &proc->cuda_proc.memory;
  size_t mem_free = 0;
  size_t ptr_size;
  void *devptr = NULL;
  
  if (cuda_mem->rw.beg == NULL)
    return NULL;
  
  blk = cuda_mem->rw.beg;
  while (NULL != blk) {
    if (blk->u.wc > 0) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] head in use ptr=%p (wc=%lu)\n",
              __FUNCTION__, __kaapi_pointer2void(blk->ptr), blk->u.wc);
      fflush(stdout);
#endif
      blk = blk->next;
      continue;
    }
    if (NULL == blk->prev)
      cuda_mem->rw.beg = blk->next;
    else
      blk->prev->next = blk->next;
    if (NULL != blk->next)
      blk->next->prev = blk->prev;
    
    ptr = blk->ptr;
    ptr_size = blk->size;
    free(blk);
    entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem,
                                         __kaapi_pointer2void(ptr));
    entry->u.block = NULL;
    
    kaapi_cuda_mem_blk_check_host(ptr);
    
    if (ptr_size >= size) {
      devptr = __kaapi_pointer2void(ptr);
    } else
      kaapi_cuda_mem_free(&ptr);
    mem_free += ptr_size;
    if (mem_free >= (size * KAAPI_CUDA_MEM_FREE_FACTOR))
      break;
  }
  if (cuda_mem->used < mem_free)
    cuda_mem->used = 0;
  else
    cuda_mem->used -= mem_free;
  
  return devptr;
}

/* TODO: consider the new counters */
/* TODO: extern interface */
static void *kaapi_cuda_mem_blk_remove(kaapi_processor_t * proc,
                                       const size_t size)
{
  void *devptr = NULL;
  
  devptr = kaapi_cuda_mem_blk_remove_ro(proc, size);
  if (devptr == NULL)
    devptr = kaapi_cuda_mem_blk_remove_rw(proc, size);
  
  return devptr;
}

static inline int
__kaapi_cuda_mem_is_full(kaapi_processor_t * proc, const size_t size)
{
  if ((proc->cuda_proc.memory.used + size) >=
      (proc->cuda_proc.memory.total))
    return 1;
  else
    return 0;
}

int kaapi_cuda_mem_mgmt_check(kaapi_processor_t * proc)
{
#if 0
  kaapi_cuda_mem_blk_t *blk;
  kaapi_cuda_mem_t *cuda_mem = &proc->cuda_proc.memory;
  kaapi_hashentries_t *entry;
  
  
  if ((cuda_mem->beg == NULL) && (cuda_mem->end == NULL))
    return 0;
  
  if ((cuda_mem->beg == NULL) && (cuda_mem->end != NULL)) {
    fprintf(stdout, "%s: kid=%lu ERROR beg != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) cuda_mem->beg, (void *) cuda_mem->end);
    fflush(stdout);
    return 1;
  }
  
  if ((cuda_mem->beg != NULL) && (cuda_mem->end == NULL)) {
    fprintf(stdout, "%s: kid=%lu ERROR beg != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) cuda_mem->beg, (void *) cuda_mem->end);
    fflush(stdout);
    return 1;
  }
  
  /* first check: beg to end */
  blk = cuda_mem->beg;
  while (blk->next != NULL)
    blk = blk->next;
  if (blk != cuda_mem->end) {	/* ERROR */
    fprintf(stdout, "%s: kid=%lu ERROR blk != end (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) blk, (void *) cuda_mem->end);
    fflush(stdout);
    return 1;
  }
  
  /* second check: end to beg */
  blk = cuda_mem->end;
  while (blk->prev != NULL)
    blk = blk->prev;
  if (blk != cuda_mem->beg) {
    fprintf(stdout, "%s: kid=%lu ERROR blk != beg (%p != %p)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            (void *) blk, (void *) cuda_mem->beg);
    fflush(stdout);
    return 1;
  }
  
  /* third check: hashmap */
  blk = cuda_mem->beg;
  while (blk != NULL) {
    entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem,
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

int kaapi_cuda_mem_alloc_(kaapi_mem_addr_t * addr, const size_t size)
{
  void *devptr = NULL;
  cudaError_t res;
  
  res = cudaMalloc(&devptr, size);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR cudaMalloc (%d) size=%lu kid=%lu\n",
            __FUNCTION__, res, size,
            (long unsigned int) kaapi_get_current_kid());
    fflush(stdout);
    abort();
  }
  *addr = (kaapi_mem_addr_t) devptr;
  
  return res;
}

int
kaapi_cuda_mem_alloc(kaapi_pointer_t * ptr,
                     const kaapi_address_space_id_t kasid,
                     const size_t size, const kaapi_access_mode_t m)
{
  void *devptr = NULL;
  cudaError_t res = cudaSuccess;
  kaapi_processor_t *const proc = kaapi_get_current_processor();
  
  if (__kaapi_cuda_mem_is_full(proc, size))
    devptr = kaapi_cuda_mem_blk_remove(proc, size);
  
out_of_memory:
  if (devptr == NULL) {
    res = cudaMalloc(&devptr, size);
    if (res == cudaErrorLaunchFailure) {
      fprintf(stdout, "%s: ERROR cudaMalloc (%d) size=%lu kid=%lu\n",
              __FUNCTION__, res, size,
              (long unsigned int) kaapi_get_current_kid());
      fflush(stdout);
      abort();
    }
    if (res != cudaSuccess) {
      devptr = kaapi_cuda_mem_blk_remove(proc, size);
      goto out_of_memory;
    }
  }
  
  ptr->ptr = (uintptr_t) devptr;
  ptr->asid = kasid;
  kaapi_cuda_mem_blk_insert(proc, ptr, size, m);
  
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%lu %p\n",
          __FUNCTION__,
          (unsigned long) kaapi_get_current_kid(), (void *) devptr);
  fflush(stdout);
#endif
  
  return res;
}

int kaapi_cuda_mem_free(kaapi_pointer_t * ptr)
{
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%lu %p\n",
          __FUNCTION__,
          (unsigned long) kaapi_get_current_kid(),
          __kaapi_pointer2void(*ptr));
  fflush(stdout);
#endif
  cudaFree(__kaapi_pointer2void(*ptr));
  ptr->ptr = 0;
  ptr->asid = 0;
  return 0;
}

static inline int
kaapi_cuda_mem_inc_use_ro(kaapi_cuda_mem_t * mem,
                          kaapi_cuda_mem_blk_t * blk)
{
  kaapi_cuda_mem_blk_t *blk_next;
  kaapi_cuda_mem_blk_t *blk_prev;
  
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] kid=%lu ptr=%p (rc=%lu)\n",
          __FUNCTION__,
          (long unsigned int) kaapi_get_current_kid(),
          __kaapi_pointer2void(blk->ptr), blk->u.rc + 1);
  fflush(stdout);
#endif
  
  blk->u.rc++;
  if (NULL == blk->next)
    return 0;
  
  blk_prev = blk->prev;
  blk_next = blk->next;
  /* remove */
  blk_next->prev = blk_prev;
  if (blk_prev != NULL)
    blk_prev->next = blk_next;
  else				/* first block */
    mem->ro.beg = blk_next;
  
  if (mem->ro.end != NULL)
    mem->ro.end->next = blk;
  blk->prev = mem->ro.end;
  blk->next = NULL;
  mem->ro.end = blk;
  
  return 0;
}

static inline int
kaapi_cuda_mem_inc_use_rw(kaapi_cuda_mem_t * mem,
                          kaapi_cuda_mem_blk_t * blk)
{
  kaapi_cuda_mem_blk_t *blk_next;
  kaapi_cuda_mem_blk_t *blk_prev;
  
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] kid=%lu ptr=%p (wc=%lu)\n",
          __FUNCTION__,
          (long unsigned int) kaapi_get_current_kid(),
          __kaapi_pointer2void(blk->ptr), blk->u.wc + 1);
  fflush(stdout);
#endif
  
  blk->u.wc++;
  if (NULL == blk->next)
    return 0;
  
  blk_prev = blk->prev;
  blk_next = blk->next;
  /* remove */
  blk_next->prev = blk_prev;
  if (blk_prev != NULL)
    blk_prev->next = blk_next;
  else				/* first block */
    mem->rw.beg = blk_next;
  
  if (mem->rw.end != NULL)
    mem->rw.end->next = blk;
  blk->prev = mem->rw.end;
  blk->next = NULL;
  mem->rw.end = blk;
  
  return 0;
}

int
kaapi_cuda_mem_inc_use(kaapi_pointer_t * ptr, const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_blk_t *blk;
  void *devptr = __kaapi_pointer2void(*ptr);
  kaapi_cuda_mem_t *cuda_mem =
  &kaapi_get_current_processor()->cuda_proc.memory;
  
  entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem, (void *) devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_blk_t *) entry->u.block;
  
  if (KAAPI_ACCESS_IS_WRITE(m))
    return kaapi_cuda_mem_inc_use_rw(cuda_mem, blk);
  else
    return kaapi_cuda_mem_inc_use_ro(cuda_mem, blk);
}

static inline int
kaapi_cuda_mem_dec_use_rw(kaapi_cuda_mem_t * mem,
                          kaapi_cuda_mem_blk_t * blk)
{
#if defined(KAAPI_DEBUG)
  if (blk->u.wc == 0) {
    fprintf(stdout, "[%s] kid=%lu ERROR double free ptr=%p (wc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(blk->ptr), 0);
    fflush(stdout);
    abort();
  }
#if defined(KAAPI_VERBOSE)
  else {
    fprintf(stdout, "[%s] kid=%lu ptr=%p (wc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(blk->ptr), blk->u.wc - 1);
    fflush(stdout);
  }
#endif				/* KAAPI_VERBOSE */
#endif				/* KAAPI_DEBUG */
  return (--blk->u.wc);
}

static inline int
kaapi_cuda_mem_dec_use_ro(kaapi_cuda_mem_t * mem,
                          kaapi_cuda_mem_blk_t * blk)
{
#if defined(KAAPI_DEBUG)
  if (blk->u.rc == 0) {
    fprintf(stdout, "[%s] kid=%lu ERROR double free ptr=%p (rc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(blk->ptr), 0);
    fflush(stdout);
    abort();
  }
#if defined(KAAPI_VERBOSE)
  else {
    fprintf(stdout, "[%s] kid=%lu ptr=%p (rc=%lu)\n",
            __FUNCTION__,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(blk->ptr), blk->u.rc - 1);
    fflush(stdout);
  }
#endif
#endif
  return (--blk->u.rc);
}

int
kaapi_cuda_mem_dec_use(kaapi_pointer_t * ptr, const kaapi_access_mode_t m)
{
  kaapi_hashentries_t *entry;
  kaapi_cuda_mem_blk_t *blk;
  void *devptr = __kaapi_pointer2void(*ptr);
  kaapi_cuda_mem_t *cuda_mem =
  &kaapi_get_current_processor()->cuda_proc.memory;
  
  entry = kaapi_big_hashmap_findinsert(&cuda_mem->kmem, (void *) devptr);
  if (entry->u.block == 0)
    return -1;
  blk = (kaapi_cuda_mem_blk_t *) entry->u.block;
  
  if (KAAPI_ACCESS_IS_WRITE(m))
    return kaapi_cuda_mem_dec_use_rw(cuda_mem, blk);
  else
    return kaapi_cuda_mem_dec_use_ro(cuda_mem, blk);
}

int
kaapi_cuda_mem_copy_htod_(kaapi_pointer_t dest,
                          const kaapi_memory_view_t * view_dest,
                          const kaapi_pointer_t src,
                          const kaapi_memory_view_t * view_src,
                          cudaStream_t stream)
{
#if 0
  fprintf(stdout, "[%s] src=%p dst=%p size=%ld\n", __FUNCTION__,
          __kaapi_pointer2void(src),
          __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
  fflush(stdout);
#endif
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG_SYS(kaapi_get_current_processor(),
                     KAAPI_PERF_ID_COMM_OUT) +=
  kaapi_memory_view_size(view_src);
#endif
  switch (view_src->type) {
    case KAAPI_MEMORY_VIEW_1D:
    {
      return kaapi_cuda_mem_1dcopy_htod_(dest, view_dest,
                                         src, view_src, stream);
      break;
    }
      
    case KAAPI_MEMORY_VIEW_2D:
    {
      return kaapi_cuda_mem_2dcopy_htod_(dest, view_dest,
                                         src, view_src, stream);
      break;
    }
      
      /* not supported */
    default:
    {
      kaapi_assert(0);
      goto on_error;
      break;
    }
  }
  
  return 0;
on_error:
  return -1;
}

int
kaapi_cuda_mem_copy_dtoh_(kaapi_pointer_t dest,
                          const kaapi_memory_view_t * view_dest,
                          const kaapi_pointer_t src,
                          const kaapi_memory_view_t * view_src,
                          cudaStream_t stream)
{
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] src=%p dst=%p size=%ld\n", __FUNCTION__,
          __kaapi_pointer2void(src),
          __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
  fflush(stdout);
#endif
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG_SYS(kaapi_get_current_processor(),
                     KAAPI_PERF_ID_COMM_IN) +=
  kaapi_memory_view_size(view_src);
#endif
  switch (view_src->type) {
    case KAAPI_MEMORY_VIEW_1D:
    {
      return kaapi_cuda_mem_1dcopy_dtoh_(dest, view_dest,
                                         src, view_src, stream);
      break;
    }
      
    case KAAPI_MEMORY_VIEW_2D:
    {
      return kaapi_cuda_mem_2dcopy_dtoh_(dest, view_dest,
                                         src, view_src, stream);
      break;
    }
      
      /* not supported */
    default:
    {
      kaapi_assert(0);
      goto on_error;
      break;
    }
  }
  
  return 0;
on_error:
  return -1;
}

int
kaapi_cuda_mem_register(kaapi_pointer_t ptr,
                        const kaapi_memory_view_t * view)
{
  cudaError_t res = cudaHostRegister((void *) __kaapi_pointer2void(ptr),
                                     kaapi_memory_view_size(view),
                                     cudaHostRegisterPortable);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR (%d) ptr=%p size=%lu kid=%lu\n",
            __FUNCTION__, res,
            (void *) __kaapi_pointer2void(ptr),
            kaapi_memory_view_size(view),
            (long unsigned int) kaapi_get_current_kid());
    fflush(stdout);
  }
  
  return 0;
}

int
kaapi_cuda_mem_1dcopy_htod_(kaapi_pointer_t dest,
                            const kaapi_memory_view_t * view_dest,
                            const kaapi_pointer_t src,
                            const kaapi_memory_view_t * view_src,
                            cudaStream_t stream)
{
  const size_t size = kaapi_memory_view_size(view_src);
  
#if KAAPI_CUDA_ASYNC
  const cudaError_t res = cudaMemcpyAsync(__kaapi_pointer2void(dest),
                                          __kaapi_pointer2void(src),
                                          size,
                                          cudaMemcpyHostToDevice,
                                          stream);
#else
  const cudaError_t res = cudaMemcpy(__kaapi_pointer2void(dest),
                                     __kaapi_pointer2void(src),
                                     size,
                                     cudaMemcpyHostToDevice);
#endif
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  
  return res;
}

int
kaapi_cuda_mem_1dcopy_dtoh_(kaapi_pointer_t dest,
                            const kaapi_memory_view_t * view_dest,
                            const kaapi_pointer_t src,
                            const kaapi_memory_view_t * view_src,
                            cudaStream_t stream)
{
  const size_t size = kaapi_memory_view_size(view_src);
  
#if KAAPI_CUDA_ASYNC
  const cudaError_t res = cudaMemcpyAsync(__kaapi_pointer2void(dest),
                                          __kaapi_pointer2void(src),
                                          size,
                                          cudaMemcpyDeviceToHost,
                                          stream);
#else
  const cudaError_t res = cudaMemcpy(__kaapi_pointer2void(dest),
                                     __kaapi_pointer2void(src),
                                     size,
                                     cudaMemcpyDeviceToHost);
#endif
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  
  return res;
}

int
kaapi_cuda_mem_2dcopy_htod_(kaapi_pointer_t dest,
                            const kaapi_memory_view_t * view_dest,
                            const kaapi_pointer_t src,
                            const kaapi_memory_view_t * view_src,
                            cudaStream_t stream)
{
  cudaError_t res;
  
#if KAAPI_VERBOSE
  fprintf(stdout,
          "[%s] kid=%lu src=%p %ldx%ld lda=%ld dst=%p %ldx%ld lda=%ld size=%ld\n",
          __FUNCTION__, (unsigned long) kaapi_get_current_kid(),
          __kaapi_pointer2void(src), view_src->size[0], view_src->size[1],
          view_src->lda, __kaapi_pointer2void(dest), view_dest->size[0],
          view_dest->size[1], view_dest->lda,
          kaapi_memory_view_size(view_src));
  fflush(stdout);
#endif
  
#if KAAPI_CUDA_ASYNC
  res = cudaMemcpy2DAsync(__kaapi_pointer2void(dest),
                          view_dest->lda * view_dest->wordsize,
                          __kaapi_pointer2void(src),
                          view_src->lda * view_src->wordsize,
                          view_dest->size[1] * view_dest->wordsize,
                          view_dest->size[0],
                          cudaMemcpyHostToDevice, stream);
#else
  res = cudaMemcpy2D(__kaapi_pointer2void(dest),
                     view_dest->lda * view_dest->wordsize,
                     __kaapi_pointer2void(src),
                     view_src->lda * view_src->wordsize,
                     view_dest->size[1] * view_dest->wordsize,
                     view_dest->size[0], cudaMemcpyHostToDevice);
#endif
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FUNCTION__, res,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(src),
            __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
    fflush(stdout);
    abort();
  }
  
  return res;
}

int
kaapi_cuda_mem_2dcopy_dtoh_(kaapi_pointer_t dest,
                            const kaapi_memory_view_t * view_dest,
                            const kaapi_pointer_t src,
                            const kaapi_memory_view_t * view_src,
                            cudaStream_t stream)
{
  cudaError_t res;
  
#if KAAPI_VERBOSE
  fprintf(stdout,
          "[%s] kid=%lu src=%p %ldx%ld lda=%ld dst=%p %ldx%ld lda=%ld size=%ld\n",
          __FUNCTION__, (unsigned long) kaapi_get_current_kid(),
          __kaapi_pointer2void(src), view_src->size[0], view_src->size[1],
          view_src->lda, __kaapi_pointer2void(dest), view_dest->size[0],
          view_dest->size[1], view_dest->lda,
          kaapi_memory_view_size(view_src));
  fflush(stdout);
#endif
  
#if KAAPI_CUDA_ASYNC
  res = cudaMemcpy2DAsync(__kaapi_pointer2void(dest),
                          view_dest->lda * view_dest->wordsize,
                          __kaapi_pointer2void(src),
                          view_src->lda * view_src->wordsize,
                          view_src->size[1] * view_src->wordsize,
                          view_src->size[0], cudaMemcpyDeviceToHost,
                          stream);
#else
  res = cudaMemcpy2D(__kaapi_pointer2void(dest),
                     view_dest->lda * view_dest->wordsize,
                     __kaapi_pointer2void(src),
                     view_src->lda * view_src->wordsize,
                     view_src->size[1] * view_src->wordsize,
                     view_src->size[0], cudaMemcpyDeviceToHost);
#endif
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FUNCTION__, res,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(src),
            __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
    fflush(stdout);
    abort();
  }
  
  return res;
}

int
kaapi_cuda_mem_copy_dtod_buffer(kaapi_pointer_t dest,
                                const kaapi_memory_view_t * view_dest,
                                const int dest_dev,
                                const kaapi_pointer_t src,
                                const kaapi_memory_view_t * view_src,
                                const int src_dev,
                                const kaapi_pointer_t host,
                                const kaapi_memory_view_t * view_host)
{
  cudaEvent_t event;
  cudaError_t res;
  cudaSetDevice(src_dev);
  res = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR cudaEventCreateWithFlags %d\n",
            __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  kaapi_processor_t *kproc = kaapi_cuda_get_proc_by_dev(src_dev);
  kaapi_cuda_mem_copy_dtoh_(host, view_host, src, view_src,
                            kaapi_cuda_get_cudastream
                            (kaapi_cuda_get_output_fifo
                             (kproc->cuda_proc.kstream)));
  res =
  cudaEventRecord(event,
                  kaapi_cuda_get_cudastream(kaapi_cuda_get_output_fifo
                                            (kproc->cuda_proc.
                                             kstream)));
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR cudaEventRecord %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  cudaSetDevice(dest_dev);
  cudaStreamWaitEvent(kaapi_cuda_HtoD_stream(), event, 0);
  return kaapi_cuda_mem_copy_htod(dest, view_dest, host, view_host);
}

int
kaapi_cuda_mem_copy_dtod_peer(kaapi_pointer_t dest,
                              const kaapi_memory_view_t * view_dest,
                              const int dest_dev,
                              const kaapi_pointer_t src,
                              const kaapi_memory_view_t * view_src,
                              const int src_dev)
{
  cudaError_t res;
  
  res = cudaMemcpyPeerAsync(kaapi_pointer2void(dest), dest_dev,
                            kaapi_pointer2void(src), src_dev,
                            kaapi_memory_view_size(view_src),
                            kaapi_cuda_HtoD_stream());
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: cudaMemcpyPeerAsync ERROR %d\n", __FUNCTION__,
            res);
    fflush(stdout);
    abort();
  }
  
  return 0;
}

int kaapi_cuda_mem_destroy(kaapi_cuda_proc_t * proc)
{
  kaapi_cuda_mem_blk_t *blk, *p;
  kaapi_cuda_mem_t *cuda_mem = &proc->memory;
  
  /* first check: beg to end */
  blk = cuda_mem->ro.beg;
  while (blk != NULL) {
    if (kaapi_pointer2void(blk->ptr) != NULL)
      kaapi_cuda_mem_free(&blk->ptr);
    p = blk;
    blk = blk->next;
    free(p);
  }
  blk = cuda_mem->rw.beg;
  while (blk != NULL) {
    if (kaapi_pointer2void(blk->ptr) != NULL)
      kaapi_cuda_mem_free(&blk->ptr);
    p = blk;
    blk = blk->next;
    free(p);
  }
  //    kaapi_big_hashmap_destroy( &cuda_mem->kmem );  
  cuda_mem->ro.beg = cuda_mem->ro.end = NULL;
  cuda_mem->rw.beg = cuda_mem->rw.end = NULL;
  
  return 0;
}

static inline int
kaapi_cuda_memory_pool_validate_host(kaapi_cuda_mem_t * const cuda_mem,
                                     kaapi_cuda_mem_blk_t * const blk)
{
  kaapi_mem_host_map_t *const cuda_map = kaapi_get_current_mem_host_map();
  const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
  kaapi_mem_host_map_t *const host_map =
  kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_data_t *kmd;
  
  kaapi_mem_host_map_find_or_insert(cuda_map, (kaapi_mem_addr_t)
                                    __kaapi_pointer2void(blk->ptr), &kmd);
  if (kaapi_mem_data_has_addr(kmd, cuda_asid)) {
    /* valid on the GPU and invalid on host ? */
    if ((!kaapi_mem_data_is_dirty(kmd, cuda_asid)) &&
        (kaapi_mem_data_is_dirty(kmd, host_asid))) {
#if defined(KAAPI_VERBOSE)
      fprintf(stdout, "[%s] %d -> %d\n",
              __FUNCTION__, cuda_asid - 1, host_asid);
      fflush(stdout);
#endif
      kaapi_mem_data_clear_dirty(kmd, host_asid);
      kaapi_data_t *src =
      (kaapi_data_t *) kaapi_mem_data_get_addr(kmd, cuda_asid);
      kaapi_data_t *dest =
      (kaapi_data_t *) kaapi_mem_data_get_addr(kmd, host_asid);
      /* TODO: optimize cudaSynchronize here */
      KAAPI_EVENT_PUSH0(kaapi_get_current_processor(),
                        kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG);
      kaapi_cuda_mem_copy_dtoh(dest->ptr, &dest->view, src->ptr,
                               &src->view);
      cudaEventRecord(cuda_mem->event, kaapi_cuda_DtoH_stream());
      KAAPI_EVENT_PUSH0(kaapi_get_current_processor(),
                        kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END);
      return 0;
    }
  }
  return 1;
}

int kaapi_cuda_memory_poll(kaapi_processor_t * const kproc)
{
  kaapi_cuda_mem_blk_t *blk;
  kaapi_cuda_mem_t *cuda_mem = &kproc->cuda_proc.memory;
  
  if (cuda_mem->rw.beg == NULL)
    return 1;
  
  if (cudaEventQuery(cuda_mem->event) != cudaSuccess)
    return 1;
  
  blk = cuda_mem->rw.beg;
  while (NULL != blk) {
    if ((blk->u.wc == 0) &&
        (!kaapi_cuda_memory_pool_validate_host(cuda_mem, blk)))
      return 0;
    blk = blk->next;
  }
  
  return 1;
}
