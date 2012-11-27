
#include <stdio.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "machine/mt/kaapi_mt_machine.h"
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_mem_cache.h"

uintptr_t kaapi_cuda_mem_alloc_(const size_t size)
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
  
  return (uintptr_t)devptr;
}

uintptr_t
kaapi_cuda_mem_alloc(
                     const kaapi_address_space_id_t kasid,
                     const size_t size,
                     const kaapi_access_mode_t m
                     )
{
  void *devptr = NULL;
  cudaError_t res = cudaSuccess;
  kaapi_processor_t *const proc = kaapi_get_current_processor();
  
  if (proc->cuda_proc.cache.is_full(proc->cuda_proc.cache.data, size))
    devptr = proc->cuda_proc.cache.remove(proc->cuda_proc.cache.data, size);
  
  if (devptr == NULL) {
    res = cudaMalloc(&devptr, size);
#if 0
    if (res == cudaErrorMemoryAllocation) {
      devptr = kaapi_cuda_mem_cache_remove(proc, size);
      goto out_of_memory;
    }
#endif
    if (res != cudaSuccess) {
      fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu size=%lu\n",
              __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
              (long unsigned int)kaapi_get_current_kid(),
              size );
      fflush(stdout);
      abort();
    }
  }
  
  proc->cuda_proc.cache.insert(proc->cuda_proc.cache.data, (uintptr_t)devptr, size, m);
  
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%lu dev=%d ptr=%p size=%lu\n",
          __FUNCTION__,
          (unsigned long) kaapi_get_current_kid(), kaapi_cuda_self_device(), (void *)devptr, size);
  fflush(stdout);
#endif
  
  return (uintptr_t)devptr;
}

int kaapi_cuda_mem_free_(void* ptr)
{
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%lu %p\n",
          __FUNCTION__,
          (unsigned long) kaapi_get_current_kid(), ptr);
  fflush(stdout);
#endif
  cudaFree(ptr);
  return 0;
}

int kaapi_cuda_mem_free(kaapi_pointer_t ptr)
{
#if KAAPI_VERBOSE
  fprintf(stdout, "[%s] kid=%lu %p\n",
          __FUNCTION__,
          (unsigned long) kaapi_get_current_kid(),
          __kaapi_pointer2void(ptr));
  fflush(stdout);
#endif
  cudaFree(__kaapi_pointer2void(ptr));
  return 0;
}

int
kaapi_cuda_mem_inc_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view,
    const kaapi_access_mode_t m)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  return kproc->cuda_proc.cache.inc_use( kproc->cuda_proc.cache.data, ptr->ptr, view, m);
}

int
kaapi_cuda_mem_dec_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  return kproc->cuda_proc.cache.dec_use(kproc->cuda_proc.cache.data, ptr->ptr, view, m);
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
    fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(src),
            __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
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
    fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(src),
            __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
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
    fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
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
    fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
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
  cudaError_t res;
  cudaStream_t stream;

  kaapi_cuda_ctx_set(src_dev);
  res = cudaStreamCreate(&stream);
  kaapi_assert_debug( res == cudaSuccess );
  res = kaapi_cuda_mem_copy_dtoh_(host, view_host, src, view_src, stream );
  kaapi_assert_debug( res == cudaSuccess );
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(),
		    kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG);
  res = cudaStreamSynchronize(stream);
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(),
		    kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END);
  kaapi_assert_debug( res == cudaSuccess );
  cudaStreamDestroy(stream);
  kaapi_cuda_ctx_set(dest_dev);

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
    fprintf(stdout, "%s:%d:%s: ERROR %s (%d) kid=%lu src=%p dst=%p size=%lu\n",
            __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(res), res,
            (long unsigned int) kaapi_get_current_kid(),
            __kaapi_pointer2void(src),
            __kaapi_pointer2void(dest), kaapi_memory_view_size(view_src));
    fflush(stdout);
    abort();
  }
  
  return 0;
}

void kaapi_cuda_mem_destroy(kaapi_cuda_proc_t * proc)
{
  proc->cache.destroy(proc->cache.data);
}

