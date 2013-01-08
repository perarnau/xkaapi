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

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "machine/mt/kaapi_mt_machine.h"

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
  
  if (kaapi_cuda_mem_cache_is_full(proc, size))
    devptr = kaapi_cuda_mem_cache_remove(proc, size);
  
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
  
  kaapi_cuda_mem_cache_insert(proc, (uintptr_t)devptr, size, m);
  
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
  return kaapi_cuda_mem_cache_inc_use(kproc, ptr->ptr, view, m);
}

int
kaapi_cuda_mem_dec_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  return kaapi_cuda_mem_cache_dec_use(kproc, ptr->ptr, view, m);
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
                                const kaapi_pointer_t src,
                                const kaapi_memory_view_t * view_src,
                                const kaapi_pointer_t host,
                                const kaapi_memory_view_t * view_host
                                )
{
  cudaError_t res;
  cudaStream_t stream;
  kaapi_processor_t* const kproc_src = kaapi_all_kprocessors[kaapi_memory_map_asid2kid(kaapi_pointer2asid(src))];
  kaapi_processor_t* const kproc_dest = kaapi_all_kprocessors[kaapi_memory_map_asid2kid(kaapi_pointer2asid(dest))];
  const int src_dev = kaapi_processor_get_cudaproc(kproc_src)->index;
  const int dest_dev = kaapi_processor_get_cudaproc(kproc_dest)->index;

  kaapi_cuda_ctx_exit(dest_dev);
  kaapi_cuda_ctx_set(src_dev);
  res = cudaStreamCreate(&stream);
  kaapi_assert_debug( res == cudaSuccess );
  res = kaapi_cuda_mem_copy_dtoh_(host, view_host, src, view_src, stream );
  kaapi_assert_debug( res == cudaSuccess );
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG);
  res = cudaStreamSynchronize(stream);
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END);
  kaapi_assert_debug( res == cudaSuccess );
  cudaStreamDestroy(stream);
  kaapi_cuda_ctx_exit(src_dev);
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

int
kaapi_cuda_mem_copy_dtoh_from_host(kaapi_pointer_t dest,
                                   const kaapi_memory_view_t * view_dest,
                                   const kaapi_pointer_t src,
                                   const kaapi_memory_view_t * view_src,
                                   kaapi_processor_id_t kid_src )
{
  cudaError_t res;
  kaapi_processor_t *const kproc = kaapi_all_kprocessors[kid_src];
  const int src_dev = kaapi_processor_get_cudaproc(kproc)->index;
  cudaStream_t stream;
  
  kaapi_cuda_ctx_set(src_dev);
  res = cudaStreamCreate(&stream);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR cudaStreamCreate %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  kaapi_cuda_mem_copy_dtoh_(dest, view_dest, src, view_src, stream);
  res = cudaStreamSynchronize(stream);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: ERROR cudaStreamSynchronize %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  cudaStreamDestroy(stream);
  kaapi_cuda_ctx_exit(src_dev);
  
  return 0;
}

void kaapi_cuda_mem_destroy(kaapi_cuda_proc_t * proc)
{
  proc->cache.destroy(proc->cache.data);
}

