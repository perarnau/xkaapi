
#ifndef KAAPI_CUDA_MEM_H_INCLUDED
#define KAAPI_CUDA_MEM_H_INCLUDED

#include "../../kaapi.h"
#include "kaapi_cuda_proc.h"
#include <cuda_runtime_api.h>

int kaapi_cuda_mem_free(kaapi_pointer_t * ptr);

int kaapi_cuda_mem_free_(void* ptr);

int kaapi_cuda_mem_alloc(kaapi_pointer_t * ptr,
			 const kaapi_address_space_id_t kasid,
			 const size_t size, const kaapi_access_mode_t m);

int kaapi_cuda_mem_alloc_(kaapi_mem_addr_t * addr, const size_t size);

int kaapi_cuda_mem_register(kaapi_pointer_t ptr,
			    const kaapi_memory_view_t * view);

/* Mark the GPU memory ptr as in use by some GPU task.
 * It returns the number of tasks using ptr. */
int
kaapi_cuda_mem_inc_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m);

/* Mark the GPU memory ptr as not any more in use by some GPU task.
 * It returns the number of tasks using ptr */
int
kaapi_cuda_mem_dec_use(kaapi_pointer_t * ptr, kaapi_memory_view_t* const view, const kaapi_access_mode_t m);

/*****************************************************************************/
/* Memory copy functions */

int kaapi_cuda_mem_copy_htod_(kaapi_pointer_t dest,
			      const kaapi_memory_view_t * view_dest,
			      const kaapi_pointer_t src,
			      const kaapi_memory_view_t * view_src,
			      cudaStream_t stream);

int kaapi_cuda_mem_copy_dtoh_(kaapi_pointer_t dest,
			      const kaapi_memory_view_t * view_dest,
			      const kaapi_pointer_t src,
			      const kaapi_memory_view_t * view_src,
			      cudaStream_t stream);

int kaapi_cuda_mem_1dcopy_htod_(kaapi_pointer_t dest,
				const kaapi_memory_view_t * view_dest,
				const kaapi_pointer_t src,
				const kaapi_memory_view_t * view_src,
				cudaStream_t stream);

int kaapi_cuda_mem_1dcopy_dtoh_(kaapi_pointer_t dest,
				const kaapi_memory_view_t * view_dest,
				const kaapi_pointer_t src,
				const kaapi_memory_view_t * view_src,
				cudaStream_t stream);

int kaapi_cuda_mem_2dcopy_htod_(kaapi_pointer_t dest,
				const kaapi_memory_view_t * view_dest,
				const kaapi_pointer_t src,
				const kaapi_memory_view_t * view_src,
				cudaStream_t stream);

int kaapi_cuda_mem_2dcopy_dtoh_(kaapi_pointer_t dest,
				const kaapi_memory_view_t * view_dest,
				const kaapi_pointer_t src,
				const kaapi_memory_view_t * view_src,
				cudaStream_t stream);

static inline int
kaapi_cuda_mem_copy_htod(kaapi_pointer_t dest,
			 const kaapi_memory_view_t * view_dest,
			 const kaapi_pointer_t src,
			 const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_copy_htod_(dest, view_dest,
				   src, view_src,
				   kaapi_cuda_HtoD_stream());
}

static inline int
kaapi_cuda_mem_copy_dtoh(kaapi_pointer_t dest,
			 const kaapi_memory_view_t * view_dest,
			 const kaapi_pointer_t src,
			 const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_copy_dtoh_(dest, view_dest,
				   src, view_src,
				   kaapi_cuda_DtoH_stream());
}

static inline int
kaapi_cuda_mem_1dcopy_htod(kaapi_pointer_t dest,
			   const kaapi_memory_view_t * view_dest,
			   const kaapi_pointer_t src,
			   const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_1dcopy_htod_(dest, view_dest,
				     src, view_src,
				     kaapi_cuda_HtoD_stream());
}

static inline int
kaapi_cuda_mem_1dcopy_dtoh(kaapi_pointer_t dest,
			   const kaapi_memory_view_t * view_dest,
			   const kaapi_pointer_t src,
			   const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_1dcopy_dtoh_(dest, view_dest,
				     src, view_src,
				     kaapi_cuda_DtoH_stream());
}

static inline int
kaapi_cuda_mem_2dcopy_htod(kaapi_pointer_t dest,
			   const kaapi_memory_view_t * view_dest,
			   const kaapi_pointer_t src,
			   const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_2dcopy_htod_(dest, view_dest,
				     src, view_src,
				     kaapi_cuda_HtoD_stream());
}

static inline int
kaapi_cuda_mem_2dcopy_dtoh(kaapi_pointer_t dest,
			   const kaapi_memory_view_t * view_dest,
			   const kaapi_pointer_t src,
			   const kaapi_memory_view_t * view_src)
{
  return kaapi_cuda_mem_2dcopy_dtoh_(dest, view_dest,
				     src, view_src,
				     kaapi_cuda_DtoH_stream());
}

/*****************************************************************************/

int kaapi_cuda_mem_sync_params(kaapi_thread_context_t * thread,
			       kaapi_taskdescr_t * td, kaapi_task_t * pc);

int kaapi_cuda_mem_sync_params_dtoh(kaapi_thread_context_t * thread,
				    kaapi_taskdescr_t * td,
				    kaapi_task_t * pc);

int kaapi_cuda_mem_mgmt_check(kaapi_processor_t * proc);

int kaapi_cuda_mem_destroy(kaapi_cuda_proc_t * proc);

static inline int kaapi_cuda_mem_register_(void *ptr, const size_t size)
{
#if KAAPI_CUDA_ASYNC
  const cudaError_t res =
      cudaHostRegister(ptr, size, cudaHostRegisterPortable);
#if 0
  if (res != cudaSuccess) {
    fprintf(stdout, "[%s] ERROR (%d) ptr=%p size=%lu kid=%lu\n",
	    __FUNCTION__, res,
	    ptr, size, (long unsigned int) kaapi_get_current_kid());
    fflush(stdout);
  }
#endif

  return res;
#else
  return 0;
#endif
}

static inline void kaapi_cuda_mem_unregister_(void *ptr)
{
#if KAAPI_CUDA_ASYNC
  cudaHostUnregister(ptr);
#endif
}

int
kaapi_cuda_mem_copy_dtod_buffer(kaapi_pointer_t dest,
				const kaapi_memory_view_t * view_dest,
				const int dest_dev,
				const kaapi_pointer_t src,
				const kaapi_memory_view_t * view_src,
				const int src_dev,
				const kaapi_pointer_t host,
				const kaapi_memory_view_t * view_host);

int
kaapi_cuda_mem_copy_dtod_peer(kaapi_pointer_t dest,
			      const kaapi_memory_view_t * view_dest,
			      const int dest_dev,
			      const kaapi_pointer_t src,
			      const kaapi_memory_view_t * view_src,
			      const int src_dev);

#endif				/* ! KAAPI_CUDA_MEM_H_INCLUDED */
