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

#ifndef KAAPI_CUDA_MEM_H_INCLUDED
#define KAAPI_CUDA_MEM_H_INCLUDED

#include "kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include <cuda_runtime_api.h>

int kaapi_cuda_mem_free(kaapi_pointer_t ptr);

int kaapi_cuda_mem_free_(void* ptr);

uintptr_t kaapi_cuda_mem_alloc( const kaapi_address_space_id_t kasid,
                              const size_t size,
                              const kaapi_access_mode_t m
                              );

uintptr_t kaapi_cuda_mem_alloc_(const size_t size);

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

int
kaapi_cuda_mem_copy_dtoh_from_host(kaapi_pointer_t dest,
                         const kaapi_memory_view_t * view_dest,
                         const kaapi_pointer_t src,
                         const kaapi_memory_view_t * view_src,
                         kaapi_processor_id_t kid_src );

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

int kaapi_cuda_mem_mgmt_check(struct kaapi_processor_t * proc);

void kaapi_cuda_mem_destroy(kaapi_cuda_proc_t * proc);

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

/**
  Transfer memory from src to dest using the host copy version.
  It assumes that the current thread is GPU and owns pointer dest.
 */
int
kaapi_cuda_mem_copy_dtod_buffer(
                                kaapi_pointer_t dest,
                                const kaapi_memory_view_t * view_dest,
                                const kaapi_pointer_t src,
                                const kaapi_memory_view_t * view_src,
                                const kaapi_pointer_t host,
                                const kaapi_memory_view_t * view_host
                                );

int
kaapi_cuda_mem_copy_dtod_peer(kaapi_pointer_t dest,
			      const kaapi_memory_view_t * view_dest,
			      const int dest_dev,
			      const kaapi_pointer_t src,
			      const kaapi_memory_view_t * view_src,
			      const int src_dev);

#endif				/* ! KAAPI_CUDA_MEM_H_INCLUDED */
