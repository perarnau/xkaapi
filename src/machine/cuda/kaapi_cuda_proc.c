/*
** kaapi_cuda_proc.h
** xkaapi
** 
** Created on Jul 2010
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_kasid.h"
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_cublas.h"

#ifdef KAAPI_CUDA_USE_POOL
#include "kaapi_cuda_pool.h"
#endif


/* exported */

int
kaapi_cuda_proc_initialize(kaapi_cuda_proc_t* proc, unsigned int idev)
{
  cudaError_t res;

  proc->is_initialized = 0;

  if ( kaapi_cuda_dev_open( proc, idev) )
    return -1;

#if KAAPI_CUDA_ASYNC
  res = cudaStreamCreate( &proc->stream );
  if (res != cudaSuccess) {
	    fprintf(stdout, "[%s] ERROR: %d\n", __FUNCTION__, res );
	    fflush(stdout);
    kaapi_cuda_dev_close( proc );
    return -1;
  }
#endif

  /* TODO */
#if 0
  res = cuDeviceGetAttribute
  (
   (int*)&proc->attr_concurrent_kernels,
   CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
   proc->dev
  );
#endif

  /* pop the context to make it floating. doing
     so allow another thread to use it, such
     as the main one with kaapi_mem_synchronize2
  */
  kaapi_cuda_cublas_init( proc );
  kaapi_cuda_sync();

#ifdef KAAPI_CUDA_USE_POOL
  kaapi_cuda_pool_init( proc );
#endif

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
			kaapi_get_current_kid() );
	fflush( stdout );
#endif
  proc->kasid_user = KAAPI_CUDA_KASID_USER_BASE + idev;
  proc->is_initialized = 1;
  proc->asid = kaapi_memory_address_space_create
    ( idev, KAAPI_MEM_TYPE_CUDA, 0x100000000UL );


  return 0;
}


int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t* proc)
{
  if (proc->is_initialized == 0)
    return -1;

#if defined(KAAPI_VERBOSE)
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		    kaapi_get_current_kid() );
    fflush( stdout );
#endif
  kaapi_cuda_cublas_finalize( proc );
#ifdef KAAPI_CUDA_ASYNC
   cudaStreamDestroy( proc->stream );
#endif

//  kaapi_cuda_ctx_pop( );
  kaapi_cuda_dev_close( proc );
  proc->is_initialized = 0;

  return 0;
}


size_t kaapi_cuda_get_proc_count(void)
{
  /* returns the number of kproc being of cuda type */
  /* todo: dont walk the kproc list every time, ok for now */
  kaapi_processor_t** pos = kaapi_all_kprocessors;
  size_t count = 0;
  size_t i;
  for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
    if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA)
      ++count;
  return count;
}

cudaStream_t kaapi_cuda_kernel_stream(void)
{
#if KAAPI_CUDA_ASYNC
    kaapi_processor_t* const self_proc =
	kaapi_get_current_processor();
    return self_proc->cuda_proc.stream;
#else
    return 0;
#endif
}

cudaStream_t kaapi_cuda_HtoD_stream(void)
{
#if KAAPI_CUDA_ASYNC
    kaapi_processor_t* const self_proc =
	kaapi_get_current_processor();
    return self_proc->cuda_proc.stream;
#else
    return 0;
#endif
}

cudaStream_t kaapi_cuda_DtoH_stream(void)
{
#if KAAPI_CUDA_ASYNC
    kaapi_processor_t* const self_proc =
	kaapi_get_current_processor();
    return self_proc->cuda_proc.stream;
#else
    return 0;
#endif
}

cudaStream_t kaapi_cuda_DtoD_stream(void)
{
#if KAAPI_CUDA_ASYNC
    kaapi_processor_t* const self_proc =
	kaapi_get_current_processor();
    return self_proc->cuda_proc.stream;
#else
    return 0;
#endif
}

