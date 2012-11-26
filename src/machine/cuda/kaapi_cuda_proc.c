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
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_mem_cache.h"

#include "kaapi_cuda_stream.h"

#if defined(KAAPI_USE_CUPTI)
#include "kaapi_cuda_trace.h"
#endif

/* number of CUDA devices */
uint32_t kaapi_cuda_count_kprocessors = 0;

/* index of kprocessors by CUDA devid */
kaapi_processor_t *kaapi_cuda_all_kprocessors[KAAPI_CUDA_MAX_DEV];
/* exported */

void kaapi_cuda_init(void)
{
  KAAPI_ATOMIC_WRITE(&kaapi_cuda_synchronize_barrier, 0);
}

int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t * proc, unsigned int idev)
{
  cudaError_t res;

  proc->is_initialized = 0;

  if ((res = kaapi_cuda_dev_open(proc, idev)) != cudaSuccess)
    return res;

  kaapi_cuda_device_sync();

  if (kaapi_default_param.cudawindowsize > 0)
    kaapi_cuda_stream_init(kaapi_default_param.cudawindowsize * 3, proc);
  else
    kaapi_cuda_stream_init(512, proc);

  /* pop the context to make it floating. doing
     so allow another thread to use it, such
     as the main one with kaapi_mem_synchronize2
   */
  kaapi_cuda_cublas_init(proc);
  kaapi_cuda_cublas_set_stream();

  if (kaapi_default_param.cudapeertopeer)
    kaapi_cuda_dev_enable_peer_access(proc);

#if defined(KAAPI_USE_CUPTI)
  if (getenv("KAAPI_RECORD_TRACE") != 0) {
    kaapi_cuda_trace_thread_init();
  }
#endif
  
  kaapi_cuda_mem_cache_init(proc);
  kaapi_cuda_device_sync();

#if KAAPI_VERBOSE
  fprintf(stdout, "%s: dev=%lu kid=%lu\n", __FUNCTION__,
	  idev, kaapi_get_current_kid());
  fflush(stdout);
#endif

  KAAPI_ATOMIC_WRITE(&proc->synchronize_flag, 0);
  proc->is_initialized = 1;
  
  kaapi_cuda_all_kprocessors[idev] = kaapi_get_current_processor();
  kaapi_cuda_count_kprocessors++;

  return 0;
}


int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t * proc)
{
  if (proc->is_initialized == 0)
    return -1;

  proc->is_initialized = 0;

  return 0;
}


size_t kaapi_cuda_get_proc_count(void)
{
  return kaapi_cuda_count_kprocessors;
}

cudaStream_t kaapi_cuda_kernel_stream(void)
{
  kaapi_processor_t *const self_proc = kaapi_get_current_processor();
  return
      kaapi_cuda_get_cudastream(kaapi_cuda_get_kernel_fifo
				(self_proc->cuda_proc.kstream));
}

cudaStream_t kaapi_cuda_HtoD_stream(void)
{
  kaapi_processor_t *const self_proc = kaapi_get_current_processor();
  return
      kaapi_cuda_get_cudastream(kaapi_cuda_get_input_fifo
				(self_proc->cuda_proc.kstream));
}

cudaStream_t kaapi_cuda_DtoH_stream(void)
{
  kaapi_processor_t *const self_proc = kaapi_get_current_processor();
  return
      kaapi_cuda_get_cudastream(kaapi_cuda_get_output_fifo
				(self_proc->cuda_proc.kstream));
}

cudaStream_t kaapi_cuda_DtoD_stream(void)
{
  kaapi_processor_t *const self_proc = kaapi_get_current_processor();
  return
      kaapi_cuda_get_cudastream(kaapi_cuda_get_output_fifo
				(self_proc->cuda_proc.kstream));
}

void kaapi_cuda_proc_poll(kaapi_processor_t * const kproc)
{
  if (KAAPI_ATOMIC_READ(&kproc->cuda_proc.synchronize_flag) == 1) {
    kaapi_cuda_sync(kproc);
    KAAPI_ATOMIC_WRITE(&kproc->cuda_proc.synchronize_flag, 0);
  } else {
    kaapi_cuda_stream_poll(kproc);
  }
}

int kaapi_cuda_proc_end_isvalid(kaapi_processor_t * const kproc)
{
  return kaapi_cuda_stream_is_empty(kproc->cuda_proc.kstream);
}

int kaapi_cuda_proc_all_isvalid(void)
{
  kaapi_processor_t **pos = kaapi_all_kprocessors;
  size_t i;

  for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
    if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA) {
      if (!kaapi_cuda_proc_end_isvalid(*pos))
	return 0;
    }

  return 1;
}

void kaapi_cuda_proc_destroy(kaapi_processor_t* const kproc)
{
#if 0
  kaapi_cuda_cublas_finalize(&kproc->cuda_proc);
  kaapi_cuda_stream_destroy(kproc->cuda_proc.kstream);
  kaapi_cuda_mem_destroy(&kproc->cuda_proc);
  kaapi_cuda_dev_close(&kproc->cuda_proc);
#endif
}
