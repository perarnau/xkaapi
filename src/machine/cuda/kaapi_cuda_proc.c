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
#include "kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_kasid.h"
#include "kaapi_cuda_error.h"


static int open_cuda_device(CUdevice* dev, CUcontext* ctx, unsigned int index)
{
  CUresult res;

  res = cuDeviceGet(dev, index);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuDeviceGet", res);
    return -1;
  }

  /* use sched_yield while waiting for sync.
     context is made current for the thread.
   */
  res = cuCtxCreate(ctx, CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST, *dev);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuCtxCreate", res);
    return -1;
  }

#if 0 /* print device attributes */
  {
    int value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, *dev);
    printf("gpu_overlap: %u\n", value);
#if CUDA_VERSION >= 3000
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_CONCCURENT_KERNELS, *dev);
    printf("conc_kernels: %u\n", value);
#endif
  }
#endif

  return 0;
}

static void close_cuda_device(CUdevice dev, CUcontext ctx)
{
  dev = dev; /* unused */
  cuCtxDestroy(ctx);
}


/* exported */

int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t* proc, unsigned int idev)
{
  CUresult res;

  proc->is_initialized = 0;

  if (open_cuda_device(&proc->dev, &proc->ctx, idev))
    return -1;

  res = cuStreamCreate(&proc->stream, 0);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamCreate", res);
    close_cuda_device(proc->dev, proc->ctx);
    return -1;
  }

  /* pop the context to make it floating. doing
     so allow another thread to use it, such
     as the main one with kaapi_mem_synchronize2
  */
  res = cuCtxPopCurrent(&proc->ctx);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuCtxPopCurrent", res);
    close_cuda_device(proc->dev, proc->ctx);
    return -1;
  }

  if (pthread_mutex_init(&proc->ctx_lock, NULL))
  {
    kaapi_cuda_error("pthread_mutex_init", 0);
    return -1;
  }

  proc->kasid_user = KAAPI_CUDA_KASID_USER_BASE + idev;

  proc->is_initialized = 1;

  return 0;
}


int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t* proc)
{
  if (proc->is_initialized == 0)
    return -1;

  cuStreamDestroy(proc->stream);

  pthread_mutex_lock(&proc->ctx_lock);
  close_cuda_device(proc->dev, proc->ctx);
  pthread_mutex_unlock(&proc->ctx_lock);
  pthread_mutex_destroy(&proc->ctx_lock);

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


unsigned int kaapi_cuda_get_first_kid(void)
{
  kaapi_processor_t** pos = kaapi_all_kprocessors;
  size_t i;
  for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
    if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA)
      return (*pos)->kid;
  return (unsigned int)-1;
}


CUstream kaapi_cuda_kernel_stream(void)
{
  kaapi_processor_t* const self_proc =
    kaapi_get_current_processor();
  return self_proc->cuda_proc.stream;
}
