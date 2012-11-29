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
#include <pthread.h>
#include <cuda_runtime_api.h>

#include "../../kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"

int kaapi_cuda_dev_open(kaapi_cuda_proc_t * proc, unsigned int index)
{
  cudaError_t res;
  
  proc->index = index;
  
  cudaSetDevice(index);
  cudaDeviceReset();
  res = cudaSetDevice(index);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: cudaSetDevice ERROR %d\n", __FUNCTION__, res);
    fflush(stdout);
    abort();
  }
  
  /* Just warm the GPU */
  cudaFree(0);
  res = cudaGetDeviceProperties(&proc->deviceProp, index);
  if (res != cudaSuccess) {
    fprintf(stdout, "%s: cudaGetDeviceProperties ERROR %d\n", __FUNCTION__,
            res);
    fflush(stdout);
    abort();
  }

  return 0;
}

void kaapi_cuda_dev_close(kaapi_cuda_proc_t * proc)
{
  cudaSetDevice(proc->index);
  cudaDeviceReset();
}

kaapi_processor_t *kaapi_cuda_mem_get_proc(void)
{
  unsigned int i;
  
  for (i = 0; i < kaapi_count_kprocessors; ++i) {
    if (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA) {
      return kaapi_all_kprocessors[i];
    }
  }
  
  return NULL;
}

int kaapi_cuda_dev_enable_peer_access(kaapi_cuda_proc_t * const proc)
{
  int dev_count;
  cudaError_t res;
  int dev;
  int canAccessPeer;
  
  memset(proc->peers, 0, sizeof(unsigned int) * KAAPI_CUDA_MAX_DEV);
  
  cudaGetDeviceCount(&dev_count);
  for (dev = 0; dev < dev_count; dev++) {
    if (dev == proc->index)
      continue;
    
    cudaDeviceCanAccessPeer(&canAccessPeer, proc->index, dev);
    if (canAccessPeer) {
      res = cudaDeviceEnablePeerAccess(dev, 0);
      if (res == cudaSuccess)
        proc->peers[dev] = 1;
      else {
        fprintf(stdout, "[%s]: ERROR %d from %d -> %d\n",
                __FUNCTION__, res, proc->index, dev);
        fflush(stdout);
      }
    }
  }
  
  return 0;
}
