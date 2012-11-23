

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

kaapi_processor_t *kaapi_cuda_get_proc_by_asid(kaapi_address_space_id_t asid)
{
  unsigned int i;
  
  for (i = 0; i < kaapi_count_kprocessors; ++i) {
    if ((kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA) &&
        (kaapi_all_kprocessors[i]->cuda_proc.kasid == asid)) {
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
