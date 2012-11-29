
#include <stdio.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"

#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_mem.h"

kaapi_atomic_t kaapi_cuda_synchronize_barrier;

int kaapi_cuda_proc_sync_all(void)
{
  kaapi_processor_t **pos = kaapi_all_kprocessors;
  size_t i;
  
  KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
  /* signal all CUDA kprocs */
  for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos) {
    if (kaapi_processor_get_type(*pos) == KAAPI_PROC_TYPE_CUDA) {
      KAAPI_ATOMIC_WRITE(&(*pos)->cuda_proc.synchronize_flag, 1);
    }
  }
  
  /* wait for GPU operations/memory etc */
  while (KAAPI_ATOMIC_READ(&kaapi_cuda_synchronize_barrier) != kaapi_cuda_get_proc_count())
  {
    kaapi_slowdown_cpu();
  }
  
  KAAPI_ATOMIC_WRITE(&kaapi_cuda_synchronize_barrier, 0);
  KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );  
  
  kaapi_assert_debug(kaapi_cuda_proc_all_isvalid());
  
  return 0;
}

int kaapi_cuda_sync(kaapi_processor_t * const kproc)
{
  kaapi_cuda_stream_t *const kstream = kproc->cuda_proc.kstream;
  
  KAAPI_EVENT_PUSH0(kproc, kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG);
  
  /* wait all kstream operations */
  kaapi_cuda_stream_waitall(kstream);
  
  /* Sync to host (KAAPI_EMPTY_ADDRESS_SPACE_ID) */
  kaapi_memory_address_space_synchronize_peer2peer(KAAPI_EMPTY_ADDRESS_SPACE_ID, kaapi_memory_map_get_current_asid());
  cudaStreamSynchronize(kaapi_cuda_DtoH_stream());
  
  KAAPI_ATOMIC_ADD(&kaapi_cuda_synchronize_barrier, 1);
  
  KAAPI_EVENT_PUSH0(kproc, kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END);
  
  return 0;
}
