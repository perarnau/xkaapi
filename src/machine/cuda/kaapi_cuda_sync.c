
#include <stdio.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_kasid.h"
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_mem.h"

kaapi_atomic_t kaapi_cuda_synchronize_barrier;

int kaapi_cuda_proc_sync_all(void)
{
  kaapi_processor_t **pos = kaapi_all_kprocessors;
  size_t i;

  /* signal all CUDA kprocs */
  for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos) {
    if (kaapi_processor_get_type(*pos) == KAAPI_PROC_TYPE_CUDA) {
      KAAPI_ATOMIC_WRITE(&(*pos)->cuda_proc.synchronize_flag, 1);
    }
  }

  /* wait for GPU operations/memory etc */
  while (KAAPI_ATOMIC_READ(&kaapi_cuda_synchronize_barrier)
	 != kaapi_cuda_get_proc_count())
    kaapi_slowdown_cpu();

  KAAPI_ATOMIC_WRITE(&kaapi_cuda_synchronize_barrier, 0);

  return 0;
}

static inline void
kaapi_cuda_sync_host_data(kaapi_mem_data_t * const kmd,
			  const kaapi_mem_asid_t host_asid,
			  const kaapi_mem_asid_t cuda_asid)
{
  kaapi_data_t *const dest = (kaapi_data_t *) kaapi_mem_data_get_addr(kmd,
								      host_asid);
  kaapi_data_t *const src = (kaapi_data_t *) kaapi_mem_data_get_addr(kmd,
								     cuda_asid);
  kaapi_cuda_mem_copy_dtoh(dest->ptr, &dest->view, src->ptr, &src->view);
#if 0
  fprintf(stdout, "[%s] kmd=%p host_asid=%lu cuda_asid=%lu\n",
	  __FUNCTION__,
	  (void *) kmd, (unsigned int) host_asid,
	  (unsigned int) cuda_asid);
  fflush(stdout);
#endif
}


static inline void kaapi_cuda_sync_memory(kaapi_processor_t * const kproc)
{
  kaapi_mem_host_map_t *host_map =
      kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_host_map_t *cuda_map = kaapi_get_current_mem_host_map();
  const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
  static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
  kaapi_big_hashmap_t *const hmap = &host_map->hmap;
  kaapi_hashentries_t *entry;
  uint32_t i;

  for (i = 0; i < map_size; ++i) {
    for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
      kaapi_mem_data_t *const kmd = entry->u.kmd;
      if (kmd == NULL)
	continue;

      if (kaapi_mem_data_has_addr(kmd, cuda_asid) &&
	  !kaapi_mem_data_is_dirty(kmd, cuda_asid) &&
	  kaapi_mem_data_is_dirty(kmd, host_asid)) {
	kaapi_cuda_sync_host_data(kmd, host_asid, cuda_asid);
      }
    }
  }
  cudaStreamSynchronize(kaapi_cuda_DtoH_stream());
}

int kaapi_cuda_sync(kaapi_processor_t * const kproc)
{
  kaapi_cuda_stream_t *const kstream = kproc->cuda_proc.kstream;

  KAAPI_EVENT_PUSH0(kproc, kaapi_self_thread(),
		    KAAPI_EVT_CUDA_CPU_SYNC_BEG);

  /* wait all kstream operations */
  kaapi_cuda_stream_waitall(kstream);

  kaapi_cuda_sync_memory(kproc);

  KAAPI_ATOMIC_ADD(&kaapi_cuda_synchronize_barrier, 1);

  KAAPI_EVENT_PUSH0(kproc, kaapi_self_thread(),
		    KAAPI_EVT_CUDA_CPU_SYNC_END);

  return 0;
}
