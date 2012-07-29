
#include "kaapi_impl.h"
#include "kaapi_mem.h"
#include "kaapi_mem_data.h"
#include "kaapi_mem_host_map.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_data.h"
#endif

kaapi_processor_id_t kaapi_all_asid2kid[KAAPI_MEM_ASID_MAX];

static inline kaapi_mem_data_t *_kaapi_mem_data_alloc(void)
{
//    kaapi_mem_data_t* kmd = malloc( sizeof(kaapi_mem_data_t) );
  kaapi_mem_data_t *kmd = calloc(1, sizeof(kaapi_mem_data_t));
  kaapi_mem_data_init(kmd);
  return kmd;
}

int
kaapi_mem_host_map_find(kaapi_mem_host_map_t * map,
			kaapi_mem_addr_t addr, kaapi_mem_data_t ** data)
{
  kaapi_hashentries_t *entry;

  entry = kaapi_big_hashmap_find(&map->hmap, (void *) addr);
  if (entry != 0)
    *data = entry->u.kmd;
  else
    return -1;

  return 0;
}

int
kaapi_mem_host_map_find_or_insert(kaapi_mem_host_map_t * map,
				  kaapi_mem_addr_t addr,
				  kaapi_mem_data_t ** kmd)
{
  kaapi_hashentries_t *entry;
#if 0
  const int res = kaapi_mem_host_map_find(map, addr, kmd);
  if (res == 0)
    return 0;
#endif

  entry = kaapi_big_hashmap_findinsert(&map->hmap, (void *) addr);
  if (entry->u.kmd == 0)
    entry->u.kmd = _kaapi_mem_data_alloc();

  /* identity mapping */
  *kmd = entry->u.kmd;

  return 0;
}

int
kaapi_mem_host_map_find_or_insert_(kaapi_mem_host_map_t * map,
				   kaapi_mem_addr_t addr,
				   kaapi_mem_data_t ** kmd)
{
  kaapi_hashentries_t *entry;
  entry = kaapi_big_hashmap_findinsert(&map->hmap, (void *) addr);
  entry->u.kmd = *kmd;

  return 0;
}

int
kaapi_mem_host_map_sync( kaapi_taskdescr_t* const td )
{
#if defined(KAAPI_USE_CUDA)
  return kaapi_cuda_data_input_host_sync(td);
#endif
  return 0;
}

