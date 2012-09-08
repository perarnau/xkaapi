
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

kaapi_mem_data_t* kaapi_mem_host_map_register_to_host(
      void* ptr,
      kaapi_memory_view_t* const view
    )
{
  kaapi_mem_host_map_t* const host_map = 
      kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
  const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
  kaapi_mem_data_t *kmd;

  /* register to host memory mapping */
  kaapi_mem_host_map_find_or_insert( host_map,
      kaapi_mem_host_map_generate_id(ptr, kaapi_memory_view_size(view)),
      &kmd );
  if (!kaapi_mem_data_has_addr(kmd, host_asid)) {
    kaapi_data_t* kdata = (kaapi_data_t*)calloc( 1, sizeof(kaapi_data_t) );
    kdata->ptr = kaapi_make_pointer(0, ptr);
    kdata->view = *view;
    kdata->kmd = kmd;
    kaapi_mem_data_set_addr( kmd, host_asid, (kaapi_mem_addr_t)kdata );
    kaapi_mem_data_clear_dirty( kmd, host_asid );
  }

  return kmd;
}

static inline void
kaapi_mem_host_map_destroy_kmd( kaapi_mem_data_t* const kmd )
{
  int i;

  for(i = 0; i < KAAPI_MEM_ASID_MAX; i++){
    if(kaapi_mem_data_has_addr(kmd, i)){
      free((void*)kaapi_mem_data_get_addr(kmd, i));
    }
  }
}

void kaapi_mem_host_map_destroy_all( kaapi_mem_host_map_t* map )
{
  static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
  kaapi_big_hashmap_t *const hmap = &map->hmap;
  kaapi_hashentries_t *entry;
  uint32_t i;

  for (i = 0; i < map_size; ++i) {
    for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
      kaapi_mem_data_t *const kmd = entry->u.kmd;
      if (kmd == NULL)
	continue;

      kaapi_mem_host_map_destroy_kmd( kmd );
      free(kmd);
      entry->u.kmd = 0;
    }
  }

  kaapi_big_hashmap_destroy( hmap );
}

