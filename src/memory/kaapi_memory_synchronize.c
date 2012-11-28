
#include <stdio.h>

#include "kaapi_impl.h"

#if defined(KAAPI_USE_CUDA)

static int
kaapi_memory_host_synchronize( void )
{
  KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
  kaapi_cuda_proc_sync_all();
  KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );
  
#if defined(KAAPI_USE_CUDA)
  kaapi_assert_debug( kaapi_cuda_proc_all_isvalid( ) );
#endif
  
  return 0;
}

#endif /* KAAPI_USE_CUDA */

int kaapi_memory_synchronize( void )
{
#if !defined(KAAPI_CUDA_NO_D2H) && defined(KAAPI_USE_CUDA)
  kaapi_memory_host_synchronize();
#endif
  return 0;
}

int kaapi_memory_synchronize_pointer( void *ptr )
{
  /* TODO */
  kaapi_assert(0);  
  return 0;
}

kaapi_pointer_t kaapi_memory_synchronize_metadata( kaapi_metadata_info_t* kdmi )
{
  return kaapi_make_nullpointer();
}

void kaapi_memory_address_space_synchronize_peer2peer(kaapi_address_space_id_t dest, kaapi_address_space_id_t src)
{
  kaapi_memory_map_t* kmap = kaapi_memory_map_get_current(kaapi_get_self_kid());
  static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
  kaapi_big_hashmap_t *const hmap = &kmap->hmap;
  kaapi_hashentries_t *entry;
  uint32_t i;
  
  for (i = 0; i < map_size; ++i) {
    for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
      kaapi_metadata_info_t* kmdi = entry->u.mdi;
      if (kmdi == NULL)
        continue;
      
      if( kaapi_metadata_info_has_data(kmdi, src) &&
          kaapi_metadata_info_is_valid(kmdi, src) &&
          !kaapi_metadata_info_is_valid(kmdi, dest)
          )
      {
        if(kaapi_metadata_info_clear_dirty_and_check(kmdi, dest)){
          kaapi_data_t* ksrc = kaapi_metadata_info_get_data(kmdi, src);
          kaapi_data_t* kdest = kaapi_metadata_info_get_data(kmdi, dest);
          kaapi_memory_copy(kdest->ptr, &kdest->view, ksrc->ptr, &ksrc->view);
        }
      }
      
    }
  }
}

void* kaapi_memory_get_host_pointer(void* const gpu_ptr)
{
  /* TODO */
  kaapi_assert(0);
  return NULL;
}

