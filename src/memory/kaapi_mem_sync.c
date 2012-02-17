
#include <stdio.h>

#include "kaapi_impl.h"
#include "memory/kaapi_mem.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_data.h"
#endif

int
kaapi_mem_sync_ptr( kaapi_data_t* kdata )
{
#if defined(KAAPI_USE_CUDA)
    if( kaapi_get_current_processor()->proc_type == KAAPI_PROC_TYPE_CUDA ){
	return kaapi_cuda_data_sync_device( kdata );
    } else {
	return kaapi_cuda_data_sync_host( kdata );
    }
#else
    return 0;
#endif
}

/**
*/
int kaapi_memory_synchronize( void )
{
    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
    kaapi_big_hashmap_t* hmap = &host_map->hmap;
    kaapi_hashentries_t* entry;
    uint32_t i;

    for (i = 0; i < map_size; ++i) {
	for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
	    const kaapi_mem_data_t *kmd = entry->u.kmd;
	    if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
		kaapi_data_t *kdata = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
			host_asid );
		kaapi_mem_sync_ptr( kdata );
	    }
	}
    }

    return 0;
}

