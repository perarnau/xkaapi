
#include "kaapi_impl.h"
#include "kaapi_mem.h"
#include "kaapi_mem_data.h"
#include "kaapi_mem_host_map.h"

#if defined(KAAPI_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

kaapi_processor_id_t	kaapi_all_asid2kid[KAAPI_MEM_ASID_MAX];

static inline kaapi_mem_data_t* 
_kaapi_mem_data_alloc( void )
{
//    kaapi_mem_data_t* kmd = malloc( sizeof(kaapi_mem_data_t) );
    kaapi_mem_data_t* kmd = calloc( 1, sizeof(kaapi_mem_data_t) );
    kaapi_mem_data_init( kmd );
    return kmd;
}

int
kaapi_mem_host_map_find( kaapi_mem_host_map_t* map, 
	kaapi_mem_addr_t addr,
	kaapi_mem_data_t** data
	)
{
    kaapi_hashentries_t* entry;

    entry = kaapi_big_hashmap_find( &map->hmap, (void*)addr );
    if (entry != 0)
	*data = entry->u.kmd;
    else
	return -1;

    return 0;
}

int
kaapi_mem_host_map_find_or_insert( kaapi_mem_host_map_t* map,
	kaapi_mem_addr_t addr,
	kaapi_mem_data_t** kmd
	)
{
    kaapi_hashentries_t* entry;
#if 0
    const int res = kaapi_mem_host_map_find( map, addr, kmd );
    if( res == 0 )
        return 0;
#endif

    entry = kaapi_big_hashmap_findinsert( &map->hmap, (void*)addr );
    if (entry->u.kmd == 0)
	entry->u.kmd = _kaapi_mem_data_alloc();

    /* identity mapping */
    *kmd = entry->u.kmd;

    return 0;
}

int
kaapi_mem_host_map_find_or_insert_( kaapi_mem_host_map_t* map,
	kaapi_mem_addr_t addr, kaapi_mem_data_t** kmd )
{
    kaapi_hashentries_t* entry;
    entry = kaapi_big_hashmap_findinsert( &map->hmap, (void*)addr );
    entry->u.kmd = *kmd;

    return 0;
}

int
kaapi_mem_host_map_sync( const kaapi_format_t* fmt, void* sp )
{
    size_t count_params = kaapi_format_get_count_params( fmt, sp );
    size_t i;
#if (defined(KAAPI_USE_CUDA) && !defined(KAAPI_CUDA_NO_D2H) && !defined(KAAPI_CUDA_NO_H2D) )
    cudaStream_t stream;
    cudaStreamCreate( &stream );
#endif

    for( i= 0; i < count_params; i++ ) {
	kaapi_access_mode_t m = fmt->get_mode_param( fmt, i, sp );

	if( m == KAAPI_ACCESS_MODE_V )
	    continue;

	kaapi_access_t access = fmt->get_access_param( fmt, i, sp );
	kaapi_data_t* kdata = kaapi_data( kaapi_data_t, &access );
#if (defined(KAAPI_USE_CUDA) && !defined(KAAPI_CUDA_NO_D2H) && !defined(KAAPI_CUDA_NO_H2D) )
	kaapi_mem_sync_data( kdata, stream );
#endif

	if( KAAPI_ACCESS_IS_WRITE(m) ) {
	    kaapi_mem_host_map_t* host_map = 
		kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
	    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
	    kaapi_mem_data_t *kmd;
	    kaapi_mem_host_map_find_or_insert( host_map, 
		    (kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
		    &kmd );
	    kaapi_mem_data_set_all_dirty_except( kmd, 
		    host_asid );
	}
    }
#if (defined(KAAPI_USE_CUDA) && !defined(KAAPI_CUDA_NO_D2H) && !defined(KAAPI_CUDA_NO_H2D) )
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
    cudaStreamSynchronize( stream );
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );
#endif

    return 0;
}

