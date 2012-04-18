
#include <stdio.h>

#include "kaapi_impl.h"
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_data.h"
#endif

#if defined(KAAPI_USE_CUDA)

int kaapi_mem_sync_data( kaapi_data_t* kdata, cudaStream_t stream )
{
    if( kaapi_get_current_processor()->proc_type == KAAPI_PROC_TYPE_CUDA ){
	return kaapi_cuda_data_sync_device( kdata );
    } else {
	return kaapi_cuda_data_sync_host( kdata, stream );
    }
}

#if defined(KAAPI_CUDA_NO_D2H)
static int
kaapi_memory_host_synchronize( void )
{
    kaapi_processor_t** pos = kaapi_all_kprocessors;
    size_t i;
    cudaStream_t stream;

    cudaStreamCreate( &stream );
    for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
	if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA) {
	    cudaStreamWaitEvent( stream, (*pos)->cuda_proc.event, 0 );
	}

    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
    cudaStreamSynchronize( stream );
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );

    return 0;
}

#else /* KAAPI_CUDA_NO_D2H */
static int
kaapi_memory_host_synchronize( void )
{
    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
    kaapi_big_hashmap_t* hmap = &host_map->hmap;
    kaapi_hashentries_t* entry;
    uint32_t i;
    cudaStream_t stream;

    cudaStreamCreate( &stream );
    for (i = 0; i < map_size; ++i) {
	for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
	    const kaapi_mem_data_t *kmd = entry->u.kmd;
	    if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
		kaapi_data_t *kdata = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
			host_asid );
		kaapi_mem_sync_data( kdata, stream );
	    }
	}
    }
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
    cudaStreamSynchronize( stream );
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );

    return 0;
}

#endif /* KAAPI_CUDA_NO_D2H */

#endif /* KAAPI_USE_CUDA */

int kaapi_memory_synchronize( void )
{
#if defined(KAAPI_USE_CUDA)
    kaapi_memory_host_synchronize();
#endif
    return 0;
}

int kaapi_memory_synchronize_pointer( void *ptr )
{
    /* TODO */
    return 0;
}
