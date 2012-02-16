
#include <stdio.h>

#include "kaapi_impl.h"
#include "memory/kaapi_mem.h"

#if defined(KAAPI_USE_CUDA)
#include <cuda_runtime_api.h>

#include "machine/cuda/kaapi_cuda_ctx.h"
#include "machine/cuda/kaapi_cuda_mem.h"
#endif

#if defined(KAAPI_USE_CUDA)
/*
 * TODO see the effects without device stream synchronization
 * */
static int
kaapi_mem_sync_transfer( const int dev, kaapi_data_t* dest, kaapi_data_t* src )
{

    kaapi_cuda_ctx_set( dev );
    cudaStream_t stream;
    cudaError_t res = cudaStreamCreate( &stream );
    if (res != cudaSuccess) {
	fprintf(stderr, "[%s] ERROR: %d\n", __FUNCTION__, res );
	fflush(stderr);
	return res;
    }
    kaapi_cuda_mem_copy_dtoh_( dest->ptr, &dest->view, src->ptr, &src->view, stream );
    res = cudaStreamSynchronize( stream );
    if (res != cudaSuccess) {
	fprintf(stderr, "[%s] ERROR: %d\n", __FUNCTION__, res );
	fflush(stderr);
	return res;
    }
    cudaStreamDestroy( stream );
    return res;
}
#endif

int
kaapi_mem_sync_ptr( kaapi_data_t* kdata )
{
    //const kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;
    kaapi_mem_asid_t valid_asid;

    kaapi_mem_host_map_find_or_insert( host_map,
	    (kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
	    &kmd );
    if( kaapi_mem_data_has_addr( kmd, host_asid ) ) {
	if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
	    valid_asid = kaapi_mem_data_get_nondirty_asid( kmd );
	    kaapi_data_t* valid_data = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd, valid_asid );
	    /* TODO here */
	    fprintf( stdout, "[%s] dirty asid=%lu(%p) valid=%lu(%p) kid=%lu\n",
		    __FUNCTION__,
		    host_asid, kaapi_pointer2void(kdata->ptr),
		    valid_asid, kaapi_pointer2void(valid_data->ptr),
		    (unsigned long)kaapi_get_current_kid() );
	    fflush(stdout);
#if defined(KAAPI_USE_CUDA)
	    kaapi_mem_sync_transfer( valid_asid, kdata, valid_data );
#endif
	    kaapi_mem_data_clear_dirty( kmd, host_asid );
	}
    } else {
	    kaapi_mem_data_set_addr( kmd, host_asid,
		    (kaapi_mem_addr_t)kdata  );
    }
    return 0;
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

