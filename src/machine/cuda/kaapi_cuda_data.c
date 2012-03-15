
#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_memory.h" /* TODO: remove this */
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_data.h"

#ifdef	KAAPI_CUDA_MEM_ALLOC_MANAGER
static inline void
xxx_kaapi_cuda_data_inc_use(
		const kaapi_mem_host_map_t* map,
		kaapi_mem_data_t* kmd,
		kaapi_data_t* src
		)
{
	const kaapi_mem_asid_t asid = kaapi_mem_host_map_get_asid(map);

	if( kaapi_mem_data_has_addr( kmd, asid ) ) {
	    kaapi_data_t* dest= (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		     asid );
	    kaapi_cuda_mem_inc_use( &dest->ptr );
	}

}

#endif /* KAAPI_CUDA_MEM_ALLOC_MANAGER */

static inline void
kaapi_cuda_data_view_convert( kaapi_memory_view_t* dest_view, const
	kaapi_memory_view_t* src_view )
{
  switch (src_view->type) 
  {
    case KAAPI_MEMORY_VIEW_1D:
	dest_view->size[0] = src_view->size[0];
	break;
    case KAAPI_MEMORY_VIEW_2D:
	dest_view->size[0] = src_view->size[0];
	dest_view->size[1] = src_view->size[1];
	dest_view->lda = src_view->size[1];
	break;
    default:
      kaapi_assert(0);
      break;
  }
    dest_view->wordsize = src_view->wordsize;
    dest_view->type = src_view->type;
//    dest_view->lda = src_view->lda;
}

/* here it checks if the CPU pointer is present in the GPU.
 * If not, it allocates.
 * Context: it executes right before a CUDA task (kernel).
 */
static inline kaapi_data_t*
xxx_kaapi_cuda_data_allocate(
		const kaapi_mem_host_map_t* cuda_map,
		kaapi_mem_data_t* kmd,
		kaapi_data_t* src
		)
{
	const kaapi_mem_asid_t asid = kaapi_mem_host_map_get_asid(cuda_map);

	if( !kaapi_mem_data_has_addr( kmd, asid ) ) {
		kaapi_data_t* dest = (kaapi_data_t*)calloc( 1, sizeof(kaapi_data_t) );
		kaapi_cuda_mem_alloc( &dest->ptr, 0UL, 
		    kaapi_memory_view_size(&src->view), 0 );
	    	kaapi_cuda_data_view_convert( &dest->view, &src->view );
		kaapi_mem_data_set_addr( kmd, asid, (kaapi_mem_addr_t)dest );
		kaapi_mem_data_set_dirty( kmd, asid );
		kaapi_mem_host_map_find_or_insert_(  cuda_map,
		    (kaapi_mem_addr_t)kaapi_pointer2void(dest->ptr),
		    &kmd );
		return dest;
	} else {
	    kaapi_data_t* dest= (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		     asid );
	    kaapi_cuda_mem_inc_use( &dest->ptr );
	    return dest;
	}

	return NULL;
}

int kaapi_cuda_data_allocate( 
	kaapi_format_t*		   fmt,
	void*              sp
)
{
#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    const kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
		count_params,
		(unsigned long)kaapi_get_current_kid(),
	        (unsigned int long)kaapi_mem_host_map_get_asid(cuda_map) );
	fflush( stdout );
#endif

	for ( i=0; i < count_params; i++ ) {
		kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
			kaapi_format_get_mode_param( fmt, i, sp) );
		if (m == KAAPI_ACCESS_MODE_V) 
			continue;

		kaapi_access_t access = kaapi_format_get_access_param( fmt,
			       	i, sp );
		kaapi_data_t* src = kaapi_data( kaapi_data_t, &access );
		kaapi_mem_host_map_find_or_insert( host_map,
			(kaapi_mem_addr_t)kaapi_pointer2void(src->ptr),
			&kmd );
		kaapi_assert_debug( kmd !=0 );

		if( !kaapi_mem_data_has_addr( kmd, host_asid ) )
		    kaapi_mem_data_set_addr( kmd, host_asid,
			    (kaapi_mem_addr_t)src );

		kaapi_data_t* dest= xxx_kaapi_cuda_data_allocate( cuda_map, kmd, src );

		/* sets new pointer to the task */
		access.data =  dest;
		kaapi_format_set_access_param( fmt, i, sp, &access );
	}

#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    fmt->name,
	    t1-t0);
#endif
	return 0;
}


/* 
 * Context: it executes right before a CUDA task (kernel).
 * The function goes through every parameter and checks if it is allocated and
 * valid on GPU memory.
 */
int kaapi_cuda_data_send( 
	kaapi_format_t*		   fmt,
	void*              sp
)
{
#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
		count_params,
		(unsigned long)kaapi_get_current_kid(),
	        (unsigned int long)kaapi_mem_host_map_get_asid(cuda_map) );
	fflush( stdout );
#endif

    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param( fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

	    kaapi_access_t access = kaapi_format_get_access_param( fmt,
			    i, sp );
	    kaapi_data_t* dev_data = kaapi_data( kaapi_data_t, &access );
	    kaapi_cuda_data_sync_device( dev_data );

	    if( KAAPI_ACCESS_IS_WRITE(m) ) {
		kaapi_mem_host_map_find_or_insert( cuda_map,
			(kaapi_mem_addr_t)kaapi_pointer2void(dev_data->ptr),
			&kmd );
		kaapi_assert_debug( kmd !=0 );
		kaapi_mem_data_set_all_dirty_except( kmd, cuda_asid );
	    }

	    //access.data = dev_data;
	    //kaapi_format_set_access_param(fmt, i, sp, &access);
    }

#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    fmt->name,
	    t1-t0);
#endif
    return 0;
}

int kaapi_cuda_data_recv( 
	kaapi_format_t*		   fmt,
	void*              sp
)
{
#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] CUDA params=%ld kid=%lu asid=%lu\n", __FUNCTION__,
		count_params,
		(unsigned long)kaapi_get_current_kid(),
	        (unsigned int long)kaapi_mem_host_map_get_asid(cuda_map) );
	fflush( stdout );
#endif

    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param( fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

	    kaapi_access_t access = kaapi_format_get_access_param( fmt,
			    i, sp );
	    kaapi_data_t* dev_data = kaapi_data( kaapi_data_t, &access );

	    if( KAAPI_ACCESS_IS_WRITE(m) ) {
		kaapi_processor_t* const self_proc =
		    kaapi_get_current_processor();
		kaapi_mem_host_map_find_or_insert( cuda_map,
			(kaapi_mem_addr_t)kaapi_pointer2void(dev_data->ptr),
			&kmd );
		kaapi_assert_debug( kmd !=0 );
		kaapi_data_t* host_data=
		    (kaapi_data_t*) kaapi_mem_data_get_addr( kmd, host_asid );
		cudaStreamWaitEvent( kaapi_cuda_DtoH_stream(),
			self_proc->cuda_proc.event, 0 );
		kaapi_cuda_mem_copy_dtoh(
			host_data->ptr, &host_data->view,
			dev_data->ptr, &dev_data->view
			);
		cudaEventCreateWithFlags( &dev_data->event,
			cudaEventDisableTiming );
		cudaEventRecord( dev_data->event,
		       kaapi_cuda_DtoH_stream() );
	    }
    }

#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    fmt->name,
	    t1-t0);
#endif
    return 0;
}

#if	KAAPI_CUDA_MEM_ALLOC_MANAGER
int 
kaapi_cuda_data_check( void )
{
    return kaapi_cuda_mem_mgmt_check( kaapi_get_current_processor() );
}
#endif

static inline int
kaapi_cuda_data_sync_device_transfer(
	kaapi_data_t* dest, const kaapi_mem_asid_t dest_asid,
	kaapi_data_t* src,  const kaapi_mem_asid_t src_asid
       	)
{
    const int dest_dev = dest_asid - 1;
    const int src_dev = src_asid - 1;

    if( src_asid == 0 ) {
#if 0
	fprintf(stdout, "%s: dest=%p %lux%lu lda=%lu src=%p %lux%lu lda=%lu\n",
		__FUNCTION__,
		kaapi_pointer2void(dest->ptr),
		dest->view.size[0], dest->view.size[1], dest->view.lda,
		kaapi_pointer2void(src->ptr),
		src->view.size[0], src->view.size[1], src->view.lda
		);
	fflush(stdout);
#endif
	kaapi_cuda_mem_copy_htod( dest->ptr, &dest->view,
		src->ptr, &src->view );
    } else {
	    kaapi_cuda_mem_copy_dtod_buffer(
		dest->ptr, &dest->view,	dest_dev,
		src->ptr, &src->view, src_dev
		);
#if 0
	int canAccessPeer;
	cudaDeviceCanAccessPeer( &canAccessPeer,
		dest_dev, src_dev );
	if( canAccessPeer ) {
	    kaapi_cuda_mem_copy_dtod_peer( 
		dest->ptr, &dest->view,	dest_dev,
		src->ptr, &src->view, src_dev
		);
	} else {
	    kaapi_cuda_mem_copy_dtod_buffer(
		dest->ptr, &dest->view,	dest_dev,
		src->ptr, &src->view, src_dev
		);
	}
#endif
    }

    return 0;
}

int
kaapi_cuda_data_sync_device( kaapi_data_t* kdata )
{
#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
    kaapi_mem_data_t *kmd;
    kaapi_mem_asid_t valid_asid;

    kaapi_mem_host_map_find_or_insert( cuda_map,
	    (kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
	    &kmd );
    if ( kaapi_mem_data_is_dirty( kmd, cuda_asid ) ) {
	valid_asid = kaapi_mem_data_get_nondirty_asid( kmd );
	kaapi_data_t* valid_data = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd, valid_asid );
	kaapi_cuda_data_sync_device_transfer( kdata, cuda_asid,
		valid_data, valid_asid );
	kaapi_mem_data_clear_dirty( kmd, cuda_asid );
    }
#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    "",
	    t1-t0);
#endif

    return 0;
}

static inline int
kaapi_cuda_data_sync_host_transfer(
	kaapi_data_t* dest, const kaapi_mem_asid_t dest_asid,
	kaapi_data_t* src,  const kaapi_mem_asid_t src_asid
	)
{
    cudaError_t res;

#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    kaapi_cuda_ctx_set( src_asid-1 );
    cudaEventSynchronize( src->event );
#if 0
    cudaStream_t stream;
    res = cudaStreamCreate( &stream );
    if (res != cudaSuccess) {
	fprintf(stdout, "%s: cudaStreamCreate ERROR %d\n", __FUNCTION__, res );
	fflush(stdout);
	abort();
    }
    /* Synchronize everything since we dont know about its execution */
    kaapi_cuda_sync();
    kaapi_cuda_mem_copy_dtoh_( dest->ptr, &dest->view, src->ptr, &src->view, stream );
    res = cudaStreamSynchronize( stream );
    if (res != cudaSuccess) {
	fprintf(stdout, "%s: cudaStreamSynchronize ERROR %d\n", __FUNCTION__, res );
	fflush(stdout);
	abort();
    }
    cudaStreamDestroy( stream );
#endif
#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    "",
	    t1-t0);
#endif

    return res;
}

int
kaapi_cuda_data_sync_host( kaapi_data_t* kdata )
{
    //const kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();
#if KAAPI_CUDA_TIME
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;
    kaapi_mem_asid_t valid_asid;

    kaapi_mem_host_map_find_or_insert( host_map,
	    (kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
	    &kmd );
    if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
	valid_asid = kaapi_mem_data_get_nondirty_asid( kmd );
	kaapi_data_t* valid_data = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd, valid_asid );
	kaapi_cuda_data_sync_host_transfer( kdata, host_asid, valid_data,
		valid_asid );
	kaapi_mem_data_clear_dirty( kmd, host_asid );
    }
#if KAAPI_CUDA_TIME
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    "",
	    t1-t0);
#endif

    return 0;
}

