
#include <stdio.h>

#include "kaapi_impl.h"
#include "../../kaapi_memory.h" /* TODO: remove this */
#include "../../memory/kaapi_mem.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_mem.h"
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

#if 0
	fprintf(stdout, "[%s] kid=%lu asid=%lu\n", __FUNCTION__,
		(unsigned long)kaapi_get_current_kid(),
	      (unsigned long int)host_asid );
	fflush( stdout );
#endif
#ifndef KAAPI_CUDA_MODE_BASIC
	if( !kaapi_mem_data_has_addr( kmd, asid ) ) {
		kaapi_data_t* dest = (kaapi_data_t*)malloc(sizeof(kaapi_data_t));
		kaapi_cuda_mem_alloc( &dest->ptr, 0UL, 
		    kaapi_memory_view_size(&src->view), 0 );
		dest->view = src->view;
		kaapi_mem_data_set_addr( kmd, asid, (kaapi_mem_addr_t)dest );
		kaapi_mem_data_set_dirty( kmd, asid );
		kaapi_mem_host_map_find_or_insert_(  cuda_map,
		    (kaapi_mem_addr_t)kaapi_pointer2void(dest->ptr),
		    &kmd );
		kaapi_cuda_mem_register( src->ptr, &src->view );
		return dest;
	} else {
	    kaapi_data_t* dest= (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		     asid );
#ifdef	KAAPI_CUDA_MEM_ALLOC_MANAGER
	    kaapi_cuda_mem_inc_use( &dest->ptr );
#endif /* KAAPI_CUDA_MEM_ALLOC_MANAGER */
	return dest;
	}
#else /* KAAPI_CUDA_MODE_BASIC */
	kaapi_data_t* dest = (kaapi_data_t*)malloc(sizeof(kaapi_data_t));
	kaapi_cuda_mem_alloc( &dest->ptr, 0UL, 
	    kaapi_memory_view_size(&src->view), 0 );
	dest->view = src->view;
	kaapi_mem_data_set_addr( kmd, asid, (kaapi_mem_addr_t)dest );
	kaapi_mem_host_map_find_or_insert_(  cuda_map,
	    (kaapi_mem_addr_t)kaapi_pointer2void(dest->ptr),
	    &kmd );
#if 0
	fprintf(stdout, "[%s] dest=%p src=%p data=%p kid=%lu asid=%lu\n", __FUNCTION__,
		kaapi_pointer2void(dest->ptr),
		kaapi_pointer2void(src->ptr),
		dest,
		(unsigned long)kaapi_get_current_kid(),
	      (unsigned long int)asid );
	fflush( stdout );
#endif
	return dest;
#endif /* KAAPI_CUDA_MODE_BASIC */
}

/* The function checks if the dest memory is valid on GPU 
 * */
static inline int
xxx_kaapi_cuda_data_send_ro(
		const kaapi_mem_host_map_t* map,
		kaapi_mem_data_t* kmd,
		kaapi_data_t* dest,
		kaapi_data_t* src
		)
{
	const kaapi_mem_asid_t host_asid= kaapi_mem_host_map_get_asid(map);
#ifndef KAAPI_CUDA_MODE_BASIC
	if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
		kaapi_cuda_mem_copy_htod( dest->ptr, &dest->view,
			src->ptr, &src->view );
		kaapi_mem_data_clear_dirty( kmd, host_asid );
	}
#else
	kaapi_cuda_mem_copy_htod( dest->ptr, &dest->view,
		src->ptr, &src->view );
#endif

	return 0;
}

/*
 * The function sets the current GPU thread as a writter 
 */
static inline int
xxx_kaapi_cuda_data_send_wr(
		const kaapi_mem_host_map_t* host_map,
		kaapi_mem_data_t* kmd,
		kaapi_data_t* dest, 
	       	kaapi_data_t* src
		)
{
#ifndef KAAPI_CUDA_MODE_BASIC
        kaapi_mem_data_set_all_dirty_except( kmd, 
	    kaapi_mem_host_map_get_asid(host_map) );
#endif
	return 0;
}

int kaapi_cuda_data_allocate( 
	kaapi_format_t*		   fmt,
	void*              sp
)
{
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

		if( !kaapi_mem_data_has_addr( kmd, host_asid ) )
		    kaapi_mem_data_set_addr( kmd, host_asid,
			    (kaapi_mem_addr_t)src );

		xxx_kaapi_cuda_data_allocate( cuda_map, kmd, src );
	}

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
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    const kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
//    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
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
	    kaapi_data_t* dest= (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		     cuda_asid );

	    if( KAAPI_ACCESS_IS_READ(m) )
		    xxx_kaapi_cuda_data_send_ro( cuda_map, kmd, dest, src );

	    if( KAAPI_ACCESS_IS_WRITE(m) )
		    xxx_kaapi_cuda_data_send_wr( cuda_map, kmd, dest, src );

	    /* sets new pointer to the task */
	    access.data =  dest;
	    kaapi_format_set_access_param( fmt, i, sp, &access );
    }

    return 0;
}

static inline int
xxx_kaapi_cuda_data_recv(
		const kaapi_mem_host_map_t* map,
		kaapi_mem_data_t* kmd,
		kaapi_data_t* h_dest, 
	       	kaapi_data_t* d_src
		)
{
	kaapi_cuda_mem_copy_dtoh( h_dest->ptr, &h_dest->view,
		d_src->ptr, &d_src->view );
#ifndef KAAPI_CUDA_MODE_BASIC
	kaapi_mem_data_clear_all_dirty( kmd );
#endif

	return 0;
}

int kaapi_cuda_data_recv( 
	kaapi_format_t*		   fmt,
	void*              sp
)
{
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    const kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    kaapi_mem_data_t *kmd;

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] CUDA params=%ld kid=%lu\n", __FUNCTION__,
	    count_params,
	    (unsigned long)kaapi_get_current_kid() );
    fflush( stdout );
#endif
    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param(fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

#ifdef KAAPI_CUDA_MODE_BASIC
	    /* This block executes in the basic case only */
	    kaapi_access_t access = kaapi_format_get_access_param( fmt,
			    i, sp );
	    kaapi_data_t* d_dev = kaapi_data( kaapi_data_t, &access );
	    kaapi_mem_host_map_find_or_insert( cuda_map,
		    (kaapi_mem_addr_t)kaapi_pointer2void(d_dev->ptr),
		    &kmd );
	    kaapi_data_t* d_host = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		    kaapi_mem_host_map_get_asid(host_map) );
#endif

	    if( KAAPI_ACCESS_IS_WRITE(m) ) {
#ifndef KAAPI_CUDA_MODE_BASIC
		/* Additional code in case of non-enabled basic mode.
		 * So, It just enters at WR parameters */
		kaapi_access_t access = kaapi_format_get_access_param( fmt,
				i, sp );
		kaapi_data_t* d_dev = kaapi_data( kaapi_data_t, &access );
		kaapi_mem_host_map_find_or_insert( cuda_map,
			(kaapi_mem_addr_t)kaapi_pointer2void(d_dev->ptr),
			&kmd );
		kaapi_data_t* d_host = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
			kaapi_mem_host_map_get_asid(host_map) );
#endif
		xxx_kaapi_cuda_data_recv( cuda_map, kmd, d_host, d_dev );
	    }

#ifdef KAAPI_CUDA_MODE_BASIC
	    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
	    kaapi_mem_data_clear_addr( kmd, cuda_asid );
	    kaapi_cuda_mem_free( &d_dev->ptr );
	    free( d_dev );
#endif
    }

    return 0;
}

int 
kaapi_cuda_data_check( void )
{
    return kaapi_cuda_mem_mgmt_check( kaapi_get_current_processor() );
}

#if 0

TODO: no use yet

int kaapi_cuda_data_send_ptr( 
	kaapi_format_t*		   fmt,
	void*			sp
)
{
    const size_t count_params = kaapi_format_get_count_params( fmt, sp );
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
	    kaapi_mem_host_map_find_or_insert( 
		    (kaapi_mem_addr_t)access.data,
		    &kmd );
	    kaapi_data_t* src;
	    if( kaapi_mem_data_has_addr( kmd, host_asid ) )
		src = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		    host_asid );
	    else{
		src= (kaapi_data_t*) malloc( sizeof(kaapi_data_t) );
		src->ptr =  kaapi_make_pointer( 0, access.data );
		src->view = kaapi_format_get_view_param(fmt, i, sp);
		kaapi_mem_data_set_addr( kmd, host_asid,
			(kaapi_mem_addr_t)src );
	    }
#if 0
    fprintf(stdout, "[%s] find=%p kmd=%p kid=%lu asid=%lu\n", __FUNCTION__,
	    kaapi_pointer2void(src->ptr), kmd,
	    (unsigned long int)kaapi_get_current_kid(),
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map) );
    fflush( stdout );
#endif

	    kaapi_data_t* dest = xxx_kaapi_cuda_data_allocate( cuda_map, kmd, src );

	    if( KAAPI_ACCESS_IS_READ(m) )
		    xxx_kaapi_cuda_data_send_ro( cuda_map, kmd, dest, src );

	    if( KAAPI_ACCESS_IS_WRITE(m) )
		    xxx_kaapi_cuda_data_send_wr( cuda_map, kmd, dest, src );

	    /* sets new pointer to the task */
	    access.data =  dest;
	    kaapi_format_set_access_param( fmt, i, sp, &access );
    }

    return 0;
}
#endif

#if 0

TODO: no use yet

int kaapi_cuda_data_recv_ptr( 
	kaapi_format_t*		   fmt,
	void*	              sp
)
{
    const size_t count_params = kaapi_format_get_count_params(fmt, sp);
    size_t i;
    const kaapi_mem_host_map_t* cuda_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    kaapi_mem_data_t *kmd;

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] CUDA params=%ld kid=%lu\n", __FUNCTION__,
	    count_params,
	    (unsigned long)kaapi_get_current_kid() );
    fflush( stdout );
#endif
    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param(fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

	    if( KAAPI_ACCESS_IS_WRITE(m) ) {
	    kaapi_access_t access = kaapi_format_get_access_param(
		    fmt, i, sp );
	    kaapi_data_t* d_dev = (kaapi_data_t*) access.data;
	    kaapi_mem_host_map_find_or_insert(
		    (kaapi_mem_addr_t)kaapi_pointer2void(d_dev->ptr),
		    &kmd );
	    kaapi_data_t* d_host = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		    kaapi_mem_host_map_get_asid(host_map) );
#if 0
    fprintf(stdout, "[%s] (%lu -> %lu) hostptr=%p devptr=%p kmd=%p kid=%lu\n", __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(cuda_map), 
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map), 
	    kaapi_pointer2void(d_host->ptr), kaapi_pointer2void(d_dev->ptr),
	    (void*)kmd,
	    (unsigned long)kaapi_get_current_kid() );
    fflush( stdout );
#endif
	    xxx_kaapi_cuda_data_recv( cuda_map, kmd, d_host, d_dev );
	    }
    }

    return 0;
}
#endif
