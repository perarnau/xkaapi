
#include "kaapi_impl.h"
#include "kaapi_mem.h"

kaapi_big_hashmap_t kmem_hm;

static inline kaapi_mem_data_t* 
_kaapi_mem_data_alloc( void )
{
    kaapi_mem_data_t* kmd = malloc( sizeof(kaapi_mem_data_t) );
    kaapi_mem_data_init( kmd );
    return kmd;
}

int
kaapi_mem_host_map_find(
	kaapi_mem_addr_t addr,
	kaapi_mem_data_t** data
	)
{
    kaapi_hashentries_t* entry;

    entry = kaapi_big_hashmap_find( &kmem_hm, (void*)addr );
    if (entry != 0)
	*data = entry->u.kmd;
    else
	return -1;

    return 0;
}

int
kaapi_mem_host_map_find_or_insert(
	kaapi_mem_addr_t addr,
	kaapi_mem_data_t** kmd
	)
{
    kaapi_hashentries_t* entry;
    const int res = kaapi_mem_host_map_find( addr, kmd );
    if( res == 0 )
        return 0;

    entry = kaapi_big_hashmap_findinsert( &kmem_hm, (void*)addr );
    if (entry->u.kmd == 0)
	entry->u.kmd = _kaapi_mem_data_alloc();

    /* identity mapping */
    *kmd = entry->u.kmd;

    return 0;
}

int
kaapi_mem_host_map_find_or_insert_( 
	kaapi_mem_addr_t addr, kaapi_mem_data_t** kmd )
{
    kaapi_hashentries_t* entry;
    const int res = kaapi_mem_host_map_find( addr, kmd );
    if( res == 0 )
        return 0;

    entry = kaapi_big_hashmap_findinsert( &kmem_hm, (void*)addr );
    entry->u.kmd = *kmd;

    return 0;
}

int
kaapi_mem_host_map_sync( const kaapi_format_t* fmt, void* sp )
{
    size_t count_params = kaapi_format_get_count_params( fmt, sp );
    size_t i;
    kaapi_mem_data_t *kmd;
    const kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);

#if 0
    fprintf( stdout, "[%s] asid=%lu task=%s params=%lu\n",
	    __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map),
	    fmt->name,
	    count_params );
    fflush(stdout);
#endif
    for( i= 0; i < count_params; i++ ) {
	kaapi_access_mode_t m = fmt->get_mode_param( fmt, i, sp );

	if( m == KAAPI_ACCESS_MODE_V )
	    continue;

	kaapi_access_t access = fmt->get_access_param( fmt, i, sp );
	kaapi_data_t* kdata = kaapi_data( kaapi_data_t, &access );
	kaapi_mem_host_map_find_or_insert( 
		(kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
		&kmd );
	if( !kaapi_mem_data_has_addr( kmd, host_asid ) )
	    kaapi_mem_data_set_addr( kmd, host_asid,
		    (kaapi_mem_addr_t)kdata  );
#if 0
    fprintf( stdout, "[%s] asid=%lu task=%s params=%lu ptr=%p kmd=%p\n",
	    __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map),
	    fmt->name,
	    count_params,
	    kaapi_pointer2void(kdata->ptr),
	    kmd );
    fflush(stdout);
#endif

	if( KAAPI_ACCESS_IS_READ(m) ) {
	    if( kaapi_mem_data_is_dirty( kmd,
		       	kaapi_mem_host_map_get_asid(host_map) ) ) {
#if 0
		fprintf( stdout, "[%s] DIRTY ptr=%p\n", __FUNCTION__,
		      kaapi_pointer2void(kdata->ptr) );
		fflush(stdout);
#endif
	    }
	}

	if( KAAPI_ACCESS_IS_WRITE(m) ) {
	    kaapi_mem_data_set_all_dirty_except( kmd, 
		    kaapi_mem_host_map_get_asid(host_map) );
	}

    }

    return 0;
}

int
kaapi_mem_host_map_sync_ptr( const kaapi_format_t* fmt, void* sp )
{
    size_t count_params = kaapi_format_get_count_params( fmt, sp );
    size_t i;
    kaapi_mem_data_t *kmd;
    const kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);

#if 0
    fprintf( stdout, "[%s] asid=%lu task=%s stack=%p params=%lu\n",
	    __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map),
	    fmt->name,
	    sp,
	    count_params );
    fflush(stdout);
#endif
    for( i= 0; i < count_params; i++ ) {
	kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		kaapi_format_get_mode_param( fmt, i, sp) );
	if( m == KAAPI_ACCESS_MODE_V )
	    continue;

	kaapi_access_t access = kaapi_format_get_access_param( fmt,
			i, sp );
#if 0
    fprintf( stdout, "[%s] asid=%lu task=%s params=%lu ptr=%p\n",
	    __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map),
	    fmt->name,
	    count_params,
	    access.data 
	    );
    fflush(stdout);
#endif
	kaapi_mem_host_map_find_or_insert( 
		(kaapi_mem_addr_t)access.data,
		&kmd );
//	kaapi_data_t* kdata = kaapi_data( kaapi_data_t, &access );
	kaapi_data_t* kdata;
	if( kaapi_mem_data_has_addr( kmd, host_asid ) ) {
		kdata = (kaapi_data_t*) kaapi_mem_data_get_addr( kmd,
		    host_asid );
	} else {
		kdata= (kaapi_data_t*) malloc( sizeof(kaapi_data_t) );
		kdata->ptr =  kaapi_make_pointer( 0, access.data );
		kdata->view = kaapi_format_get_view_param(fmt, i, sp);
		kaapi_mem_data_set_addr( kmd, host_asid,
			(kaapi_mem_addr_t)kdata );
	}

	if( KAAPI_ACCESS_IS_READ(m) ) {
	    if( kaapi_mem_data_is_dirty( kmd,
		       	kaapi_mem_host_map_get_asid(host_map) ) ) {
		fprintf( stdout, "[%s] DIRTY ptr=%p\n", __FUNCTION__,
		      kaapi_pointer2void(kdata->ptr) );
		fflush(stdout);
	    }
	}

	if( KAAPI_ACCESS_IS_WRITE(m) ) {
	    kaapi_mem_data_set_all_dirty_except( kmd, 
		    kaapi_mem_host_map_get_asid(host_map) );
	}
	access.data =  kdata;
	kaapi_format_set_access_param( fmt, i, sp, &access );
    }

    return 0;
}
