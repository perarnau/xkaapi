
#include "kaapi_impl.h"
#include "kaapi_mem.h"

kaapi_big_hashmap_t kmem_hm;

static inline kaapi_mem_data_t* 
_kaapi_mem_data_alloc( void )
{
    kaapi_mem_data_t* kmd = malloc( sizeof(kaapi_mem_data_t) );
    return kmd;
}

int
kaapi_mem_host_map_find( kaapi_mem_host_map_t* m, kaapi_mem_addr_t addr,
	kaapi_mem_data_t** data )
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
kaapi_mem_host_map_find_or_insert( kaapi_mem_host_map_t* m,
	kaapi_mem_addr_t addr, kaapi_mem_data_t** kmd )
{
    kaapi_hashentries_t* entry;
    const int res = kaapi_mem_host_map_find( m, addr, kmd );
    if( res == 0 )
        return 0;

    entry = kaapi_big_hashmap_findinsert( &kmem_hm, (void*)addr );
    if (entry->u.kmd == 0)
	entry->u.kmd = _kaapi_mem_data_alloc();

    /* identity mapping */
    kaapi_mem_data_init( entry->u.kmd );
    kaapi_mem_data_set_addr( entry->u.kmd, kaapi_mem_host_map_get_asid(m), addr );
    *kmd = entry->u.kmd;

    return 0;
}

int
kaapi_mem_host_map_sync( const kaapi_format_t* fmt, kaapi_task_t* task )
{
    void* sp = task->sp;
    size_t count_params = kaapi_format_get_count_params( fmt, sp );
    size_t i;
    kaapi_mem_data_t *kmd;
    kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();

#if KAAPI_VERBOSE
    fprintf( stdout, "[%s] asid=%lu task=%s params=%lu\n",
	    __FUNCTION__,
	    (unsigned long int)kaapi_mem_host_map_get_asid(host_map),
	    fmt->name,
	    count_params );
    fflush(stdout);
#endif
    for( i= 0; i < count_params; i++ ) {
	kaapi_access_mode_t m = fmt->get_mode_param( fmt, i, sp );
//	kaapi_access_mode_t m = kaapi_format_get_mode_param( fmt, i, sp );
//	m = KAAPI_ACCESS_GET_MODE( m );
//	kaapi_access_t access = kaapi_format_get_access_param(
//		fmt, i, sp );

	if( m == KAAPI_ACCESS_MODE_V )
	    continue;

	kaapi_access_t access = fmt->get_access_param( fmt, i, sp );
	kaapi_data_t* kdata = kaapi_data( kaapi_data_t, &access );
	kaapi_mem_host_map_find_or_insert( 
		host_map,
		(kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
		&kmd );

	if( KAAPI_ACCESS_IS_WRITE(m) ) {
	    kaapi_mem_data_set_all_dirty_except( kmd, 
		    kaapi_mem_host_map_get_asid(host_map) );
	}

	if( KAAPI_ACCESS_IS_READ(m) ) {
	    if( kaapi_mem_data_is_dirty( kmd,
		       	kaapi_mem_host_map_get_asid(host_map) ) ) {
		fprintf( stdout, "[%s] DIRTY ptr=%p\n", __FUNCTION__,
		      kaapi_pointer2void(kdata->ptr) );
		fflush(stdout);
	    }
	}
    }

    return 0;
}
