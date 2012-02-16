
#include <stdio.h>

#include "kaapi_impl.h"
#include "memory/kaapi_mem.h"

int
kaapi_mem_sync_ptr( kaapi_access_t access )
{
    //const kaapi_mem_host_map_t* host_map = kaapi_get_current_mem_host_map();
    const kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_data_t *kmd;
    kaapi_mem_asid_t valid_asid;

    kaapi_data_t* kdata = kaapi_data( kaapi_data_t, &access );
    kaapi_mem_host_map_find_or_insert( host_map,
	    (kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr),
	    &kmd );
    if( kaapi_mem_data_has_addr( kmd, host_asid ) ) {
	if ( kaapi_mem_data_is_dirty( kmd, host_asid ) ) {
	    valid_asid = kaapi_mem_data_get_nondirty_asid( kmd );
	    kaapi_data_t* valid_data = kaapi_mem_data_get_addr( kmd, valid_asid );
	    fprintf( stdout, "[%s] dirty asid=%lu(%p) valid=%lu(%p) kid=%lu\n",
		    __FUNCTION__,
		    host_asid, kaapi_pointer2void(kdata->ptr),
		    valid_asid, kaapi_pointer2void(valid_data->ptr),
		    (unsigned long)kaapi_get_current_kid() );
	    fflush(stdout);
//	    kaapi_mem_data_clear_dirty( kmd, host_asid );
	}
    } else {
	    kaapi_mem_data_set_addr( kmd, host_asid,
		    (kaapi_mem_addr_t)kdata  );
    }
    return 0;
}
