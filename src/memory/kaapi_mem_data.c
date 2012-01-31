
#include "kaapi_impl.h"
#include "kaapi_mem.h"

kaapi_mem_asid_t
kaapi_mem_data_get_nondirty_asid( const kaapi_mem_data_t* kmd )
{
    kaapi_mem_asid_t asid= 0;

    for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
	if( !kaapi_mem_data_is_dirty( kmd, asid ) )
	    break;


    return asid;
}

