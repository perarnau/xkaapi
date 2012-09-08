
#ifndef KAAPI_MEM_HOST_MAP_H_INCLUDED
#define KAAPI_MEM_HOST_MAP_H_INCLUDED

#include "kaapi_mem.h"

extern kaapi_processor_id_t	kaapi_all_asid2kid[KAAPI_MEM_ASID_MAX];

static inline int
kaapi_mem_host_map_init( kaapi_mem_host_map_t* map, kaapi_processor_id_t kid, kaapi_mem_asid_t asid )
{
#if KAAPI_VERBOSE
    fprintf( stdout, "[%s] asid=%lu\n", __FUNCTION__,
	    (unsigned long int)asid );
    fflush(stdout);
#endif
  map->asid = asid;
  kaapi_all_asid2kid[asid] = kid;

  kaapi_big_hashmap_init( &map->hmap, 0 );  
  return 0;
}

static inline kaapi_mem_asid_t
kaapi_mem_host_map_get_asid( const kaapi_mem_host_map_t* map )
{ return map->asid; }

static inline kaapi_processor_id_t
kaapi_mem_asid2kid( kaapi_mem_asid_t asid )
{
    if( asid == 0 )
	return (kaapi_processor_id_t)0;
    return kaapi_all_asid2kid[asid];
}

int
kaapi_mem_host_map_find( kaapi_mem_host_map_t*, kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_find_or_insert( kaapi_mem_host_map_t*,
	kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_find_or_insert_( kaapi_mem_host_map_t*, 
	kaapi_mem_addr_t, kaapi_mem_data_t**);

struct kaapi_taskdescr_t;

int
kaapi_mem_host_map_sync( struct kaapi_taskdescr_t* const );

static inline kaapi_mem_addr_t
kaapi_mem_host_map_generate_id( void* const ptr, const size_t size )
{
  return (kaapi_mem_addr_t)( ((kaapi_mem_addr_t)ptr) + (kaapi_mem_addr_t)size );
}

static inline kaapi_mem_addr_t
kaapi_mem_host_map_generate_id_by_data( kaapi_data_t* const kdata )
{
  return (kaapi_mem_addr_t)(
      ((kaapi_mem_addr_t)kaapi_pointer2void(kdata->ptr)) + 
	(kaapi_mem_addr_t)kaapi_memory_view_size(&kdata->view)
      );
}

extern kaapi_mem_data_t* kaapi_mem_host_map_register_to_host(
      void* ptr,
      kaapi_memory_view_t* const view
    );

#endif /* KAAPI_MEM_HOST_MAP_H_INCLUDED */
