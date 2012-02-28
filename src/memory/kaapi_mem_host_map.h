
#ifndef KAAPI_MEM_HOST_MAP_H_INCLUDED
#define KAAPI_MEM_HOST_MAP_H_INCLUDED

#include "kaapi_mem.h"

static inline int
kaapi_mem_host_map_init( kaapi_mem_host_map_t* map, kaapi_mem_asid_t asid )
{
#if KAAPI_VERBOSE
    fprintf( stdout, "[%s] asid=%lu\n", __FUNCTION__,
	    (unsigned long int)asid );
    fflush(stdout);
#endif
  map->asid = asid;
  map->data.beg = map->data.end = NULL;
  kaapi_big_hashmap_init( &map->data.hblocks, 0 );  

  kaapi_big_hashmap_init( &map->hmap, 0 );  
  return 0;
}

static inline kaapi_mem_asid_t
kaapi_mem_host_map_get_asid( const kaapi_mem_host_map_t* map )
{ return map->asid; }

int
kaapi_mem_host_map_find( const kaapi_mem_host_map_t*, kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_find_or_insert( const kaapi_mem_host_map_t*,
	kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_find_or_insert_( const kaapi_mem_host_map_t*, 
	kaapi_mem_addr_t, kaapi_mem_data_t**);

int
kaapi_mem_host_map_sync( const kaapi_format_t* , void* );

#endif /* KAAPI_MEM_HOST_MAP_H_INCLUDED */
