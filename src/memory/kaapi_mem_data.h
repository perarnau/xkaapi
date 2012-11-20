
#ifndef KAAPI_MEM_DATA_H_INCLUDED
#define KAAPI_MEM_DATA_H_INCLUDED

#include "kaapi_mem.h"
#include "kaapi_impl.h"

static inline void
kaapi_mem_data_init( kaapi_mem_data_t *m )
{
  m->next = NULL;
  kaapi_bitmap_clear_64( &m->valid_bits );
  kaapi_bitmap_clear_64( &m->addr_bits );
  memset(&m->addr, 0, KAAPI_MEM_ASID_MAX*sizeof(kaapi_mem_addr_t));
}

static inline void
kaapi_mem_data_set_dirty( kaapi_mem_data_t *m, kaapi_mem_asid_t asid)
{
  kaapi_bitmap_unset_64( &m->valid_bits, asid );
}

static inline void
kaapi_mem_data_set_all_dirty_except( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  kaapi_bitmap_clear_64( &m->valid_bits );
  kaapi_bitmap_set_64( &m->valid_bits, asid );
}

static inline void
kaapi_mem_data_clear_dirty( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  kaapi_bitmap_set_64( &m->valid_bits, asid );
}

/* test if the previous value was dirty */
static inline int
kaapi_mem_data_clear_dirty_and_check( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return (kaapi_bitmap_fetch_and_set_64( &m->valid_bits, asid ) == 0);
}

static inline unsigned int
kaapi_mem_data_is_dirty( const kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return (kaapi_bitmap_get_64( &m->valid_bits, asid ) == 0);
}

static inline void
kaapi_mem_data_set_next( kaapi_mem_data_t* m, kaapi_mem_data_t* next )
{
    m->next = next;
}

static inline kaapi_mem_data_t*
kaapi_mem_data_get_next( kaapi_mem_data_t* m )
{
    return m->next;
}

static inline void
kaapi_mem_data_set_addr(kaapi_mem_data_t* m,
	kaapi_mem_asid_t asid, kaapi_mem_addr_t addr )
{
  m->addr[asid] = addr;
  kaapi_bitmap_set_64( &m->addr_bits, asid );
}

static inline kaapi_mem_addr_t
kaapi_mem_data_get_addr( const kaapi_mem_data_t* m, const kaapi_mem_asid_t asid )
{
  return  m->addr[asid];
}

static inline unsigned int
kaapi_mem_data_has_addr( const kaapi_mem_data_t* m, const kaapi_mem_asid_t asid )
{
  return kaapi_bitmap_get_64( &m->addr_bits, asid );
}

static inline void
kaapi_mem_data_clear_addr( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  kaapi_bitmap_unset_64( &m->addr_bits, asid );
}

/* It returns the first asid non-dirty of kmd data */
static inline kaapi_mem_asid_t
kaapi_mem_data_get_nondirty_asid( const kaapi_mem_data_t* kmd )
{
  return ( kaapi_bitmap_first1_64(&kmd->valid_bits) - 1);
}

#endif /* KAAPI_MEM_DATA_H_INCLUDED */
