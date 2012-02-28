
#ifndef KAAPI_MEM_DATA_H_INCLUDED
#define KAAPI_MEM_DATA_H_INCLUDED

#include "kaapi_mem.h"

static inline void
kaapi_mem_data_init( kaapi_mem_data_t *m )
{
  m->parent = NULL;
  m->dirty_bits = 0UL;
  m->addr_bits = 0UL;
}

static inline void
kaapi_mem_data_set_dirty( kaapi_mem_data_t *m, kaapi_mem_asid_t asid)
{
  m->dirty_bits |= 1 << asid;
}

static inline void
kaapi_mem_data_set_all_dirty_except( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  m->dirty_bits = ~(1 << asid);
}

static inline void
kaapi_mem_data_clear_dirty( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  m->dirty_bits &= ~(1 << asid);
}

static inline void
kaapi_mem_data_clear_all_dirty( kaapi_mem_data_t* m )
{
  m->dirty_bits = 0UL;
}

static inline unsigned int
kaapi_mem_data_is_dirty( const kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return m->dirty_bits & (1 << asid);
}

static inline void
kaapi_mem_data_set_parent( kaapi_mem_data_t* m,
	kaapi_mem_data_t* parent )
{
    m->parent = parent;
}

static inline kaapi_mem_data_t*
kaapi_mem_data_get_parent( kaapi_mem_data_t* m )
{
    return m->parent;
}

static inline void
kaapi_mem_data_set_addr(kaapi_mem_data_t* m,
	kaapi_mem_asid_t asid, kaapi_mem_addr_t addr )
{
  m->addr[asid] = addr;
  m->addr_bits |= 1 << asid;
}

static inline kaapi_mem_addr_t
kaapi_mem_data_get_addr( const kaapi_mem_data_t* m, const kaapi_mem_asid_t asid )
{
  return  m->addr[asid];
}

static inline unsigned int
kaapi_mem_data_has_addr( const kaapi_mem_data_t* m, const kaapi_mem_asid_t asid )
{
  return m->addr_bits & (1 << asid);
}

static inline void
kaapi_mem_data_clear_addr( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  m->addr_bits &= ~(1 << asid);
}

/* It returns the first asid non-dirty of kmd data */
static inline kaapi_mem_asid_t
kaapi_mem_data_get_nondirty_asid( const kaapi_mem_data_t* kmd )
{
    kaapi_mem_asid_t asid= 0;

    for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
	if( !kaapi_mem_data_is_dirty( kmd, asid ) )
	    break;

    return asid;
}

/* It returns a non-dirty asid but different of current_asid */
static inline kaapi_mem_asid_t
kaapi_mem_data_get_nondirty_asid_( const kaapi_mem_data_t* kmd,
       kaapi_mem_asid_t current_asid )
{
    kaapi_mem_asid_t asid= 0;

    for (asid = 0; asid < KAAPI_MEM_ASID_MAX; ++asid)
	if( (!kaapi_mem_data_is_dirty( kmd, asid )) &&
		(current_asid != asid) )
	    break;

    return asid;
}

#endif /* KAAPI_MEM_DATA_H_INCLUDED */
