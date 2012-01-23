/*
** kaapi_mem.h
** xkaapi
** 
**
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#ifndef KAAPI_MEM_H_INCLUDED
# define KAAPI_MEM_H_INCLUDED


#include <sys/types.h>
#include <stdint.h>

void
kaapi_mem_init( void );

void
kaapi_mem_destroy( void );

/* kaapi_mem_addr_t is a type large enough to
   contain all the addresses of all memory spaces.
*/
typedef uintptr_t kaapi_mem_addr_t;

/* address space identifier
 */
typedef unsigned int kaapi_mem_asid_t;

typedef struct kaapi_mem_data_t {
#define KAAPI_MEM_ASID_MAX 32
    kaapi_mem_addr_t addr[KAAPI_MEM_ASID_MAX];

    unsigned int dirty_bits;
    unsigned int addr_bits;
} kaapi_mem_data_t;

static inline void
kaapi_mem_data_init( kaapi_mem_data_t *m )
{
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

static inline unsigned int
kaapi_mem_data_is_dirty( const kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return m->dirty_bits & (1 << asid);
}

static inline void
kaapi_mem_data_set_addr(kaapi_mem_data_t* m,
	kaapi_mem_asid_t asid, kaapi_mem_addr_t addr )
{
  m->addr[asid] = addr;
  m->addr_bits |= 1 << asid;
}

static inline kaapi_mem_addr_t
kaapi_mem_data_get_addr( const kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return  m->addr[asid];
}

static inline unsigned int
kaapi_mem_data_has_addr( const kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  return m->addr_bits & (1 << asid);
}

static inline void
kaapi_mem_data_clear_addr( kaapi_mem_data_t* m, kaapi_mem_asid_t asid )
{
  m->addr_bits &= ~(1 << asid);
}

kaapi_mem_asid_t
kaapi_mem_data_get_nondirty_asid( const kaapi_mem_data_t* );

typedef struct kaapi_mem_host_map_t {
    kaapi_mem_asid_t asid;
} kaapi_mem_host_map_t;

static inline int
kaapi_mem_host_map_init( kaapi_mem_host_map_t* map, kaapi_mem_asid_t asid )
{
#if KAAPI_VERBOSE
    fprintf( stdout, "[%s] asid=%lu\n", __FUNCTION__,
	    (unsigned long int)asid );
    fflush(stdout);
#endif
  map->asid = asid;
  return 0;
}

static inline kaapi_mem_asid_t
kaapi_mem_host_map_get_asid( kaapi_mem_host_map_t* map )
{ return map->asid; }

int
kaapi_mem_host_map_find( kaapi_mem_host_map_t*, kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_find_or_insert( kaapi_mem_host_map_t*, kaapi_mem_addr_t, kaapi_mem_data_t** );

int
kaapi_mem_host_map_sync( const kaapi_format_t* , kaapi_task_t* );

#if 0
/* kaapi_mem_mapping is the set of the remote addr
   associated with a given address. it contains the
   meta data for coherency protocol encoded by the
   dirty and addr bitmaps, one bit per asid.
*/
typedef struct kaapi_mem_mapping
{
  struct kaapi_mem_mapping* next;

  /* unsigned int bitmaps means 32 max asids */
#define KAAPI_MEM_ASID_MAX 32
  kaapi_mem_addr_t addrs[KAAPI_MEM_ASID_MAX];

  /* meta, one bit per asid */
  unsigned int dirty_bits;
  unsigned int addr_bits;

} kaapi_mem_mapping_t;



/* a map contains all the mappings of a given as.
 */

typedef struct kaapi_mem_map
{
  kaapi_mem_asid_t     asid;
  kaapi_mem_mapping_t* head;
} kaapi_mem_map_t;

/* Put here to avoid conflict in processor mapping
*/
static inline int kaapi_mem_map_initialize(kaapi_mem_map_t* map, kaapi_mem_asid_t asid)
{
  map->asid = asid;
  map->head = NULL;
  return 0;
}

/*
*/
void kaapi_mem_map_cleanup(kaapi_mem_map_t*);

/*
*/
int kaapi_mem_map_find
  (kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);


/*
*/
int kaapi_mem_map_find_or_insert
  (kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);

/*
*/
int kaapi_mem_map_find_inverse
  (kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_mapping_t**);

/*
*/
int kaapi_mem_map_find_with_asid
(kaapi_mem_map_t*, kaapi_mem_addr_t, kaapi_mem_asid_t, kaapi_mem_mapping_t**);

/*
*/
void kaapi_mem_synchronize(kaapi_mem_addr_t, size_t);

int kaapi_mem_synchronize2(kaapi_mem_addr_t, size_t);
int kaapi_mem_synchronize3(kaapi_mem_mapping_t*, size_t);

/*
 */
void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);

#endif

#endif /* ! KAAPI_MEM_H_INCLUDED */
