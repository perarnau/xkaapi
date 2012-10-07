/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
** clement.pernet@imag.fr
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
#ifndef _KAAPI_HASHMAP_H_
#define _KAAPI_HASHMAP_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif

#include "config.h"
#include "kaapi_error.h"
#include "kaapi_defs.h"


/* ============================= Basic type ============================ */
/** \ingroup DFG
*/
typedef struct kaapi_gd_t {
  kaapi_access_mode_t         last_mode;    /* last access mode to the data */
  void*                       last_version; /* last verion of the data, 0 if not ready */
} kaapi_gd_t;

/* fwd decl
*/
struct kaapi_version_t;
struct kaapi_metadata_info_t;
struct kaapi_mem_data_t;
struct kaapi_cuda_mem_cache_blk_t;

/** pair of pointer,int 
    Used to display tasklist
*/
typedef struct kaapi_pair_ptrint_t {
  void*               ptr;
  uintptr_t           tag;
  kaapi_access_mode_t last_mode;
} kaapi_pair_ptrint_t;


/* ============================= Hash table for WS ============================ */

/*
*/
#define KAAPI_BLOCENTRIES_SIZE 2048

/* Generic blocs with KAAPI_BLOCENTRIES_SIZE entries
*/
#define KAAPI_DECLARE_BLOCENTRIES(NAME, TYPE) \
typedef struct NAME {\
  TYPE         data[KAAPI_BLOCENTRIES_SIZE]; \
  uintptr_t    pos;  /* next free in data */\
  struct NAME* next; /* link list of bloc */\
  void*        ptr;  /* memory pointer of allocated bloc */\
} NAME


/*
*/
typedef struct kaapi_hashentries_t {
  union { /* depending of the kind of hash table... */
    kaapi_gd_t                    value;
    struct kaapi_version_t*       version;     /* for static scheduling */
    kaapi_pair_ptrint_t           data;        /* used for print tasklist */
    struct kaapi_metadata_info_t* mdi;         /* store of metadata info */
    struct kaapi_taskdescr_t*     td;          /* */
    struct kaapi_mem_data_t*	 kmd;
    struct kaapi_cuda_mem_cache_blk_t*    block;
  } u;
  const void*                     key;
  struct kaapi_hashentries_t*     next; 
} kaapi_hashentries_t;

KAAPI_DECLARE_BLOCENTRIES(kaapi_hashentries_bloc_t, kaapi_hashentries_t);


/* ========================================================================== */
/* Hashmap default size.
   Warning in kapai_hashmap_t, entry_map type should have a size that is
   equal to KAAPI_HASHMAP_SIZE.
*/
#define KAAPI_HASHMAP_SIZE 64

static inline uint64_t _key_to_mask(uint32_t k)
{ return ((uint64_t)1) << k; }

/*
*/
typedef struct kaapi_hashmap_t {
  kaapi_hashentries_t* entries[KAAPI_HASHMAP_SIZE];
  kaapi_hashentries_bloc_t* currentbloc;
  kaapi_hashentries_bloc_t* allallocatedbloc;
  uint64_t entry_map; /* type size must at least KAAPI_HASHMAP_SIZE */
} kaapi_hashmap_t;


/* ========================================================================== */
/* Big hashmap_big
   Used for bulding readylist
*/
#define KAAPI_HASHMAP_BIG_SIZE 65536

/*
*/
typedef struct kaapi_big_hashmap_t {
  kaapi_hashentries_t* entries[KAAPI_HASHMAP_BIG_SIZE];
  kaapi_hashentries_bloc_t* currentbloc;
  kaapi_hashentries_bloc_t* allallocatedbloc;
} kaapi_big_hashmap_t;



/* ========================================================================== */
/** Compute a hash value from a string
*/
extern uint32_t kaapi_hash_value_len(const char * data, size_t len);

/*
*/
extern uint32_t kaapi_hash_value(const char * data);


/**
 * Compression 64 -> 7 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^7 + X^3 + 1
 */
static inline uint32_t kaapi_hash_ulong7(uint64_t v)
{
  v ^= (v >> 32);
  v ^= (v >> 16);
  v ^= (v >> 8);
  if (v & 0x00000080) v ^= 0x00000009;
  return (uint32_t) (v&0x0000007F);
}


/**
 * Compression 64 -> 6 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^6 + X + 1
 */
static inline uint32_t kaapi_hash_ulong6(uint64_t v)
{
  v ^= (v >> 32);
  v ^= (v >> 16);
  v ^= (v >> 8);
  if (v & 0x00000040) v ^= 0x00000003;
  if (v & 0x00000080) v ^= 0x00000006;
  return (uint32_t) (v&0x0000003F);
}

/**
 * Compression 64 -> 5 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^5 + X^2 + 1
 */
static inline uint32_t kaapi_hash_ulong5(uint64_t v)
{
  v ^= (v >> 32);
  v ^= (v >> 16);
  v ^= (v >> 8);
  if (v & 0x00000020) v ^= 0x00000005;
  if (v & 0x00000040) v ^= 0x0000000A;
  if (v & 0x00000080) v ^= 0x00000014;
  return (uint32_t) (v&0x0000001F);
}


/** Hash value for pointer.
    Used for data flow dependencies
*/
static inline uint32_t kaapi_hash_ulong(uint64_t v)
{
#if 1
  v ^= (v >> 32);
  v ^= (v >> 16);
  v ^= (v >> 8);
  return (uint32_t) ( v & 0x0000FFFF);
#else  /* */
  uint64_t val = v >> 3;
  v = (v & 0xFFFF) ^ (v>>32);
  return (uint32_t)v;
#endif
}


/*
*/
static inline kaapi_hashentries_t* _get_hashmap_entry(kaapi_hashmap_t* khm, uint32_t key)
{
  kaapi_assert_debug(key < (8 * sizeof(khm->entry_map)));

  if (khm->entry_map & _key_to_mask(key))
    return khm->entries[key];

  return 0;
}


/*
*/
static inline void _set_hashmap_entry
(kaapi_hashmap_t* khm, uint32_t key, kaapi_hashentries_t* entries)
{
  kaapi_assert_debug(key < (8 * sizeof(khm->entry_map)));
  khm->entries[key] = entries;
  khm->entry_map |= _key_to_mask(key);
}


/*
*/
extern int kaapi_hashmap_init( kaapi_hashmap_t* khm, kaapi_hashentries_bloc_t* initbloc );

/*
*/
extern int kaapi_hashmap_clear( kaapi_hashmap_t* khm );

/*
*/
extern int kaapi_hashmap_destroy( kaapi_hashmap_t* khm );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_findinsert( kaapi_hashmap_t* khm, const void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_find( kaapi_hashmap_t* khm, const void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_insert( kaapi_hashmap_t* khm, const void* ptr );

/*
*/
extern kaapi_hashentries_t* get_hashmap_entry( kaapi_hashmap_t* khm, uint32_t key);

/*
*/
extern void set_hashmap_entry( kaapi_hashmap_t* khm, uint32_t key, kaapi_hashentries_t* entries);

/*
*/
extern int kaapi_big_hashmap_init( kaapi_big_hashmap_t* khm, kaapi_hashentries_bloc_t* initbloc );

/*
*/
extern int kaapi_big_hashmap_destroy( kaapi_big_hashmap_t* khm );

/*
*/
extern kaapi_hashentries_t* kaapi_big_hashmap_findinsert( kaapi_big_hashmap_t* khm, const void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_big_hashmap_find( kaapi_big_hashmap_t* khm, const void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_big_hashmap_insert( kaapi_big_hashmap_t* khm, const void* ptr );



#if defined(__cplusplus)
}
#endif

#endif
