/*
 ** xkaapi
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
#include "kaapi_impl.h"
//#include "kaapi_memory.h"

/** Global Hash map of all mapping
*/
#ifdef SMALL_HASH
static kaapi_hashmap_t kmdi_hm;
#else
static kaapi_big_hashmap_t kmdi_hm;
#endif


/*
*/
void kaapi_memory_init(void)
{
#ifdef SMALL_HASH
  kaapi_hashmap_init( &kmdi_hm, 0 );  
#else
  kaapi_big_hashmap_init( &kmdi_hm, 0 );  
#endif
}


/*
*/
void kaapi_memory_destroy(void)
{
#ifdef SMALL_HASH
  kaapi_hashmap_destroy( &kmdi_hm );
#else
  kaapi_big_hashmap_destroy( &kmdi_hm );
#endif
}

/**
*/
kaapi_metadata_info_t* kaapi_memory_find_metadata( void* ptr )
{
  kaapi_hashentries_t* entry;
  
#ifdef SMALL_HASH
  entry = kaapi_hashmap_find(&kmdi_hm, ptr);
#else
  entry = kaapi_big_hashmap_find(&kmdi_hm, ptr);
#endif
  if (entry ==0) return 0;
  return entry->u.mdi;
}


/**
*/
static inline kaapi_metadata_info_t* _kaapi_memory_allocate_mdi()
{
#if 0//defined(KAAPI_USE_NUMA)
  kaapi_metadata_info_t* mdi = numa_alloc_local( sizeof(kaapi_metadata_info_t) );
#else
  kaapi_metadata_info_t* mdi = malloc( sizeof(kaapi_metadata_info_t) );
#endif
  if (mdi ==0) return 0;
  mdi->validbits = 0;
#if defined(KAAPI_DEBUG)
  memset(mdi->data, 0, sizeof(kaapi_data_t)*KAAPI_MAX_ADDRESS_SPACE );
  memset(mdi->version, 0, sizeof(kaapi_version_t*)*KAAPI_MAX_ADDRESS_SPACE );
#endif
  return mdi;
}


/**
*/
kaapi_metadata_info_t* kaapi_mem_findinsert_metadata( void* ptr )
{
  kaapi_hashentries_t* entry;
  
#ifdef SMALL_HASH
  entry = kaapi_hashmap_findinsert(&kmdi_hm, ptr);
#else
  entry = kaapi_big_hashmap_findinsert(&kmdi_hm, ptr);
#endif
  if (entry->u.mdi ==0)
    entry->u.mdi = _kaapi_memory_allocate_mdi();
  return entry->u.mdi;
}


/**
*/
kaapi_version_t** _kaapi_metadata_info_bind_data( 
    kaapi_metadata_info_t* kmdi, 
    kaapi_address_space_id_t kasid, 
    void* ptr, const kaapi_memory_view_t* view
)
{
  uint16_t lid = _kaapi_memory_address_space_getlid( kasid );
  kaapi_assert_debug( lid < KAAPI_MAX_ADDRESS_SPACE );
  kmdi->data[lid].ptr  = kaapi_make_pointer(kasid, ptr);
  kmdi->data[lid].view = *view;
  kmdi->data[lid].mdi = kmdi;
  kmdi->validbits |= (1UL << lid);
  return &kmdi->version[lid];
}


/**
*/
kaapi_metadata_info_t* kaapi_memory_bind( 
  kaapi_address_space_id_t kasid, 
  int flag, 
  void* ptr, 
  size_t size 
)
{
  kaapi_hashentries_t* entry;
  
#ifdef SMALL_HASH
  entry = kaapi_hashmap_findinsert(&kmdi_hm, ptr);
#else
  entry = kaapi_big_hashmap_findinsert(&kmdi_hm, ptr);
#endif
  if (entry->u.mdi ==0)
    entry->u.mdi = _kaapi_memory_allocate_mdi();

  kaapi_assert_debug( entry->u.mdi->data[_kaapi_memory_address_space_getlid( kasid )].ptr.ptr == 0 );
  kaapi_memory_view_t view = kaapi_memory_view_make1d(size, 1);
  _kaapi_metadata_info_bind_data( entry->u.mdi, kasid, ptr, &view );
  return entry->u.mdi;
}


/**
*/
kaapi_metadata_info_t* kaapi_memory_bind_view( 
  kaapi_address_space_id_t kasid, 
  int flag, 
  void* ptr, 
  const kaapi_memory_view_t* view 
)
{
  kaapi_hashentries_t* entry;
  
#ifdef SMALL_HASH
  entry = kaapi_hashmap_findinsert(&kmdi_hm, ptr);
#else
  entry = kaapi_big_hashmap_findinsert(&kmdi_hm, ptr);
#endif
  if (entry->u.mdi ==0)
    entry->u.mdi = _kaapi_memory_allocate_mdi();

  kaapi_assert_debug( entry->u.mdi->data[_kaapi_memory_address_space_getlid( kasid )].ptr.ptr == 0 );
  _kaapi_metadata_info_bind_data( entry->u.mdi, kasid, ptr, view );
  return entry->u.mdi;
}


/** 
*/
void* kaapi_memory_unbind( kaapi_metadata_info_t* kmdi )
{
  kaapi_assert(0); // TODO
  void* retval = 0;
  return retval;
}
