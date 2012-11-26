/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
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

/** Map all ASID pointers. **/
static kaapi_memory_map_t         kaapi_memory_all_local_maps[KAAPI_MAX_ADDRESS_SPACE];

/** Map processors to one address space (kid -> asid)  **/
static kaapi_address_space_id_t   kaapi_memory_all_kid2asid[KAAPI_MAX_PROCESSOR];

/** Map address space to processor (asid -> kid) **/
static kaapi_processor_id_t       kaapi_memory_all_asid2kid[KAAPI_MAX_ADDRESS_SPACE];

void kaapi_memory_map_init( void )
{
  int i;
  
  for( i= 0; i < KAAPI_MAX_PROCESSOR; i++ )
  {
    kaapi_memory_all_kid2asid[i] = KAAPI_EMPTY_ADDRESS_SPACE_ID;
  }
  for( i= 0; i < KAAPI_MAX_ADDRESS_SPACE; i++ )
  {
    kaapi_big_hashmap_init( &kaapi_memory_all_local_maps[i].hmap, 0 );
    kaapi_memory_all_local_maps[i].asid = KAAPI_EMPTY_ADDRESS_SPACE_ID;
    kaapi_memory_all_asid2kid[i] = 0;
  }
}

void kaapi_memory_map_destroy( void )
{
  int i;
  
  for( i= 0; i < KAAPI_MAX_ADDRESS_SPACE; i++ )
  {
    kaapi_big_hashmap_destroy( &kaapi_memory_all_local_maps[i].hmap );
  }
}

int kaapi_memory_map_create( kaapi_processor_id_t kid, kaapi_address_space_id_t kasid )
{
  if( kaapi_memory_address_space_gettype(kasid) == KAAPI_MEM_TYPE_CPU )
  {
    kaapi_memory_all_local_maps[kaapi_memory_address_space_getlid(KAAPI_EMPTY_ADDRESS_SPACE_ID)].asid = kasid;
    kaapi_memory_all_kid2asid[kid] = KAAPI_EMPTY_ADDRESS_SPACE_ID;
    kaapi_memory_all_asid2kid[kaapi_memory_address_space_getlid(KAAPI_EMPTY_ADDRESS_SPACE_ID)] = kid;
  }
  if( kaapi_memory_address_space_gettype(kasid) == KAAPI_MEM_TYPE_CUDA )
  {
    kaapi_memory_all_local_maps[kaapi_memory_address_space_getlid(kasid)].asid = kasid;
    kaapi_memory_all_kid2asid[kid] = kasid;
    kaapi_memory_all_asid2kid[kaapi_memory_address_space_getlid(kasid)] = kid;
  }
  
  return 0;
}

kaapi_address_space_id_t kaapi_memory_map_kid2asid(  kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >= 0) && (kid < KAAPI_MAX_PROCESSOR) );
  return kaapi_memory_all_kid2asid[kid];
}

kaapi_processor_id_t kaapi_memory_map_asid2kid( kaapi_address_space_id_t kasid )
{
  return kaapi_memory_all_asid2kid[kaapi_memory_address_space_getlid(kasid)];
}

kaapi_address_space_id_t kaapi_memory_map_get_current_asid( void )
{
  return kaapi_memory_map_kid2asid(kaapi_get_self_kid());
}

kaapi_memory_map_t* kaapi_memory_map_get_current( kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >= 0) && (kid < KAAPI_MAX_PROCESSOR) );
  const kaapi_address_space_id_t kasid = kaapi_memory_map_kid2asid(kid);
  return &kaapi_memory_all_local_maps[kaapi_memory_address_space_getlid(kasid)];
}

kaapi_metadata_info_t* kaapi_memory_map_find_or_insert( kaapi_memory_map_t* kmap, void* ptr )
{
  kaapi_hashentries_t *entry;
  
  entry = kaapi_big_hashmap_findinsert(&kmap->hmap, (void *)ptr);
  if (entry->u.mdi == 0)
    entry->u.mdi = kaapi_metadata_info_alloc();
  
  return entry->u.mdi;
}

kaapi_metadata_info_t* kaapi_memory_map_find( kaapi_memory_map_t* kmap, void* ptr )
{
  kaapi_hashentries_t *entry;
  
  entry = kaapi_big_hashmap_find(&kmap->hmap, (void *)ptr);  
  return entry->u.mdi;
}


