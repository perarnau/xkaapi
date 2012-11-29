/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
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

#ifndef _KAAPI_MEMORY_MAP_H_
#define _KAAPI_MEMORY_MAP_H_

#include "kaapi_impl.h"

typedef struct kaapi_memory_map_t {
  kaapi_address_space_id_t    asid;
  kaapi_big_hashmap_t         hmap;
} kaapi_memory_map_t;

void kaapi_memory_map_init( void );

void kaapi_memory_map_destroy( void );

int kaapi_memory_map_create( kaapi_processor_id_t kid, kaapi_address_space_id_t kasid );

kaapi_address_space_id_t kaapi_memory_map_kid2asid(  kaapi_processor_id_t kid );

kaapi_processor_id_t kaapi_memory_map_asid2kid( kaapi_address_space_id_t kasid );

kaapi_processor_id_t kaapi_memory_map_lid2kid( uint16_t lid );

kaapi_address_space_id_t kaapi_memory_map_get_current_asid( void );

kaapi_memory_map_t* kaapi_memory_map_get_current( kaapi_processor_id_t kid );

/** Find or create the metadata info for ptr in kmap. */
kaapi_metadata_info_t* kaapi_memory_map_find_or_insert( kaapi_memory_map_t* kmap, void* ptr );

/** Insert into kmap the kmdi using as key ptr. */
kaapi_metadata_info_t* kaapi_memory_map_find_and_insert( kaapi_memory_map_t* kmap, void* ptr,
                                                        kaapi_metadata_info_t* kmdi);

/** Find into kmap metadata info associated with ptr. */
kaapi_metadata_info_t* kaapi_memory_map_find( kaapi_memory_map_t* kmap, void* ptr );

static inline kaapi_address_space_id_t kaapi_memory_map_get_asid( const kaapi_memory_map_t* kmap )
{
  return kmap->asid;
}

#endif /* _KAAPI_MEMORY_MAP_H_ */