/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
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

/**
*/
int kaapi_versionallocator_init( kaapi_version_allocator_t* va )
{
  va->allallocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/**
*/
int kaapi_versionallocator_destroy( kaapi_version_allocator_t* va )
{
  while (va->allallocatedbloc !=0)
  {
    kaapi_version_bloc_t* curr = va->allallocatedbloc;
    va->allallocatedbloc = curr->next;
    free (curr);
  }
  va->allallocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}


/**
*/
kaapi_version_t* kaapi_versionallocator_allocate( kaapi_version_allocator_t* va )
{
  /* allocate new entry */
  if (va->currentbloc == 0) 
  {
    va->currentbloc = malloc( sizeof(kaapi_version_bloc_t) );
    va->currentbloc->next = va->allallocatedbloc;
    va->allallocatedbloc = va->currentbloc;
    va->currentbloc->pos = 0;
  }
  
  kaapi_version_t* entry = &va->currentbloc->data[va->currentbloc->pos];
  if (++va->currentbloc->pos == KAAPI_BLOCENTRIES_SIZE)
  {
    va->currentbloc = 0;
  }
#if defined(kAAPI_DEBUG)
  memset( entry, 0, sizeof(entry) );
#endif
  return entry;
}


/**
*/
int kaapi_data_version_allocator_init( kaapi_data_version_allocator_t* va )
{
  va->allallocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}


/**
*/
int kaapi_data_version_allocator_destroy( kaapi_data_version_allocator_t* va )
{
  while (va->allallocatedbloc !=0)
  {
    kaapi_data_version_bloc_t* curr = va->allallocatedbloc;
    va->allallocatedbloc = curr->next;
    free (curr);
  }
  va->allallocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}


/**
*/
kaapi_data_version_t* kaapi_data_version_allocate(kaapi_threadgroup_t thgrp)
{
  kaapi_data_version_allocator_t* va = &thgrp->data_version_allocator;
  /* allocate new entry */
  if (va->currentbloc == 0) 
  {
    va->currentbloc = malloc( sizeof(kaapi_data_version_bloc_t) );
    va->currentbloc->next = va->allallocatedbloc;
    va->allallocatedbloc = va->currentbloc;
    va->currentbloc->pos = 0;
  }
  
  kaapi_data_version_t* entry = &va->currentbloc->data[va->currentbloc->pos];
  if (++va->currentbloc->pos == KAAPI_BLOCENTRIES_SIZE)
  {
    va->currentbloc = 0;
  }
  memset(entry, 0, sizeof(*entry) );
  return entry;
}

