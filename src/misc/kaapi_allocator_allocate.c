/*
** kaapi_abort.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
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

#define KAAPI_ALLOCATED_BLOCSIZE 128*4096

static kaapi_allocator_bloc_t* _kaapi_allocator_blocfreelist[64] = 
{
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0
};
static int _kaapi_indexfreelist = 0;

/**/
int kaapi_allocator_destroy( kaapi_allocator_t* va )
{
  while (va->allocatedbloc !=0)
  {
    kaapi_allocator_bloc_t* curr = va->allocatedbloc;
    va->allocatedbloc = curr->next;
    if (_kaapi_indexfreelist == 64) 
    {
#if defined(KAAPI_USE_NUMA)
      numa_free(curr, KAAPI_ALLOCATED_BLOCSIZE );
#else
      free (curr);
#endif
    }
    else {
      _kaapi_allocator_blocfreelist[_kaapi_indexfreelist++] = curr;
    }
  }
  va->allocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/**/
void* _kaapi_allocator_allocate_slowpart( kaapi_allocator_t* va, size_t size )
{
  void* entry;

  /* round size to double size */
  kaapi_assert_debug( KAAPI_ALLOCATED_BLOCSIZE > size );

  if (_kaapi_indexfreelist ==0)
  {
#if defined(KAAPI_USE_NUMA)
    va->currentbloc = (kaapi_allocator_bloc_t*)numa_alloc_local( KAAPI_ALLOCATED_BLOCSIZE );
#else
    va->currentbloc = (kaapi_allocator_bloc_t*)malloc( KAAPI_ALLOCATED_BLOCSIZE );
#endif
  }
  else 
  {
    va->currentbloc = _kaapi_allocator_blocfreelist[--_kaapi_indexfreelist];
  }

  va->currentbloc->next = va->allocatedbloc;
  va->allocatedbloc = va->currentbloc;
  va->currentbloc->pos = 0;
  
  entry = &va->currentbloc->data[va->currentbloc->pos];
  va->currentbloc->pos += size;
  KAAPI_DEBUG_INST( memset( entry, 0, size*sizeof(double) ) );
  return entry;
}
