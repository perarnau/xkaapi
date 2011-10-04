/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#ifndef _KAAPI_ALLOCATOR_H_
#define _KAAPI_ALLOCATOR_H_ 1

#include "config.h"
#include "kaapi_defs.h"
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*
*/
#define KAAPI_BLOCALLOCATOR_SIZE 8*4096

/* Macro to define a generic bloc allocator of byte.
*/
typedef struct kaapi_allocator_bloc_t {
  double                           data[KAAPI_BLOCALLOCATOR_SIZE/sizeof(double)
                                        - sizeof(uintptr_t) - sizeof(struct kaapi_allocator_bloc_t*)];
  uintptr_t                        pos;  /* next free in data */
  struct kaapi_allocator_bloc_t*   next; /* link list of bloc */
} kaapi_allocator_bloc_t;

typedef struct kaapi_allocator_t {
  kaapi_allocator_bloc_t* currentbloc;
  kaapi_allocator_bloc_t* allocatedbloc;
} kaapi_allocator_t;

#define KAAPI_DECLARE_GENBLOCENTRIES(ALLOCATORNAME) \
  typedef kaapi_allocator_t ALLOCATORNAME

/**/
static inline int kaapi_allocator_init( kaapi_allocator_t* va ) 
{
  va->allocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/**/
static inline int kaapi_allocator_destroy( kaapi_allocator_t* va )
{
  while (va->allocatedbloc !=0)
  {
    kaapi_allocator_bloc_t* curr = va->allocatedbloc;
    va->allocatedbloc = curr->next;
    free (curr);
  }
  va->allocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/* Here size is size in Bytes
*/
extern void* _kaapi_allocator_allocate_slowpart( kaapi_allocator_t* va, size_t size );


/**/
static inline void* kaapi_allocator_allocate( kaapi_allocator_t* va, size_t size )
{
  void* retval;
  /* round size to double size */
  size = (size+sizeof(double)-1)/sizeof(double);
  const size_t sz_max = KAAPI_BLOCALLOCATOR_SIZE/sizeof(double)-sizeof(uintptr_t)-sizeof(kaapi_allocator_bloc_t*);
  if ((va->currentbloc != 0) && (va->currentbloc->pos + size < sz_max))
  {
    retval = &va->currentbloc->data[va->currentbloc->pos];
    va->currentbloc->pos += size;
    KAAPI_DEBUG_INST( memset( retval, 0, size*sizeof(double) ) );
    return retval;
  }
  return _kaapi_allocator_allocate_slowpart(va, size);
}


#if defined(__cplusplus)
}
#endif

#endif
