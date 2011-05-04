/*
** kaapi_hashmap.c
** xkaapi
** 
** 
** Copyright 2010 INRIA.
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
#include "kaapi_impl.h"

/*
*/
kaapi_hashentries_t* kaapi_big_hashmap_findinsert( kaapi_big_hashmap_t* khm, void* ptr )
{
  uint32_t hkey = kaapi_hash_ulong((unsigned long)ptr);

  hkey = hkey % KAAPI_HASHMAP_BIG_SIZE;
  kaapi_hashentries_t* list_hash = khm->entries[hkey];
  kaapi_hashentries_t* entry = list_hash;
  while (entry != 0)
  {
    if (entry->key == ptr) return entry;
    entry = entry->next;
  }
  
  /* allocate new entry */
  if (khm->currentbloc == 0) 
  {
#if defined(KAAPI_USE_NUMA)
    khm->currentbloc = numa_alloc_local( sizeof(kaapi_hashentries_bloc_t) );
#else
    khm->currentbloc = malloc( sizeof(kaapi_hashentries_bloc_t) );
#endif
    khm->currentbloc->next = khm->allallocatedbloc;
    khm->allallocatedbloc = khm->currentbloc;
    khm->currentbloc->pos = 0;
  }
  
  entry = &khm->currentbloc->data[khm->currentbloc->pos];
  entry->key = ptr;
  memset(&entry->u.value, 0, sizeof(entry->u.value) );

  if (++khm->currentbloc->pos == KAAPI_BLOCENTRIES_SIZE)
  {
    khm->currentbloc = 0;
  }
  entry->next = list_hash;
  khm->entries[hkey] = entry;
  return entry;
}
