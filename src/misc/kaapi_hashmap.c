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
** theo.trouillon@imag.fr
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
int kaapi_hashmap_init( kaapi_hashmap_t* khm, kaapi_hashentries_bloc_t* initbloc )
{
  khm->allallocatedbloc = 0;
  khm->currentbloc = initbloc;
  if (initbloc !=0)
    khm->currentbloc->pos = 0;
  return 0;
}

/*
*/
int kaapi_hashmap_clear( kaapi_hashmap_t* khm )
{
  memset( &khm->entries, 0, sizeof(khm->entries) );
  if (khm->currentbloc !=0)
    khm->currentbloc->pos = 0;
  return 0;
}


/*
*/
int kaapi_hashmap_destroy( kaapi_hashmap_t* khm )
{
  while (khm->allallocatedbloc !=0)
  {
    kaapi_hashentries_bloc_t* curr = khm->allallocatedbloc;
    khm->allallocatedbloc = curr->next;
    free (curr);
  }
  return 0;
}


/*
*/
kaapi_hashentries_t* kaapi_hashmap_find( kaapi_hashmap_t* khm, void* ptr )
{
  kaapi_uint32_t hkey = kaapi_hash_value_len( (const char*)&ptr, sizeof( void* ) );
#if defined(KAAPI_DEBUG_LOURD)
fprintf(stdout," [@=%p, hkey=%u]", ptr, hkey);
#endif
  hkey = hkey % KAAPI_HASHMAP_SIZE;
  kaapi_hashentries_t* list_hash = khm->entries[ hkey ];
  kaapi_hashentries_t* entry = list_hash;
  while (entry != 0)
  {
    if (entry->key == ptr) return entry;
    entry = entry->next;
  }
  return NULL;
}

kaapi_hashentries_t* kaapi_hashmap_add( kaapi_hashmap_t* khm, void* ptr )
{
  
  kaapi_uint32_t hkey = kaapi_hash_value_len( ptr, sizeof( void* ) );
  hkey = hkey % KAAPI_HASHMAP_SIZE;
  kaapi_hashentries_t* list_hash = khm->entries[ hkey ];
  kaapi_hashentries_t* entry = list_hash;


  /* allocate new entry */
  if (khm->currentbloc == 0) 
  {
    khm->currentbloc = malloc( sizeof(kaapi_hashentries_bloc_t) );
    khm->currentbloc->next = khm->allallocatedbloc;
    khm->allallocatedbloc = khm->currentbloc;
    khm->currentbloc->pos = 0;
  }
  
  entry = &khm->currentbloc->data[khm->currentbloc->pos];
  entry->key = ptr;
  entry->value.last_version = 0;
  entry->value.last_mode = KAAPI_ACCESS_MODE_VOID;
  if (++khm->currentbloc->pos == KAAPI_BLOCENTRIES_SIZE)
  {
    khm->currentbloc = 0;
  }
  entry->next = list_hash;
  khm->entries[ hkey ] = entry;
  return entry;
}

