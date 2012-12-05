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

#include <stdio.h>

#include "kaapi_impl.h"

static int
kaapi_memory_host_synchronize( void )
{
#if defined(KAAPI_USE_CUDA)
  kaapi_cuda_proc_sync_all();
#endif /* KAAPI_USE_CUDA */
  return 0;
}

int kaapi_memory_synchronize( void )
{
  kaapi_memory_host_synchronize();
  return 0;
}

int kaapi_memory_synchronize_pointer( void *ptr )
{
  /* TODO */
  kaapi_assert(0);  
  return 0;
}

kaapi_pointer_t kaapi_memory_synchronize_metadata( kaapi_metadata_info_t* kdmi )
{
  return kaapi_make_nullpointer();
}

void kaapi_memory_address_space_synchronize_peer2peer(kaapi_address_space_id_t dest, kaapi_address_space_id_t src)
{
  kaapi_memory_map_t* kmap = kaapi_memory_map_get_current(kaapi_get_self_kid());
  static const uint32_t map_size = KAAPI_HASHMAP_BIG_SIZE;
  kaapi_big_hashmap_t *const hmap = &kmap->hmap;
  kaapi_hashentries_t *entry;
  uint32_t i;
  
  for (i = 0; i < map_size; ++i) {
    for (entry = hmap->entries[i]; entry != NULL; entry = entry->next) {
      kaapi_metadata_info_t* kmdi = entry->u.mdi;
      if (kmdi == NULL)
        continue;
      
      if( kaapi_metadata_info_has_data(kmdi, src) &&
          kaapi_metadata_info_is_valid(kmdi, src) &&
          !kaapi_metadata_info_is_valid(kmdi, dest)
          )
      {
        if(kaapi_metadata_info_clear_dirty_and_check(kmdi, dest)){
          kaapi_data_t* ksrc = kaapi_metadata_info_get_data(kmdi, src);
          kaapi_data_t* kdest = kaapi_metadata_info_get_data(kmdi, dest);
          kaapi_memory_copy(kdest->ptr, &kdest->view, ksrc->ptr, &ksrc->view);
        }
      }
      
    }
  }
}

void* kaapi_memory_get_host_pointer(void* const gpu_ptr)
{
  kaapi_metadata_info_t* kmdi = kaapi_memory_find_metadata(gpu_ptr);
  const kaapi_address_space_id_t host_kasid = KAAPI_EMPTY_ADDRESS_SPACE_ID;

  if(kaapi_metadata_info_has_data(kmdi, host_kasid))
  {
    kaapi_data_t* kdata = kaapi_metadata_info_get_data(kmdi, host_kasid);
    kaapi_assert_debug(kdata != 0);
    return kaapi_pointer2void(kdata->ptr);
  }
  
  return NULL;
}

void* kaapi_memory_get_host_pointer_and_validate(void* const gpu_ptr)
{
  kaapi_metadata_info_t* kmdi = kaapi_memory_find_metadata(gpu_ptr);
  const kaapi_address_space_id_t host_kasid = KAAPI_EMPTY_ADDRESS_SPACE_ID;
  
  if(kaapi_metadata_info_has_data(kmdi, host_kasid))
  {
    kaapi_data_t* kdata = kaapi_metadata_info_get_data(kmdi, host_kasid);
    kaapi_assert_debug(kdata != 0);
    kaapi_metadata_info_clear_dirty(kmdi, host_kasid);
    return kaapi_pointer2void(kdata->ptr);
  }
  
  return NULL;
}

void kaapi_memory_evict_pointer(uintptr_t ptr, size_t size)
{
  kaapi_metadata_info_t* kmdi = kaapi_memory_find_metadata((void*)ptr);
  const kaapi_address_space_id_t kasid = kaapi_memory_map_get_current_asid();
  const kaapi_address_space_id_t host_kasid = KAAPI_EMPTY_ADDRESS_SPACE_ID;
  
  /* no address nor valid here ? */
  if((!kaapi_metadata_info_has_data(kmdi, kasid)) ||
     (!kaapi_metadata_info_is_valid(kmdi, kasid))  )
    goto evict;
  
  /* already valid on host */
  if((!kaapi_metadata_info_has_data(kmdi, host_kasid)) ||
     kaapi_metadata_info_is_valid(kmdi, host_kasid)  )
    goto evict;

  /* transfer to the host memory */
  kaapi_data_t* ksrc = kaapi_metadata_info_get_data(kmdi, kasid);
  kaapi_data_t* kdest = kaapi_metadata_info_get_data(kmdi, host_kasid);
  /* Sync: mandatory by device side */
  kaapi_memory_copy(kdest->ptr, &kdest->view, ksrc->ptr, &ksrc->view);
  kaapi_metadata_info_clear_dirty(kmdi, host_kasid);
  
evict: /* clear pointer reference to this address space */
  kaapi_metadata_info_clear_data(kmdi, kasid);
}