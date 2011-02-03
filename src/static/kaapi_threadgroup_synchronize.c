/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threadctxts.
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

#if 0//defined(KAAPI_USE_NETWORK)
typedef struct kaapi_rdma_data_t {
  kaapi_pointer_t     raddr;
  kaapi_memory_view_t view;
} kaapi_rdma_data_t;
static void kaapi_threadgroup_synchronizeservice(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
  kaapi_rdma_data_t rd;
  memcpy(&rd, buffer, sizeof(rd));
  const void* src;
  kaapi_memory_view_t view_src;

  printf("%i::[kaapi_memory_receive request] (@raddr:%p, size:%lu)\n", 
      kaapi_network_get_current_globalid(),
      rd.raddr, kaapi_memory_view_size(&rd.view) 
  );

//  kaapi_network_rdma( source, rd.raddr, &rd.view, (const void*)src, &view_src ); 
}
#endif



/**
*/
int kaapi_threadgroup_synchronize(kaapi_threadgroup_t thgrp )
{
  /* update the list of version in the hashmap to suppress reference to previously executed task */
  for (int i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    kaapi_hashentries_t* entry = _get_hashmap_entry(&thgrp->ws_khm, i);
    while (entry != 0)
    {
      kaapi_version_t* ver = entry->u.version;
      if (ver->writer_mode != KAAPI_ACCESS_MODE_VOID)
      {
        kaapi_address_space_id_t asid = kaapi_threadgroup_tid2asid(thgrp, -1);

printf("%i::Synchronize: copie: src:%p, size:%u -> dest:%p, size:%u, asid_src:",
    kaapi_network_get_current_globalid(),
    (void*)ver->writer.addr, kaapi_memory_view_size(&ver->writer.view),
    (void*)ver->orig.addr, kaapi_memory_view_size(&ver->orig.view) );
kaapi_memory_address_space_fprintf( stdout, ver->writer.asid );
printf("tag:%i \n", (int)ver->tag);

#if 0// defined(KAAPI_USE_NETWORK)
        if (kaapi_memory_address_space_getgid(ver->writer.asid) != kaapi_network_get_current_globalid())
        {
          /* remote address space -> communication */

          kaapi_rdma_data_t rd;
          rd.raddr = (kaapi_pointer_t)dest;
          rd.view  = *view_src;

          kaapi_network_am(
              kaapi_memory_address_space_getgid(ver->writer.asid),
              kaapi_threadgroup_synchronizeservice, 
              &rd, sizeof(rd) 
          );
          return 0;
        }
#endif
        /* */
        kaapi_memory_copy( asid, (kaapi_pointer_t)ver->orig.addr, &ver->orig.view, 
                           ver->writer.asid, (kaapi_pointer_t)ver->writer.addr, &ver->writer.view );
      }
      entry = entry->next;
    }
  }
#if defined(KAAPI_USE_NETWORK)
  kaapi_memory_global_barrier();
#endif
}