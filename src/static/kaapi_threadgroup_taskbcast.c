/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
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

#if defined(KAAPI_USE_NETWORK)
/* network service to push ready list */
static void kaapi_threadgroup_signalservice(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
#if 0
  printf("[kaapi_threadgroup_signalservice] begin receive signal tag\n"); fflush(stdout);
#endif
  kaapi_pointer_t rsignal;
  kaapi_assert_debug( sizeof(rsignal) == sz_buffer );
  memcpy(&rsignal, buffer, sizeof(rsignal));
  kaapi_tasklist_pushsignal( rsignal );
#if 0
  printf("[kaapi_threadgroup_signalservice] end receive signal tag\n"); fflush(stdout);
#endif
}
#endif

/* */
int kaapi_threadgroup_bcast( kaapi_threadgroup_t thgrp, kaapi_address_space_id_t asid_src, kaapi_comsend_t* com)
{
  kaapi_comsend_raddr_t* lraddr;
  while (com != 0)
  {
#if 0
    printf("Bcast data:%p\n", com->data); fflush(stdout);
#endif
    lraddr = &com->front;
    while (lraddr !=0)
    {

      /* copy (com->data, com->size) to (lraddr->raddr, lraddr->rsize) */
      /* warning for GPU device -> communication to the device, that will post transfert */
#if 1
      printf("%i::[bcast] memory copy dest@:%p, src@:%p, size:%lu, asid_dest:",
          kaapi_network_get_current_globalid(),
          (void*)lraddr->raddr, 
          (void*)com->data, 
          kaapi_memory_view_size(&com->view)); 
      kaapi_memory_address_space_fprintf( stdout, lraddr->asid );
      printf(", asid_src:");
      kaapi_memory_address_space_fprintf( stdout, asid_src ); 
      printf("\n");
      fflush(stdout);
#endif
      
      kaapi_memory_copy( lraddr->asid, lraddr->raddr, &lraddr->rview, 
                         asid_src, (kaapi_pointer_t)com->data, &com->view );

      /* signal remote thread in lraddr->asid */
      int tid = kaapi_memory_address_space_getuser( lraddr->asid );
      if (thgrp->tid2gid[tid] == thgrp->localgid)
      {
        kaapi_tasklist_pushsignal( lraddr->rsignal );
      }
      else {
#if defined(KAAPI_USE_NETWORK)
#if 0
        printf("Remote signal raddr:%p\n", (void*)lraddr->raddr); fflush(stdout);
#endif
        /* remote address space -> communication */
        kaapi_network_am(
            kaapi_memory_address_space_getgid(lraddr->asid),
            kaapi_threadgroup_signalservice, 
            &lraddr->rsignal, sizeof(kaapi_pointer_t) 
        );
#else
        kaapi_assert_debug( 0 );
#endif
      }

      lraddr = lraddr->next;
    }
    com = com->next;
  }
  return 0;
}
