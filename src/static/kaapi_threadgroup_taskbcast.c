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

/* network service to push ready list */
static void kaapi_threadgroup_signalservice(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
  printf("Super! I'm receive signal tag\n"); fflush(stdout);
  kaapi_pointer_t rsignal;
  kaapi_assert_debug( sizeof(rsignal) == sz_buffer );
  memcpy(&rsignal, buffer, sizeof(rsignal));
  kaapi_tasklist_pushsignal( rsignal );
}

/* */
int kaapi_threadgroup_bcast( kaapi_threadgroup_t thgrp, kaapi_address_space_t asid_src, kaapi_comsend_t* com)
{
  kaapi_comsend_raddr_t* lraddr;
  while (com != 0)
  {
    printf("Bcast tag:%llu\n", com->tag); fflush(stdout);
    lraddr = &com->front;
    while (lraddr !=0)
    {

      /* copy (com->data, com->size) to (lraddr->raddr, lraddr->rsize) */
      /* warning for GPU device -> communication to the device, that will post transfert */
      kaapi_memory_copy( lraddr->asid, lraddr->raddr, &lraddr->rview, asid_src, (kaapi_pointer_t)com->data, &com->view );

      /* signal remote thread in lraddr->asid */
      int tid = kaapi_threadgroup_asid2tid( thgrp, lraddr->asid );
      if (thgrp->tid2gid[tid] == thgrp->localgid)
      {
        kaapi_tasklist_pushsignal( lraddr->rsignal );
      }
      else {
        printf("Remote signal tag:%llu\n", com->tag); fflush(stdout);
        /* remote address space -> communication */
        kaapi_network_am(
            kaapi_memory_address_space_getgid(lraddr->asid),
            kaapi_threadgroup_signalservice, 
            &lraddr->rsignal, sizeof(kaapi_pointer_t) 
        );
      }

      lraddr = lraddr->next;
    }
    com = com->next;
  }
  return 0;
}
