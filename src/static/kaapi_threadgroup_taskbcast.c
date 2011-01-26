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

/* */
int kaapi_threadgroup_bcast( kaapi_threadgroup_t thgrp, kaapi_comsend_t* com)
{
  kaapi_comsend_raddr_t* lraddr;
  while (com != 0)
  {
    printf("Bcast tag:%llu\n", com->tag);
    lraddr = &com->front;
    while (lraddr !=0)
    {

      /* copy (com->data, com->size) to (lraddr->raddr, lraddr->rsize) */
      /* this code should be in the memory library */
      kaapi_assert_debug (kaapi_memory_view_size(&com->view) == kaapi_memory_view_size(&lraddr->rview));
      switch (com->view.type) 
      {
        case KAAPI_MEM_VIEW_1D:
        {
          memcpy( (void*)lraddr->raddr, (const void*)com->data, com->view.size[0] );
        } break;

        case KAAPI_MEM_VIEW_2D:
        {
          size_t i;
          size_t llda  = com->view.lda;
          size_t rlda  = lraddr->rview.lda;
          const char* laddr = (const char*)com->data;
          char* raddr = (char*)lraddr->raddr;
          
          if (kaapi_memory_view_iscontiguous(&com->view) && kaapi_memory_view_iscontiguous(&lraddr->rview))
          {
              memcpy( raddr, laddr, kaapi_memory_view_size( &com->view) );
          }
          else 
          {
            for (i=0; i<com->view.size[0]; ++i, laddr += llda, raddr += rlda)
              memcpy( raddr, laddr, com->view.size[1] );
          }

          break;
        }
        default:
          kaapi_assert(0);
          break;
      }

      /* memory barrier if required */
      kaapi_writemem_barrier();
      
      /* signal remote thread in lraddr->asid */
      int tid = kaapi_threadgroup_asid2tid( thgrp, lraddr->asid );
      if (thgrp->tid2gid[tid] == thgrp->localgid)
      {
        kaapi_tasklist_ready_pushsignal( thgrp->threadctxts[tid]->readytasklist, lraddr->rsignal );
      }
      else {
        //communication
      }

      lraddr = lraddr->next;
    }
    com = com->next;
  }
  return 0;
}
