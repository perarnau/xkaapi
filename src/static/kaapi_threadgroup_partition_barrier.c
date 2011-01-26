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

static int kaapi_threadgroup_resolved_for( kaapi_threadgroup_t thgrp, int tid, kaapi_comlink_t* cl )
{
  kaapi_comsend_t* com;
  kaapi_comsend_raddr_t* lraddr;
  while (cl != 0)
  {
    com = cl->u.send;
    lraddr = &com->front;
    while (lraddr !=0)
    {
      int rtid = kaapi_threadgroup_asid2tid( thgrp, lraddr->asid );
      kaapi_comrecv_t* recv = kaapi_recvcomlist_find_tag( thgrp->threadctxts[rtid]->list_recv, com->tag );
      kaapi_assert_debug( recv != 0);
      lraddr->rsignal = (kaapi_pointer_t)recv;
      lraddr->raddr   = (kaapi_pointer_t)recv->data;
      lraddr->rview   = recv->view;
      lraddr = lraddr->next;
    }
    cl = cl->next;
  }
  return 0;
}


static int kaapi_threadgroup_update_recv( kaapi_threadgroup_t thgrp, int tid, kaapi_comlink_t* cl )
{
  while (cl !=0)
  {
    ++thgrp->threadctxts[tid]->readytasklist->count_recv;
    cl = cl->next;
  }
  return 0;
}

int kaapi_threadgroup_barrier_partition( kaapi_threadgroup_t thgrp )
{
  /* this version is only for multicore machine */
  if ((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0)
    kaapi_threadgroup_save(thgrp);

  for (int i=-1; i<thgrp->group_size; ++i)
  {
    kaapi_threadgroup_resolved_for( thgrp, i, thgrp->threadctxts[i]->list_send );
    kaapi_threadgroup_update_recv( thgrp, i, thgrp->threadctxts[i]->list_recv );
  }
  return 0;
}