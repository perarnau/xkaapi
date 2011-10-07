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

/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
int kaapi_thread_tasklistready_pushactivated( 
    kaapi_readytasklist_t*  rtl, 
    kaapi_activationlink_t* head 
)
{
  kaapi_taskdescr_t** base = rtl->base;
  kaapi_workqueue_index_t local_beg =  rtl->next; /* reuse the position of the previous executed task */

  kaapi_taskdescr_t* td;
  int task_pushed = 0;
  while (head !=0)
  {
    td = head->td;
    if (kaapi_taskdescr_activated(td))
    {
      /* if non local -> push on remote queue ? */
      if ( kaapi_cpuset_empty(&td->u.acl.mapping) )
      {
        /* push on local queue */
        kaapi_assert_debug((char*)&base[local_beg] > (char*)(kaapi_self_thread()->sp_data));
        base[local_beg--] = head->td;
        task_pushed = 1;
      }
      else 
      {
        /* push on a remote queue */
        kaapi_affinity_queue_t* queue = kaapi_sched_affinity_lookup_queue(&td->u.acl.mapping);
        if (queue == 0)
        {
          kaapi_assert_debug((char*)&base[local_beg] > (char*)(kaapi_self_thread()->sp_data));
          base[local_beg--] = head->td;
          task_pushed = 1;
        }
        else {
          /* push the task in the bounded queue */
          KAAPI_ATOMIC_INCR(&td->tl->count_thief);
          kaapi_sched_affinity_thief_pushtask(queue, td );
        }
      }
    }
    head = head->next;
  }
  rtl->next = local_beg+1;   /* position of the last pushed task (next to execute) */
  rtl->task_pushed = task_pushed;
  return task_pushed;
}