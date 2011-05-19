/*
** kaapi_sched_stealprocessor.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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


extern unsigned long kaapi_numa_get_kid_binding(unsigned int);

/** Most important assumption here:
    kaapi_sched_lock was locked.
*/
int kaapi_sched_stealprocessor(
  kaapi_processor_t* kproc, 
  kaapi_listrequest_t* lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_request_t* request;
  kaapi_reply_t* reply;
  kaapi_tasksteal_arg_t* stealarg;
  kaapi_wsqueuectxt_cell_t* cell;
  kaapi_thread_context_t* thread;
  kaapi_task_t* task;
  unsigned int war;
  unsigned int thiefid;
  unsigned int numaid;

  /* test should be done before calling the function */
  kaapi_assert_debug( !kaapi_listrequest_iterator_empty(lrrange) );
  
  /* first request */
  request = kaapi_listrequest_iterator_get( lrequests, lrrange );

  /* steal victim local queue first */
 steal_bound_task:
  if (request == 0) return 0;

  thiefid = kaapi_request_getthiefid(request);
  numaid = (unsigned int)kaapi_numa_get_kid_binding(kproc->kid);
  reply = kaapi_request_getreply(request);
  stealarg = (void*)&reply->udata;

  if (kaapi_pop_bound_task_numaid(numaid, &thread, &task, &war) == -1)
    goto steal_readylist;

  kaapi_task_splitter_dfg_single(thread, task, war, request);

  request = kaapi_listrequest_iterator_next( lrequests, lrrange );

  goto steal_bound_task;

 steal_readylist:
  if ((request !=0) && !kaapi_sched_readyempty(kproc))
  {
    kaapi_thread_context_t* const thread =
      kaapi_sched_stealready( kproc, thiefid);

    if (thread != 0)
    {
      /* reply */
      reply->u.s_thread = thread;
      _kaapi_request_reply(request, KAAPI_REPLY_S_THREAD);
      request = kaapi_listrequest_iterator_next( lrequests, lrrange );
      goto steal_bound_task;
    }
  }

  cell = kproc->lsuspend.tail;
  if ( !kaapi_listrequest_iterator_empty(lrrange) && (cell !=0))
  {
    kaapi_thread_context_t* thread = cell->thread;
    if (thread != 0)
    {
      kaapi_sched_stealstack( thread, 0, lrequests, lrrange );

#if 0 /* not working */
      /* some get stolen */
      if (request != kaapi_listrequest_iterator_get( lrequests, lrrange ))
	goto steal_bound_task;
#endif
    }
  }

  /* steal current thread */
  if ( (kproc->thread !=0) && (kproc->issteal ==0))
  {
    kaapi_thread_context_t* const thread = kproc->thread;

    /* signal that count thefts are waiting */
    kaapi_sched_stealstack( thread, 0, lrequests, lrrange );

#if 0 /* not working */
    /* some get stolen */
    if (request != kaapi_listrequest_iterator_get( lrequests, lrrange ))
      goto steal_bound_task;
#endif
  }

  return 0;
}
