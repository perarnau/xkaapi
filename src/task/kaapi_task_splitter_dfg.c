/*
** kaapi_task_splitter_dfg.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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

/** Return the number of splitted parts (at most 1 if the task may be steal)
*/
int kaapi_task_splitter_dfg( 
  kaapi_thread_context_t* thread, 
  kaapi_task_t* task, 
  unsigned int war_param, 
  kaapi_listrequest_t* lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_request_t*        request    = 0;
  kaapi_tasksteal_arg_t*  argsteal;
  kaapi_reply_t*          stealreply;

#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&thread->proc->lock) != 0 );
#endif
  kaapi_assert_debug( task !=0 );
  kaapi_assert_debug( kaapi_task_state_issteal( kaapi_task_getstate( task) ) );

//printf("Task: %p steal in thread: %p\n", (void*)task, (void*)thread ); fflush(stdout);
  /* find the first request in the list */
  request = kaapi_listrequest_iterator_get( lrequests, lrrange );
  kaapi_assert(request !=0);

  stealreply = kaapi_request_getreply(request);
  
  /* - create the task steal that will execute the stolen task
     The task stealtask stores:
       - the original thread
       - the original task pointer
       - the pointer to shared data with R / RW access data
       - and at the end it reserve enough space to store original task arguments
     The original body is saved as the extra body of the original task data structure.
  */
  argsteal = (kaapi_tasksteal_arg_t*)stealreply->udata;
  argsteal->origin_thread         = thread;
  argsteal->origin_task           = task;
  argsteal->war_param             = war_param;  
  stealreply->u.s_task.body       = kaapi_tasksteal_body;

  _kaapi_request_reply( request, KAAPI_REPLY_S_TASK); /* success of steal */
  
  /* update next request to process */
  kaapi_listrequest_iterator_next( lrequests, lrrange );
  return 1;
}
