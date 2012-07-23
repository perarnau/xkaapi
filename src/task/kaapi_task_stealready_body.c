/*
** kaapi_task_steal.c
** xkaapi
** 
**
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
#include <stdio.h> // debug 
#include <inttypes.h>

/** New version.
    - the task kaapi_taskstealready_body is the first pushed task on a steal operation on a tasklist (i.e. when a taskdescr_t was stolen).
    - the task
*/
void kaapi_taskstealready_body( void* taskarg, kaapi_thread_t* uthread  )
{
  kaapi_thread_context_t*     thread;
  kaapi_taskstealready_arg_t* arg;
  kaapi_frame_t*              frame;
  kaapi_tasklist_t*           tasklist;

  thread = kaapi_self_thread_context();

  /* get information of the task to execute */
  arg = (kaapi_taskstealready_arg_t*)taskarg;
  
#if !defined(TASKLIST_REPLY_ONETD)  
  kaapi_assert_debug(arg->td_beg < arg->td_end);
#else
#endif

  /* create a new tasklist on the stack of the running thread
  */
  tasklist = (kaapi_tasklist_t*)kaapi_thread_pushdata(uthread, sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist, 0 );

  /* Execute the orinal body function with the original args */
  frame = (kaapi_frame_t*)uthread;
  kaapi_assert_debug( frame == thread->stack.sfp );

  thread->stack.sfp[1] = *frame;
  thread->stack.sfp = ++frame;

  /* Fill the task list with ready stolen tasks.
  */
#if defined(TASKLIST_REPLY_ONETD)  
  kaapi_tasklistready_push_init_fromsteal( 
    tasklist, 
    &arg->td, 
    &arg->td+1
  );
  kaapi_processor_incr_workload(kaapi_get_current_processor(), 1);
#else
  kaapi_thread_tasklistready_push_init_fromsteal( 
    tasklist, 
    arg->td_beg, 
    arg->td_end
  );
  kaapi_processor_incr_workload(kaapi_get_current_processor(), arg->td_end-arg->td_beg);
#endif
  
#if defined(TASKLIST_ONEGLOBAL_MASTER) && !defined(TASKLIST_REPLY_ONETD)
  /* to synchronize steal operation and the recopy of TD on the non master tasklist */
  if (arg->victim_tasklist != 0)
    KAAPI_ATOMIC_DECR( &arg->victim_tasklist->pending_stealop );
#endif
  
  kaapi_writemem_barrier();
  frame->tasklist = tasklist;

  /* start execution */
  kaapi_sched_sync_(thread);

  kaapi_sched_lock(&thread->stack.lock);
  frame->tasklist = 0;
  --thread->stack.sfp;
#if 0
  /* one thief less */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->master->count_thief) >0 );
#endif
  
  /* report the number of executed task to the master tasklist */
  /* TODO change it */
  KAAPI_ATOMIC_ADD( &arg->master_frame_tasklist->cnt_exec, KAAPI_ATOMIC_READ(&tasklist->cnt_exec) );

  kaapi_sched_unlock( &thread->stack.lock );

#if !defined(TASKLIST_REPLY_ONETD)
  /* synchronize steal operation and the recopy of TD on the non master tasklist */
  while (KAAPI_ATOMIC_READ( &tasklist->pending_stealop ) !=0)
    kaapi_slowdown_cpu();
#endif
  
  /* do not destroy tasklist: no usefull data */

}
