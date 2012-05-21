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

/**
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
  kaapi_tasklist_init( tasklist, thread );

  /* Execute the orinal body function with the original args */
  frame = (kaapi_frame_t*)uthread;
  kaapi_assert_debug( frame == thread->stack.sfp );

  thread->stack.sfp[1] = *frame;
  thread->stack.sfp = ++frame;

  /* link tasklist with its master for terminaison on count_thief */
  tasklist->master     = arg->master_tasklist;
  tasklist->t_infinity = arg->master_tasklist->t_infinity;


  /* Fill the task list with ready stolen tasks.
  */
#if defined(TASKLIST_REPLY_ONETD)  
  kaapi_thread_tasklistready_push_init_fromsteal( 
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

  /* keep the first task to execute outside the workqueue */
  tasklist->context.chkpt = 2;
#if defined(KAAPI_USE_CUDA)
    if( kaapi_processor_get_type(kaapi_get_current_processor()) == KAAPI_PROC_TYPE_CUDA )
	tasklist->context.td    =  kaapi_thread_tasklist_commit_ready_and_steal_gpu( tasklist );
    else
#else
  tasklist->context.td    =  kaapi_thread_tasklist_commit_ready_and_steal( tasklist );
#endif
  
  kaapi_writemem_barrier();
  frame->tasklist = tasklist;

  /* start execution */
  kaapi_sched_sync_(thread);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->count_thief) == 0);

  kaapi_sched_lock(&thread->stack.lock);
  frame->tasklist = 0;
  --thread->stack.sfp;
#if 0
  /* one thief less */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->master->count_thief) >0 );
#endif
  
  /* report the number of executed task to the master tasklist */
  KAAPI_ATOMIC_ADD( &tasklist->master->cnt_exec, KAAPI_ATOMIC_READ(&tasklist->cnt_exec) );

#if !defined(TASKLIST_ONEGLOBAL_MASTER) 
  /* decrement the number of reader on the tasklist */
  KAAPI_ATOMIC_DECR( &tasklist->master->count_thief );  
#endif

  kaapi_sched_unlock( &thread->stack.lock );

#if defined(TASKLIST_ONEGLOBAL_MASTER) && !defined(TASKLIST_REPLY_ONETD)
  /* synchronize steal operation and the recopy of TD on the non master tasklist */
  while (KAAPI_ATOMIC_READ( &tasklist->pending_stealop ) !=0)
    kaapi_slowdown_cpu();
#endif
  
  kaapi_tasklist_destroy( tasklist );

}
