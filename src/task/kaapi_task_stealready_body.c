/*
** kaapi_task_steal.c
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
  
  kaapi_assert(arg->origin_td_beg < arg->origin_td_end);

  /* Execute the orinal body function with the original args */
  frame = (kaapi_frame_t*)uthread;
  kaapi_assert_debug( frame == thread->sfp );

  thread->sfp[1] = *frame;
  frame = ++thread->sfp;

  /* create a new tasklist on the stack of the running thread
  */
  tasklist = (kaapi_tasklist_t*)malloc(sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist );
  kaapi_thread_tasklist_init( tasklist, thread );
  tasklist->master    = arg->origin_tasklist;

  /* Fill the task list with ready stolen tasks.
     Because we do not have cactus stack, we recopy
     stolen tasks into the new stacks (only pointer to
     tasks).
     Not that from this task list execution entry,
     we do not store ready tasks into linked list:
     - they are directly pushed into the workqueue.
  */
  kaapi_thread_tasklist_push_stealready_init( tasklist, thread, arg->origin_td_beg, arg->origin_td_end);
  kaapi_thread_tasklist_commit_ready( tasklist );
  
  /* keep the first task to execute outside the workqueue */
  tasklist->context.chkpt = 2;

#if 0//defined(KAAPI_DEBUG)
  kaapi_sched_lock( &thread->proc->lock );
#endif
  frame->tasklist = tasklist;
#if 0//defined(KAAPI_DEBUG)
  kaapi_sched_unlock( &thread->proc->lock );
#endif

  /* start execution */
  kaapi_sched_sync_(thread);
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->count_thief) == 0);

  kaapi_sched_lock(&thread->proc->lock);
  frame->tasklist = 0;
  /* one thief less */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&tasklist->master->count_thief) >0 );
  KAAPI_ATOMIC_DECR( &tasklist->master->count_thief );
  kaapi_sched_unlock( &thread->proc->lock );
  
  kaapi_tasklist_destroy( tasklist );
  free(tasklist);
  --thread->sfp;
}
