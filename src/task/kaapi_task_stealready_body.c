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
#include <stdio.h>


/**
*/
void kaapi_taskstealready_body( void* taskarg, kaapi_thread_t* uthread  )
{
  kaapi_thread_context_t*     thread;
  kaapi_taskstealready_arg_t* arg;
  kaapi_frame_t*              frame;
  kaapi_tasklist_t*           tasklist;
  int err;

  thread = kaapi_self_thread_context();

  /* get information of the task to execute */
  arg = (kaapi_taskstealready_arg_t*)taskarg;
  
  /* Execute the orinal body function with the original args */
  frame = (kaapi_frame_t*)uthread;
  kaapi_assert_debug( frame == thread->sfp );

  /* create a new tasklist: should be very fast allocation,
     its a root tasklist for this thread. 
     May allocated it at thread creation time
  */
  tasklist = (kaapi_tasklist_t*)malloc(sizeof(kaapi_tasklist_t));
  kaapi_tasklist_init( tasklist );
  kaapi_tasklist_pushback_ready( tasklist, arg->origin_td );
  
  /* reserve the tasklist for 32 tasks (at most) */
  tasklist->cnt_tasks = 32;

  kaapi_writemem_barrier();
  frame->tasklist = tasklist;

  /* exec the spawned subtasks */
  err = kaapi_thread_execframe_tasklist( thread );
  kaapi_assert( (err == 0) || (err == ECHILD) );

  KAAPI_ATOMIC_ADD( &arg->origin_tasklist->count_exec, 
      tasklist->cnt_exectasks );

#if 0
  printf("%i::[subtasklist] exec tasks: %llu\n", 
      kaapi_get_self_kid(), tasklist->cnt_exectasks 
  );
  fflush(stdout);
#endif

  kaapi_sched_lock(&thread->proc->lock);
  frame->tasklist = 0;
  kaapi_sched_unlock(&thread->proc->lock);
  
  kaapi_tasklist_destroy( tasklist );
  free(tasklist);
}
