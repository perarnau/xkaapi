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
#include "kaapi_tasklist.h"

/** kaapi_threadgroup_execframe
    Use the list of ready task to execute program.
    Once a task has been executed from the ready list, then call execframe
    of the possible created tasks.
    If a task activates new ready task, then beguns by executes these tasks
    prior to other tasks.
*/
int kaapi_thread_execframe_readylist( kaapi_thread_context_t* thread )
{
  kaapi_task_t*              pc;      /* cache */
  kaapi_taskdescr_t*         td;      /* cache */
  kaapi_task_body_t          body;
  kaapi_frame_t*             fp;
  kaapi_tasklist_t*          tasklist;
  kaapi_syncrecv_t*          recv;
  unsigned int               proc_type;
#if defined(KAAPI_USE_PERFCOUNTER)
  uint32_t                   cnt_tasks = 0;
#endif

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);
  tasklist = thread->sfp->tasklist;
  kaapi_assert_debug( tasklist != 0);
  
  /* get the processor type to select correct entry point */
  proc_type = thread->proc->proc_type;
    
  fp = (kaapi_frame_t*)thread->sfp;

  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc      = fp->sp;
  thread->sfp[1].sp      = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);

  while (!kaapi_tasklist_isempty( tasklist ))
  {
    td = kaapi_tasklist_pop( tasklist );

    /* execute td->task */
    if (td !=0)
    {
      pc = td->task;
      if (pc !=0)
      {
        /* get the correct body for the proc type */
        if (td->fmt ==0)
        { /* currently some internal tasks do not have format */
          body = kaapi_task_getuserbody( td->task );
        }
        else 
        {
          body = td->fmt->entrypoint_wh[proc_type];
        }
        kaapi_assert_debug(body != 0);

        /* here... call the task*/
        body( pc->sp, (kaapi_thread_t*)thread->sfp );
      }
#if defined(KAAPI_USE_PERFCOUNTER)
      ++cnt_tasks;
#endif
    }
    
    /* recv incomming synchronisation 
       - process it before the activation list of the executed
       in order to force directly activated task to be executed first.
    */
    recv = kaapi_tasklist_popsignal( tasklist );
    if ( recv != 0 ) /* here may be a while loop */
    {
      kaapi_tasklist_merge_activationlist( tasklist, &recv->list );
      --tasklist->count_recv;
    }
    
    /* bcast? management of the output communication after puting input comm */
    if (td !=0) 
    {
      /* post execution: new tasks created */
      if (unlikely(fp->sp > thread->sfp->sp))
      {
        int err = kaapi_thread_execframe( thread );
        if ((err == EWOULDBLOCK) || (err == EINTR)) return err;
        kaapi_assert_debug( err == 0 );
      }

      /* push in the front the activated tasks */
      if (!kaapi_activationlist_isempty(&td->list))
        kaapi_tasklist_merge_activationlist( tasklist, &td->list );

      /* do bcast after child execution (they can produce output data) */
      if (td->bcast !=0) 
        kaapi_tasklist_merge_activationlist( tasklist, td->bcast );      
    }
    
  } /* while */

  /* pop frame */
  --thread->sfp;

#if 0
  printf("%i::[kaapi_threadgroup_execframe] end tid:%i, #task:%p, #recvlist:%p, #wc_recv:%i\n", 
    thread->the_thgrp->localgid, 
    thread->partid, 
    (void*)tasklist->front,
    (void*)tasklist->recvlist,
    (int)tasklist->count_recv);
#endif
  
  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
  if (kaapi_tasklist_isempty(tasklist))
  {
    kaapi_thread_signalend_exec( thread );

#if 0
    if (thread->partid != -1)
    { 
      /* detach the thread: may it should be put into the execframe function */
      kaapi_sched_lock(&thread->proc->lock);
      thread->proc->thread = 0;
      kaapi_sched_unlock(&thread->proc->lock);
    }
#endif

    return ECHILD;
  }
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
}
