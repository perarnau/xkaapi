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

/** Use the readylist and list of task to execute tasks.
    If the list doesnot have ready tasks but all the tasks
    are note executed, then return EWOULDBLOCK.
    At the begining of the algorithm, if the td_ready container is
    not allocated, the function allocated it. It is a stack
    ready tasks : each time a task becomes ready, it is pushed on the
    top of the task. 
    The stack can be steal : the workqueue ensure consistency between
    push/pop and steal operations.
    
    If the execution of a task creates new tasks, then the function execute
    then using the standard DFG execframe function.
*/
int kaapi_cuda_thread_execframe_tasklist( kaapi_thread_context_t* thread )
{
  kaapi_task_t*              pc;      /* cache */
  kaapi_workqueue_index_t    local_beg, local_end;
  kaapi_tasklist_t*          tasklist;
  kaapi_taskdescr_t**        td_top;  /*cache of tasklist->td_top */
  kaapi_taskdescr_t*         td;
  kaapi_task_body_t          body;
  kaapi_frame_t*             fp;
  kaapi_activationlink_t*    curr;
  unsigned int               proc_type;
  int                        task_pushed = 0;
  int                        err;
  uint32_t                   cnt_exec = 0;

  printf("%s\n", __FUNCTION__);

  kaapi_assert_debug( thread->sfp >= thread->stackframe );
  kaapi_assert_debug( thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL );
  tasklist = thread->sfp->tasklist;
  kaapi_assert_debug( tasklist != 0 );
  
  /* get the processor type to select correct entry point */
  proc_type = thread->proc->proc_type;
    
  /* alloc to much tasks descr ? -> */
  if (tasklist->td_ready == 0)
  {
    tasklist->td_ready = 
      (kaapi_taskdescr_t**)malloc( 
            (size_t) (sizeof(kaapi_taskdescr_t*) * tasklist->cnt_tasks) 
    );
    kaapi_workqueue_index_t ntasks = 0;

    tasklist->recv = tasklist->recvlist.front;

    curr = tasklist->readylist.front;
    /* WARNING: the task pushed into the ready list should be in the push in the front 
       if we want to ensure at least creation order 
    */
    td_top = tasklist->td_ready + tasklist->cnt_tasks;
    while (curr !=0)
    {
      *--td_top = curr->td;
      ++ntasks;
      curr = curr->next;
    }
    /* the initial workqueue is [-ntasks, 0) at the begining of td_top; */
    kaapi_writemem_barrier();
    tasklist->td_top = tasklist->td_ready + tasklist->cnt_tasks;
    kaapi_writemem_barrier();
    kaapi_workqueue_init(&tasklist->wq_ready, -ntasks, 0);
  }

  /* here we assume that execframe was already called 
     - only reset td_top
  */
  td_top = tasklist->td_top;
  
  /* jump to previous state if return from suspend 
     (if previous return from EWOULDBLOCK)
  */
  switch (tasklist->context.chkpt) {
    case 1:
      td = tasklist->context.td;
      fp = tasklist->context.fp;
      goto redo_frameexecution;
    default:
      break;
  };

  /* push the frame for the next task to execute */
  fp = (kaapi_frame_t*)thread->sfp;
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
    err = kaapi_workqueue_pop(&tasklist->wq_ready, &local_beg, &local_end, 1);
    if (err ==0)
    {
      task_pushed = 0;
      td = td_top[local_beg];
      kaapi_assert_debug( td != 0);
      KAAPI_DEBUG_INST( td_top[local_beg] = 0 );

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

	if ((uintptr_t)body == (uintptr_t)kaapi_taskmove_body)
	{
	  /* todo: push in cuda stream */
	  printf("taskmove: %lx\n", (uintptr_t)td);
	}
	else
	{
	  printf("taskop: %lx\n", (uintptr_t)td);
	}

        /* here... call the task*/
        body( pc->sp, (kaapi_thread_t*)thread->sfp );
        ++cnt_exec;
        
        /* new tasks created ? */
        if (unlikely(fp->sp > thread->sfp->sp))
        {
redo_frameexecution:
          err = kaapi_thread_execframe( thread );
          if ((err == EWOULDBLOCK) || (err == EINTR)) 
          {
            tasklist->context.chkpt  = 1;
            tasklist->context.td     = td;
            tasklist->context.fp     = fp;
            tasklist->cnt_exectasks += cnt_exec;
            return err;
          }
          kaapi_assert_debug( err == 0 );
        }
      }

      /* push in the front the activated tasks */
      if (!kaapi_activationlist_isempty(&td->list))
      {
        kaapi_activationlink_t* curr_activated = td->list.front;
        while (curr_activated !=0)
        {
          if (kaapi_taskdescr_activated(curr_activated->td))
	  {
	    printf("activated(%lx)\n", (uintptr_t)curr_activated->td);
            /* if non local -> push on remote queue ? */
            td_top[local_beg--] = curr_activated->td;
	  }
	  else
	  {
	    printf("notYetActivated(%lx)\n", (uintptr_t)curr_activated->td);
	  }
          curr_activated = curr_activated->next;
        }
        task_pushed |= 1;
      }

      /* do bcast after child execution (they can produce output data) */
      if (td->bcast !=0) 
      {
        kaapi_activationlink_t* curr_activated = td->bcast->front;
        while (curr_activated !=0)
        {
          /* bcast task are always ready */
          td_top[local_beg--] = curr_activated->td;
          curr_activated      = curr_activated->next;
        }
        task_pushed |= 1;
      }      
    }
    
    /* recv incomming synchronisation 
       - process it before the activation list of the executed
       in order to force directly activated task to be executed first.
    */
    if (tasklist->recv !=0)
    {
      td_top[local_beg--] = tasklist->recv->td;
      tasklist->recv      = tasklist->recv->next;
      task_pushed        |= 1;
    }

    /* ok, now push pushed task into the wq */
    if (task_pushed)
    {
      kaapi_workqueue_push(&tasklist->wq_ready, 1+local_beg);
      task_pushed = 0;
    }
  } /* while */

  /* pop frame */
  --thread->sfp;

  /* update executed tasks */
  tasklist->cnt_exectasks += cnt_exec;
  
  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
  if (kaapi_tasklist_isempty(tasklist))
  {
    tasklist->context.chkpt = 0;
#if defined(KAAPI_DEBUG)
    tasklist->context.td = 0;
    tasklist->context.fp = 0;
#endif    
    return ECHILD;
  }
  tasklist->context.chkpt = 2;
  tasklist->context.td = 0;
  tasklist->context.fp = 0;
  return EWOULDBLOCK;
}
