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
int kaapi_thread_execframe_tasklist( kaapi_thread_context_t* thread )
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
  int                        err =0;
  uint32_t                   cnt_exec = 0; /* executed tasks during one call of execframe_tasklist */
  kaapi_workqueue_index_t    ntasks = 0;

  kaapi_assert_debug( thread->sfp >= thread->stackframe );
  kaapi_assert_debug( thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL );
  tasklist = thread->sfp->tasklist;
  kaapi_assert_debug( tasklist != 0 );

  /* here... begin execute frame tasklist*/
  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_FRAME_TL_BEG );

  /* get the processor type to select correct entry point */
  proc_type = thread->proc->proc_type;
    
  /* td_ready is an array of pointers to task descriptor which
     uses the current thread's execution stack:
     - sp and pc are not used for executing task in the ready list
     - td_top has the same value as sp (first free task)
     - the workqueue refers interval [beg,end), beg <= end <=0. which points
     to all task descriptors td_top[i] for all i in [beg,end)
     - when thread pop a task, it increases beg
     - when thief steals tasks, it decreases the end of the work queue 'end'
  */
  if (tasklist->td_ready == 0)
  {
    fp = (kaapi_frame_t*)thread->sfp;

    tasklist->recv = tasklist->recvlist.front;
    curr = tasklist->readylist.front;
    /* WARNING: the task pushed into the ready list should be in the push in the front 
       if we want to ensure at least creation order 
    */
    tasklist->td_top = td_top = (kaapi_taskdescr_t**)fp->sp;
    while (curr !=0)
    {
      kaapi_assert_debug((char*)td_top > (char*)fp->sp_data);
      *--td_top = curr->td;
      ++ntasks;
      curr = curr->next;
    }
    tasklist->td_ready = tasklist->td_top; /* unused ? */

    /* the initial workqueue is [-ntasks, 0) relative to td_top; 
       td_top[0] serves as a marker of the end
    */
    kaapi_sched_lock( &thread->proc->lock );
    kaapi_workqueue_init(&tasklist->wq_ready, -ntasks, 0);
    kaapi_sched_unlock( &thread->proc->lock );
    /* here thieves can steal wq */
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
      td        = tasklist->context.td;
      fp        = tasklist->context.fp;
      local_beg = tasklist->context.local_beg;
      goto redo_frameexecution;

    case 2:
      err = 0;
      local_beg = tasklist->context.local_beg;
      goto execute_first;

    default:
      break;
  };
  
  /* force previous write before next write */
  //kaapi_writemem_barrier();

  while (!kaapi_tasklist_isempty( tasklist ))
  {
    err = kaapi_workqueue_pop(&tasklist->wq_ready, &local_beg, &local_end, 1);
    if (err ==0)
    {
execute_first:
      task_pushed = 0;
      td = td_top[local_beg];
      kaapi_assert_debug( td != 0);

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

        /* push the frame for the running task: pc/sp = one before td (which is in the stack)à */
        fp = (kaapi_frame_t*)thread->sfp;
        thread->sfp[1].sp = (kaapi_task_t*)(((uintptr_t)td_top+local_beg-sizeof(kaapi_task_t)+1) & ~0xF);
        thread->sfp[1].pc = thread->sfp[1].sp;
        thread->sfp[1].sp_data = fp->sp_data;
        kaapi_writemem_barrier();
        fp = ++thread->sfp;
        kaapi_assert_debug((char*)fp->sp > (char*)fp->sp_data);
        kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);
        
        /* start execution of the user body of the task */
        KAAPI_DEBUG_INST(kaapi_assert( td->exec_date == 0 ));
        kaapi_event_push1(thread->proc, thread, KAAPI_EVT_TASK_BEG, pc );
        body( pc->sp, (kaapi_thread_t*)thread->sfp );
        kaapi_event_push1(thread->proc, thread, KAAPI_EVT_TASK_END, pc );  
        KAAPI_DEBUG_INST( td->exec_date = kaapi_get_elapsedns() );
        ++cnt_exec;

        /* force memory barrier to ensure correct view of output data to
           activated tasks that can be theft
        */
        kaapi_mem_barrier();

        /* new tasks created ? */
        if (unlikely(fp->sp > thread->sfp->sp))
        {
redo_frameexecution:
          err = kaapi_thread_execframe( thread );
          if (err == EWOULDBLOCK)
          {
            tasklist->context.chkpt     = 1;
            tasklist->context.td        = td;
            tasklist->context.fp        = fp;
            tasklist->context.local_beg = local_beg;
            return EWOULDBLOCK;
          }
          kaapi_assert_debug( err == 0 );
        }
        
        /* pop the frame, even if not used */
        fp = --thread->sfp;
      }

      /* push in the front the activated tasks */
      if (!kaapi_activationlist_isempty(&td->list))
      {
        kaapi_activationlink_t* curr_activated = td->list.front;
        kaapi_assert_debug( curr_activated != 0 );
        while (curr_activated !=0)
        {
          if (kaapi_taskdescr_activated(curr_activated->td))
          {
            /* if non local -> push on remote queue ? */
            kaapi_assert_debug((char*)&td_top[local_beg] > (char*)fp->sp_data);
            td_top[local_beg--] = curr_activated->td;
            task_pushed = 1;
          }
          curr_activated = curr_activated->next;
        }
        /* force barrier such that activated tasks that have produce result may be read */
        kaapi_mem_barrier();
      }

      /* do bcast after child execution (they can produce output data) */
      if (td->bcast !=0) 
      {
        kaapi_activationlink_t* curr_activated = td->bcast->front;
        while (curr_activated !=0)
        {
          /* bcast task are always ready */
          kaapi_assert_debug((char*)&td_top[local_beg] > (char*)fp->sp_data);
          td_top[local_beg--] = curr_activated->td;
          curr_activated      = curr_activated->next;
        }
        task_pushed = 1;
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
      /* keep last pushed task */
      ++local_beg;
      
      /* ABA problem here if we suppress lock/unlock? seems to be true */
      kaapi_sched_lock( &thread->proc->lock );
      kaapi_workqueue_push(&tasklist->wq_ready, local_beg+1); /* push last indice == local_beg +1 */
      kaapi_sched_unlock( &thread->proc->lock );
      goto execute_first;
    }
  } /* while */

  /* here... end execute frame tasklist*/
  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_FRAME_TL_END );

  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
  if (kaapi_tasklist_isempty(tasklist))
  {
    int retval;
    tasklist->context.chkpt = 0;
#if defined(KAAPI_DEBUG)
    tasklist->context.td = 0;
    tasklist->context.fp = 0;
#endif 

    /* else: main tasklist, wait a little before return EWOULDBLOCK */
    for (int i=0; (KAAPI_ATOMIC_READ(&tasklist->count_thief) != 0) && (i<1000); ++i)
      kaapi_slowdown_cpu();

    kaapi_sched_lock(&thread->proc->lock);
    retval = KAAPI_ATOMIC_READ(&tasklist->count_thief);
    kaapi_sched_unlock(&thread->proc->lock);

    if (retval ==0)
      return 0;
    
    /* they are no more ready task, 
       the tasklist is not completed, 
       then return EWOULDBLOCK 
    */
    return EWOULDBLOCK;
  }
  
  /* should only occurs with partitioning: incomming recv task ! */
  kaapi_assert(0);
  tasklist->context.chkpt = 2;
  tasklist->context.td = 0;
  tasklist->context.fp = 0;
  return EWOULDBLOCK;
}
