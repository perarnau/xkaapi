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
  kaapi_task_t*              pc;         /* cache */
  kaapi_tasklist_t*          tasklist;
  kaapi_taskdescr_t*         td;
  kaapi_task_body_t          body;
  kaapi_frame_t*             fp;
  unsigned int               proc_type;
  int                        err =0;
  uint32_t                   cnt_exec = 0; /* executed tasks during one call of execframe_tasklist */

  kaapi_assert_debug( thread->sfp >= thread->stackframe );
  kaapi_assert_debug( thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL );
  tasklist = thread->sfp->tasklist;
  kaapi_assert_debug( tasklist != 0 );

  /* here... begin execute frame tasklist*/
  kaapi_event_push0(thread->proc, thread, KAAPI_EVT_FRAME_TL_BEG );

  /* get the processor type to select correct entry point */
  proc_type = thread->proc->proc_type;

  /*
  */
  if (tasklist->td_ready == 0)
  {
    kaapi_thread_tasklist_pushready_init( tasklist, thread );
    kaapi_thread_tasklist_commit_ready( tasklist );
    goto execute_first;
  }
  
  /* jump to previous state if return from suspend 
     (if previous return from EWOULDBLOCK)
  */
  switch (tasklist->context.chkpt) {
    case 1:
      td = tasklist->context.td;
      fp = tasklist->context.fp;
      goto redo_frameexecution;

    case 2:
      /* nothing to restart: the state of the tasklist should ok */
      goto execute_first;

    default:
      break;
  };
  
  /* force previous write before next write */
  //kaapi_writemem_barrier();

  while (!kaapi_tasklist_isempty( tasklist ))
  {
    err = kaapi_thread_tasklist_pop( tasklist );
    if (err ==0)
    {
execute_first:
      td = kaapi_thread_tasklist_getpoped( tasklist );
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
        thread->sfp[1].sp = kaapi_thread_tasklist_getsp(tasklist); 
        thread->sfp[1].pc = thread->sfp[1].sp;
        thread->sfp[1].sp_data = fp->sp_data;
        /* kaapi_writemem_barrier(); */
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
            return EWOULDBLOCK;
          }
          kaapi_assert_debug( err == 0 );
        }
        
        /* pop the frame, even if not used */
        fp = --thread->sfp;
      }

      /* push in the front the activated tasks */
      if (!kaapi_activationlist_isempty(&td->list))
	kaapi_thread_tasklist_pushready( tasklist, td->list.front );

      /* do bcast after child execution (they can produce output data) */
      if (td->bcast !=0) 
        kaapi_thread_tasklist_pushready( tasklist, td->bcast->front );
    }
    
    /* recv incomming synchronisation 
       - process it before the activation list of the executed
       in order to force directly activated task to be executed first.
    */
    if (tasklist->recv !=0)
    {
    }

    /* ok, now push pushed task into the wq */
    if (kaapi_thread_tasklist_commit_ready( tasklist ))
      goto execute_first;
            
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
