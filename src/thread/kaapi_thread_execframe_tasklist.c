/*
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
#include "kaapi_tasklist.h"
#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_host_map.h"

#if defined(KAAPI_NO_CPU)

int kaapi_thread_execframe_tasklist( kaapi_thread_context_t* thread )
{
  kaapi_stack_t* const stack = &thread->stack;
  kaapi_tasklist_t*          tasklist;
  tasklist = stack->sfp->tasklist;
  if (tasklist->master ==0)
  {
    while( KAAPI_ATOMIC_READ(&tasklist->cnt_exec) != tasklist->total_tasks )
      kaapi_slowdown_cpu();
  }
    return 0;
}

#else /* KAAPI_NO_CPU */

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
  kaapi_stack_t* const stack = &thread->stack;
  kaapi_task_t*              pc;         /* cache */
  kaapi_tasklist_t*          tasklist;
  kaapi_taskdescr_t*         td;
  kaapi_task_body_t          body;
  kaapi_frame_t*             fp;
  unsigned int               proc_type;
  int                        err =0;
  uint32_t                   cnt_exec; /* executed tasks during one call of execframe_tasklist */

  kaapi_assert_debug( stack->sfp >= stack->stackframe );
  kaapi_assert_debug( stack->sfp < stack->stackframe+KAAPI_MAX_RECCALL );
  tasklist = stack->sfp->tasklist;

  /* here... begin execute frame tasklist*/
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_BEG );

  /* get the processor type to select correct entry point */
  proc_type = stack->proc->proc_type;
  
  /* */
  cnt_exec = 0;
  
  kaapi_assert_debug( tasklist != 0 );

  /* jump to previous state if return from suspend 
     (if previous return from EWOULDBLOCK)
  */
  switch (tasklist->context.chkpt) {
    case 1:
      td = tasklist->context.td;
      fp = tasklist->context.fp;
      goto redo_frameexecution;

    case 2:
      /* set up the td to start from previously select task */
      td = tasklist->context.td;
      goto execute_first;

    default:
      break;
  };
  
  /* force previous write before next write */
    KAAPI_DEBUG_INST(kaapi_tasklist_t save_tasklist __attribute__((unused)) = *tasklist; )

  while (!kaapi_tasklist_isempty( tasklist ))
  {
    err = kaapi_readylist_pop( &tasklist->rtl, &td );

    if (err ==0)
    {
      kaapi_processor_decr_workload( stack->proc, 1 );
execute_first:
      pc = td->task;
      if (pc !=0)
      {
        /* get the correct body for the proc type */
        if (td->fmt ==0)
        { /* currently some internal tasks do not have format */
          body = kaapi_task_getbody( pc );
        }
        else 
        {
          body = td->fmt->entrypoint_wh[proc_type];          
        }
        kaapi_assert_debug(body != 0);

        /* push the frame for the running task: pc/sp = one before td (which is in the stack)à */
        fp = (kaapi_frame_t*)stack->sfp;
        stack->sfp[1] = *fp;

        stack->sfp = ++fp;
        kaapi_assert_debug((char*)fp->sp > (char*)fp->sp_data);
        kaapi_assert_debug( stack->sfp - stack->stackframe <KAAPI_MAX_RECCALL);
        
#if defined(KAAPI_USE_CUDA) && !defined(KAAPI_CUDA_NO_D2H)
	if ( td->fmt != 0 )
		kaapi_mem_host_map_sync( td->fmt, pc->sp );
#endif

//#if defined(KAAPI_VERBOSE)
#if 0
  if( td->fmt != 0 )
      fprintf(stdout, "[%s] kid=%lu td=%p name=%s (counter=%d,wc=%d)\n", 
	      __FUNCTION__,
		(long unsigned int)kaapi_get_current_kid(),
	      (void*)td, td->fmt->name,
	      KAAPI_ATOMIC_READ(&td->counter),
	      td->wc
	      );
  else
      fprintf(stdout, "[%s] kid=%lu td=%p (counter=%d,wc=%d)\n", 
	      __FUNCTION__,
		(long unsigned int)kaapi_get_current_kid(),
	      (void*)td,
	      KAAPI_ATOMIC_READ(&td->counter),
	      td->wc
	     );
  fflush(stdout);
#endif
        /* start execution of the user body of the task */
        KAAPI_DEBUG_INST(kaapi_assert( td->u.acl.exec_date == 0 ));
        KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_STATIC_TASK_BEG );
        body( pc->sp, (kaapi_thread_t*)stack->sfp );
        KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_STATIC_TASK_END );
        KAAPI_DEBUG_INST( td->u.acl.exec_date = kaapi_get_elapsedns() );
        ++cnt_exec;

        /* new tasks created ? */
        if (unlikely(fp->sp > stack->sfp->sp))
        {
redo_frameexecution:
          err = kaapi_stack_execframe( &thread->stack );
          if (err == EWOULDBLOCK)
          {
            tasklist->context.chkpt     = 1;
            tasklist->context.td        = td;
            tasklist->context.fp        = fp;
            KAAPI_ATOMIC_ADD(&tasklist->cnt_exec, cnt_exec);
            return EWOULDBLOCK;
          }
          kaapi_assert_debug( err == 0 );
        }
        
        /* pop the frame, even if not used */
        stack->sfp = --fp;
      }

      kaapi_tasklist_push( tasklist, td );
    }
    
    KAAPI_DEBUG_INST(save_tasklist = *tasklist;)

  } /* while */

    if( !kaapi_readytasklist_isempty( stack->proc->rtl ) ){
	if( kaapi_readylist_pop( stack->proc->rtl, &td ) == 0 )
	    goto execute_first;
    }

  /* here... end execute frame tasklist*/
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_END );
  
  KAAPI_ATOMIC_ADD(&tasklist->cnt_exec, cnt_exec);

  kaapi_assert( kaapi_tasklist_isempty(tasklist) );

  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
#if defined(TASKLIST_ONEGLOBAL_MASTER)  
  if (tasklist->master ==0)
  {
    /* this is the master thread */
    for (int i=0; (KAAPI_ATOMIC_READ(&tasklist->cnt_exec) != tasklist->total_tasks) && (i<100); ++i)
      kaapi_slowdown_cpu();
      
    int isterm = KAAPI_ATOMIC_READ(&tasklist->cnt_exec) == tasklist->total_tasks;
    if (isterm) return 0;

    tasklist->context.chkpt = 0;
#if defined(KAAPI_DEBUG)
    tasklist->context.td = 0;
    tasklist->context.fp = 0;
#endif 
    return EWOULDBLOCK;
  }
  return 0;

#else // #if defined(TASKLIST_ONEGLOBAL_MASTER)  
  
  int retval;
  tasklist->context.chkpt = 0;
#if defined(KAAPI_DEBUG)
  tasklist->context.td = 0;
  tasklist->context.fp = 0;
#endif 

  /* else: wait a little until count_thief becomes 0 */
  for (int i=0; (KAAPI_ATOMIC_READ(&tasklist->count_thief) != 0) && (i<100); ++i)
    kaapi_slowdown_cpu();

  /* lock thief under stealing before reading counter:
     - there is no work to steal, but need to synchronize with currentl thieves
  */
//  kaapi_sched_lock(&stack->lock);
//  kaapi_sched_unlock(&stack->lock);
  retval = KAAPI_ATOMIC_READ(&tasklist->count_thief);

  if (retval ==0) 
  {
    return 0;
  }
  
  /* they are no more ready task, 
     the tasklist is not completed, 
     then return EWOULDBLOCK 
  */
//printf("EWOULDBLOCK case 2: master:%i\n", tasklist->master ? 0 : 1);
  return EWOULDBLOCK;
#endif // #if !defined(TASKLIST_ONEGLOBAL_MASTER)  

}
#endif
