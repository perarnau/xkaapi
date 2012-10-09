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
          kaapi_mem_host_map_sync( td );
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
          err = kaapi_stack_execframe( &thread->stack );
          if (err == EWOULDBLOCK)
          {
#if defined(KAAPI_USE_PERFCOUNTER)
            KAAPI_PERF_REG(stack->proc, KAAPI_PERF_ID_TASKS) += cnt_exec;
#endif
            KAAPI_ATOMIC_ADD(&tasklist->cnt_exec, cnt_exec);
            return EWOULDBLOCK;
          }
          kaapi_assert_debug( err == 0 );
        }
        
        /* pop the frame, even if not used */
        stack->sfp = --fp;
      }
      
      kaapi_readytasklist_pushactivated( &tasklist->rtl, td );
    }
    
    KAAPI_DEBUG_INST(save_tasklist = *tasklist;)
    
  } /* while */

  if( !kaapi_readytasklist_isempty(stack->proc->rtl) ){
    if( kaapi_readylist_pop(stack->proc->rtl, &td) == 0 )
	    goto execute_first;
  }
  
  if( !kaapi_readytasklist_isempty(stack->proc->rtl_remote) ){
    if( kaapi_readylist_pop(stack->proc->rtl_remote, &td) == 0 )
	    goto execute_first;
  }
  
  KAAPI_ATOMIC_ADD(&tasklist->cnt_exec, cnt_exec);
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(stack->proc, KAAPI_PERF_ID_TASKS) += cnt_exec;
#endif

  /* here... end execute frame tasklist*/
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_END );
  
  kaapi_assert( kaapi_tasklist_isempty(tasklist) );
  
  if ((tasklist->master ==0) && (KAAPI_ATOMIC_READ(&tasklist->cnt_exec) != tasklist->total_tasks))
    return EWOULDBLOCK;
  
  return 0;
  
}
#endif
