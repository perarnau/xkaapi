/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** Joao.Lima@imag.fr
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

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_tasklist.h"

#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_data.h"
#include "kaapi_cuda_cublas.h"

#include "kaapi_cuda_event.h"

#include "kaapi_cuda_stream.h"

/** \ingroup TASK
    Register a task format 
*/
static inline kaapi_task_body_t
kaapi_format_get_task_body_by_arch(
	const kaapi_format_t*	const	fmt,
	unsigned int		arch
	)
{
    return fmt->entrypoint_wh[arch];
}

/* cuda task body */
typedef void (*kaapi_cuda_task_body_t)(
	void*,
	cudaStream_t
    );

static inline void
kaapi_cuda_thread_tasklist_activate_deps(
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
	)
{
    /* push in the front the activated tasks */
    if( !kaapi_activationlist_isempty(&td->u.acl.list) )
	kaapi_thread_tasklistready_pushactivated( tasklist,
		td->u.acl.list.front );

    /* do bcast after child execution (they can produce output data) */
    if( td->u.acl.bcast !=0 ) 
	kaapi_thread_tasklistready_pushactivated( tasklist,
		td->u.acl.bcast->front );

    kaapi_thread_tasklist_commit_ready( tasklist );
    kaapi_thread_tasklist_commit_ready( tasklist->master );
}

#if defined(KAAPI_CUDA_DATA_CACHE_WT)
static int
kaapi_cuda_gpu_task_callback3_sync_host(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
    kaapi_cuda_data_output_dev_dec_use( kstream, tasklist, td );
    kaapi_cuda_thread_tasklist_activate_deps( tasklist, td );  
    return 0;
}
#endif

/* call back to push ready task into the tasklist after terminaison of a task */
static int
kaapi_cuda_gpu_task_callback2_after_kernel(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] END kid=%lu td=%p name=%s (counter=%d,wc=%d)\n", 
	  __FUNCTION__,
	    (long unsigned int)kaapi_get_current_kid(),
	  (void*)td, td->fmt->name,
	  KAAPI_ATOMIC_READ(&td->counter),
	  td->wc
	  );
  fflush(stdout);
#endif
#if !defined(KAAPI_CUDA_NO_H2D)
#if defined(KAAPI_CUDA_DATA_CACHE_WT)
    /* write-through policy */
    kaapi_cuda_data_async_recv( kstream, tasklist, td );
    kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_D2H, 
	   kaapi_cuda_gpu_task_callback3_sync_host, tasklist, td );
#else /* KAAPI_CUDA_DATA_CACHE_WT */
    /* default write-back policy (lazy) */
    kaapi_cuda_data_output_dev_dec_use( kstream, tasklist, td );
    kaapi_cuda_thread_tasklist_activate_deps( tasklist, td );  
#endif /* KAAPI_CUDA_DATA_CACHE_WT */
#else /* !KAAPI_CUDA_NO_H2D */
    kaapi_cuda_thread_tasklist_activate_deps( tasklist, td );  
#endif /* !KAAPI_CUDA_NO_H2D */
    return 0;
}

static int
kaapi_cuda_gpu_task_callback1_exec_task(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
    kaapi_task_t*              pc;         /* cache */
    kaapi_cuda_task_body_t body = (kaapi_cuda_task_body_t)
	kaapi_format_get_task_body_by_arch(
		td->fmt,
		KAAPI_PROC_TYPE_CUDA
	    );
    kaapi_assert_debug(body != 0);
#if defined(KAAPI_TASKLIST_POINTER_TASK)
    pc = td->task;
#else
    pc = &td->task;
#endif
#if defined(KAAPI_VERBOSE)
  fprintf(stdout, "[%s] INIT kid=%lu td=%p name=%s (counter=%d,wc=%d)\n", 
	  __FUNCTION__,
	    (long unsigned int)kaapi_get_current_kid(),
	  (void*)td, td->fmt->name,
	  KAAPI_ATOMIC_READ(&td->counter),
	  td->wc
	  );
  fflush(stdout);
#endif
    kaapi_assert_debug(pc != 0);
    kaapi_cuda_ctx_push( );
    body( kaapi_task_getargs(pc), kaapi_cuda_kernel_stream() );
#ifndef	    KAAPI_CUDA_ASYNC /* Synchronous execution */
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread_context(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
	kaapi_cuda_sync();
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread_context(), KAAPI_EVT_CUDA_CPU_SYNC_END );
#endif
    kaapi_cuda_ctx_pop( );
    kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_KER, 
	    kaapi_cuda_gpu_task_callback2_after_kernel, tasklist, td );
    return 0;
}

static int
kaapi_cuda_gpu_task_callback0_sync_gpu(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
#if !defined(KAAPI_CUDA_NO_H2D)
    kaapi_cuda_ctx_push( );
    kaapi_cuda_data_input_alloc( kstream, tasklist, td );
    kaapi_cuda_data_input_dev_sync( kstream, tasklist, td );
    kaapi_cuda_ctx_pop( );
#endif
    kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_H2D, 
	    kaapi_cuda_gpu_task_callback1_exec_task, tasklist, td );
    return 0;
}

static int
kaapi_cuda_host_task_callback1_exec_task(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
    kaapi_task_t*              pc;
#if defined(KAAPI_TASKLIST_POINTER_TASK)
    pc = td->task;
#else
    pc = &td->task;
#endif
    kaapi_assert_debug(pc != 0);
    kaapi_task_body_t body = kaapi_format_get_task_body_by_arch(
	    td->fmt, 
	    KAAPI_PROC_TYPE_HOST
	);
    kaapi_assert_debug(body != 0);
    body( kaapi_task_getargs(pc), 0 );
    kaapi_cuda_thread_tasklist_activate_deps( tasklist, td );  
    return 0;
}

static int
kaapi_cuda_host_task_callback0_sync_host(
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*    tasklist,
	kaapi_taskdescr_t*   td
    )
{
#if !defined(KAAPI_CUDA_NO_D2H)
    kaapi_cuda_ctx_push( );
    kaapi_cuda_data_input_host_sync( kstream, tasklist, td );
    kaapi_cuda_ctx_pop( );
#endif
    kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_D2H, 
	   kaapi_cuda_host_task_callback1_exec_task, tasklist, td );
    return 0;
}

int kaapi_cuda_thread_execframe_tasklist( kaapi_thread_context_t* thread )
{
  kaapi_stack_t* const stack = &thread->stack;
  kaapi_task_t*              pc;         /* cache */
  kaapi_tasklist_t*          tasklist;
  kaapi_taskdescr_t*         td;
  kaapi_frame_t*             fp;
  int                        err =0;
  uint32_t                   cnt_exec; /* executed tasks during one call of execframe_tasklist */
//  uint32_t                   cnt_pushed;
  kaapi_cuda_stream_t*	    kstream;

  kaapi_assert_debug( stack->sfp >= stack->stackframe );
  kaapi_assert_debug( stack->sfp < stack->stackframe+KAAPI_MAX_RECCALL );
  tasklist = stack->sfp->tasklist;
  kaapi_assert_debug( tasklist != 0 );

  /* here... begin execute frame tasklist*/
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_BEG );

  /* get the processor type to select correct entry point */
//  proc_type = stack->proc->proc_type;
  
  /* */
  cnt_exec = 0;
  
  /* */
//  cnt_pushed = 0;
  
  kstream = stack->proc->cuda_proc.kstream;

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
  //kaapi_writemem_barrier();
    KAAPI_DEBUG_INST(kaapi_tasklist_t save_tasklist = *tasklist; )

  while (!kaapi_tasklist_isempty( tasklist )) {
execute_pop:
    err = kaapi_readylist_pop_gpu( &tasklist->rtl, &td );

    if (err == 0) {
      kaapi_processor_decr_workload( stack->proc, 1 );
execute_first:
#if defined(KAAPI_TASKLIST_POINTER_TASK)
      pc = td->task;
#else
      pc = &td->task;
#endif

    kaapi_assert_debug(pc != 0);
    /* push the frame for the running task: pc/sp = one before td (which is in the stack)à */
    fp = (kaapi_frame_t*)stack->sfp;
    stack->sfp[1] = *fp;

    /* kaapi_writemem_barrier(); */
    stack->sfp = ++fp;
    kaapi_assert_debug((char*)fp->sp > (char*)fp->sp_data);
    kaapi_assert_debug( stack->sfp - stack->stackframe <KAAPI_MAX_RECCALL);
    
    /* start execution of the user body of the task */
    KAAPI_DEBUG_INST(kaapi_assert( td->u.acl.exec_date == 0 ));
    KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_STATIC_TASK_BEG );

#if defined(KAAPI_VERBOSE)
  if( td->fmt == 0 )
      fprintf(stdout, "[%s] kid=%lu td=%p (counter=%d,wc=%d)\n", 
	      __FUNCTION__,
		(long unsigned int)kaapi_get_current_kid(),
	      (void*)td,
	      KAAPI_ATOMIC_READ(&td->counter),
	      td->wc
	     );
  fflush(stdout);
#endif
    /* get the correct body for the proc type */
    if( td->fmt == 0 ){
	/* currently some internal tasks do not have format */
	kaapi_task_body_t body = kaapi_task_getbody( pc );
        kaapi_assert_debug(body != 0);
	body( kaapi_task_getargs(pc), (kaapi_thread_t*)stack->sfp );
    } else if( kaapi_format_get_task_body_by_arch( td->fmt, KAAPI_PROC_TYPE_CUDA ) == 0 ) {
	/* execute a CPU task in the GPU thread */
	kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_D2H, 
	       kaapi_cuda_host_task_callback0_sync_host, tasklist, td );
    } else {
	kaapi_cuda_stream_push2( kstream, KAAPI_CUDA_OP_H2D, 
		kaapi_cuda_gpu_task_callback0_sync_gpu, tasklist, td );
	kaapi_cuda_test_stream( kstream );
#if defined(KAAPI_USE_WINDOW)
    /* The slicing window is applied to all streams */
    while( 
	    (kaapi_default_param.cudawindowsize <=
	    kaapi_cuda_get_active_count_fifo( kaapi_cuda_get_input_fifo(kstream)))
	||
	    (kaapi_default_param.cudawindowsize <=
	    kaapi_cuda_get_active_count_fifo( kaapi_cuda_get_kernel_fifo(kstream)))
	||
	    (kaapi_default_param.cudawindowsize <=
	    kaapi_cuda_get_active_count_fifo( kaapi_cuda_get_output_fifo(kstream)))
	 )
    {
	kaapi_cuda_test_stream( kstream );
    }
#endif
	kaapi_cuda_test_stream( kstream );
    }
    KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_STATIC_TASK_END );
    KAAPI_DEBUG_INST( td->u.acl.exec_date = kaapi_get_elapsedns() );
    ++cnt_exec;

    /* new tasks created ? */
    if (unlikely(fp->sp > stack->sfp->sp)) {
    redo_frameexecution:
      err = kaapi_stack_execframe( &thread->stack );
      if (err == EWOULDBLOCK)
      {
    printf("EWOULDBLOCK case 1\n");
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

    /* activate non-CUDA tasks now */
    if ( td->fmt == 0 )
	kaapi_cuda_thread_tasklist_activate_deps( tasklist, td );  

    } /* err == 0 */
    
    /* recv incomming synchronisation 
       - process it before the activation list of the executed
       in order to force directly activated task to be executed first.
    */
    if (tasklist->recv !=0)
    {
    }


    /* ok, now push pushed task into the wq and restore the next td to execute */
#if 0
    if ( (td = kaapi_thread_tasklist_commit_ready_and_steal( tasklist )) !=0 )
	  goto execute_first;
#endif
    //kaapi_thread_tasklist_commit_ready( tasklist );
            
    KAAPI_DEBUG_INST(save_tasklist = *tasklist;)

  } /* while */

    /* finish all GPU CUDA operations */
    while( kaapi_cuda_waitfirst_stream( kstream ) != KAAPI_CUDA_STREAM_EMPTY ){
///	if ( (td = kaapi_thread_tasklist_commit_ready_and_steal( tasklist )) !=0 )
	err = kaapi_readylist_pop_gpu( &tasklist->rtl, &td );
	if( err == 0 )
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
