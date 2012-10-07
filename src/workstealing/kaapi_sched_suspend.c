/*
** kaapi_sched_suspend.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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

#if defined(KAAPI_USE_CUDA)
# include "../machine/cuda/kaapi_cuda_execframe.h"
# include "../machine/cuda/kaapi_cuda_threadgroup_execframe.h"
#endif


/* this version is close to the kaapi_sched_idle, except that a condition of
   wakeup is to test that suspended condition is false.
   The main reason of this function is because we have no yet reported
   context switch to scheduler code as in C++ Kaapi.   
*/
int kaapi_sched_suspend ( kaapi_processor_t* kproc, int (*fcondition)(void* ), void* arg_fcondition )
{
  int err;
  kaapi_request_status_t  ws_status;
  kaapi_thread_context_t* thread;
  kaapi_thread_context_t* thread_condition;
  kaapi_thread_context_t* tmp;

  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->thread !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );

  /* do not account suspension if condition if true */
  if (fcondition(arg_fcondition) !=0) 
    return 0;

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
  KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_TASK_END );  
  KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
#endif

  /* here is the reason of suspension */
  thread_condition = kproc->thread;
  kaapi_assert_debug( kproc == thread_condition->stack.proc);

  /* such threads are sticky: the control flow will return to this call 
     with the same thread as active thread on the kproc.
     Without thread user's context switch, only this control flow could wakeup
     the thread.
  */
  kaapi_assert_debug( kaapi_cpuset_empty( &thread_condition->affinity ) );

  /* put context in the list of suspended contexts: no critical section with respect of thieves */
  kaapi_setcontext(kproc, 0);
  kaapi_wsqueuectxt_push( kproc, thread_condition );
  
  /* ok now we can mark that the kproc is idle */
  kproc->isidle = 1;

  do {
#if defined(KAAPI_USE_NETWORK)
    kaapi_network_poll();
#endif

    /* end of a parallel region ? */
    if (kaapi_suspendflag)
      kaapi_mt_suspend_self(kproc);

    /* wakeup a context: either a ready thread (first) or a suspended thread.
       Precise the suspended thread 'thread_condition' in order to wakeup it first and task_condition.
    */
    thread = kaapi_sched_wakeup(kproc, kproc->kid, thread_condition, fcondition, arg_fcondition); // thread_condition, task_condition);
    if (thread !=0)
    {
      if (thread == thread_condition)
      {
        tmp = kproc->thread;
        kaapi_setcontext( kproc, thread_condition );

        /* push kproc context into free list */
        if (tmp !=0)
          kaapi_context_free( kproc, tmp );
#if 0
        /* ok suspended thread is ready for execution */
        kaapi_assert((tasklist !=0) || (thread->stack.sfp->pc == task_condition));
#endif

#if defined(KAAPI_USE_PERFCOUNTER)
        kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
        KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_SCHED_IDLE_END );
#endif
        kproc->isidle = 0;
        return 0;
      }

      tmp = kproc->thread;
      kaapi_setcontext( kproc, thread );

      /* push kproc context into free list */
      if (tmp !=0)
        kaapi_context_free( kproc, tmp );


      goto redo_execution;
    }

    /* always allocate a thread before emitting a steal request */
    if (kproc->thread ==0)
    {
      thread = kaapi_context_alloc(kproc,(size_t)-1);
      kaapi_assert_debug( thread != 0 );
      
      kaapi_setcontext(kproc, thread);
    }

    /* steal request */
    ws_status = kproc->emitsteal(kproc);
    if (ws_status != KAAPI_REQUEST_S_OK)
      continue;

redo_execution:
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
    KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_SCHED_IDLE_END );
    KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_TASK_BEG );  
#endif

#if defined(KAAPI_USE_CUDA)
    if (kproc->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      if (kproc->thread->sfp->tasklist == 0)
        err = kaapi_cuda_execframe( kproc->thread );
      else /* assumed kaapi_threadgroup_execframe */
        err = kaapi_cuda_threadgroup_execframe(kproc->thread);
    }
    else
#endif /* KAAPI_USE_CUDA */
    if (kproc->thread->stack.sfp->tasklist ==0)
      err = kaapi_stack_execframe(&kproc->thread->stack);
    else
      err = kaapi_thread_execframe_tasklist(kproc->thread);

#if defined(KAAPI_USE_PERFCOUNTER)
    KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_TASK_END );  
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
    KAAPI_EVENT_PUSH0( kproc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
#endif
    kaapi_assert( err != EINVAL);
    kproc->isidle = 1;

    if (err == EWOULDBLOCK) 
    {
      /* push it: suspended because top task is not ready */
      thread = kproc->thread;
      kaapi_setcontext(kproc, 0);
      kaapi_wsqueuectxt_push( kproc, thread );

#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif
    } 
    /* WARNING: this case is used by static scheduling in order to detach a thread context 
       from a thread at the end of an iteration. See kaapi_thread_execframe, kaapi_tasksignalend_body.
       Previous code: without the if
    */
    else if ((err == ECHILD) || (err == EINTR))
    {
      /* detach thread */
      kaapi_setcontext(kproc, 0);
    }
#if defined(KAAPI_DEBUG)
    else 
    {
      kaapi_assert( kaapi_frame_isempty( kproc->thread->stack.sfp ) );
      kaapi_assert( kproc->thread->stack.sfp == kproc->thread->stack.stackframe );
      kaapi_stack_reset(&kproc->thread->stack);
      kaapi_synchronize_steal(kproc);
    }
#endif
    kproc->isidle = 1;

  } while (1);
  
  return EINTR;
}
