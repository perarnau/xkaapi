/*
** kaapi_sched_suspend.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:01 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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

/* this version is close to the kaapi_sched_idle, except that a condition of
   wakeup is to test that suspended condition is false
*/
int kaapi_sched_suspend ( kaapi_processor_t* kproc )
{
  int err;
  kaapi_thread_context_t* ctxt;
  kaapi_thread_context_t* thread_condition;
  kaapi_task_t*           task_condition;
  kaapi_thread_context_t* thread;

  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif

  /* here is the reason of suspension */
  thread_condition = kproc->thread;
  kaapi_assert_debug( kproc == thread_condition->proc);

  task_condition = thread_condition->sfp->pc;
  if (kaapi_task_getbody(task_condition) != kaapi_suspend_body) return 0;

  /* such threads are sticky: the control flow is on return to this call and
     without thread user context switch only this activation frame could wakeup
     the thread
  */
  kaapi_assert_debug( thread_condition->affinity == 0 );

  /* put context in the list of suspended contexts: no critical section with respect of thieves */
  kaapi_setcontext(kproc, 0);
  kaapi_wsqueuectxt_push( kproc, thread_condition );

  do {
    /* wakeup a context: either a ready thread (first) or a suspended thread.
       Precise the suspended thread 'thread_condition' in order to wakeup it first
    */
    ctxt = kaapi_sched_wakeup(kproc, kproc->kid, thread_condition);
    if (ctxt ==0)
    {
      kaapi_assert_debug(kproc->thread == 0);
      kproc->thread = 0;
    }
    else {
      kaapi_setcontext( kproc, ctxt );
      kaapi_assert_debug(kproc->thread == ctxt );
    }

    if (kproc->thread == thread_condition) 
    {
      kaapi_assert(kproc->thread->sfp->pc == task_condition);
#if defined(KAAPI_USE_PERFCOUNTER)
      kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif
      return 0;
    }
#if defined(KAAPI_DEBUG)
    {
      kaapi_wsqueuectxt_cell_t* cell;
      int found = 0;
      if (kproc->readythread == ctxt_condition)
        found =1;
        
      /* else search iff ctxt_condition is yet in suspend list, it should ! */
      if (!found)
      {
        cell = kproc->lsuspend.head;
        while (cell !=0)
        {
          if (cell->thread == thread_condition) { found=1; break; }
          kaapi_wsqueuectxt_cell_t* nextcell = cell->next;
          cell = nextcell;
        }
      }
      kaapi_assert_m(found !=0, "cannot find ctxt_condition in lists");
    }
#endif    

    if (kproc->thread !=0)
      goto redo_execution;

/* warning: to avoid steal of processor ! */
//continue;    

    /* else steal a task */
    if (kproc->thread ==0)
    {
      ctxt = kaapi_context_alloc(kproc);
      kaapi_setcontext(kproc, ctxt);

      /* on return, either a new thread has been stolen, either a task as been put into ctxt or thread ==0 */
      thread = kaapi_sched_emitsteal( kproc );

      if (kaapi_frame_isempty(ctxt->sfp))
      {
        /* push it into the free list */
        kaapi_setcontext( kproc , 0);
        kaapi_lfree_push( kproc, ctxt );
      }
      if (thread ==0) {
        //kaapi_sched_advance(kproc);
        continue;
      }

      kaapi_setcontext(kproc, thread);
    }

redo_execution:
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif
    err = kaapi_stack_execframe( kproc->thread );
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif
    kaapi_assert( err != EINVAL);

    ctxt = kproc->thread;

    /* update */
    kaapi_setcontext(kproc, 0);

    if (err == EWOULDBLOCK) 
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif
      /* push it: suspended because top task is not ready */
      kaapi_wsqueuectxt_push( kproc, ctxt );
    } 
    /* WARNING: this case is used by static scheduling in order to detach a thread context 
       from a thread at the end of an iteration. See kaapi_tasksignalend_body.
       Previous code: without the if
    */
    else if (ctxt != 0) 
    {
      /* push it into freelist */
      kaapi_lfree_push( kproc, ctxt );
    }
  } while (1);
  return EINTR;
}
