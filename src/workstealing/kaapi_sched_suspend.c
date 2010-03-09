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
  kaapi_thread_context_t* tmp;
  kaapi_thread_context_t* ctxt_condition;
  kaapi_task_t*           task_condition;
  kaapi_stack_t*          stack;

  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif

  /* here is the reason of suspension */
  ctxt_condition = kproc->ctxt;
  task_condition = ctxt_condition->pfsp->pc;
  if (kaapi_task_getbody(task_condition) != kaapi_suspend_body) return 0;
  
  /* put context is list of suspended contexts: critical section with respect of thieves */
  kproc->ctxt = 0;
  kaapi_wsqueuectxt_push( &kproc->lsuspend, ctxt_condition );

  do {
    /* wakeup a context */
    kproc->ctxt = kaapi_sched_wakeup(kproc);

    if (kproc->ctxt == ctxt_condition) 
    {
      kaapi_assert(kproc->ctxt->frame_sp == task_condition);
#if defined(KAAPI_USE_PERFCOUNTER)
      kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif
      return 0;
    }

    /* else steal a task */
    if (kproc->ctxt ==0)
    {
      ctxt = kaapi_context_alloc(kproc);
      ctxt->hasrequest = 0;
      kaapi_setcontext(kproc, ctxt);

      stack = kaapi_sched_emitsteal( kproc );

      if (kaapi_stack_isempty(stack))
      {
        kaapi_sched_advance(kproc);

        /* push it into the free list */
        tmp = kproc->ctxt;
        kproc->ctxt = 0;
        KAAPI_STACK_PUSH( &kproc->lfree, tmp );
        continue;
      }
      if (stack != kproc->ctxt)
      {
        /* push it into the free list */
        tmp = kproc->ctxt;
        kproc->ctxt = 0;

        KAAPI_STACK_PUSH( &kproc->lfree, tmp );
        kaapi_setcontext(kproc, stack);
      }
    }

#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif
    err = kaapi_stack_execframe( kproc->ctxt );
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif
    kaapi_assert( err != EINVAL);

    ctxt = kproc->ctxt;

    /* update   */
    kproc->ctxt = 0;

    if (err == EWOULDBLOCK) 
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif
      /* push it: suspended because top task is not ready */
      kaapi_wsqueuectxt_push( &kproc->lsuspend, ctxt );
    } else {
      /* push it: free */
      KAAPI_STACK_PUSH( &kproc->lfree, ctxt );
    }
  } while (1);
  return EINTR;
}
