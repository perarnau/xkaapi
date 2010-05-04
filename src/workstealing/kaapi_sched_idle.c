/*
** kaapi_sched_idle.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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

/* \TODO: voir ici si les fonctions (a adapter si besoin): makecontext, getcontext, setcontext 
   seront suffisante dans le cas d'un suspend durant l'execution d'une tâche.
*/
void kaapi_sched_idle ( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* ctxt;
  kaapi_thread_context_t* tmp;
  kaapi_thread_context_t* thread;
  int err;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );
  kaapi_assert_debug( kproc->thread !=0 );

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif
  do {

/*    usleep( 10000 );*/
/*pthread_yield_np();*/

    /* terminaison ? */
    if (kaapi_isterminated())
    {
      return;
    }
    
    ctxt = 0;
    /* local wake up first */
    for (int i=0; i<5; ++i)
    {
      if (!kaapi_sched_suspendlist_empty(kproc))
      {
        ctxt = kaapi_sched_wakeup(kproc); 
        if (ctxt !=0) break;
      }
    }

    if (ctxt !=0) /* push kproc->ctxt to free and set ctxt as new ctxt */
    {
      /* push kproc context into free list */
      tmp = kproc->thread;

      /* update */
      kproc->thread = 0;

      KAAPI_STACK_PUSH( &kproc->lfree, tmp );

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
      goto redo_execute;
    }
    
    /* steal request */
    thread = kaapi_sched_emitsteal( kproc );

    /* next assert if ok because we do not steal thread... */
    kaapi_assert_debug( (thread == 0) || (thread == kproc->thread) );

    if ((thread ==0) || (kaapi_frame_isempty(thread->sfp))) 
    {
      kaapi_sched_advance(kproc);
      continue;
    }
    kaapi_assert_debug( thread != 0);
    

redo_execute:

    /* printf("Thief, 0x%p, pc:0x%p,  #task:%u\n", stack, stack->pc, stack->sp - stack->pc ); */
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif
    err = kaapi_stack_execframe( kproc->thread );

#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif

    if (err == EWOULDBLOCK) 
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif
      kaapi_thread_context_t* ctxt = kproc->thread;
      /* update */
      kaapi_setcontext(kproc, 0);

      /* push it: suspended because top task is not ready */
      kaapi_wsqueuectxt_push( &kproc->lsuspend, ctxt );

      if (kaapi_sched_suspendlist_empty(kproc))
        kproc->thread = 0;
      else
        kaapi_setcontext(kproc, kaapi_sched_wakeup(kproc) ); 
      if (kproc->thread !=0) goto redo_execute;

      /* else reallocate a context */
      ctxt = kaapi_context_alloc(kproc);

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
    }
  } while (1);
  
}
