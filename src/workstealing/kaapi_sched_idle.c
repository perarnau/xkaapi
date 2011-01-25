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

/* \TODO: voir ici si les fonctions (a adapter si besoin): makecontext, getcontext, setcontext 
   seront suffisante dans le cas d'un suspend durant l'execution d'une tâche.
*/
void kaapi_sched_idle ( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* ctxt;
  kaapi_thread_context_t* thread;
  int err;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );
  kaapi_assert_debug( kproc->thread !=0 );

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif
  do {

    /* terminaison ? */
    if (kaapi_isterminated())
    {
      return;
    }
    
    ctxt = 0;
    /* local wake up first, inline test to avoid function call */
    if (!kaapi_sched_readyempty(kproc) || !kaapi_wsqueuectxt_empty(kproc) )
    {
      ctxt = kaapi_sched_wakeup(kproc, kproc->kid, 0, 0); 

      if (ctxt !=0) /* push kproc->thread to free and set ctxt as new ctxt */
      {
        /* push kproc context into free list */
        kaapi_sched_lock(&kproc->lock);
        kaapi_lfree_push( kproc, kproc->thread );
        kaapi_sched_unlock(&kproc->lock);

        /* set new context to the kprocessor */
        kaapi_setcontext(kproc, ctxt);
        goto redo_execute;
      }
    }

    /* steal request */
    kaapi_assert_debug( kproc->thread !=0 );
    ctxt = kproc->thread;
    thread = kaapi_sched_emitsteal( kproc );

    if (thread ==0) 
      continue;

    if (thread != ctxt)
    {
      /* also means ctxt is empty, so push ctxt into the free list */
      kaapi_setcontext( kproc , 0);
      /* wait end of thieves before releasing a thread */
      kaapi_sched_lock(&kproc->lock);
      kaapi_lfree_push( kproc, ctxt );
      kaapi_sched_unlock(&kproc->lock);
    }
    kaapi_setcontext(kproc, thread);

redo_execute:
//  kaapi_thread_print( stdout, kproc->thread );

#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
#endif

#if defined(KAAPI_USE_CUDA)
    if (kproc->proc_type == KAAPI_PROC_TYPE_CUDA)
      err = kaapi_cuda_execframe( kproc->thread );
    else
#endif /* KAAPI_USE_CUDA */
    err = (*kproc->thread->execframe)( kproc->thread );

#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif

    if (err == EWOULDBLOCK) 
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_SUSPEND);
#endif
      ctxt = kproc->thread;
      /* update */
      kaapi_setcontext(kproc, 0);

      /* push it: suspended because top task is not ready */
      kaapi_wsqueuectxt_push( kproc, ctxt );

      ctxt = kaapi_sched_wakeup(kproc, kproc->kid, 0, 0);
      if (ctxt !=0)
      {
        kaapi_setcontext(kproc, ctxt ); 
        goto redo_execute;
      }
      
      /* else reallocate a context */
      ctxt = kaapi_context_alloc(kproc);

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
    }

    /* WARNING: this case is used by static scheduling in order to detach a thread context 
       from a thread at the end of an iteration. See kaapi_thread_execframe kaapi_tasksignalend_body
       Previous code: without the test else if () {... }
    */
    else if (err == EINTR) 
    {
      /* used to detach the thread of the processor in order to reuse it ... */
      ctxt = kaapi_context_alloc(kproc);

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
    }
  } while (1);
  
}
