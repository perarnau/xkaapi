/*
 ** kaapi_sched_idle.c
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
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
# include "../../machine/cuda/kaapi_cuda_execframe.h"
# include "../../machine/cuda/kaapi_cuda_threadgroup_execframe.h"
#endif

/* \TODO: voir ici si les fonctions (a adapter si besoin): makecontext, getcontext, setcontext 
 seront suffisante dans le cas d'un suspend durant l'execution d'une tâche.
 */
void kaapi_sched_idle ( kaapi_processor_t* kproc )
{
  kaapi_request_status_t  ws_status;
  kaapi_thread_context_t* thread;
  kaapi_thread_context_t* tmp;
  int err;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );
  
  /* currently kprocessor are created with an attached thread */
  kaapi_assert_debug( kproc->thread !=0 );
  
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
  kaapi_event_push0(kproc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
#endif

  do 
  {
#if defined(KAAPI_USE_NETWORK)
    kaapi_network_poll();
#endif
    if (kaapi_suspendflag)
      kaapi_mt_suspend_self(kproc);
    
    /* terminaison ? */
    if (kaapi_isterminated())
    {
      kaapi_event_push0(kproc, 0, KAAPI_EVT_SCHED_IDLE_END );
      return;
    }
    
    /* try to wake up suspended thread first, inline test to avoid function call */
    if (!kaapi_wsqueuectxt_empty(kproc))
    {
      thread = kaapi_sched_wakeup(kproc, kproc->kid, 0, 0); 
      
      if (thread !=0) /* push kproc->thread to freelist and set thread as the new ctxt */
      {
        tmp = kproc->thread;
        /* set new context to the kprocessor */
        kaapi_setcontext(kproc, thread);

        /* push kproc context into free list */
        if (tmp) 
          kaapi_context_free( kproc, tmp );
        
        goto redo_execute;
      }
    }

    /* always allocate a thread before emitting a steal request */
    if (kproc->thread ==0)
    {
      thread = kaapi_context_alloc(kproc, (size_t)-1);
      kaapi_assert_debug( thread != 0 );
      
      kaapi_setcontext(kproc, thread);
    }

    /* steal request */
    ws_status = kproc->emitsteal(kproc);
    if (ws_status != KAAPI_REQUEST_S_OK)
      continue;

redo_execute:    
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_USER_STATE );
    kaapi_event_push0(kproc, 0, KAAPI_EVT_SCHED_IDLE_END );
#endif

#if defined(KAAPI_USE_CUDA)
    if (kproc->proc_type == KAAPI_PROC_TYPE_CUDA)
    {
      if (kproc->thread->sfp->tasklist == 0)
        err = kaapi_thread_execframe(&kproc->thread->stack);
      else
        err = kaapi_cuda_thread_execframe_tasklist(&kproc->thread->stack);
    }
    else
#endif /* KAAPI_USE_CUDA */
      if (kproc->thread->stack.sfp->tasklist ==0)
        err = kaapi_stack_execframe(&kproc->thread->stack);
      else
        err = kaapi_thread_execframe_tasklist( kproc->thread );
    
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
    kaapi_event_push0(kproc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
#endif
    
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
     from a thread at the end of an iteration. See kaapi_thread_execframe kaapi_tasksignalend_body
     Previous code: without the test else if () {... }
     */
    else if ((err == ECHILD) || (err == EINTR))
    {
      /* detach thread */
      kaapi_setcontext(kproc, 0);
    }
#if defined(KAAPI_DEBUG)
    else 
    {
      kaapi_stack_reset(&kproc->thread->stack);
      kaapi_synchronize_steal(kproc);
#if 0
      kaapi_assert( kaapi_frame_isempty( kproc->thread->stack.sfp ) );
      kaapi_assert( kproc->thread->stack.sfp == kproc->thread->stack.stackframe );
      kaapi_assert( kproc->thread->stack.sfp->pc == kproc->thread->stack.task );
      kaapi_assert( kproc->thread->stack.sfp->sp == kproc->thread->stack.task );
      kaapi_assert( kproc->thread->stack.sfp->sp_data == kproc->thread->stack.data );
#endif
    }
#endif
  } while (1);
  
}
