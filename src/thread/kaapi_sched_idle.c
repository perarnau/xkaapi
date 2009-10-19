/*
** kaapi_sched_idle.c
** ckaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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

void kaapi_sched_idle ( kaapi_processor_t* proc )
{
  int i,victim;
  kaapi_steal_processor_t* victim_processor;
  kaapi_thread_descr_t* td __attribute__((__unused__))= (kaapi_thread_descr_t*)pthread_getspecific(kaapi_current_thread_key);

  if (proc ==0) 
  {
    /* */
    proc = pthread_getspecific( kaapi_current_processor_key );
    if (proc ==0) 
    {
      proc = kaapi_allocate_processor();
      pthread_setspecific( kaapi_current_processor_key, proc );
    }
  }
  
redo_post:
  /** Try to wakeup suspendend thread, from the most recent inserted thread to the last recent
  */
  if (!KAAPI_WORKQUEUE_EMPTY( &proc->_sc_thread._suspended_thread ))
  {
    kaapi_queue_cellsuspended_t* currbloc = proc->_sc_thread._suspended_thread._bottom_bloc;
    
    redo_test:
    for (i= currbloc->_bottom-1; i >= currbloc->_top; --i)
    {
      if ( (KAAPI_ATOMIC_READ(&currbloc->_data[i]._status) == 1) && (currbloc->_data[i]._fwakeup !=0) )
        if ( (*currbloc->_data[i]._fwakeup)(currbloc->_data[i]._arg_fwakeup) )
      {
        KAAPI_ATOMIC_WRITE(&currbloc->_data[i]._status, 0);
        kaapi_thread_descr_t* thread = currbloc->_data[i]._thread;
        if (thread ==0) continue;
        currbloc->_data[i]._thread = 0;

////WARNING/ HERE GARBAGE !!! if (i==currbloc->_top) currbloc->_top = currbloc->_bottom = 0;

        ckaapi_assert( thread->_state == KAAPI_THREAD_SUSPEND )

        /* activate thread */
        thread->_state = KAAPI_THREAD_RUNNING;
        thread->_proc = proc;
        proc->_sc_thread._active_thread = thread;
        /* jmp to thread context, never store processor context */
#if defined(KAAPI_USE_UCONTEXT)
        setcontext( &thread->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
        _longjmp( thread->_ctxt,  (int)(long)thread);
#endif
      }
    }
    if (currbloc->_prevbloc !=0) 
    {
      currbloc = currbloc->_prevbloc; 
      goto redo_test; 
    }
  }
  
  /** Try to get a ready thread 
  */
  if (!KAAPI_WORKQUEUE_EMPTY( &proc->_sc_thread._ready_list )) 
  {
    kaapi_queue_cellready_t* currbloc = proc->_sc_thread._ready_list._top_bloc;
    
    redo_test_ready:
    for (i= currbloc->_bottom-1; i >= currbloc->_top; --i)
    {
      if (KAAPI_WORKQUEUE_BLOC_STEAL( currbloc, i ))
      {
        kaapi_thread_descr_t* thread = currbloc->_data[i]._thread;
        currbloc->_data[i]._thread = 0;

//WARNING/ HERE GARBAGE !!!        if (i==currbloc->_top) currbloc->_top = currbloc->_bottom = 0;

        /* activate thread */
        thread->_state = KAAPI_THREAD_RUNNING;
        thread->_proc = proc;
        proc->_sc_thread._active_thread = thread;
#if defined(KAAPI_USE_UCONTEXT)
        setcontext( &thread->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
        _longjmp( thread->_ctxt,  (int)(long)thread);
#endif
      }
    }
    if (currbloc->_prevbloc !=0)
    {
      currbloc = currbloc->_prevbloc;
      goto redo_test_ready;
    }
  }
  
  /** Select a victim 
  */
  victim = kaapi_steal_processor_select_victim(  proc->_the_steal_processor );
  
  /** Is terminated 
  */
  if ((victim ==-1) || kaapi_stealapi_term) goto terminate_program;

  victim_processor = kaapi_all_stealprocessor[victim];

  /* Post non blocking request 
  */
  kaapi_thief_request_post( victim_processor, proc->_the_steal_processor, &proc->_the_steal_processor->_request );

  /* a way to release processor ... and wait for other stealer ? */
  kaapi_yield();
  
  /* lock victim and process my request and may be other request 
  */
  ckaapi_assert( 0 == kaapi_mutex_lock( &victim_processor->_lock ) );
  while (kaapi_thief_request_status(&proc->_the_steal_processor->_request) == KAAPI_REQUEST_S_POSTED)
  {
    /* here request should be cancelled... */
    kaapi_steal_processor( victim_processor );
  }
  ckaapi_assert( 0 == kaapi_mutex_unlock( &victim_processor->_lock ) );

  ckaapi_assert( kaapi_thief_request_status(&proc->_the_steal_processor->_request) != KAAPI_REQUEST_S_POSTED );

  /* test if my request is ok
  */
  if (kaapi_thief_request_ok(&proc->_the_steal_processor->_request)) 
    goto execute_work;
  
  /* if not redo steal
  */
  goto redo_post;
  
execute_work:
  /* Do the local computation
  */
  kaapi_thief_execute( proc->_the_steal_processor, &proc->_the_steal_processor->_request );
  
  goto redo_post;

terminate_program:

  kaapi_barrier_td_setactive( &kaapi_stealapi_barrier_term, 0 );
  
  /* Jump to the kernel thread, if not currently running */
#if defined(KAAPI_USE_UCONTEXT)
  setcontext( &proc->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _longjmp( proc->_ctxt,  (int)(long)proc);
#endif
}
