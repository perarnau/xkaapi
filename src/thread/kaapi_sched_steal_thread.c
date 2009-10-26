/*
** kaapi_sched_steal.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:02 2009
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
#include <string.h>


/** Method to execute a thread after a successful steal op
*/
void kaapi_steal_thief_entrypoint_thread(struct kaapi_steal_processor_t* kpsp, struct kaapi_steal_request_t* request)
{
  kaapi_thread_descr_t* thread;
  kaapi_processor_t* proc = (kaapi_processor_t*) ((char*)kpsp - sizeof(kaapi_processor_t));

  /* test if p has already an active thread or not */
  memcpy( &thread, &request->_data[0], sizeof(kaapi_thread_descr_t*));

  /* jmp to thread context */
  thread->_state = KAAPI_THREAD_S_RUNNING;
  thread->_proc = proc;
  proc->_sc_thread._active_thread = thread;
#if defined(KAAPI_USE_UCONTEXT)
  setcontext( &thread->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _longjmp( thread->_ctxt,  (int)(long)thread);
#endif
}


/**
*/
int kaapi_sched_steal_sc_thread(
    kaapi_steal_processor_t* kpsp, kaapi_steal_context_t* sc,
    int count, kaapi_steal_request_t** requests
)
{
  int i, retval = count;
  kaapi_processor_t* proc = (kaapi_processor_t*)((char*)kpsp - sizeof(kaapi_processor_t));
  kaapi_steal_thread_context_t* scthread = (kaapi_steal_thread_context_t*)sc;

  kaapi_assert_debug( &proc->_sc_thread == scthread );
  
  /* my be concurrency between thief and the victim, kpss */
  if (count ==0) return 0;
  
  /* try to steal ready thread first
    - here a test should be ensure that thief could steal a thread
  */
  if (KAAPI_WORKQUEUE_EMPTY( &scthread->_ready_list )) return 0;
  
  kaapi_queue_cellready_t* currbloc = scthread->_ready_list._top_bloc;
  
  redo_test:
  for (i= currbloc->_top; (count>0) && (i<currbloc->_bottom); ++i)
  {
    if (KAAPI_WORKQUEUE_BLOC_STEAL( currbloc, i ))
    {
      kaapi_thread_descr_t* thread = currbloc->_data[i]._thread;
      int idxreq = CPU_INTERSECT( &kpsp->_list_request._cpuset, &thread->_cpuset);
      if (idxreq !=-1)
      {
        currbloc->_data[i]._thread = 0;
        if (i==currbloc->_top) ++currbloc->_top;
        
        /* reply to thief */
        requests[idxreq]->_entrypoint = &kaapi_steal_thief_entrypoint_thread;
        memcpy( &requests[idxreq]->_data[0], &thread, sizeof(kaapi_thread_descr_t*));
        kaapi_thief_reply_request( sc, requests, idxreq, 1 );
        CPU_CLR( idxreq, &kpsp->_list_request._cpuset );
        --count;
      }
      else {
        KAAPI_WORKQUEUE_BLOC_ABORTSTEAL( currbloc, i );
      }
    }
  }
  if (currbloc->_nextbloc !=0)
  {
    currbloc = currbloc->_nextbloc;
    goto redo_test;
  }

  return retval-count;
}

