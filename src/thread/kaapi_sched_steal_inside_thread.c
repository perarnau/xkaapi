/*
** kaapi_sched_steal.c
** ckaapi
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

/**
*/
int kaapi_sched_steal_sc_inside(
    struct kaapi_steal_processor_t* kpsp, struct kaapi_steal_context_t* sc,
    int count, kaapi_steal_request_t** requests
)
{
  int i, retval = count;
  kaapi_processor_t* proc = (kaapi_processor_t*)((char*)kpsp - sizeof(kaapi_processor_t));
  kaapi_steal_thread_context_t* scthread = (kaapi_steal_thread_context_t*)&proc->_sc_thread;

  /* my be concurrency between thief and the victim, kpss */
  if (count ==0) return 0;
  
  /* try to steal inside suspended thread first
  */
  if (KAAPI_WORKQUEUE_EMPTY( &scthread->_suspended_thread )) return 0;
  
  kaapi_queue_cellsuspended_t* currbloc = proc->_sc_thread._suspended_thread._top_bloc;
  
  redo_test:
  for (i= currbloc->_top; (count>0) && (i<currbloc->_bottom); ++i)
  {
    /* critical section (using cas) could we avoid it ? */
    if (KAAPI_WORKQUEUE_BLOC_STEAL( currbloc, i ))
    {
      kaapi_thread_descr_t* thread = currbloc->_data[i]._thread;
      if (thread->_splitter !=0)
        count -= (*thread->_splitter)( kpsp, sc, count, requests );
      KAAPI_WORKQUEUE_BLOC_ABORTSTEAL( currbloc, i );
    }
  }
  if (currbloc->_nextbloc !=0)
  {
    currbloc = currbloc->_nextbloc;
    goto redo_test;
  }

  /* Try to steal the active thread
     Processor was locked by the caller: active thread could not change
  */
  if (scthread->_active_thread == 0) return 0;
  
  /* It is to the responsability to the _splitter method to ensure syncronisation between caller and callee */
  if (scthread->_active_thread->_splitter !=0)
    count -= (*scthread->_active_thread->_splitter)( kpsp, sc, count, requests );

  return retval-count;
}

