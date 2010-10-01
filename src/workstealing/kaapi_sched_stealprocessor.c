/*
** kaapi_sched_stealprocessor.c
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

/*
   Utiliser un iterator sur le lrequests qui a ete crée dans la partie "atomic" de
   la lecture de l'état du lrequest.
   - doit permettre aussi d'autoriser la vol en concurrence. Sur un même bloc de request.
   => listready, suspendlist concurrente.
*/

/** 
*/
int kaapi_sched_stealprocessor(
  kaapi_processor_t* kproc, 
  kaapi_listrequest_t* lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_request_t* request;

  /* test should be done before calling the function */
  kaapi_assert_debug( !kaapi_listrequest_iterator_empty(lrrange) );
  
  request = kaapi_listrequest_iterator_get( lrequests, lrrange );

  /* try to steal ready thread */
  while (!kaapi_readylist_empty(&kproc->lready))
  {
    kaapi_thread_context_t* thread;
    thread = kaapi_listready_steal( &kproc->lready, kaapi_request_getthiefid(request) );
    if (thread != 0)
    {
      /* reply */
      _kaapi_request_reply(request, 1);
      request = kaapi_listrequest_iterator_next( lrequests, lrrange );
      if (kaapi_listrequest_iterator_empty(lrrange)) return 0;
    }
  }
  
#if 0
for (int i=0; i<1; ++i)
  if (1)
  { /* WARNING do not try to steal inside suspended stack */
    kaapi_wsqueuectxt_cell_t* cell;
    cell = kproc->lsuspend.tail;
    while ((cell !=0) && (count >0))
    {
      stealok = KAAPI_ATOMIC_CAS( &cell->state, 0, 1);
      if (stealok)
      {
        kaapi_thread_context_t* thread = cell->thread;
        if (thread !=0)
        {
          replycount += kaapi_sched_stealstack( thread, 0, count, kproc->hlrequests.requests );
          count = KAAPI_ATOMIC_READ( &kproc->hlrequests.count );
        }
        KAAPI_ATOMIC_CAS( &cell->state, 1, 0); /* may be ==2 -> wakeuped by the owner */
      }
      cell = cell->prev;
    }
  }
#endif

  /* steal current thread */
  kaapi_thread_context_t*  thread = kproc->thread;
  if ( (thread !=0) 
    && (kproc->issteal ==0) /* last: if a thread is stealing, its current thread will be used to receive work... */ 
  )
  {
    /* signal that count thefts are waiting */
    kaapi_sched_stealstack( thread, 0, lrequests, lrrange );
  }  
  return 0;
}
