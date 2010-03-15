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

/** 
*/
int kaapi_sched_stealprocessor(kaapi_processor_t* kproc)
{
  kaapi_thread_context_t*  thread;
  int count =0;
  int stealok = 0;
#if defined(KAAPI_CONCURRENT_WS)
  int replycount = 0;
#endif

  count = KAAPI_ATOMIC_READ( &kproc->hlrequests.count );
  kaapi_assert_debug( count > 0 );
  if (count ==0) return 0;
  
  /* a second read my only view a count greather or equal to the previous because here
     the caller is in critical section to reply to posted requests.
  */
  kaapi_assert_debug( count <= KAAPI_ATOMIC_READ( &kproc->hlrequests.count ) );

#if defined(KAAPI_CONCURRENT_WS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->lock) == 1+_kaapi_get_current_processor()->kid );
#endif
  
  if (1)
  { /* WARNING do not try to steal inside suspended stack */
    kaapi_wsqueuectxt_cell_t* cell;
    cell = kproc->lsuspend.tail;
    while ((cell !=0) && (count >0))
    {
      stealok = KAAPI_ATOMIC_CAS( &cell->state, 0, 1);
      if (stealok)
      {
        kaapi_readmem_barrier(); /* to force view of cell->thread */
        kaapi_thread_context_t* thread = cell->thread;
        if (thread !=0)
        {
          replycount += kaapi_sched_stealstack( thread, 0, count, kproc->hlrequests.requests );
          count = KAAPI_ATOMIC_READ( &kproc->hlrequests.count );
        }
        KAAPI_ATOMIC_WRITE( &cell->state, 0 );
      }
      cell = cell->prev;
    }
  }
  
  /* steal current thread */
  thread = kproc->thread;
  if ((count >0) && (thread !=0) && (kproc->issteal ==0))
  {
#if defined(KAAPI_CONCURRENT_WS)
    /* if concurrent WS, then steal directly the current stack of the victim processor
    */
    kaapi_assert_debug( count <= KAAPI_ATOMIC_READ( &kproc->hlrequests.count ) );
    /* signal that count thefts are waiting */
    replycount += kaapi_sched_stealstack( thread, 0, count, kproc->hlrequests.requests );
#else
#warning  "TO REDO"
    /* signal that count thefts are waiting */
    kaapi_threadcontext2stack(thread)->hasrequest = count;
    thread->errcode |= 0x1; /* interrupt the executor flag to request steal... */

    /* busy wait: on return the negative value of correct reply or the ctxt_top is no more the active contexte */
    while ((kaapi_threadcontext2stack(thread)->hasrequest !=0) && (thread == kproc->thread))
    {
      if (kaapi_isterminated()) break;
    }
#endif
  }  
  return 0;
}
