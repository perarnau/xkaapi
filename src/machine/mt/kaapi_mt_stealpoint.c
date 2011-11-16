/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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

#define KAAPI_USE_AGGREGATION

#if 0

#if defined(KAAPI_USE_AGGREGATION)
int kaapi_sched_stealstack_helper( kaapi_stealcontext_t* stc )
{
  kaapi_thread_context_t*	self_thread;
  kaapi_processor_t*      kproc;

  kaapi_listrequest_t*    victim_hlr;
  kaapi_listrequest_iterator_t lri;
  
  self_thread = stc->s_thread;
  kproc = self_thread->proc;
    
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->thread !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );

  victim_hlr = &kproc->hlrequests;

#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQ);
#endif

#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->lock) !=0 );
#endif
  /* here becomes an aggregator... the trylock has synchronized memory */
  kaapi_listrequest_iterator_init(victim_hlr, &lri);
#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->lock) !=0 );
#endif

  /* (3)
     process all requests on the victim kprocessor and reply failed to remaining requests
     Warning: In this version the aggregator has a lock on the victim processor.
  */
  if (!kaapi_listrequest_iterator_empty(&lri) ) 
  {
#if defined(KAAPI_DEBUG)
    int count_req = kaapi_listrequest_iterator_count(&lri);
    kaapi_assert( (count_req >0) );
    kaapi_bitmap_value_t savebitmap = (kaapi_bitmap_value_t)(lri.bitmap | (1UL << lri.idcurr));
    for (int i=0; i<count_req; ++i)
    {
      int firstbit = kaapi_bitmap_first1_and_zero( &savebitmap );
      kaapi_assert( firstbit != 0);
      kaapi_assert( victim_hlr->requests[firstbit-1].reply != 0 );
    }
#endif  

#if defined(KAAPI_SCHED_LOCK_CAS)
    kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->lock) !=0 );
#endif
    kaapi_sched_stealprocessor( kproc, victim_hlr, &lri );
#if defined(KAAPI_SCHED_LOCK_CAS)
    kaapi_assert_debug( KAAPI_ATOMIC_READ(&kproc->lock) !=0 );
#endif

    /* reply failed for all others requests */
    kaapi_request_t* request = kaapi_listrequest_iterator_get( victim_hlr, &lri );
    kaapi_assert_debug( !kaapi_listrequest_iterator_empty(&lri) || (request ==0) );

    while (request !=0)
    {
      _kaapi_request_reply(request, KAAPI_REPLY_S_NOK);
      request = kaapi_listrequest_iterator_next( victim_hlr, &lri );
      kaapi_assert_debug( !kaapi_listrequest_iterator_empty(&lri) || (request ==0) );
    }
    
    return 1;
  }

#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALOP);
#endif

  return 0;  
}

#else // KAAPI_USE_AGGREGATION
#error "Should use aggregation ! else not implemented"
#endif

#endif