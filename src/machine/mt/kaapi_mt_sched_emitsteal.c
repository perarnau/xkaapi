/*
** kaapi_mt_sched_idle.c
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

typedef struct kaapi_flatemitsteal_context {
  kaapi_listrequest_t lr;   /* bit map used to post request */
} kaapi_flatemitsteal_context;

/* no concurrency here: always called before starting threads */
int kaapi_sched_flat_emitsteal_init(kaapi_processor_t* kproc)
{
  kaapi_flatemitsteal_context* ctxt;
  kproc->emitsteal_ctxt = ctxt = malloc(sizeof(kaapi_flatemitsteal_context));
  if (kproc->emitsteal_ctxt ==0) return ENOMEM;
  kaapi_listrequest_init( &ctxt->lr );
  return 0;
}


/*
*/
kaapi_request_status_t kaapi_sched_flat_emitsteal ( kaapi_processor_t* kproc )
{
  kaapi_atomic_t               status __attribute__((aligned(8)));
  kaapi_victim_t               victim;
  kaapi_request_t*             self_request;
  int                          err;
  kaapi_listrequest_iterator_t lri;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->thread !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );
  
  if (kaapi_count_kprocessors <2) return KAAPI_REQUEST_S_NOK;
  
  /* allocate thief task data on the stack */
  kproc->thief_task = 0;
  
redo_select:
  /* select the victim processor */
  err = (*kproc->fnc_select)( kproc, &victim, KAAPI_SELECT_VICTIM );
  if (unlikely(err !=0)) goto redo_select;
  /* never pass by this function for a processor to steal itself */
  if (kproc == victim.kproc) return KAAPI_REQUEST_S_NOK;
  kaapi_assert_debug( (victim.kproc->kid >=0) && (victim.kproc->kid <kaapi_count_kprocessors));

  /* mark current processor as stealing */
  kproc->issteal = 1;

  /* (1) 
     Fill & Post the request to the victim processor 
  */
  kaapi_flatemitsteal_context* victim_stealctxt 
    = (kaapi_flatemitsteal_context*)victim.kproc->emitsteal_ctxt;
    
  self_request = &kaapi_requests_list[kproc->kid];
  kaapi_assert_debug( self_request->ident == kproc->kid );
  kaapi_request_post( 
    &victim_stealctxt->lr,
    self_request,
    &status, 
    &kproc->thread->stealreserved_task, 
    &kproc->thread->stealreserved_arg
  );
  
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQ);
#endif

  /* (2)
     lock and re-test if they are yet posted requests on victim or not 
     if during tentaive of locking, a reply occurs, then return with reply
  */
#if defined(KAAPI_SCHED_LOCK_CAS)
  while (!kaapi_sched_trylock( victim.kproc ))
  {
    if (kaapi_reply_test( reply ) ) 
      goto return_value;

#if defined(KAAPI_USE_NETWORK)
    kaapi_network_poll();
#endif
    kaapi_slowdown_cpu();
  }
#else /* cannot rely on kaapi_sched_trylock... */
acquire:
  if (KAAPI_ATOMIC_DECR(&victim.kproc->lock) ==0) goto enter;
  while (KAAPI_ATOMIC_READ(&victim.kproc->lock) <=0)
  {
    if (kaapi_request_status_test( &status )) 
      goto return_value;
#if defined(KAAPI_USE_NETWORK)
    kaapi_network_poll();
#endif
    kaapi_slowdown_cpu();
  }
  goto acquire;
enter:
#endif

#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) !=0 );
#endif
  /* here becomes an aggregator... the trylock has synchronized memory */
  kaapi_listrequest_iterator_init(&victim_stealctxt->lr, &lri);
#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) !=0 );
#endif

  /* (3)
     process all requests on the victim kprocessor and reply failed to remaining requests
     Warning: In this version the aggregator has a lock on the victim processor.
  */
  if (!kaapi_listrequest_iterator_empty(&lri) ) 
  {
    kaapi_request_t* request;

#if defined(KAAPI_SCHED_LOCK_CAS)
    kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) !=0 );
#endif
    kaapi_sched_stealprocessor( victim.kproc, &victim_stealctxt->lr, &lri );
#if defined(KAAPI_SCHED_LOCK_CAS)
    kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) !=0 );
#endif

    /* reply failed for all others requests */
    request = kaapi_listrequest_iterator_get( &victim_stealctxt->lr, &lri );
    kaapi_assert_debug( !kaapi_listrequest_iterator_empty(&lri) || (request ==0) );
    
    while (request !=0)
    {
      kaapi_request_replytask(request, KAAPI_REQUEST_S_NOK);
      request = kaapi_listrequest_iterator_next( &victim_stealctxt->lr, &lri );
      kaapi_assert_debug( !kaapi_listrequest_iterator_empty(&lri) || (request ==0) );
    }
  }

  /* unlock the victim kproc after processing the steal operation */
  kaapi_sched_unlock( &victim.kproc->lock );

  if (kaapi_request_status_test( &status ))
    goto return_value;
  
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALOP);
#endif

  kproc->issteal = 0;
  return KAAPI_REQUEST_S_NOK;
  
return_value:
  /* mark current processor as no stealing anymore */
  kproc->issteal = 0;
  kaapi_assert_debug( (kaapi_request_status_get(&status) != KAAPI_REQUEST_S_POSTED) ); 

  /* test if my request is ok */
  kaapi_request_syncdata( self_request );

  switch (kaapi_request_status_get(&status))
  {
    case KAAPI_REQUEST_S_OK:
      kproc->thief_task = self_request->thief_task;
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_SUCCESS );
      return KAAPI_REQUEST_S_OK;

    case KAAPI_REQUEST_S_NOK:
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_FAILED );
      return KAAPI_REQUEST_S_NOK;

    case KAAPI_REQUEST_S_ERROR:
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_ERROR );
      return KAAPI_REQUEST_S_ERROR;

    default:
      kaapi_assert_debug_m(0, "Bad request status" );
  }
  
  return KAAPI_REQUEST_S_NOK;  
}

