/*
** xkaapi
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com 
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
#if defined(KAAPI_USE_NUMA)
  kproc->emitsteal_ctxt = ctxt = numa_alloc_local(sizeof(kaapi_flatemitsteal_context));
#else
  kproc->emitsteal_ctxt = ctxt = malloc(sizeof(kaapi_flatemitsteal_context));
#endif
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
  kaapi_request_t*             request;
  int                          err;
  kaapi_listrequest_iterator_t lri;

#if defined(KAAPI_USE_PERFCOUNTER)
  uintptr_t serial;
#endif

  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->thread !=0 );
  kaapi_assert_debug( kproc == kaapi_get_current_processor() );
  
  
  if (kproc->mailbox.head != 0 )
  {
    kaapi_task_withlink_t* taskwl;

    /* pop the first item at kproc->mailbox.head
       lock free MPSC FIFO queue must be used here.
    */
    kaapi_atomic_lock(&kproc->lock);
    taskwl = kproc->mailbox.head;
    if (kproc->mailbox.tail == taskwl)
      kproc->mailbox.tail = 0;
    kproc->mailbox.head = taskwl->next;
    kaapi_atomic_unlock(&kproc->lock);
    
    /* push the task into local queue */
    *kaapi_thread_toptask(&kproc->thread->stack.stackframe[0]) = taskwl->task;
    kaapi_thread_pushtask(&kproc->thread->stack.stackframe[0]);

    return KAAPI_REQUEST_S_OK;
  }
  
  if (kaapi_count_kprocessors <2) 
    return KAAPI_REQUEST_S_NOK;
    
redo_select:
  /* select the victim processor */
  err = (*kproc->fnc_select)( kproc, &victim, KAAPI_SELECT_VICTIM );
  if (unlikely(err !=0)) 
  {
    if (kaapi_isterm) return 0;
    goto redo_select;
  }

#if 1 // TG: test, steal also allow to steal stack from myself. Else only wakeup
  /* never pass by this function for a processor to steal itself */
  if (kproc == victim.kproc) 
    return KAAPI_REQUEST_S_NOK;
#endif

  /* quick test to detect if thread has no work */
  if (kaapi_processor_has_nowork(victim.kproc))
  {
    (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_FAILED );
    goto redo_select;
  }
  kaapi_assert_debug( (victim.kproc->kid >=0) && (victim.kproc->kid <kaapi_count_kprocessors));

#if 0
  fprintf(stdout, "[%s] kid=%lu kvictim=%lu\n", 
	  __FUNCTION__,
	    (long unsigned int)kaapi_get_current_kid(),
	    (long unsigned int)victim.kproc->kid
	  );
  fflush(stdout);
#endif

  /* (1) 
     Fill & Post the request to the victim processor 
  */
  kaapi_flatemitsteal_context* victim_stealctxt 
    = (kaapi_flatemitsteal_context*)victim.kproc->emitsteal_ctxt;
    
  kaapi_stack_reset( &kproc->thread->stack );

  self_request = &kaapi_global_requests_list[kproc->kid];
  kaapi_assert_debug( self_request->ident == kproc->kid );

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_IFUSE_TRACE(kproc,
    self_request->victim = (uintptr_t)victim.kproc->kid;
    serial = ++kproc->serial;
    self_request->serial = serial;
    KAAPI_EVENT_PUSH2(kproc, 0, KAAPI_EVT_STEAL_OP, 
        (uintptr_t)victim.kproc->kid, 
        self_request->serial
    );
  );
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQ);
#endif

  KAAPI_DEBUG_INST(kproc->victim_kproc = victim.kproc;)
  kaapi_request_post( 
    &victim_stealctxt->lr,
    self_request,
    &status, 
    &kproc->thread->stack.stackframe[0] 
  );
  

  /* (2)
     lock and re-test if they are yet posted requests on victim or not 
     if during tentaive of locking, a reply occurs, then return with reply
  */
  while (!kaapi_sched_trylock( &victim.kproc->lock ))
  {
    if (kaapi_request_status_test( &status )) 
      goto return_value;

#if defined(KAAPI_USE_NETWORK)
    kaapi_network_poll();
#endif
    kaapi_slowdown_cpu();
  }
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_EVENT_PUSH2(kproc, 0, KAAPI_EVT_REQUESTS_BEG, 
                    (uintptr_t)victim.kproc->kid, serial );
#endif

  /* here becomes an aggregator... the trylock has synchronized memory */
  kaapi_listrequest_iterator_init(&victim_stealctxt->lr, &lri);

  /* (3)
     process all requests on the victim kprocessor and reply failed to remaining requests
     Warning: In this version the aggregator has a lock on the victim processor 
     steal context (i.e. the list of requests).
  */
  if (!kaapi_listrequest_iterator_empty(&lri) ) 
  {
#if defined(KAAPI_USE_PERFCOUNTER)
    kaapi_assert_debug( sizeof(kaapi_atomic64_t) <= sizeof(kaapi_perf_counter_t) );
    KAAPI_ATOMIC_ADD64( 
      (kaapi_atomic64_t*)&KAAPI_PERF_REG(victim.kproc, KAAPI_PERF_ID_STEALIN),
      kaapi_listrequest_iterator_count(&lri)
    );
#endif
    kaapi_sched_stealprocessor( victim.kproc, &victim_stealctxt->lr, &lri );

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

  KAAPI_DEBUG_INST(kproc->victim_kproc = 0;)

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_EVENT_PUSH2(kproc, 0, KAAPI_EVT_REQUESTS_END, 
            (uintptr_t)victim.kproc->kid, serial
  );
#endif
  /* unlock the victim kproc after processing the steal operation */
  kaapi_sched_unlock( &victim.kproc->lock );

  if (kaapi_request_status_test( &status ))
    goto return_value;

  /* my request may not have been replied BUT have to leave this function
     to test for runtime finalization, among other things. Thus, have to
     wait for my own reply.
   */
  while (1)
  {
    if (kaapi_request_status_test(&status)) 
      goto return_value;

    /* runtime may be finalizing. in this case,
       cancel the request and leave. note that
       some might be replying, in which case I
       have to wait for the reply.
     */
    if (kaapi_isterm)
    {
      const int err = kaapi_bitmap_unset
         ( &victim_stealctxt->lr.bitmap, (int)self_request->ident );

      /* bit unset by me, no one saw my request and leaving is safe */
      if (err == 0) return KAAPI_REQUEST_S_NOK;

      /* otherwise, wait for the reply */
      /* note: could be optimized a bit, but finalization code */

      /* FIXME
	   should wait for the request but does not work. maybe
	   the request bitmap is being destroyed.
       */
      return KAAPI_REQUEST_S_NOK;
      /* FIXME */
    }
  }
  
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALOP);
#endif

  return KAAPI_REQUEST_S_NOK;
  
return_value:
  /* mark current processor as no stealing anymore */
  kaapi_assert_debug( (kaapi_request_status_get(&status) != KAAPI_REQUEST_S_POSTED) ); 

  /* test if my request is ok */
  kaapi_request_syncdata( self_request );

  switch (kaapi_request_status_get(&status))
  {
    case KAAPI_REQUEST_S_OK:
       /* Currently only verify that the returned pointer are into the area given by stackframe[0]
          - Else it means that the result of the steal are tasks that lies outside the stackframe,
          then the start up in that case should be different:
            - first, execute the task between [req->frame.pc and req->frame.sp] in the context
            of the current stack of frame (i.e. pushed task must be pushed locally).
          This requires 1/ change the interface of execframe in order to add pc, sp in signature.
          2/ change the callee to execframe 3/ suppress this comment.
       */
      kaapi_assert_debug( ((uintptr_t)self_request->frame.sp <= (uintptr_t)kproc->thread->stack.stackframe[0].pc)
                       && ((uintptr_t)self_request->frame.sp >= (uintptr_t)kproc->thread->stack.stackframe[0].sp_data)
                       && ((uintptr_t)self_request->frame.sp_data >= (uintptr_t)kproc->thread->stack.stackframe[0].sp_data)
                       && ((uintptr_t)self_request->frame.sp_data <= (uintptr_t)kproc->thread->stack.stackframe[0].sp)
      );
      /* also assert that pc does not have changed (only used at runtime to execute task) */
      kaapi_assert_debug( kproc->thread->stack.stackframe[0].pc == self_request->frame.pc);
      kproc->thread->stack.stackframe[0].sp_data = self_request->frame.sp_data;
      kaapi_writemem_barrier();
      kproc->thread->stack.stackframe[0].sp = self_request->frame.sp;
        
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_SUCCESS );
#if defined(KAAPI_USE_PERFCOUNTER)
      KAAPI_EVENT_PUSH3(kproc, 0, KAAPI_EVT_RECV_REPLY, 
                        self_request->victim, self_request->serial, 1 );
#endif
      return KAAPI_REQUEST_S_OK;

    case KAAPI_REQUEST_S_NOK:
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_FAILED );
#if defined(KAAPI_USE_PERFCOUNTER)
      KAAPI_EVENT_PUSH3(kproc, 0, KAAPI_EVT_RECV_REPLY, 
                        self_request->victim, self_request->serial, 0 );
#endif
      return KAAPI_REQUEST_S_NOK;

    case KAAPI_REQUEST_S_ERROR:
      (*kproc->fnc_select)( kproc, &victim, KAAPI_STEAL_ERROR );
#if defined(KAAPI_USE_PERFCOUNTER)
      KAAPI_EVENT_PUSH3(kproc, 0, KAAPI_EVT_RECV_REPLY, 
                        self_request->victim, self_request->serial, 0 );
#endif
      return KAAPI_REQUEST_S_ERROR;

    default:
      kaapi_assert_debug_m(0, "Bad request status" );
  }
  
  return KAAPI_REQUEST_S_NOK;  
}

