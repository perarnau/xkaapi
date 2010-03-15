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

kaapi_thread_context_t* kaapi_sched_emitsteal ( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* retval;
  kaapi_victim_t          victim;
  int i, replycount, err, ok;
  int count;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->thread !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

    /* clear thief stack/thread that will receive tasks */
  kaapi_thread_clear( kproc->thread );

redo_select:
  /* select the victim processor */
  err = (*kproc->fnc_select)( kproc, &victim );
  if (err !=0) goto redo_select;
  /* never pass by this function for a processor to steal itself */
  if (kproc == victim.kproc) return 0;

  /* mark current processor as stealing */
  kproc->issteal = 1;

  /* (1) 
     Fill & Post the request to the victim processor 
  */
  replycount = 0;
  kaapi_request_post( kproc, &kproc->reply, &victim );
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQ);
#endif
  kaapi_assert_debug( KAAPI_ATOMIC_READ( &victim.kproc->hlrequests.count ) < KAAPI_MAX_PROCESSOR );

#if 0
  fprintf(stdout,"%i kproc post steal to:%p\n", kproc->kid, (void*)victim.kproc );
  fflush(stdout);
#endif

#if 0 /* experimental, release CPU */
//  pthread_yield();
#endif

  /* (2)
     lock and retest if they are yet posted requests on victim or not 
     if during tentaive of locking, a reply occurs, then return with reply
  */
#if 0 /* experimental, release CPU */
  int counter;
#endif

  while (1)
  {
    /* if lock sucess then steal of all processors in the request array */
    ok = KAAPI_ATOMIC_CAS(&victim.kproc->lock, 0, 1+kproc->kid);
    if (ok) break;
//TODO    if (kproc->hasrequest) kproc->thread->hasrequest = 0;   /* current stack never accept steal request */

    /* here is not yet an exponential backoff, but the cas should not
       to busy to avoid memory transaction: do multiple tests on the reply field
    */
    for (i=0; i<10; ++i)
    {
      if (kaapi_reply_test( &kproc->reply ) ) 
        /* return with out trying to lock / unlock the victim: 
           an other processor or myself has replied 
        */
        goto return_value;

  #if 0 /* experimental but should release CPU */
      if ((counter & 0xFF) ==0) {
        counter =0;
        /*pthread_yield();*/
      }
  #endif
    }
  }
#if 0
  fprintf(stdout,"%i kproc enter critical section to:%p\n", kproc->kid, (void*)victim.kproc );
  fflush(stdout);
#endif

  kaapi_assert_debug( ok );
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) == 1+kproc->kid );
  
  count = KAAPI_ATOMIC_READ( &victim.kproc->hlrequests.count );
  /* here the current processor may see different value of the memory (no sequential consistency) :
     - count >0 but array of requests empty: there is no memory barrier between write 
     of the status of the request and the increment (see request_post)
     - count ==0 but the array of requests is not empty: if the write of the request status and 
     atomic increment is reorder in request_post.
     The only valid assumption is that:
       (status == POSTED) => data fields of the request are correctly set
  */
  kaapi_assert_debug( (0 <= count) && (count < KAAPI_MAX_PROCESSOR) );

  /* (3)
     process all requests on the victim kprocessor and reply failed to remaining requests
  */
  if (count >0) 
  {
    kaapi_readmem_barrier();
    kaapi_sched_stealprocessor( victim.kproc );
#if defined(KAAPI_USE_PERFCOUNTER)
    ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALOP);
#endif
  }
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) == 1+kproc->kid );

  /* reply at least to my request if not replied
  */
  if (!kaapi_reply_test( &kproc->reply ))
  {
    kaapi_assert_debug( kaapi_request_ok( &victim.kproc->hlrequests.requests[ kproc->kid ] ) )
    kaapi_assert_debug( victim.kproc->hlrequests.requests[kproc->kid].proc == victim.kproc);
#if 0
    fprintf(stdout,"%i kproc reply failed to:%p, @req=%p\n", kproc->kid, (void*)victim.kproc, (void*)&victim.kproc->hlrequests.requests[i] );
    fflush(stdout);
#endif
    _kaapi_request_reply( &victim.kproc->hlrequests.requests[kproc->kid], 0, 0 );
  }

#if 0
  /* reply to all requests ??? 
  */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {
    if (kaapi_request_ok(&victim.kproc->hlrequests.requests[i]))
    {
#if 0
      fprintf(stdout,"%i kproc reply to:%p, @req=%p\n", kproc->kid, (void*)victim.kproc, (void*)&victim.kproc->hlrequests.requests[i] );
      fflush(stdout);
#endif
      /* user version that do not decrement the counter */
      kaapi_assert_debug( victim.kproc->hlrequests.requests[i].proc == victim.kproc);
      _kaapi_request_reply( &victim.kproc->hlrequests.requests[i], 0, 0 );
      ++replycount;
    }
  }
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) == 1+kproc->kid );
#endif

  /* unlock  */ 
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim.kproc->lock) == 1+kproc->kid );  
#if 0
  fprintf(stdout,"%i kproc leave critical section to:%p\n", kproc->kid, (void*)victim.kproc );
  fflush(stdout);
#endif
  KAAPI_ATOMIC_WRITE(&victim.kproc->lock, 0);

  /* */
  kaapi_assert_debug(kaapi_reply_test( &kproc->reply ));

return_value:
  /* mark current processor as no stealing */
  kproc->issteal = 0;

  kaapi_assert_debug( kaapi_request_status(&kproc->reply) != KAAPI_REQUEST_S_POSTED );

  /* test if my request is ok
  */
  if (!kaapi_reply_ok(&kproc->reply))
  {
    return 0;
  }
#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
  
  /* get the work (stack) and return it
  */
  retval = kaapi_request_data(&kproc->reply);

  return retval;
}
