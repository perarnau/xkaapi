/*
** kaapi_sched_advance.c
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

int kaapi_advance ( void )
{
#if 1/*defined(KAAPI_CONCURRENT_WS)*/
  /* high level function -> empty macro */
  return 0;
#else
  return kaapi_sched_advance( kaapi_get_current_processor() );
#endif
}


/*
*/
int kaapi_sched_advance ( kaapi_processor_t* kproc )
{
  int count;
  kaapi_stack_t* stack = kaapi_threadcontext2stack(kproc->thread);
  count = stack->hasrequest;
  if (count ==0) return 0;

#if defined(KAAPI_USE_PERFCOUNTER)
  int saved_state = kaapi_perf_thread_state(kproc);
  kaapi_perf_thread_stopswapstart(kproc, KAAPI_PERF_SCHEDULE_STATE );
#endif
#if 0 //TODO
  kaapi_stealpoint_isactive(kproc->thread,0);
#endif
  count = stack->hasrequest;
  
  if (count !=0) 
  {
#if 0
    replycount = 0;
    for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    {
      if (kaapi_request_ok(&kproc->hlrequests.requests[i]))
      {
        /* user version that do not decrement the counter */
        _kaapi_request_reply( kproc->ctxt, 0, &kproc->hlrequests.requests[i], 0, 0, 0 );
        ++replycount;
        if (replycount == count) break;
      }
    }

    /* assert on the counter of victim processor request count */
    if (replycount >0)
    {
      KAAPI_ATOMIC_SUB( &kproc->hlrequests.count, replycount );
      kaapi_assert_debug( KAAPI_ATOMIC_READ( &kproc->hlrequests.count ) >= 0 );
    }
#endif
    stack->hasrequest = 0;
  }
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_perf_thread_stopswapstart(kproc, saved_state );
#endif
  return 0;
}


