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
  kaapi_thread_context_t*  ctxt_top;
  int count =0;
  int replycount = 0;

  count = KAAPI_ATOMIC_READ( &kproc->hlrequests.count );
  if (count ==0) return 0;
  
  ctxt_top = KAAPI_STACK_TOP( &kproc->lsuspend );
  while ((ctxt_top !=0) && (count >0))
  {
    replycount += kaapi_sched_stealstack( ctxt_top, 0 );
    count = KAAPI_ATOMIC_READ( &kproc->hlrequests.count );
    ctxt_top = KAAPI_STACK_NEXT_FIELD( ctxt_top );
  }
  
  ctxt_top = kproc->ctxt;
  if ((count >0) && (ctxt_top !=0) && (kproc->issteal ==0))
  {
    /* if concurrent WS, then steal directly the current stack of the victim processor
       else set flag to 0 on the stack and wait reply
    */
#if defined(KAAPI_CONCURRENT_WS)
    replycount += kaapi_sched_stealstack( ctxt_top, 0 );
#else
    /* signal that count thefts are waiting */
    ctxt_top->hasrequest = count;

    /* busy wait: on return the negative value of correct reply or the ctxt_top is no more the active contexte */
    while ((ctxt_top->hasrequest !=0) && (ctxt_top == kproc->ctxt))
    {
      if (kaapi_isterminated()) break;
    }
#endif
  }
  
#if defined(KAAPI_USE_PERFCOUNTER)
  kproc->cnt_stealreq += replycount;
  ++kproc->cnt_stealop;
#endif

  return 0;
}
