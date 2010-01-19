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

kaapi_stack_t* kaapi_sched_emitsteal ( kaapi_processor_t* kproc )
{
  kaapi_stack_t*       stack;
  kaapi_victim_t       victim;
  int err;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc->ctxt !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

  /* */
  if (!KAAPI_STACK_EMPTY(&kproc->lsuspend))
  {
    /* TODO try top wakeup a waiting stack ? */  
  }
  
  /* clear stack */
  kaapi_stack_clear( kproc->ctxt );

redo_select:
  /* try to steal a victim processor */
  err = (*kproc->fnc_select)( kproc, &victim );
  if (err !=0) goto redo_select;

  /* mark current processor as stealing */
  kproc->issteal = 1;

  /* Fill & Post the request to the victim processor */
  kaapi_request_post( kproc, &kproc->reply, &victim );
  /*printf("%i:: Post to kproc=%i\n", kproc->kid, victim.kproc->kid );*/

#if defined(KAAPI_CONCURRENT_WS)
  while (!kaapi_reply_test( &kproc->reply ))
  {
    kaapi_sched_advance( victim.kproc );
  }

#else /* COOPERATIVE */

  while (!kaapi_reply_test( &kproc->reply ))
  {
    /* here request should be cancelled... */
    kaapi_sched_advance( kproc );
    if (kaapi_isterminated()) 
    {
      kproc->issteal = 0;
      return 0;
    }
    if (!kaapi_reply_test( &kproc->reply ))
#if defined(__linux__)
      pthread_yield();
#elif defined(__APPLE__)
      pthread_yield_np();
#endif
  }
#endif
  /* mark current processor as no stealing */
  kproc->issteal = 0;

  kaapi_assert_debug( kaapi_request_status(&kproc->reply) != KAAPI_REQUEST_S_POSTED );

  /* test if my request is ok
  */
  if (!kaapi_reply_ok(&kproc->reply))
  {
    return 0;
  }
  
  /* Reset original ctxt and do the local computation
  */
  stack = kaapi_request_data(&kproc->reply);

  return stack;
}
