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

void kaapi_sched_idle ( kaapi_processor_t* kproc )
{
  kaapi_victim_t       victim;
  kaapi_reply_t        reply;
  int err;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

#if 0
  /* */
  if (current_proc->tosuspend_thread !=0)
  {
    KAAPI_STACK_PUSH( current_proc->suspended_threads, current_proc->tosuspend_thread );  
    current_proc->tosuspend_thread->thread->state = KAAPI_THREAD_S_SUSPEND;
  }
#endif
    
redo_post:
  /* terminaison ? */
  if (kaapi_isterminated()) goto terminate_program;

  /* try to steal a victim processor */
  err = (*kproc->fnc_select)( kproc, &victim );
  if (err !=0) goto redo_post;

  /* Fill & Post the request to the victim processor */
  kaapi_stack_clear(&kproc->stack);
  kaapi_request_post( kproc, &reply, &victim );

  while (!kaapi_reply_test( &reply ))
  {
    /* here request should be cancelled... */
    kaapi_sched_advance( kproc );
  }
  if (kaapi_isterminated()) return;

  kaapi_assert_debug( kaapi_request_status(&reply) != KAAPI_REQUEST_S_POSTED );

  /* test if my request is ok
  */
  if (!kaapi_reply_ok(&reply)) 
  {
    goto redo_post;
  }
  
  /* Do the local computation
  */
  {
    kaapi_stack_t* stack = kaapi_request_data(&reply);
/*    printf("Thief, 0x%x, pc:0x%x,  #task:%u\n", stack, stack->pc, stack->sp - stack->pc ); */
    err = kaapi_stack_execall( stack );
  }
  if (err == EWOULDBLOCK) 
  {
    /* push the task in a list of suspended stack and and reallocate a new one 
       The C-stack is not saved.
    */
    kaapi_thread_context_t* ctxt = 0;
    kaapi_getcontext( kproc, ctxt );
    KAAPI_STACK_PUSH( &kproc->lsuspend, ctxt );

    /* make a new context */
    kaapi_makecontext( kproc, ctxt, 0, 0 );

    /* set the context on the current processor */
    kaapi_setcontext( kproc, ctxt );
  }
  
  goto redo_post;

terminate_program:

  kaapi_barrier_td_setactive( &kaapi_term_barrier, 0 );  
}
