/*
** kaapi_sched_idle.c
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

/* \TODO: utiliser ici les fonctions (a adapter si besoin): makecontext, getcontext, setcontext 
   stack: objet de retour du vol
   context: associe au flot d'execution, ie instancier a partir d'une stack ???
*/
void kaapi_sched_idle ( kaapi_processor_t* kproc )
{
  kaapi_thread_context_t* ctxt;
  kaapi_stack_t* stack;
  int err;
  
  double t0;
  double t1;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( kproc == _kaapi_get_current_processor() );

  /* push it into the free list */
  
  do {
    /* terminaison ? */
    if (kaapi_isterminated()) break;

    /* local wake up first */
    ctxt = kaapi_sched_wakeup(kproc); 
    if (ctxt !=0) 
    {
      /* push kproc context into free list */
      KAAPI_LOG(50, "[IDLE] free ctxt 0x%x\n", ctxt);
      KAAPI_STACK_PUSH( &kproc->lfree, kproc->ctxt );

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
      goto redo_execute;
    }
    
    /* steal request */
    stack = kaapi_sched_emitsteal( kproc );
    if (kaapi_stack_isempty(stack)) continue;
    kaapi_assert_debug( stack != 0);
    
    if (stack != kproc->ctxt)
    {
      ctxt = kproc->ctxt;
      kproc->ctxt = 0;

      KAAPI_LOG(50, "[IDLE] free ctxt 0x%x\n", ctxt);
      /* push it into the free list */
      KAAPI_STACK_PUSH( &kproc->lfree, ctxt );

      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, stack);
    }

redo_execute:
    t0 = kaapi_get_elapsedtime();
    /* printf("Thief, 0x%x, pc:0x%x,  #task:%u\n", stack, stack->pc, stack->sp - stack->pc ); */
    KAAPI_LOG(50, "[IDLE] execute ctxt 0x%x\n", kproc->ctxt);
    err = kaapi_stack_execall( kproc->ctxt );
    t1 = kaapi_get_elapsedtime();
    KAAPI_LOG(50, "[IDLE] Work for %fs\n", t1-t0);

    if (err == EWOULDBLOCK) 
    {
      kaapi_thread_context_t* ctxt = kproc->ctxt;
      kproc->ctxt = 0;

      KAAPI_LOG(50, "[IDLE] suspend ctxt 0x%x\n", ctxt);
      /* push it: suspended because top task is not ready */
      KAAPI_STACK_PUSH( &kproc->lsuspend, ctxt );

      kproc->ctxt = kaapi_sched_wakeup(kproc); 
      if (kproc->ctxt !=0) goto redo_execute;

      /* else reallocate a context */
      ctxt = kaapi_context_alloc(kproc);
      /* set new context to the kprocessor */
      kaapi_setcontext(kproc, ctxt);
    }
  } while (1);
  
}