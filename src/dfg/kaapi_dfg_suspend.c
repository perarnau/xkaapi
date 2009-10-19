/*
** ckaapi
** 
** Created on Tue Mar 31 15:22:28 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#include "kaapi_dfg.h"


static int kaapi_dfg_closure_is_term(void *arg)
{
  kaapi_dfg_closure_t* clo = (kaapi_dfg_closure_t*)arg;
  
  return KAAPI_DFG_CLOSURE_GETSTATE(clo) == KAAPI_CLOSURE_TERM;
}


/** This function is call in place of the normal execution of closure
    if the state is not KAAPI_CLOSURE_INIT, meaning that full data flow evaluation is
    required before continuing the execution.
    The implementation will try to steal work from other processor
    and then switch to the new work.
*/
void kaapi_dfg_suspend(  
  kaapi_dfg_stack_t* s, 
  kaapi_dfg_frame_t* f, 
  kaapi_dfg_closure_t* c
)
{
redo:
  if (KAAPI_DFG_CLOSURE_GETSTATE(c) == KAAPI_CLOSURE_INIT) 
  {
    KAAPI_CLOSURE_EXECUTE( s, f, c );
  }
  else if (KAAPI_DFG_CLOSURE_GETSTATE(c) == KAAPI_CLOSURE_EXEC) 
  {
    return;
  }
  else if (c->_state == KAAPI_CLOSURE_STEAL) 
  { 
    kaapi_t thread = kaapi_self();
    ckaapi_assert( thread->_scope == KAAPI_PROCESS_SCOPE );
    kaapi_sched_suspend ( thread->_proc, thread, &kaapi_dfg_closure_is_term, c );
    ckaapi_assert( KAAPI_DFG_CLOSURE_GETSTATE(c) == KAAPI_CLOSURE_TERM );
  }
  if (KAAPI_DFG_CLOSURE_GETSTATE(c) == KAAPI_CLOSURE_TERM) 
  { /* execute the terminaison code */
    printf("Try to execute the terminaison code term of closure @:0x%x, spin waiting\n", c);
    KAAPI_CLOSURE_EXECUTE( s, f, c );
  }
}
