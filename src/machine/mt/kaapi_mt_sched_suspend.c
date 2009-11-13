/*
** kaapi_sched_suspend.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:01 2009
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

int kaapi_sched_suspend ( kaapi_thread_processor_t* proc, kaapi_cellsuspended_t* cell )
{
  int previous_errno = errno;
  kaapi_thread_descr_process_t* thread;
  
  kaapi_assert_debug( proc !=0 );
  kaapi_assert_debug( cell !=0 );
  kaapi_assert_debug( cell->thread !=0 );
  kaapi_assert_debug( proc->active_thread == cell->thread );

  thread = (kaapi_thread_descr_process_t*)cell->thread;
  
  /* push cell in list of suspended thread */
  proc->tosuspend_thread = cell;

  /* capture the current context of the suspended thread */
  kaapi_getcontext( &thread->ctxt );

  /* to return in case of return from setjmp / getcontext */
  if (thread->state != KAAPI_THREAD_S_SUSPEND) 
  {
    errno = previous_errno;
    return 0;
  }
  
  kaapi_thread_descr_process_t* new_active = 
      kaapi_allocate_thread_descriptor(KAAPI_PROCESS_SCOPE, 1, thread->ctxt.cstacksize, thread->ctxt.kstacksize );
  proc->active_thread = new_active;

  kaapi_makecontext( new_active, kaapi_sched_idle, proc);
  kaapi_setcontext( proc, new_active )

  /* never reach this point */
  kaapi_assert( false );
  return 0;
}
