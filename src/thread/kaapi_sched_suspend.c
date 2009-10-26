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

int kaapi_sched_suspend ( kaapi_thread_descr_processor_t* proc, kaapi_thread_descr_t* thread, kaapi_test_wakeup_t fwakeup, void* argwakeup )
{
  int previous_errno = errno;
  
  kaapi_assert_debug( proc !=0 );
  kaapi_assert_debug( thread != 0 );
  kaapi_assert_debug( thread->_scope == KAAPI_PROCESS_SCOPE );
  kaapi_assert_debug( proc == thread->_proc );
  kaapi_assert_debug( proc->_active_thread == thread );

  proc->_active_thread = 0;
  thread->_state = KAAPI_THREAD_S_SUSPEND;
  
  /* capture the current context of the suspended thread */
#if defined(KAAPI_USE_UCONTEXT)
  getcontext( &thread->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _setjmp( thread->_ctxt );
#endif

  /* case of return from setjmp / getcontext */
  if (thread->_state == KAAPI_THREAD_S_SUSPEND)
  {
    /* put thread in internal list */
    KAAPI_WORKQUEUE_SUSPEND_PUSH( &proc->_suspended_thread, thread, fwakeup, argwakeup );
    kaapi_sched_idle( proc );
  }

  errno = previous_errno;

  return 0;
}
