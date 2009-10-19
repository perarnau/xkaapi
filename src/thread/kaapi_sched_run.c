/*
** ckaapi
** 
** Created on Tue Mar 31 15:17:57 2009
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

void* kaapi_sched_run_processor( void* arg )
{
  kaapi_processor_t* proc = (kaapi_processor_t*)arg;
  ckaapi_assert( proc != 0);
  
  ckaapi_assert ( 0 == pthread_setspecific( kaapi_current_processor_key, proc ) );
  
  /* change the scope of the running thread to be a KAAPI_PROCESSOR_SCOPE */
  kaapi_thread_descr_t* td = (kaapi_thread_descr_t*)pthread_getspecific(kaapi_current_thread_key);
  td->_proc  = proc;

#if defined(KAAPI_USE_UCONTEXT)
  getcontext( &proc->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _setjmp( proc->_ctxt );
#endif
  if (kaapi_stealapi_term) return 0;
  kaapi_sched_idle(proc);
  return 0;
}


/*
*/
kaapi_processor_t* kaapi_sched_get_processor()
{
  kaapi_processor_t* proc = 0;
  proc = pthread_getspecific( kaapi_current_processor_key );
  return proc;
}
