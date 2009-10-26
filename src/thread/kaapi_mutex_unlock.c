/*
** kaapi_mutex_unlock.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:16 2009
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


/**
*/
int kaapi_mutex_unlock(kaapi_mutex_t *mutex)
{
  kaapi_t thread = kaapi_self();
  
  kaapi_assert ( 0 == pthread_mutex_lock (&mutex->_mutex) );
  
  if ((mutex->_type == KAAPI_MUTEX_RECURSIVE) && (mutex->_owner == thread))
  {
    if (mutex->_nb_lock > 1)
    { 
      mutex->_nb_lock--;
      kaapi_assert ( 0 == pthread_mutex_unlock (&mutex->_mutex) );
      return 0;
    }
    else
    {
      mutex->_owner = NULL;
      mutex->_nb_lock = 0;
    }
  }
  
  if (KAAPI_QUEUE_EMPTY (mutex))
  {
    KAAPI_ATOMIC_WRITE( &mutex->_lock, 0);
    
    kaapi_assert ( 0 == pthread_mutex_unlock (&mutex->_mutex) );
    
    return 0;
  }
  
  KAAPI_QUEUE_POP_FRONT (mutex, thread);
  thread->_state = KAAPI_THREAD_S_RUNNING;
  if (mutex->_type == KAAPI_MUTEX_RECURSIVE)
  { 
    mutex->_owner = thread;
  }
  
  
  /* Only for SYSTEM_SCOPE threads to wakeup the kernel thread */
  if (thread->_scope == KAAPI_SYSTEM_SCOPE)
    kaapi_assert ( 0 == pthread_cond_signal (&thread->th.s._cond) );

  kaapi_assert ( 0 == pthread_mutex_unlock (&mutex->_mutex) ); 
  
  return 0;
}
