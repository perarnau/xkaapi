/*
** kaapi_mutex_lock.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:32 2009
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

static int kaapi_test_and_lock_mutex (void *arg)
{
  kaapi_test_and_lock__t* ktl = (kaapi_test_and_lock__t*)arg;
  
  if (KAAPI_ATOMIC_CAS( &ktl->mutex->_lock, 0, 1 ))
  {
    if (ktl->mutex->_type == KAAPI_MUTEX_RECURSIVE)
    {
      ktl->mutex->_owner = ktl->thread;
      ktl->mutex->_nb_lock++;
    }
    
    return 1;
  }

  return 0;
}


int kaapi_mutex_lock (kaapi_mutex_t *mutex)
{
  kaapi_thread_descr_t* thread = kaapi_self_internal();
  kaapi_assert_debug( (thread->scope == KAAPI_SYSTEM_SCOPE)||(thread->scope == KAAPI_PROCESSOR_SCOPE) );
  
  if ((mutex->_type == KAAPI_MUTEX_RECURSIVE) && (mutex->_owner == thread))
  {
    mutex->_nb_lock++;
    return 0;
  }
  
  /* try to lock */
  if (KAAPI_ATOMIC_CAS (&mutex->_lock, 0, 1))
  {
    if (mutex->_type == KAAPI_MUTEX_RECURSIVE)
    { 
      mutex->_owner = thread;
      mutex->_nb_lock++;     /* TG <=> set to 1 ? */
    }
    return 0;
  }
  
  /* cannot lock: put myself into the queue of the mutex */
  if (thread->scope == KAAPI_SYSTEM_SCOPE)
  {
    kaapi_assert( 0 == pthread_mutex_lock (&mutex->_mutex) );
    
    /* retry with critical section among system threads */
    if (KAAPI_ATOMIC_CAS (&mutex->_lock, 0, 1))
    {
      if (mutex->_type == KAAPI_MUTEX_RECURSIVE)
      { 
        mutex->_owner = thread;
        mutex->_nb_lock++;
      }
      kaapi_assert( 0 == pthread_mutex_unlock (&mutex->_mutex) );
      return 0;
    }

    /* link cell is allocated on the stack */
    { kaapi_cellsuspended_t cell;
      cell.thread = thread;
      cell.f_test_wakeup = 0;
      cell.f_wakeup = 0;
      thread->state = KAAPI_THREAD_S_SUSPEND;

      KAAPI_FIFO_PUSH( &mutex->_list, &cell);
        
      while (thread->state != KAAPI_THREAD_S_RUNNING)
        kaapi_assert( 0 == pthread_cond_wait (&thread->th.s.cond, &mutex->_mutex) );
    }
    
    if (mutex->_type == KAAPI_MUTEX_RECURSIVE)
    { 
      mutex->_owner = thread;
      mutex->_nb_lock++;
    }
    kaapi_assert( 0 == pthread_mutex_unlock (&mutex->_mutex) );
    
    return 0;
  }

  { /* new lexical scope to allow declaration here */
    kaapi_cellsuspended_t cell;
    cell.thread        = 0;         /* should put the user thread ! */
    cell.f_test_wakeup = &kaapi_test_and_lock_mutex;
    cell.f_wakeup      = 0;
    cell.arg_fwakeup   = mutex;
    kaapi_sched_suspend (thread, &cell );
  }

  return 0;
}


/* TG: better algorithm with exponential backoff */
int kaapi_mutex_spinlock (kaapi_mutex_t *mutex)
{
  while (!KAAPI_ATOMIC_CAS (&mutex->_lock, 0, 1)) sched_yield();

  return 0;
}
