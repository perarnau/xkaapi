/*
** kaapi_cond_timedwait.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:43 2009
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

static int kaapi_timed_test_and_lock_mutex (void *arg)
{
  kaapi_timed_test_and_lock__t* kttl = (kaapi_timed_test_and_lock__t*)arg;

  if (kttl->retval == EINVAL)
  {
    /* get the current date may be by using clock_gettime(CLOCK_REALTIME, &now) (not on darwin) */
    struct timeval now;
    gettimeofday(&now,0);
    if (now.tv_sec > kttl->abstime->tv_sec) {
      kttl->retval = ETIMEDOUT;
      return 1;
    }
    else if (now.tv_sec < kttl->abstime->tv_sec) return 0; 
    else if (now.tv_usec > (kttl->abstime->tv_nsec / 1000))
    {
      kttl->retval = ETIMEDOUT;
      return 1;
    }
    else return 0;
  }
  
  return 1;
}

int kaapi_cond_timedwait(kaapi_cond_t *__restrict cond, kaapi_mutex_t *__restrict mutex, const struct timespec *__restrict abstime)
{
  kaapi_t thread = kaapi_self ();
  
  if (thread->_scope == KAAPI_SYSTEM_SCOPE)
  {
    int err = 0;
    
    xkaapi_assert ( 0 == pthread_mutex_lock (&cond->_mutex) );
    
    kaapi_mutex_unlock (mutex);
    
    thread->_state = KAAPI_THREAD_SUSPEND;
    KAAPI_QUEUE_PUSH_FRONT(&cond->_th_q, thread);
    
    while (thread->_state != KAAPI_THREAD_RUNNING)
    {
      err = pthread_cond_timedwait (&thread->_cond, &cond->_mutex, abstime);
      if (err == ETIMEDOUT) 
      {
        thread->_state = KAAPI_THREAD_RUNNING;
        KAAPI_QUEUE_REMOVE(&cond->_th_q, thread);
        xkaapi_assert (0 == pthread_mutex_unlock (&cond->_mutex));
        kaapi_mutex_lock (mutex);
        return err;
      }
    }
    xkaapi_assert (err == 0);
    kaapi_mutex_lock (mutex);
    xkaapi_assert (0 == pthread_mutex_unlock (&cond->_mutex));
    
    return 0;
  }
  
  // opérations atomic?
  kaapi_mutex_unlock (mutex);
  
  /* ICI: revoir la gestion des threads en attente: un signal qui reveil un thread suspendu
     devrait pouvoir aussi le deplacer dans sa liste des threads ready...
  */
  kaapi_timed_test_and_lock__t kttl;
  kttl.thread  = thread;
  kttl.abstime = abstime;
  kttl.mutex   = mutex;
  kttl.retval  = EINVAL;
  KAAPI_QUEUE_PUSH_FRONT (&cond->_kttl_q, &kttl);
  kaapi_sched_suspend (thread->_proc, thread, &kaapi_timed_test_and_lock_mutex, &kttl );
  return kttl.retval;
}