/*
** kaapi_cond_signal.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:47 2009
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


int kaapi_cond_signal(kaapi_cond_t *cond)
{
  kaapi_assert ( 0 == pthread_mutex_lock (&cond->_mutex) );

  if (!KAAPI_QUEUE_EMPTY(&cond->_th_q))
  {
    kaapi_t thread;
    if (!KAAPI_QUEUE_EMPTY(&cond->_th_q))
    {
      KAAPI_QUEUE_POP_FRONT(&cond->_th_q, thread);
      kaapi_assert ( 0 == pthread_mutex_unlock (&cond->_mutex) );
      thread->_state      = KAAPI_THREAD_S_RUNNING;
    }
  }
  else if (!KAAPI_QUEUE_EMPTY(&cond->_kttl_q))
  {
    kaapi_timed_test_and_lock__t *kttl;
    KAAPI_QUEUE_POP_FRONT(&cond->_kttl_q, kttl);
    
    if (kttl->abstime ==0) kttl->retval = 0;
    else {
      struct timeval now;
      gettimeofday(&now,0);
      if (now.tv_sec > kttl->abstime->tv_sec) kttl->retval = ETIMEDOUT;
      else if (now.tv_sec < kttl->abstime->tv_sec) kttl->retval = 0;
      else if (now.tv_usec > kttl->abstime->tv_nsec / 1000) kttl->retval = ETIMEDOUT;
    }
/*
   // TG: why not here ?
    //thread->_state      = KAAPI_THREAD_S_RUNNING;
*/
  }

  kaapi_assert ( 0 == pthread_mutex_unlock (&cond->_mutex) );

  return 0;
}
