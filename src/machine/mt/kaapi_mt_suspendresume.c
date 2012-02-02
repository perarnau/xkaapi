/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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

/*
*/
volatile int kaapi_suspendflag;

/*
*/
kaapi_atomic_t kaapi_suspendedthreads;

#define USE_POSIX_CONDITION 0

/*
*/
static pthread_cond_t   wakeupcond_threads;
static pthread_mutex_t  wakeupmutex_threads;

void kaapi_mt_suspendresume_init(void)
{
  kaapi_assert( 0 == pthread_cond_init(&wakeupcond_threads, 0) );
  kaapi_assert( 0 == pthread_mutex_init(&wakeupmutex_threads, 0) );
  kaapi_suspendflag = 0;
  KAAPI_ATOMIC_WRITE(&kaapi_suspendedthreads, 0);
}


void kaapi_mt_suspend_self( kaapi_processor_t* kproc )
{
#if USE_POSIX_CONDITION
  if (kaapi_suspendflag)
  {
    KAAPI_ATOMIC_INCR( &kaapi_suspendedthreads );
    pthread_mutex_lock(&wakeupmutex_threads);
    pthread_cond_wait(&wakeupcond_threads, &wakeupmutex_threads);
    memset(&kproc->fnc_selecarg, 0, sizeof(kproc->fnc_selecarg) );
    pthread_mutex_unlock(&wakeupmutex_threads);
  }
#else
  KAAPI_ATOMIC_INCR( &kaapi_suspendedthreads );
  while (kaapi_suspendflag !=0)
    kaapi_slowdown_cpu();
#endif
}


/* should always be called by the main thread only
*/
void kaapi_mt_suspend_threads(void)
{
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kaapi_suspendedthreads) == 0 );
  kaapi_writemem_barrier();
  kaapi_suspendflag = 1;
  while (KAAPI_ATOMIC_READ(&kaapi_suspendedthreads) != (kaapi_count_kprocessors-1))
  {
    kaapi_slowdown_cpu();
  }
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kaapi_suspendedthreads) == (kaapi_count_kprocessors-1) );
}

/*
*/
void kaapi_mt_resume_threads(void)
{
  kaapi_assert_debug( KAAPI_ATOMIC_READ(&kaapi_suspendedthreads) == (kaapi_count_kprocessors-1) );
#if USE_POSIX_CONDITION
  pthread_mutex_lock(&wakeupmutex_threads);
  kaapi_suspendflag = 0;
  KAAPI_ATOMIC_WRITE(&kaapi_suspendedthreads, 0);
  pthread_cond_broadcast(&wakeupcond_threads);
  pthread_mutex_unlock(&wakeupmutex_threads);  
#else
  KAAPI_ATOMIC_WRITE_BARRIER(&kaapi_suspendedthreads, 0);
  kaapi_suspendflag = 0;
#endif
}

