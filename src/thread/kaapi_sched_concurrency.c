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


/** Create and initialize and start concurrency kernel thread to execute user threads
    TODO: a voir le cas du thread main : kaapi_processor ?
*/
int kaapi_setconcurrency( int concurrency )
{
  int i;
  int stsize;
  void* staddr;
  kaapi_t thread;
  
  if ((concurrency <0) || (concurrency > KAAPI_MAX_PROCESSOR)) return EINVAL;
  
  if (KAAPI_ATOMIC_READ(&kaapi_stealapi_barrier_term) !=1) return ENOSYS; /* cannot be changed dynamically should be implemented */
  
  kaapi_processor_t** all_processors = kaapi_allocate_processors( concurrency-1, default_param.cpuset );
  
#if defined(KAAPI_USE_SCHED_AFFINITY)
  cpu_set_t cpuset;
#endif

  /* initialize worker thread */
  for (i=1; i<concurrency; ++i)
  {
    /* set attr to the posix thread */
    kaapi_attr_t attr;
    ckaapi_assert ( 0 == kaapi_attr_init( &attr ) );
    ckaapi_assert ( 0 == kaapi_attr_setdetachstate( &attr, 1 ) );
    attr._scope = KAAPI_PROCESSOR_SCOPE;
#if defined(KAAPI_USE_SCHED_AFFINITY)
    if (default_param.usecpuset !=0)
    {
      CPU_ZERO( &cpuset );
      CPU_SET ( (1+i)%default_param.cpucount, &cpuset);
      kaapi_attr_setaffinity( &attr, sizeof(cpuset), &cpuset );
    }
#endif

    stsize = KAAPI_STACK_MIN;
    staddr = malloc(stsize);

    ckaapi_assert ( 0 == kaapi_attr_setstacksize( &attr, stsize ));
    ckaapi_assert ( 0 == kaapi_attr_setstackaddr( &attr, staddr ));

    /* increment number of running thread */
    kaapi_barrier_td_setactive( &kaapi_stealapi_barrier_term, 1 );
    ckaapi_assert( 0 == kaapi_create( &thread, &attr, &kaapi_sched_run_processor, all_processors[i-1] ));    
  }
  
  return 0;
}


/**
*/
int kaapi_getconcurrency(void )
{
  return KAAPI_ATOMIC_READ(&kaapi_stealapi_barrier_term);
}
