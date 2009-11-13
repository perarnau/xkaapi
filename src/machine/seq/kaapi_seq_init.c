/*
** kaapi_seq_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
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
#include "kaapi_config.h"
#include "kaapi_seq_machine.h"
#include <stdlib.h>



/**
*/
void __attribute__ ((constructor)) kaapi_init(void)
{
  int i;
  kaapi_thread_processor_t* kaapi_main_processor;
  kaapi_thread_descr_t* td_main_thread;  
  default_param.stacksize = getpagesize()*4;

  kaapi_assert( 0 == pthread_key_create( &kaapi_current_thread_key, 0 ) );
  td_main_thread = kaapi_allocate_thread_descriptor(KAAPI_SYSTEM_SCOPE, 1, 0, default_param.stacksize);
  pthread_setspecific( kaapi_current_thread_key, td_main_thread );
  
  /* init dataspecific table */
  for (i = 0; i < KAAPI_KEYS_MAX; i++)
  {
    kaapi_global_keys[i].dest = &_kaapi_dummy;
    kaapi_global_keys[i].next = i+1;
  }
  kaapi_global_keys[KAAPI_KEYS_MAX - 1].next = -1;

  /* should be the first kaapi key with value == 0 */
  kaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );

  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(default_param.syscpucount);
    sysctl(mib, 2, &default_param.syscpucount, &len, 0, 0);
  }
#else
  #warning "Could not compute number of physical cpu of the system."
  default_param.syscpucount = 1;
#endif
  
  default_param.cpucount  = default_param.syscpucount;
  default_param.usecpuset = 0;

  /* initialize the default attr */
  kaapi_attr_init( &kaapi_default_attr );

  
  /* initialize kaapi_main_processor */
  kaapi_main_processor = kaapi_allocate_processor();
  kaapi_assert( kaapi_main_processor !=0 );
  kaapi_assert( 0 == pthread_setspecific(kaapi_current_processor_key, kaapi_main_processor ) );

  /* setup kaapi_sched_select_victim_function */
  kaapi_sched_select_victim_function = &kaapi_sched_select_victim_rand;
  
  /* TODO : set to 1  default_param.cpuset; */
  printf("[KAAPI::INIT] Current thread is: %lu\n", (unsigned long)pthread_self() );
  fflush( stdout );
}

/**
*/
void __attribute__ ((destructor)) kaapi_fini(void)
{
  printf("[KAAPI::TERM] Current thread is: %lu\n", (unsigned long)pthread_self() );
  fflush( stdout );

#if defined(KAAPI_USE_SCHED_AFFINITY)
  free(kaapi_kproc2cpu);
  free(kaapi_cpu2kproc);
#endif
}
