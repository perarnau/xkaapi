/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#if defined (_WIN32) || defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#include <errno.h>


/* == 0 if lib is not initialized
*/
static kaapi_atomic_t kaapi_count_init = {0};

/*
*/
kaapi_rtparam_t kaapi_default_param = {
   .startuptime = 0,
   .stacksize   = 64*4096, /**/
   .cpucount    = 0,
   .kproc_list  = 0,
   .kid2cpu     = 0,
   .cpu2kid     = 0
};


/** \ingroup WS
    Initialize from xkaapi runtime parameters from command line
    \param argc [IN] command line argument count
    \param argv [IN] command line argument vector
    \retval 0 in case of success 
    \retval EINVAL because of error when parsing then KAAPI_CPUSET string
    \retval E2BIG because of a cpu index too high in KAAPI_CPUSET
*/
static int kaapi_setup_param()
{
  const char* wsselect;
  const char* emitsteal;
    
  /* compute the number of cpu of the system */
#if defined(__linux__)
  kaapi_default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(__APPLE__)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(kaapi_default_param.syscpucount);
    sysctl(mib, 2, &kaapi_default_param.syscpucount, &len, 0, 0);
  }
#elif defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  kaapi_default_param.syscpucount = sys_info.dwNumberOfProcessors;
#else
#  warning "Could not compute number of physical cpu of the system. Default value==1"
  kaapi_default_param.syscpucount = 1;
#endif
  /* adjust system limit, if library is compiled with greather number of processors that available */
  if (kaapi_default_param.syscpucount > KAAPI_MAX_PROCESSOR_LIMIT)
    kaapi_default_param.syscpucount = KAAPI_MAX_PROCESSOR_LIMIT;

  kaapi_default_param.use_affinity = 0;

  kaapi_default_param.cpucount  = kaapi_default_param.syscpucount;
  
  if (getenv("KAAPI_DISPLAY_PERF") !=0)
    kaapi_default_param.display_perfcounter = 1;
  else
    kaapi_default_param.display_perfcounter = 0;

  if (getenv("KAAPI_STACKSIZE") !=0)
    kaapi_default_param.stacksize = atoll(getenv("KAAPI_STACKSIZE"));

  /* workstealing selection function */
  wsselect = getenv("KAAPI_WSSELECT");
  kaapi_default_param.wsselect = &kaapi_sched_select_victim_rand;
  if (wsselect !=0)
  {
    if (strcmp(wsselect, "rand") ==0)
      kaapi_default_param.wsselect = &kaapi_sched_select_victim_rand;
    else if (strcmp(wsselect, "workload") ==0)
      kaapi_default_param.wsselect = &kaapi_sched_select_victim_workload_rand;
    else if (strcmp(wsselect, "first0") ==0)
      kaapi_default_param.wsselect = &kaapi_sched_select_victim_rand_first0;
    else if (strcmp(wsselect, "hierarchical") ==0)
      kaapi_default_param.wsselect = &kaapi_sched_select_victim_hierarchy;
  #if 0
    else if (strcmp(wsselect, "pws") ==0)
      kaapi_default_param.wsselect = &kaapi_sched_select_victim_pws;
  #endif
    else {
      fprintf(stderr, "***Kaapi: bad value for variable KAAPI_WSSELECT\n");
      return EINVAL;
    }
  }

  kaapi_default_param.emitsteal          = kaapi_sched_flat_emitsteal;
  kaapi_default_param.emitsteal_initctxt = kaapi_sched_flat_emitsteal_init;
  emitsteal = getenv("KAAPI_EMITSTEAL");
  if (emitsteal != NULL)
  {
    if (strcmp(emitsteal, "hws") == 0)
    {
      kaapi_default_param.emitsteal          = kaapi_hws_emitsteal;
      kaapi_default_param.emitsteal_initctxt = kaapi_hws_emitsteal_init;
    }
    else if (strcmp(emitsteal, "flat") == 0)
    {
      kaapi_default_param.emitsteal          = kaapi_sched_flat_emitsteal;
      kaapi_default_param.emitsteal_initctxt = kaapi_sched_flat_emitsteal_init;
    }
    else {
      fprintf(stderr, "***Kaapi: bad value for variable KAAPI_EMITSTEAL\n");
      return EINVAL;
    }
  }
  
  return 0;
}



/**
*/
int kaapi_init(int flag, int* argc, char*** argv)
{
  if (KAAPI_ATOMIC_INCR(&kaapi_count_init) !=1) 
    return EALREADY;

  kaapi_init_basicformat();
  
  /* set up runtime parameters */
  kaapi_assert_m( 0 == kaapi_setup_param(), "kaapi_setup_param" );
  
#if defined(KAAPI_USE_NETWORK)
  kaapi_network_init(argc, argv);
#endif

  kaapi_memory_init();
  int err = kaapi_mt_init();

  if (flag)
    kaapi_begin_parallel(KAAPI_SCHEDFLAG_DEFAULT);
  return err;
}


/**
*/
int kaapi_finalize(void)
{
  kaapi_sched_sync();
  if (KAAPI_ATOMIC_DECR(&kaapi_count_init) !=0) 
    return EAGAIN;

  kaapi_memory_destroy();

  kaapi_mt_finalize();

#if defined(KAAPI_USE_NETWORK)
  kaapi_network_finalize();
#endif

  return 0;
}


/* Counter of enclosed parallel/begin calls.
*/
static kaapi_atomic_t kaapi_parallel_stack = {0};

/** begin parallel & and parallel
    - it should not have any concurrency on the first increment
    because only the main thread is running before parallel computation
    - after that serveral threads may declare parallel region that
    will implies concurrency
*/
void kaapi_begin_parallel( int schedflag )
{
  if (schedflag & KAAPI_SCHEDFLAG_STATIC)
  {
    kaapi_thread_set_unstealable(1);
  }
  /* if not static then wakeup thread here */
  else if (KAAPI_ATOMIC_INCR(&kaapi_parallel_stack) == 1)
  {
    kaapi_mt_resume_threads();
  }
}


/**
*/
void kaapi_end_parallel(int flag)
{
  if (flag & KAAPI_SCHEDFLAG_STATIC) 
  { /* end of the static parallel region: compute readylist + 
       set thread stealable + resume thread if 1rst // region*/
    kaapi_sched_computereadylist();
    kaapi_thread_set_unstealable(0);
    if (KAAPI_ATOMIC_INCR(&kaapi_parallel_stack) == 1)
      kaapi_mt_resume_threads();
  }
  
  if ((flag & KAAPI_SCHEDFLAG_NOWAIT) ==0) 
    kaapi_sched_sync();

  if (KAAPI_ATOMIC_DECR(&kaapi_parallel_stack) == 0)
  {
    kaapi_mt_suspend_threads();
  }
}


