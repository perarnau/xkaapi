/*
** kaapi_init.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com 
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
#include <stdlib.h>
#include <inttypes.h> 
#include "kaapi_impl.h"

#if defined(KAAPI_DEBUG)
#  include <unistd.h>
#  include <sys/time.h>
#  include <signal.h>
#endif

#if defined(KAAPI_USE_PERFCOUNTER)
#include <signal.h>

#include "machine/cuda/kaapi_cuda_trace.h"
#endif

/*
*/
kaapi_request_t kaapi_global_requests_list[KAAPI_MAX_PROCESSOR+1];


/*
*/
uint32_t volatile kaapi_count_kprocessors = 0;


/*
*/
kaapi_processor_t** kaapi_all_kprocessors = 0;



/*
*/
#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
__thread kaapi_thread_t**        kaapi_current_thread_key;
__thread kaapi_processor_t*      kaapi_current_processor_key;
__thread kaapi_thread_context_t* kaapi_current_thread_context_key;
__thread kaapi_threadgroup_t     kaapi_current_threadgroup_key;
#else
pthread_key_t kaapi_current_processor_key;

kaapi_thread_t* kaapi_self_thread(void)
{
  return (kaapi_thread_t*)kaapi_get_current_processor()->thread->stack.sfp;
}
kaapi_threadgroup_t kaapi_self_threadgroup(void)
{
  return kaapi_get_current_processor()->thread->thgrp;
}
void kaapi_set_threadgroup(kaapi_threadgroup_t thgrp)
{
  kaapi_get_current_processor()->thread->thgrp = thgrp;
}
#endif

/* 
*/
kaapi_atomic_t kaapi_term_barrier = { 0 };

/* 
*/
volatile int kaapi_isterm = 0;

#ifdef KAAPI_MAX_PROCESSOR_GENERIC
void (*kaapi_bitmap_clear)( kaapi_bitmap_t* b );
int (*kaapi_bitmap_value_empty)( kaapi_bitmap_value_t* b );
void (*kaapi_bitmap_value_set)( kaapi_bitmap_value_t* b, int i );
void (*kaapi_bitmap_value_copy)( kaapi_bitmap_value_t* retval, kaapi_bitmap_value_t* b);
void (*kaapi_bitmap_swap0)( kaapi_bitmap_t* b, kaapi_bitmap_value_t* v );
int (*kaapi_bitmap_set)( kaapi_bitmap_t* b, int i );
int (*kaapi_bitmap_count)( kaapi_bitmap_value_t b );
int (*kaapi_bitmap_value_first1_and_zero)( kaapi_bitmap_value_t* b );
#endif


/** Initialize a request
    \param kpsr a pointer to a kaapi_steal_request_t
*/
static inline void kaapi_request_init( struct kaapi_request_t* pkr, uintptr_t ident )
{
  *(uintptr_t*)&pkr->ident    = ident;
  pkr->status   = 0;
}


/**
*/
int kaapi_mt_init(void)
{
  kaapi_processor_t*      kproc;
  kaapi_thread_context_t* thread;
  kaapi_task_t*           task;
  const char* volatile    version __attribute__((unused));

  kaapi_isterm = 0;
  version = get_kaapi_version();

  /* It should be the only location where request are initialized */
  for (int i=0; i<KAAPI_MAX_PROCESSOR+1; ++i)
    kaapi_request_init(&kaapi_global_requests_list[i], i);
  
  /* build the memory hierarchy
     update kaapi_default_param data structure fields:
      * kid2cpu
      * cpu2kid
      * memory
   */
  kaapi_hw_init();
  
#if 0 // DEPRECATED_ATTRIBUTE
  /* Build global scheduling queue on the hierarchy */
  kaapi_sched_affinity_initialize();  
#endif


#ifdef KAAPI_MAX_PROCESSOR_GENERIC
  /* Choosing an implementation depending on the available ones */ 
#  ifdef KAAPI_MAX_PROCESSOR_32
  if (kaapi_default_param.cpucount <= 32) {
    kaapi_bitmap_clear           = &kaapi_bitmap_clear_32;
    kaapi_bitmap_value_empty     = &kaapi_bitmap_value_empty_32;
    kaapi_bitmap_value_set       = &kaapi_bitmap_value_set_32;
    kaapi_bitmap_value_copy      = &kaapi_bitmap_value_copy_32;
    kaapi_bitmap_swap0           = &kaapi_bitmap_swap0_32;
    kaapi_bitmap_set             = &kaapi_bitmap_set_32;
    kaapi_bitmap_count           = &kaapi_bitmap_count_32;
    kaapi_bitmap_value_first1_and_zero = &kaapi_bitmap_value_first1_and_zero_32;
  } else
#  endif
#  ifdef KAAPI_MAX_PROCESSOR_64
  if (kaapi_default_param.cpucount <= 64) {
    kaapi_bitmap_clear           = &kaapi_bitmap_clear_64;
    kaapi_bitmap_value_empty     = &kaapi_bitmap_value_empty_64;
    kaapi_bitmap_value_set       = &kaapi_bitmap_value_set_64;
    kaapi_bitmap_value_copy      = &kaapi_bitmap_value_copy_64;
    kaapi_bitmap_swap0           = &kaapi_bitmap_swap0_64;
    kaapi_bitmap_set             = &kaapi_bitmap_set_64;
    kaapi_bitmap_count           = &kaapi_bitmap_count_64;
    kaapi_bitmap_value_first1_and_zero = &kaapi_bitmap_value_first1_and_zero_64;
  } else
#  endif
#  ifdef KAAPI_MAX_PROCESSOR_128
  if (kaapi_default_param.cpucount <= 128) {
    kaapi_bitmap_clear           = &kaapi_bitmap_clear_128;
    kaapi_bitmap_value_empty     = &kaapi_bitmap_value_empty_128;
    kaapi_bitmap_value_set       = &kaapi_bitmap_value_set_128;
    kaapi_bitmap_value_copy      = &kaapi_bitmap_value_copy_128;
    kaapi_bitmap_swap0           = &kaapi_bitmap_swap0_128;
    kaapi_bitmap_set             = &kaapi_bitmap_set_128;
    kaapi_bitmap_count           = &kaapi_bitmap_count_128;
    kaapi_bitmap_value_first1_and_zero = &kaapi_bitmap_value_first1_and_zero_128;
  } else
#  endif
#  ifdef KAAPI_MAX_PROCESSOR_LARGE
  if (kaapi_default_param.cpucount) {
    kaapi_bitmap_clear           = &kaapi_bitmap_clear_large;
    kaapi_bitmap_value_empty     = &kaapi_bitmap_value_empty_large;
    kaapi_bitmap_value_set       = &kaapi_bitmap_value_set_large;
    kaapi_bitmap_value_copy      = &kaapi_bitmap_value_copy_large;
    kaapi_bitmap_swap0           = &kaapi_bitmap_swap0_large;
    kaapi_bitmap_set             = &kaapi_bitmap_set_large;
    kaapi_bitmap_count           = &kaapi_bitmap_count_large;
    kaapi_bitmap_value_first1_and_zero = &kaapi_bitmap_value_first1_and_zero_large;
  } else
#  endif
  {
    fprintf(stderr, "Too many processors\nAborting\n");
    exit(1);
  }
#endif

  /* initialize the kprocessor key */
#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  kaapi_current_thread_key = 0;
  kaapi_current_threadgroup_key = 0;
#else
  kaapi_assert( 0 == pthread_key_create( &kaapi_current_processor_key, 0 ) );
#endif

  kaapi_default_param.startuptime = kaapi_get_elapsedns();
    
#if defined(KAAPI_USE_PERFCOUNTER)
  /* call prior setconcurrency */
  kaapi_perf_global_init();
  if (getenv("KAAPI_RECORD_TRACE") !=0) {
    kaapi_cuda_trace_init();
  }
  /* kaapi_perf_thread_init(); */
#endif
    
  /* create the kprocessor AFTER topology !!! */
  kaapi_assert_m( 0 == kaapi_setconcurrency(), "kaapi_setconcurrency" );
  kproc = kaapi_get_current_processor();

  /* destroy the procinfo list, thread args no longer valid */
  kaapi_procinfo_list_free(kaapi_default_param.kproc_list);
  free(kaapi_default_param.kproc_list);
  kaapi_default_param.kproc_list = 0;

/*** TODO BEG: this code should but outside machine specific init*/
  /* push dummy task in exec mode */
  thread = kaapi_self_thread_context();
  task = kaapi_thread_toptask(kaapi_threadcontext2thread(thread));
  kaapi_task_init_withstate( 
    task, 
    (kaapi_task_body_t)kaapi_taskstartup_body, 
    thread->stack.proc,
    KAAPI_TASK_STATE_EXEC
  );
  kaapi_thread_pushtask(kaapi_threadcontext2thread(thread));
  thread->stack.sfp->pc        = thread->stack.sfp->sp;

  /* push the current frame that correspond to the execution of the startup task */
  thread->stack.sfp[1].pc      = thread->stack.sfp->sp;
  thread->stack.sfp[1].sp      = thread->stack.sfp->sp;
  thread->stack.sfp[1].sp_data = thread->stack.sfp->sp_data;
  ++thread->stack.sfp;

  /* dump output information */
  if (kaapi_default_param.display_perfcounter != KAAPI_NO_DISPLAY_PERF)
  {
    printf("[KAAPI::INIT] use #physical cpu:%u, start time:%15f\n", kaapi_default_param.cpucount,kaapi_get_elapsedtime());
    fflush( stdout );
  }

  KAAPI_EVENT_PUSH1(kproc, kproc->thread, KAAPI_EVT_TASK_BEG, task );
  

#if defined(KAAPI_DEBUG)
  /* set signal handler print callstack */
  {
    struct sigaction sa;
    sa.sa_handler = _kaapi_signal_dump_backtrace;
    sa.sa_flags = SA_RESTART;
    sigemptyset (&sa.sa_mask);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
  }
#if defined(KAAPI_USE_PERFCOUNTER)
  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    /* set signal handler to flush event */
    struct sigaction sa;
    sa.sa_handler = _kaapi_signal_dump_counters;
    sa.sa_flags = SA_RESTART;
    sigemptyset (&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    sigaction(SIGTSTP, &sa, NULL);
    /* other signals: SIGABRT, SIGBUS, SIGTERM, SIGSEGV are handled by backtrace_sighandler */
  }
#endif
  
#elif defined(KAAPI_USE_PERFCOUNTER)
  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    /* set signal handler to flush event */
    struct sigaction sa;
    sa.sa_handler = _kaapi_signal_dump_counters;
    sa.sa_flags = SA_RESTART;
    sigemptyset (&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGTSTP, &sa, NULL);
  }
#endif
  
  /* */
#if defined(KAAPI_DEBUG)
  /* set alarm */
  if (getenv("KAAPI_DUMP_PERIOD") !=0)
  {
    kaapi_default_param.alarmperiod = atoi(getenv("KAAPI_DUMP_PERIOD"));
    if (kaapi_default_param.alarmperiod <0) return EINVAL;
    if (kaapi_default_param.alarmperiod ==0) return 0;
    
    /* set signal handler on SIGALRM + set alarm */
    struct sigaction sa;
    sa.sa_handler = _kaapi_signal_dump_state;
    sa.sa_flags = SA_RESTART;
    sigemptyset (&sa.sa_mask);
    sigaction(SIGALRM , &sa, NULL);

    /* set periodic alarm */
    alarm( kaapi_default_param.alarmperiod );

  }
#endif
  
  return 0;
}


/**
*/
int kaapi_mt_finalize(void)
{
  unsigned int i;
  kaapi_processor_t* kproc;

  kaapi_assert(kaapi_count_kprocessors >0);

  kproc = kaapi_get_current_processor();
  kaapi_assert(kaapi_all_kprocessors[0] == kproc);  

  KAAPI_EVENT_PUSH0( kproc, kproc->thread, KAAPI_EVT_TASK_END );

  /* if thread suspended, then resume all threads
  */
  if (kaapi_suspendflag)
    kaapi_mt_resume_threads();

#if defined(KAAPI_USE_PERFCOUNTER)
  /*  */
  kaapi_perf_thread_stop(kaapi_all_kprocessors[0]);
#endif

  if (kaapi_default_param.display_perfcounter != KAAPI_NO_DISPLAY_PERF)
  {  
    printf("[KAAPI::TERM] end time:%15f, delta: %15f(s)\n", kaapi_get_elapsedtime(), 
          (double)(kaapi_get_elapsedns()-kaapi_default_param.startuptime)*1e-9 );
    fflush( stdout );
  }

  /* wait end of the initialization */
  kaapi_isterm = 1;
  kaapi_barrier_td_setactive(&kaapi_term_barrier, 0);
  kaapi_barrier_td_waitterminated(&kaapi_term_barrier);

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_EVENT_PUSH0(kaapi_all_kprocessors[0], 0, KAAPI_EVT_KPROC_STOP);
#if defined(KAAPI_USE_CUDA)
  if (getenv("KAAPI_RECORD_TRACE") !=0) {
      kaapi_cuda_trace_finalize();
  }
#endif /* KAAPI_USE_CUDA */
  kaapi_perf_thread_fini(kaapi_all_kprocessors[0]);
  kaapi_perf_global_fini();
#endif

#if defined(KAAPI_USE_CUDA)
  kaapi_mem_host_map_destroy_all(kaapi_get_current_mem_host_map());
#endif

  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
#if defined(KAAPI_USE_CUDA)
    /* initialize cuda processor */
    if (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA)
      kaapi_cuda_proc_cleanup(&kaapi_all_kprocessors[i]->cuda_proc);
#endif /* KAAPI_USE_CUDA */
    kaapi_wsqueuectxt_destroy(&kaapi_all_kprocessors[i]->lsuspend);
    kaapi_processor_destroy(kaapi_all_kprocessors[i]);
    kaapi_processor_free(kaapi_all_kprocessors[i]);
    kaapi_all_kprocessors[i]= 0;
  }
  free( kaapi_all_kprocessors );
  kaapi_all_kprocessors =0;

  /* TODO: destroy topology data structure */
  return 0;
}

