/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
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
#ifndef _KAAPI_IMPL_H
#define _KAAPI_IMPL_H 1

#if defined(__cplusplus)
extern "C" {
#endif

#include "kaapi_config.h"
#include "kaapi_atomic.h"
#include "kaapi_datastructure.h"
/* must be included before kaapi.h in order to avoid including kaapi_type.h : */
#include "kaapi_private_structure.h"
#include "kaapi.h"
#include "kaapi_param.h"

#include <pthread.h>
#include <limits.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h> /* getpagesize */
#include <stdint.h> /* fixed size int */

#if defined(KAAPI_USE_APPLE)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif


#ifdef HAVE_NUMA_H
#  include <numa.h>
#endif

#include "kaapi_error.h"

/* Extend the scope for processor 
*/
#define KAAPI_PROCESSOR_SCOPE (KAAPI_PROCESS_SCOPE + 1)

#include "kaapi_time.h"
#include "kaapi_task.h"

/* ========================================================================= */
/* Dynamic list allocated by bloc of pagesize() size array
 */



/* ========================================================================= */
/* INTERNAL structure of the request at the level of kaapi_steal_thread_context_t
   This data structure only concern the _data of a kaapi_steal_request_t
 */
  extern pthread_key_t kaapi_current_processor_key;   /* should be the first key (0) point to */
  
  /** Spin lock: extension to posix interface
  */
  int kaapi_mutex_spinlock (kaapi_mutex_t *mutex);

  /* INTERNAL interface
   */
  void* kaapi_start_system_handler(void *arg);
  
  /** Get the workstealing concurrency number, i.e. the number of kernel activities to execute the user level thread.
      If kaapi_setconcurrency was no called before then return 0, else return the number set by kaapi_setconcurrency.
   */
  int kaapi_getconcurrency(void );
  
  /**
   */
  int kaapi_setconcurrency( int concurrency );
  
  /* ========================================================================= */
  /* INTERNAL scheduler Interface 
   */
  
  /**
  */
  kaapi_thread_descr_processor_t* kaapi_sched_get_processor();


  /** Entry point to run worker kernel threads
      \param arg: pointer to the kaapi_processor_t structure
  */
  void* kaapi_sched_run_processor( void* arg );

  /** Allocate the array of processors
  */
  kaapi_thread_descr_processor_t** kaapi_allocate_processors( int kproc, cpu_set_t cpuset);

  /** Allocate one processor
      Inherited the cpuset from the running thread.
  */
  kaapi_thread_descr_processor_t* kaapi_allocate_processor();

#if 0 /* TG TO REDO API SPEC */
  /** Steal thread in a kaapi_steal_thread_context_t 
  */
  int kaapi_sched_steal_sc_thread(
      struct kaapi_steal_processor_t* kpss, struct kaapi_steal_context_t* sc,
      int count, kaapi_steal_request_t** requests
  );
  
  /** Method to execute in case of success during stealing a kaapi_sched_steal_sc_thread
  */
  void kaapi_steal_thief_entrypoint_thread(struct kaapi_steal_processor_t* kpsp, struct kaapi_steal_request_t* request);


  /** Steal work from thread in a kaapi_steal_thread_context_t 
  */
  int kaapi_sched_steal_sc_inside(
      struct kaapi_steal_processor_t* kpss, struct kaapi_steal_context_t* sc,
      int count, kaapi_steal_request_t** requests
  );
  
#endif

  /** Called when a thread is suspended on an instruction
  */
  int kaapi_sched_suspend   ( kaapi_thread_descr_processor_t* proc, 
                              kaapi_thread_descr_t* thread, 
                              kaapi_test_wakeup_t fwakeup, 
                              void* argwakeup );

  
  /** Enter in the infinite loop of trying to steal work.
      Never return from this function...
      If proc is null pointer, then the function allocate a new kaapi_processor_t and assigns it to the current processor.
      This method may be called by 'host' thread in order to become executor thread.
      The method returns in case of terminaison.
  */
  void kaapi_sched_idle      ( kaapi_thread_descr_processor_t* proc );
  
  /* Return 1 iff it should redo call to its entrypoint, else never return
   */
  int kaapi_sched_terminate_or_redo (kaapi_thread_descr_processor_t* proc, kaapi_thread_descr_t* thread);
  
  /** Client interface to steal a given kaapi_processor_t
   */
/*  kaapi_thread_descr_t* kaapi_sched_do_steal( kaapi_processor_t* proc ); */
  
  /**
   */
  struct kaapi_thread_descr_t* kaapi_allocate_thread_descriptor( int scope, int detachstate, size_t c_stacksize, size_t k_stacksize );
  
  /**
   */
  void kaapi_deallocate_thread_descriptor( struct kaapi_thread_descr_t* thread );
  
  /* ========================================================================= */
  /* INTERNAL numa Interface 
   */
#if defined(HAVE_NUMA_H)
  
#endif
  
  
  /* ========================================================================= */
  /* Initialization / destruction functions
   */
  
  void __attribute__ ((constructor)) kaapi_init(void);
  
  void __attribute__ ((destructor)) kaapi_fini(void);
  
  /**
   */
  void _kaapi_dummy(void*);
  
  /** To force reference to kaapi_init.c in order to link against kaapi_init and kaapi_fini
   */
  static void __attribute__((unused)) __kaapi_dumy_dummy(void)
  {
    _kaapi_dummy(NULL);
  }
  
  /* ========================================================================== */
  /** kaapi_get_elapsedtime
      The function kaapi_get_elapsedtime() will return the elapsed time since an epoch.
  */
  extern double kaapi_get_elapsedtime();

#ifdef __cplusplus
}
#endif

#endif /* _KAAPI_IMPL_H */
