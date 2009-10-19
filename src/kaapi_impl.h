/*
** kaapi_impl.h
** ckaapi
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
// must be included before kaapi.h in order to avoid including kaapi_type.h :
#include "kaapi_private_structure.h"
#include "kaapi.h"
#include "kaapi_param.h"

#include <pthread.h>
#include <limits.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h> /* getpagesize */

#if defined(KAAPI_USE_APPLE)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif

#ifdef HAVE_NUMA_H
#  include <numa.h>
#endif

#include "kaapi_error.h"
  
#define KAAPI_PROCESSOR_SCOPE (KAAPI_PROCESS_SCOPE + 1)

#include "kaapi_stealapi.h"

/* ========================================================================= */
/* Dynamic list allocated by bloc of pagesize() size array
 */

/* ========================================================================= */
/* INTERNAL thread descriptor
 */ 
struct kaapi_thread_descr_t {
  volatile kaapi_thread_state_t  _state;           /* see definition */
  int                            _scope;           /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_run_entrypoint_t         _run_entrypoint;  /* the entry point */
  void*                          _arg_entrypoint;  /* the argument of the entry point */
  void*                          _return_value;    /* */
  kaapi_thread_descr_t*          _td;              /* point to the kaapi_thread_descr_t */

  pthread_t                      _pthid;           /* iff scope system or processor */
  pthread_cond_t                 _cond;            /* use by system scope thread to be signaled */  
  KAAPI_QUEUE_FIELD(struct kaapi_thread_descr_t);  /* system scope attribut to be used in Queue, Stack and Fifo. */

  kaapi_thread_context_t         _ctxt;            /* process scope thread context */
  kaapi_processor_t*             _proc;            /* iff scope == PROCESS_SCOPE, always schedule on a processor */
  kaapi_splitter_t               _splitter;        /* code to call if work of a thread is steal */

  pthread_mutex_t                _mutex_join;
  pthread_cond_t                 _cond_join;
  
  int                            _detachstate;      /* 1 iff the thread is created in datachstate (then x_join fields are not init */
  cpu_set_t                      _cpuset;           /* cpuset attribut */
  size_t                         _stacksize;        /* stack size attribut */
  void*                          _stackaddr;        /* thread stack base pointer */
  void**                         _key_table;        /* thread data specific */
};


/* ========================================================================= */
/* INTERNAL processor descriptor, extend kaapi_steal_processor by adding more
 * fields such as a kaapi_steal_thread_context_t to store user threads
 */

#define KAAPI_WORKQUEUE_BLOC_MAXITEM(type) ((4096 - 2*(sizeof(int) + sizeof(void*)))/sizeof(type))

/**
*/
#define KAAPI_WORKQUEUE_BLOC(name, type) \
typedef struct name { \
  int _top; \
  int _bottom; \
  struct name* _nextbloc; \
  struct name* _prevbloc; \
  type _data[ KAAPI_WORKQUEUE_BLOC_MAXITEM(type) ]; \
} name


/**
*/
#define KAAPI_WORKQUEUE_BLOC_INIT( kpwq ) \
  (kpwq)->_top = (kpwq)->_bottom = 0

/*
*/
#define KAAPI_WORKQUEUE_BLOC_STEAL( kpwq, i ) \
  KAAPI_ATOMIC_CAS( &(kpwq)->_data[i]._status, 1, 0 )

#define KAAPI_WORKQUEUE_BLOC_ABORTSTEAL( kpwq, i ) \
  KAAPI_ATOMIC_WRITE( &(kpwq)->_data[i]._status, 1 )


/** Should declared at the first position inside a structure !!!!
*/
#define KAAPI_WORKQUEUE_FIELD(name) \
  struct name* _first_bloc; \
  struct name* _top_bloc; \
  struct name* _bottom_bloc


#define KAAPI_WORKQUEUE_INIT( kpwq ) \
  (kpwq)->_first_bloc = (kpwq)->_top_bloc = (kpwq)->_bottom_bloc = 0 \

#define KAAPI_WORKQUEUE_EMPTY( kpwq ) \
  ( ((kpwq)->_bottom_bloc == 0 ) || \
    ( ((kpwq)->_top_bloc == (kpwq)->_bottom_bloc) && ((kpwq)->_bottom_bloc->_top == (kpwq)->_bottom_bloc->_bottom) ) \
  )

/** This type should begin with same kind (size) of fields that KAAPI_WORKQUEUE_BLOC(name)
*/
typedef struct kaapi_workqueue_bloc_t {
  int   _top;
  int   _bottom;
  void* _nextbloc;
  void* _prevbloc;
} kaapi_workqueue_bloc_t;

/** This type should begin with same kind (size) of fields that KAAPI_WORKQUEUE_FIELD(name)
*/
typedef struct kaapi_workqueue_head_t {
  kaapi_workqueue_bloc_t* _first_bloc;
  kaapi_workqueue_bloc_t* _top_bloc;
  kaapi_workqueue_bloc_t* _bottom_bloc;
} kaapi_workqueue_head_t;

/** Allocate a new bloc for the workqueue pointed by ptr
*/
void kaapi_workqueue_alloc_bloc(void* ptr );

/**
*/
typedef struct kaapi_cellsuspended_t {
  kaapi_atomic_t        _status;      /* status : present (1) empty (0) */
  kaapi_thread_descr_t* _thread;      /* the thread */
  kaapi_test_wakeup_t   _fwakeup;     /* != 0 if the thread is suspended */
  void*                 _arg_fwakeup; /* arg to func twakeup */
} kaapi_cellsuspended_t;

KAAPI_WORKQUEUE_BLOC(kaapi_queue_cellsuspended_t, kaapi_cellsuspended_t);

/**
*/
typedef struct kaapi_cellready_t {
  kaapi_atomic_t               _status;      /* status : present (1) empty (0) */
  kaapi_thread_descr_t*        _thread;      /* the thread */
} kaapi_cellready_t;

KAAPI_WORKQUEUE_BLOC(kaapi_queue_cellready_t, kaapi_cellready_t);

typedef struct kaapi_workqueue_suspended_t {
  KAAPI_WORKQUEUE_FIELD(kaapi_queue_cellsuspended_t);
} kaapi_workqueue_suspended_t;


#define KAAPI_WORKQUEUE_SUSPEND_PUSH( kpwr, td, fwakeup, argwakeup ) \
  if ((kpwr)->_bottom_bloc ==0) kaapi_workqueue_alloc_bloc( kpwr );\
  else if ((kpwr)->_bottom_bloc->_bottom >= KAAPI_WORKQUEUE_BLOC_MAXITEM(kaapi_cellsuspended_t)) kaapi_workqueue_alloc_bloc( kpwr );\
  (kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom]._thread = td; \
  (kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom]._fwakeup = fwakeup; \
  (kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom]._arg_fwakeup = argwakeup; \
  kaapi_writemem_barrier(); \
  KAAPI_ATOMIC_WRITE(&(kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom++]._status, 1)
  
typedef struct kaapi_workqueue_ready_t {
  KAAPI_WORKQUEUE_FIELD(kaapi_queue_cellready_t);
} kaapi_workqueue_ready_t;

#define KAAPI_WORKQUEUE_READY_PUSH( kpwr, td ) \
  if ((kpwr)->_bottom_bloc ==0) kaapi_workqueue_alloc_bloc( kpwr );\
  else if ((kpwr)->_bottom_bloc->_bottom >= KAAPI_WORKQUEUE_BLOC_MAXITEM(kaapi_cellready_t)) kaapi_workqueue_alloc_bloc( kpwr );\
  (kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom]._thread = td; \
  kaapi_writemem_barrier(); \
  KAAPI_ATOMIC_WRITE(&(kpwr)->_bottom_bloc->_data[(kpwr)->_bottom_bloc->_bottom++]._status, 1)
  
/* ========================================================================= */
/* INTERNAL steal context for threads managed by a processor 
   Such context allows to steal ready thread or suspended thread that becomes ready
 */
typedef struct kaapi_steal_thread_context_t {
  kaapi_steal_context_t        _sc;
  kaapi_thread_descr_t*        _active_thread;    /* the current active thread */
  kaapi_workqueue_suspended_t  _suspended_thread; /* the suspended thread that directly call sched_idle */
  kaapi_workqueue_ready_t      _ready_list;       /* ready threads, FIFO order */
} kaapi_steal_thread_context_t;

/**
*/
#define kaapi_steal_thread_context_init( kpstc ) \
{\
  kaapi_steal_context_init( &(kpstc)->_sc );\
  (kpstc)->_active_thread = 0;\
  KAAPI_WORKQUEUE_INIT( &(kpstc)->_suspended_thread );\
  KAAPI_WORKQUEUE_INIT( &(kpstc)->_ready_list );\
}

/**
*/
#define kaapi_steal_thread_context_destroy( kpstc ) 


/* ========================================================================= */
/* INTERNAL a processor 
 */
struct kaapi_processor_t {
  kaapi_steal_processor_t*     _the_steal_processor;                  /* the processor view as a steal processor */
  unsigned int                 _seed;                                 /* used for random generator rand_r */
  kaapi_thread_context_t       _ctxt;
  kaapi_thread_descr_t*        _term_thread;                          /* the current terminated user thread that could be reused... */
  kaapi_steal_thread_context_t _sc_thread;                            /* */
  kaapi_steal_context_t        _sc_inside;                            /* steal inside a thread */
};

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
  kaapi_processor_t* kaapi_sched_get_processor();


  /** Entry point to run worker kernel threads
      \param arg: pointer to the kaapi_processor_t structure
  */
  void* kaapi_sched_run_processor( void* arg );

  /** Allocate the array of processors
  */
  kaapi_processor_t** kaapi_allocate_processors( int kproc, cpu_set_t cpuset);

  /** Allocate one processor
      Inherited the cpuset from the running thread.
  */
  kaapi_processor_t* kaapi_allocate_processor();

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
  

  /** Called when a thread is suspended on an instruction
  */
  int kaapi_sched_suspend   ( kaapi_processor_t* proc, kaapi_thread_descr_t* thread, kaapi_test_wakeup_t fwakeup, void* argwakeup );

  
  /** Enter in the infinite loop of trying to steal work.
      Never return from this function...
      If proc is null pointer, then the function allocate a new kaapi_processor_t and assigns it to the current processor.
      This method may be called by 'host' thread in order to become executor thread.
      The method returns in case of terminaison.
  */
  void kaapi_sched_idle      ( kaapi_processor_t* proc );
  
  /* Return 1 iff it should redo call to its entrypoint, else never return
   */
  int kaapi_sched_terminate_or_redo (kaapi_processor_t* proc, kaapi_thread_descr_t* thread);
  
  /** Client interface to steal a given kaapi_processor_t
   */
//  kaapi_thread_descr_t* kaapi_sched_do_steal( kaapi_processor_t* proc );
  
  /**
   */
  struct kaapi_thread_descr_t* allocate_thread_descriptor( int scope, int detachstate );
  
  /**
   */
  void deallocate_thread_descriptor( struct kaapi_processor_t* proc, struct kaapi_thread_descr_t* thread );
  
  /* ========================================================================= */
  /* INTERNAL numa Interface 
   */
#if defined(HAVE_NUMA_H)
  
#endif
  
  
  /* ========================================================================= */
  /* Initialization / destruction functions
   */
  
  //void __attribute__ ((constructor)) my_init(void);
  
  //void __attribute__ ((destructor)) my_fini(void);
  
  /**
   */
  void _kaapi_dummy(void*);
  
  /** To force reference to kaapi_init.c in order to link against kaapi_init and kaapi_fini
   */
  static void __attribute__((unused)) __kaapi_dumy_dummy(void)
  {
    _kaapi_dummy(NULL);
  }
  

#ifdef __cplusplus
}
#endif

#endif // _KAAPI_IMPL_H
