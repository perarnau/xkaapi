/*
** kaapi_private_structure.h
** xkaapi
** 
** Created on Tue Mar 31 15:18:09 2009
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
#ifndef _KAAPI_PRIVATE_STRUCTURE_H
#define _KAAPI_PRIVATE_STRUCTURE_H 1

/* Avoid multiple definitions with public types. */
#define _KAAPI_TYPE_H 1

#ifdef KAAPI_USE_APPLE
#  include <time.h>
#endif

#include "kaapi_config.h"
#include "kaapi_datastructure.h"
#include "kaapi_atomic.h"
#include <pthread.h>
#include <limits.h>
#include <stddef.h>
#include "kaapi_common.h"


/* ========================================================================== */
/** Typedef
 */
#if defined(KAAPI_USE_SETJMP)
#  include <setjmp.h>
typedef jmp_buf kaapi_thread_context_t;
#elif defined(KAAPI_USE_UCONTEXT)
#  include <ucontext.h>
typedef ucontext_t kaapi_thread_context_t;
#else
#  error "Unknown context definition"
#endif

/* fwd decl */
struct kaapi_thread_descr_t;
typedef struct kaapi_thread_descr_t kaapi_thread_descr_t;
typedef void* (*kaapi_run_entrypoint_t)(void*);
typedef int (*kaapi_test_wakeup_t)(void*);

/** \defgroup THREAD Thread
    This group defines the functions of the subset of POSIX Thread implemented in Kaapi.
*/
/** \defgroup WS Workstealing
    This group defines the functions to manage workstealing
*/


/* ========================================================================== */
/** Atomic type
    \ingroup THREAD
 */
typedef struct kaapi_atomic_t {
  volatile int _counter;
} kaapi_atomic_t;


/* ========================================================================== */
/** Termination dectecting barrier
    \ingroup THREAD
*/
typedef kaapi_atomic_t kaapi_barrier_td_t;


/* ========================================================================== */
/** Mutex attribut data structure
    \ingroup THREAD
 */
typedef struct kaapi_mutexattr_t {
  int _type;
} kaapi_mutexattr_t;

/** Mutex attribut
    \ingroup THREAD
 */
typedef struct kaapi_mutex_t {
  kaapi_atomic_t        _lock;   /* 0: free, 1: locked */
  kaapi_thread_descr_t* _owner;
  int                   _nb_lock;
  int                   _type;
  pthread_mutex_t       _mutex;
  KAAPI_QUEUE_DECLARE_FIELD (kaapi_thread_descr_t);
} kaapi_mutex_t;

typedef struct kaapi_test_and_lock__t {
  kaapi_thread_descr_t*       thread;
  kaapi_mutex_t*              mutex;
  KAAPI_QUEUE_FIELD (struct kaapi_test_and_lock__t);
} kaapi_test_and_lock__t;


/* ========================================================================= */
typedef struct kaapi_timed_test_and_lock__t {
  kaapi_thread_descr_t  *thread;
  const struct timespec *abstime;
  kaapi_mutex_t*         mutex;
  int                    retval;
  KAAPI_QUEUE_FIELD (struct kaapi_timed_test_and_lock__t);
} kaapi_timed_test_and_lock__t;

KAAPI_QUEUE_DECLARE( kaapi_kttl_queue_t, kaapi_timed_test_and_lock__t );

/** Condition attribut data structure
    \ingroup THREAD
 */
typedef struct kaapi_condattr_t {
  int _unused;
} kaapi_condattr_t;


KAAPI_QUEUE_DECLARE( kaapi_system_thread_queue_t, struct kaapi_thread_descr_t );

/** Condition data structure
    \ingroup THREAD
 */
typedef struct kaapi_cond_t {
  pthread_mutex_t             _mutex;
  kaapi_system_thread_queue_t _th_q;   /* List of only SYSTEM_SCOPE threads. */
  kaapi_kttl_queue_t          _kttl_q; /* List of only PROCESS_SCOPE threads. */
} kaapi_cond_t;


/* ========================================================================= */
/** Thread descr attribut
    \ingroup THREAD
 */
typedef struct kaapi_attr_t {
  int                      _detachstate;
  int                      _scope;
#if 0
  cpu_set_t                _cpuset;
#endif
  size_t                   _stacksize;
  void*                    _stackaddr;
} kaapi_attr_t;

extern kaapi_attr_t kaapi_default_attr;

#define KAAPI_ATTR_INITIALIZER kaapi_default_attr


/* ========================================================================= */
/** Thread identifier
    \ingroup THREAD
    A thread identifier is a system wide identifier of thread that allows
    to retreive the location of the thread. 
    The identifier is unique between all threads on all process.
*/
typedef kaapi_uint32_t kaapi_thread_id_t;


/** Request send by a processor
    \ingroup WS
*/
typedef struct kaapi_steal_request_t {
  KAAPI_INHERITE_FROM_SSTRUCT_T;                  /* read-write synchro status field */
  kaapi_thread_id_t              _tid;            /* system wide id of the thief (in) or the victim (out) */
  kaapi_uint32_t                 _site;           /* identifier of the thief process (in) or the victim process (out) */
  kaapi_stack_t*                 _thiefstack;     /* system wide index of the thief stack where to store result of the thief */
  char _data[1];                                  /* where to store application dependent data */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_steal_request_t;
 
#define KAAPI_REQUEST_DATA_SIZE (sizeof(kaapi_steal_request_t) - offsetof( kaapi_steal_request_t, data))

/** Status of a request
    \ingroup WS
*/
enum kaapi_steal_request_status_t {
  KAAPI_REQUEST_S_CREATED = 0,
  KAAPI_REQUEST_S_POSTED  = 1,
  KAAPI_REQUEST_S_SUCCESS = 2,
  KAAPI_REQUEST_S_FAIL    = 3,
  KAAPI_REQUEST_S_ERROR   = 4,
  KAAPI_REQUEST_S_QUIT    = 5
};


/* ========================================================================= */
/** Kaapi thread state 
    \ingroup THREAD
    In Kaapi, 2 kind of thread exists: POSIX like process scope thread (called user thread),
    POSIX like system scope thread (called kernel thread).
    A kernel thread responsible for stealing work is called k-processor. During work stealing
    algorithm, a K-processor tries to steal work, this work is called k-thread. A K-thread is
    a user level thread. 
    At the beginning of the execution, a K-processor executes the main K-thread with is defined
    by the control flow executing the main entry point of the program.

    \dot
    digraph StateThread {
      Created   [label="Created", shape=ellipse, style=filled, color=gray]
      Running   [label="Running", shape=ellipse, style=filled, color=gray]
      Suspended [label="Suspended", shape=ellipse, style=filled, color=blue]
      Stopped   [label="Stopped", shape=ellipse, style=filled, color=gray]
      Thief     [label="Thief", shape=ellipse, style=filled, color=indianred]

      Terminated [label="Terminated", shape=ellipse, style=filled, color=gray]

      Created -> Running [label ="Start of the thread"]
      Running -> Suspended [label="Thread suspended on a lock,\na condition or\nany other synchronization"]
      Running -> Thief [label="The K-thread is\n stealing work"]
      Running -> Stopped [label="Signal to stop"]
      Suspended -> Running [label="Condition is satisfied"]
      Thief   -> Running [label="The K-processor runs K-thread"]
      Thief   -> Stopped [label="The K-processor was stopped"]
      Thief   -> Suspended [label="Post unbound request"]
      Stopped -> Running [label="Signal to continue"]
      Stopped -> Thief [label="Signal continue\non a K-processor "]
      Suspended -> Stopped [label="Signal to stop"]
      Suspended  -> Thief [label="Receive reply"]
      Stopped -> Suspended [label="Thread suspended,\nSignal to continue"]
      Running -> Terminated [label="End of the computation"]
    }
    \enddot
    
    A REVOIR:
    Depending of the type of the thread, not all states are reachable. POSIX (user threads and kernel threads)
    will be able to take each transition except those move to a state in read ('Thief' state).
    Moreover, K-processor thread may by suspended (blue state) only when is post request with unbound delay in the
    replay (such that doing remote work stealing operation).
    If the K-processor switches to running, it means that a K-thread has being to executed by the K-processor.
    When the K-thread suspends, the K-processor switches to execute the work stealing algorithm until work was found
    (2 changements d'état: K-thread Running -> Suspend, K-processor Running -> Thief.... ça fait beaucoup.
    
    When a K-thread (=user level thread) is under beging suspended, it call a function kaapi_sched_suspend that
    will but it on the queue of suspended thread of its processor. In order that the thread may be wakeup from
    the queue, it evaluates a condition_function on the suspended threads and some data. In this way, a K-thread
    may be suspended because it tries to acquire a lock, waiting on a condition or waiting on a data flow constraints.
*/
typedef enum kaapi_thread_state_t {
  KAAPI_THREAD_S_ALLOCATED  = 0,
  KAAPI_THREAD_S_CREATED    = 1,
  KAAPI_THREAD_S_RUNNING    = 2,
  KAAPI_THREAD_S_SUSPEND    = 3,
  KAAPI_THREAD_S_STEALING   = 4,
  KAAPI_THREAD_S_TERMINATED = 5,
  KAAPI_THREAD_S_DESTROY    = 6
} kaapi_thread_state_t;

/** Linked list used to manage list of suspended threads on a condition or the list of ready threads
    \ingroup THREAD
    The cell data structure should has a scope at least until the thread resume its executions.
    Typically this data structure is an automatic variable allocated in the call of kaapi_sched_suspend:
    when the thread resume its executions, the thread has been removed from the workqueue and the cell
    data structure could be deleted.
*/
typedef struct kaapi_cellsuspended_t {
  struct kaapi_thread_descr_t*  _thread;      /* the thread */
  kaapi_test_wakeup_t           _fwakeup;     /* != 0 if the thread is suspended */
  void*                         _arg_fwakeup; /* arg to func twakeup */
  struct kaapi_cellsuspended_t* _next;
} kaapi_cellsuspended_t;


/** Workqueue of suspended threads
    \ingroup THREAD
*/
typedef struct kaapi_workqueue_suspended_t {
  kaapi_cellsuspended_t* _head;
} kaapi_workqueue_suspended_t;

KAAPI_FIFO_DECLARE( kaapi_workqueue_ready_t, struct kaapi_thread_descr_t);

/** This data structure defines a work stealer processor thread
    \ingroup THREAD WS
    WARNING: Note that begining of the data structure should shared the same fields, in the same order,
    that the kaapi_thread_descr_system_t data structure.
*/
typedef struct kaapi_thread_descr_processor_t {
  pthread_t                        _pthid;             /* iff scope system or processor */
  pthread_cond_t                   _cond;              /* use by system scope thread to be signaled */  
  struct kaapi_thread_descr_t*     _stealer_thread;    /* the thread that will store results of a steal */
  struct kaapi_thread_descr_t*     _active_thread;     /* the active thread or 0 */
  struct kaapi_thread_descr_t*     _kill_thread;       /* the thread to kill */

  kaapi_workqueue_ready_t          _ready_threads;     /* the suspended thread that directly call sched_idle */
  kaapi_workqueue_suspended_t      _suspended_threads; /* the suspended thread that directly call sched_idle */
} __attribute__ ((aligned (KAAPI_CACHE_LINE))) kaapi_thread_descr_processor_t;


/** System scope thread attribut
    \ingroup THREAD
*/
typedef struct kaapi_thread_descr_system_t {
  pthread_t                        _pthid;           /* iff scope system or processor */
  pthread_cond_t                   _cond;            /* use by system scope thread to be signaled */  
} kaapi_thread_descr_system_t;

/** Process scope thread descriptor
    \ingroup THREAD
*/
typedef struct kaapi_thread_descr_process_t {
  kaapi_thread_context_t           _ctxt;           /* process scope thread context */
  kaapi_thread_descr_processor_t*  _proc;           /* iff scope == PROCESS_SCOPE, always schedule by a processor */

} kaapi_thread_descr_process_t;


/** Thread data structure
    \ingroup THREAD
*/
struct kaapi_thread_descr_t {
  volatile kaapi_thread_state_t    _state;            /* see definition */
  kaapi_uint16_t                   _scope;            /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_uint16_t                   _pagesize;         /* number of pages or the thread descriptor */
  
  KAAPI_QUEUE_FIELD(struct kaapi_thread_descr_t);     /* system scope attribut to be used in Queue, Stack and Fifo. */
  kaapi_stack_t                    _stack;            /* stack of tasks (rfo, dfg, adaptive) for user threads (process & system only) */
  kaapi_run_entrypoint_t           _run_entrypoint;   /* the entry point */
  void*                            _arg_entrypoint;   /* the argument of the entry point */
  void*                            _return_value;     /* POSIX return value */
  kaapi_uint16_t                   _detachstate;      /* 1 iff the thread is created in datachstate (then x_join fields are not init */
  kaapi_uint16_t                   _affinity;         /* affinity attribut */
  size_t                           _stacksize;        /* stack size attribut */
  void*                            _stackaddr;        /* thread stack base pointer */
  void**                           _key_table;        /* thread data specific */

  union {
    kaapi_thread_descr_process_t   p;
    kaapi_thread_descr_system_t    s;
    kaapi_thread_descr_processor_t k;
  } th;
};


/* ========================================================================= */
/** Once data structure
    \ingroup THREAD
*/
typedef kaapi_atomic_t kaapi_once_t;

/* ========================================================================= */
/** Dataspecific data structures and functions
    \ingroup THREAD
 */
typedef void (*kaapi_dataspecific_destructor_t)(void *);

typedef struct kaapi_global_key{
  kaapi_dataspecific_destructor_t dest;
  long                            next;
} kaapi_global_key;

extern kaapi_global_key kaapi_global_keys[KAAPI_KEYS_MAX];
extern kaapi_atomic_t kaapi_global_keys_front;


/* ========================================================================= */
/** Once data structure
 */
#if defined(KAAPI_USE_SCHED_AFFINITY)
/** Mapping of virtual kaapi processor to physical cpu.
 In the execution model of KAAPI, a set of physical cpus is used to schedule kernel threads created by xkaapi.
 This set of physicall cpus is called virtual kaapi processors: KAAPI maintains a mapping of a virtual kaapi
 processor to a physical cpu.
 A kernel thread in KAAPI could be attached to a cpuset of virtual kaapi processors that is mapped on physical
 cpu.
 kaapi_countcpu is the number of physical cpus (or virtual kaapi processors) used, (number of bits in KAAPI_CPUSET)
 kaapi_kproc2cpu[i] is equal to k means that the i-th virtual processor is mapped onto the k-th CPU into the OS
 */
extern int  kaapi_countcpu;
extern int* kaapi_kproc2cpu;
#endif

#if defined(HAVE_NUMA_H)
/** Data structure that defines the hierarchy of NUMA nodes
 */
typedef struct kaapi_neighbor_t {
  int  count; 
  int  kprocs[2];       /* of size count */
} kaapi_neighbor_t;

typedef enum kaapi_level_t {
  KAAPI_L1 = 0,
  KAAPI_L2 = 1,
  KAAPI_L3 = 2
} kaapi_level_t;

/** Give for kaapi processor its neighbor at a given level
 kaapi_kprocneighbor[kprocid][LEVEL] returns the list of kaapi processors that shared memory at level LEVEL with 
 kaapi processor kprocid.
 */
typedef kaapi_neighbor_t kaapi_kprocneighbor_t[KAAPI_L3+1];

extern kaapi_kprocneighbor_t* kaapi_kprocneighbor;
#endif

/* global data: array of all K-processors  */
extern kaapi_thread_descr_t** kaapi_all_processors;

/* points to the current running thread
 If the running thread is of kind PROCESSOR_SCOPE, then it is a kaapi processor and the running (active)
 thread is thread->_proc->_active_thread
 */
extern pthread_key_t kaapi_current_thread_key;


/* ========================================================================== */
/** Condition for task
    \ingroup TASK
*/
typedef struct kaapi_task_condition_t {
  kaapi_task_body_t  save_body;            /** C function that represent the body to execute*/
  void*              save_arg_condition;   /** used to restore correct task information */
} kaapi_task_condition_t;


#endif /* _KAAPI_PRIVATE_STRUCTURE_H */
