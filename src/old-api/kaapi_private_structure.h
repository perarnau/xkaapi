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

/** \defgroup THREAD Thread
    This group defines the functions of the subset of POSIX Thread implemented in Kaapi.
*/
/** \defgroup WS Workstealing
    This group defines the functions to manage workstealing
*/

/* fwd declaration */
struct kaapi_thread_processor_t;
struct kaapi_thread_descr_t;

/* ========================================================================= */
/** Thread identifier
    \ingroup THREAD
    A thread identifier is a system wide identifier of thread that allows
    to retreive the location of the thread. 
    The identifier is unique between all threads on all process.
    
    Bits are currently reserved in this maner:
    31|30: 00 -> process scope thread
    31|30: 01 -> processor scope thread
    31|30: 11 -> system scope thread
*/
typedef kaapi_uint32_t kaapi_thread_id_t;

/* ========================================================================== */
/** Table of thread data specific 
    \ingroup THREAD
    This is an array of size KAAPI_KEYS_MAX pointer to user thread data specific
*/
typedef void** kaapi_key_table_t;


/* ========================================================================== */
#define KAAPI_CONTEXT_SAVE_CSTACK  0x1   /* 0001 */
#define KAAPI_CONTEXT_SAVE_KSTACK  0x2   /* 0010 */
#define KAAPI_CONTEXT_SAVE_ALL     0x3   /* 0011 */

#if defined(KAAPI_USE_SETJMP)
#  include <setjmp.h>
#elif defined(KAAPI_USE_UCONTEXT)
#  include <ucontext.h>
#else
#  error "Unknown context definition"
#endif



/* fwd decl */
typedef struct kaapi_thread_descr_t kaapi_thread_descr_t;
typedef void* (*kaapi_run_entrypoint_t)(void*);
typedef int (*kaapi_test_wakeup_t)(void*);
typedef void (*kaapi_thread_wakeup_t)(void*);


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


/** Linked list used to manage list of suspended threads on a condition or the list of ready threads
    \ingroup THREAD
    The cell data structure should has a scope at least until the thread resume its executions.
    Typically this data structure is an automatic variable allocated in the call of kaapi_sched_suspend:
    when the thread resume its executions, the thread has been removed from the workqueue of suspended threads 
    and the cell data structure has been removed.
    
    When a running k-processor or an other thread tries to wakeup a thread, it iterates through the list: 
    for each item of kaapi_cellsuspended_t, it first test to wakeup the suspended thread by calling _fwakeup.
    This function should return !=0 value iff the thread may wakeup. It that case, 2/ the cell data structure
    is removed from the queue and 3/ the thread is wakeuped by calling the method fwakeup (if it is no null). 
    If _fwakeup function is null then the thread is started on the k-processor.
    The previous operations always in the order 1/; if 1/ is true then 2/, then 3/.
*/
typedef struct kaapi_cellsuspended_t {
  struct kaapi_thread_descr_t*  thread;            /* the thread */
  kaapi_test_wakeup_t           f_test_wakeup;     /* return != 0 iff the thread could be wakeuped */
  kaapi_thread_wakeup_t         f_wakeup;          /* called to wakeup the thread */
  void*                         arg_fwakeup;       /* arg to func twakeup */
  KAAPI_FIFO_CELL_FIELD(struct kaapi_cellsuspended_t);
} kaapi_cellsuspended_t;


/** Workqueue of suspended threads
    \ingroup THREAD
*/
typedef struct kaapi_workqueue_suspended_t {
  KAAPI_FIFO_DECLARE_FIELD(struct kaapi_cellsuspended_t);
} kaapi_workqueue_suspended_t;

/** Workqueue of ready threads
    \ingroup THREAD
*/
typedef struct kaapi_workqueue_ready_t {
  KAAPI_FIFO_DECLARE_FIELD(struct kaapi_cellsuspended_t);
} kaapi_workqueue_ready_t;


/* ========================================================================== */
/** Mutex attribut data structure
    \ingroup THREAD
 */
typedef struct kaapi_mutexattr_t {
  int _type;
} kaapi_mutexattr_t;

/** Mutex data structure
    \ingroup THREAD
 */
typedef struct kaapi_mutex_t {
  kaapi_atomic_t              _lock;   /* 0: free, 1: locked */
  kaapi_thread_descr_t*       _owner;
  int                         _nb_lock;
  int                         _type;
  pthread_mutex_t             _mutex;
  kaapi_workqueue_suspended_t _list;
} kaapi_mutex_t;

typedef struct kaapi_test_and_lock__t {
  kaapi_thread_descr_t*       thread;
  kaapi_mutex_t*              mutex;
  kaapi_workqueue_suspended_t list;
} kaapi_test_and_lock__t;

/** Ticket Mutex data structure
    \ingroup THREAD
    This mutex is only used in internal function of Kaapi
 */
typedef struct kaapi_ticketmutex_t {
  kaapi_atomic_t        _queueticket;      /* if _queueticket == dequeueticket then lock acquired */
  kaapi_atomic_t        _dequeueticket;    /* incremented on unlock */
} kaapi_ticketmutex_t;


/* ========================================================================= */
typedef struct kaapi_timed_test_and_lock__t {
  kaapi_thread_descr_t  *thread;
  const struct timespec *abstime;
  kaapi_mutex_t*         mutex;
  int                    retval;
  KAAPI_FIFO_CELL_FIELD (struct kaapi_timed_test_and_lock__t);
} kaapi_timed_test_and_lock__t;

/*KAAPI_QUEUE_DECLARE( kaapi_kttl_queue_t, kaapi_timed_test_and_lock__t );*/

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
  kaapi_workqueue_suspended_t _list_s_threads;  /* List of only SYSTEM_SCOPE threads. */
  kaapi_workqueue_suspended_t _list_p_threads;  /* List of only PROCESS_SCOPE threads. */
} kaapi_cond_t;


/* ========================================================================= */
/** Thread descr attribut
    \ingroup THREAD
 */
typedef struct kaapi_attr_t {
  int                      _detachstate;
  int                      _scope;
#if 0 && !defined(KAAPI_USE_SCHED_AFFINITY)
 /* affinity should be redisign without cpuset_t (do not scale) */
  cpu_set_t                _cpuset;
#endif
  size_t                   _stacksize;
  void*                    _stackaddr;
} kaapi_attr_t;

extern kaapi_attr_t kaapi_default_attr;

#define KAAPI_ATTR_INITIALIZER kaapi_default_attr

/* ========================================================================== */
typedef enum kaapi_futur_state_t {
  KAAPI_FUTUR_S_DETACHED      = 0x1,
  KAAPI_FUTUR_MASK_DETACHED   = 0x1,
  KAAPI_FUTUR_S_TERMINATED    = 0x2,
  KAAPI_FUTUR_MASK_TERMINATED = 0x2
} kaapi_futur_state_t;


/**
*/
typedef struct kaapi_thread_futur_t {
  void* volatile result;
  int            state;     /* see state above */
  kaapi_cond_t   condition;
} kaapi_thread_futur_t;



/* ========================================================================== */
/** Request send by a processor
    \ingroup WS
*/
typedef struct kaapi_steal_request_t {
  KAAPI_INHERITE_FROM_SSTRUCT_T;                  /* read-write synchro status field */
  kaapi_thread_id_t              _tid;            /* system wide id of the thief (in) or the victim (out) */
  kaapi_stack_t*                 _thiefstack;     /* system wide index of the thief stack where to store result of the thief */
  char _data[1];                                  /* where to store application dependent data */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_steal_request_t;

/** Size of _data field in kaapi_steal_request_t
    \ingroup WS
*/
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
  KAAPI_THREAD_S_WAITING    = 5,
  KAAPI_THREAD_S_TERMINATED = 6,
  KAAPI_THREAD_S_DESTROY    = 7

} kaapi_thread_state_t;


/** Thread data structure
    \ingroup THREAD
*/
typedef struct kaapi_thread_descr_t {
  volatile kaapi_uint8_t           state;            /* see definition above */
  kaapi_uint8_t                    scope;            /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_uint16_t                   affinity;         /* affinity attribut */  
  kaapi_uint16_t                   pagesize;         /* number of pages or the thread descriptor */
  kaapi_thread_futur_t*            futur;            /* futur return value: ==0 if the thread is deatached */
} kaapi_thread_descr_t;

/** Process scope thread descriptor
    \ingroup THREAD
*/
typedef struct kaapi_thread_descr_process_t {
  volatile kaapi_uint8_t           state;            /* see definition above */
  kaapi_uint8_t                    scope;            /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_uint16_t                   affinity;         /* affinity attribut */  
  kaapi_uint16_t                   pagesize;         /* number of pages or the thread descriptor */
  kaapi_thread_futur_t*            futur;            /* futur return value: ==0 if the thread is deatached */

  kaapi_thread_context_t           ctxt;           /* process scope thread context */
  kaapi_thread_processor_t*        proc;            /* attached processor */
} kaapi_thread_descr_process_t;

/** This data structure defines a work stealer processor thread
    \ingroup THREAD WS
    WARNING: Note that begining of the data structure should shared the same fields, in the same order,
    that the kaapi_thread_descr_system_t data structure.
*/
typedef struct kaapi_thread_processor_t {
  volatile kaapi_uint8_t           state;            /* see definition above */
  kaapi_uint8_t                    scope;            /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_uint16_t                   affinity;         /* affinity attribut */  
  kaapi_uint16_t                   pagesize;         /* number of pages or the thread descriptor */
  kaapi_thread_futur_t*            futur;            /* futur return value: ==0 if the thread is deatached */

  kaapi_thread_descr_process_t*    active_thread;     /* current context of the kaapi_thread_descr_process_t  */
  kaapi_cellsuspended_t*           tosuspend_thread;  /* cell & thread to put in list of suspended thread */
  kaapi_workqueue_ready_t          ready_threads;     /* the suspended thread that directly call sched_idle */
  kaapi_workqueue_suspended_t      suspended_threads; /* the suspended thread that directly call sched_idle */
  
} kaapi_thread_processor_t;

#define KAAPI_PROCESSOR_GETINDEX( proc ) \
  ((proc)->_ctxt.tid & 0xFF )

/** System scope thread attribut
    \ingroup THREAD
*/
typedef struct kaapi_thread_descr_system_t {
  volatile kaapi_uint8_t           state;            /* see definition above */
  kaapi_uint8_t                    scope;            /* contention scope of the lazy thread == PROCESS_SCOPE */
  kaapi_uint16_t                   affinity;         /* affinity attribut */  
  kaapi_uint16_t                   pagesize;         /* number of pages or the thread descriptor */
  kaapi_thread_futur_t*            futur;            /* futur return value: ==0 if the thread is deatached */

  kaapi_thread_id_t                tid;             /* thread id, system wide */
  kaapi_stack_t                    kstack;           /* */
  kaapi_key_table_t                dataspecific;    /* data specific table */
  pthread_t                        pthid;           /* iff scope system or processor */
  void*                            arg_entrypoint;  /* due to use of a trampoline function */
  void*                            (*entrypoint)(void*);  /* due to use of a trampoline function */
  pthread_cond_t                   cond;            /* use by system scope thread to sleep and wakeup system thread */  
} kaapi_thread_descr_system_t;



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
