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
#include <stdint.h>
#include <pthread.h>
#include <limits.h>

/* ========================================================================= */
/* Kaapi name for stdint typedefs.
 */
typedef uint8_t  kaapi_uint8_t;
typedef uint16_t kaapi_uint16_t;
typedef uint32_t kaapi_uint32_t;
typedef int8_t   kaapi_int8_t;
typedef int16_t  kaapi_int16_t;
typedef int32_t  kaapi_int32_t;

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

typedef struct kaapi_thread_descr_t kaapi_thread_descr_t;
typedef struct kaapi_processor_t kaapi_processor_t;
typedef void* (*kaapi_run_entrypoint_t)(void*);
typedef int (*kaapi_test_wakeup_t)(void*);


/* ========================================================================== */
/** Cpu_set
 */
#ifndef HAVE_CPU_SET_T
#define HAVE_CPU_SET_T
#include <strings.h> /* for bzero in cpu set */


/*#ifndef __cpu_set_t_defined */
/* Size definition for CPU sets. All */
#  define __CPU_SETSIZE	64
#  define __NCPUBITS	(8 * sizeof (__cpu_mask))

/* Type for array elements in 'cpu_set_t'.  */
typedef unsigned long int __cpu_mask;

/* Data structure to describe CPU mask.  */
typedef struct {
  __cpu_mask __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;


#define CPU_ZERO( pcpuset ) bzero( pcpuset, sizeof(cpu_set_t) )
  
#define CPU_SET( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] |= \
(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))

#define CPU_CLR( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] &= \
~(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))

#define CPU_ISSET( cpu, pcpuset ) (pcpuset)->__bits[ cpu / __NCPUBITS] & \
(1UL << (cpu -(cpu / __NCPUBITS)*__NCPUBITS))


/*#endif*/
#endif

#define CPU_SETALL( pcpuset ) \
  { int i; \
    for (i=0; i<__CPU_SETSIZE / __NCPUBITS; ++i)\
      (pcpuset)->__bits[ i ] = ~0UL;\
  }

/* min_index_cpu_set[ cpu_set & 0xFF ] returns the smallest index i such that 2^i is the least significant bit of cpu_set & 0xFF
   kaapi_min_index_cpu_set[0] = -1.
*/
extern int kaapi_min_index_cpu_set[256]; 

/* Returns the least significant index i such that 2^i is the least significant bit of the intersection 
   If no intersection returns -1
*/
static inline int CPU_INTERSECT( cpu_set_t* pcpuset1, cpu_set_t* pcpuset2 ) 
{
  int i, j;
  __cpu_mask retval;
  for (i=0; i< __CPU_SETSIZE / __NCPUBITS; ++i)
  {
    retval = (pcpuset1->__bits[i] & pcpuset2->__bits[i]);
    if (retval !=0) 
    {
      for (j=0; j<__NCPUBITS/8; ++j)
      {
        int idx = kaapi_min_index_cpu_set[ (retval >> j*8) & 0xFF ];
        if (idx !=-1) return idx;
      }
    }
  }
  return -1;
}


/* ========================================================================== */
/** Atomic type
 */
typedef struct kaapi_atomic_t {
  volatile int _counter;
} kaapi_atomic_t;


/* ========================================================================== */
/** Termination dectecting barrier
*/
typedef kaapi_atomic_t kaapi_barrier_td_t;

/* ========================================================================== */
/** Mutex attribut, data structure
 */
typedef struct kaapi_mutexattr_t {
  int _type;
} kaapi_mutexattr_t;

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
/** Condition attribut, data structure
 */
typedef struct kaapi_timed_test_and_lock__t {
  kaapi_thread_descr_t  *thread;
  const struct timespec *abstime;
  kaapi_mutex_t*         mutex;
  int                    retval;
  KAAPI_QUEUE_FIELD (struct kaapi_timed_test_and_lock__t);
} kaapi_timed_test_and_lock__t;

KAAPI_QUEUE_DECLARE( kaapi_system_thread_queue_t, struct kaapi_thread_descr_t );
KAAPI_QUEUE_DECLARE( kaapi_kttl_queue_t, kaapi_timed_test_and_lock__t );

typedef struct kaapi_condattr_t {
  int _unused;
} kaapi_condattr_t;

typedef struct kaapi_cond_t {
  pthread_mutex_t             _mutex;
  kaapi_system_thread_queue_t _th_q;   /* List of only SYSTEM_SCOPE threads. */
  kaapi_kttl_queue_t          _kttl_q; /* List of only PROCESS_SCOPE threads. */
} kaapi_cond_t;


/* ========================================================================= */
/** Thread descr, attribut
 */
typedef enum kaapi_thread_state_t {
  KAAPI_THREAD_ALLOCATED,
  KAAPI_THREAD_LAZY,
  KAAPI_THREAD_CREATED,
  KAAPI_THREAD_RUNNING,
  KAAPI_THREAD_SUSPEND,
  KAAPI_THREAD_TERMINATED
} kaapi_thread_state_t;

typedef struct kaapi_attr_t {
  int                      _detachstate;
  int                      _scope;
  cpu_set_t                _cpuset;
  size_t                   _stacksize;
  void*                    _stackaddr;
} kaapi_attr_t;

extern kaapi_attr_t kaapi_default_attr;

#define KAAPI_ATTR_INITIALIZER kaapi_default_attr


/* ========================================================================= */
/** Once_t
*/
typedef kaapi_atomic_t kaapi_once_t;

/* ========================================================================= */
/** Dataspecific data structures and functions
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

/* global data */
extern kaapi_processor_t** kaapi_all_processors;

/* points to the current running thread
 If the running thread is of kind PROCESSOR_SCOPE, then it is a kaapi processor and the running (active)
 thread is thread->_proc->_active_thread
 */
extern pthread_key_t kaapi_current_thread_key;



/* ========================================================================= */
/** Task Condition structure 
*/
struct kaapi_task_t;
struct kaapi_stack_t;
struct kaapi_task_condition_t;

/*------- begin same as in kaapi_type.h.in !!!!! */
/** \defgroup Constants for number for fixed size parameter of a task
*/
/*@{*/
#define KAAPI_TASK_MAX_DATA  24 /* allows 3 double */
#define KAAPI_TASK_MAX_IDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint32_t))
#define KAAPI_TASK_MAX_SDATA (KAAPI_TASK_MAX_DATA/sizeof(kaapi_uint16_t))
#define KAAPI_TASK_MAX_DDATA (KAAPI_TASK_MAX_DATA/sizeof(double))
#define KAAPI_TASK_MAX_FDATA (KAAPI_TASK_MAX_DATA/sizeof(float))
#define KAAPI_TASK_MAX_PDATA (KAAPI_TASK_MAX_DATA/sizeof(void*))
/*@}*/

/* ========================================================================= */
/** Task body 
*/
typedef void (*kaapi_task_body_t)(struct kaapi_task_t* /*task*/, struct kaapi_stack_t* /* thread */);

/** Kaapi task definition
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  /* state is the public member: initialized with the values above. It is used
     to initializ. istate is the internal state.
*/     
  kaapi_uint32_t     format;   /** */
  union {
    kaapi_uint16_t   locality; /** locality number see documentation */
    kaapi_uint16_t   event;    /** in case of adaptive task */
  };
  union {
    kaapi_uint16_t   state;    /** State of the task see above + flags in order to initialze both in 1 op */
    struct { 
      kaapi_uint8_t  xstate;   /** State of the task see above */
      kaapi_uint8_t  flags;    /** flags of the task see above */
    };
  };
  kaapi_task_body_t  body;     /** C function that represent the body to execute*/
  union { /* union data to view task's immediate data with type information. Be carreful: this is an (anonymous) union  */
    kaapi_uint32_t idata[ KAAPI_TASK_MAX_IDATA ];
    kaapi_uint16_t sdata[ KAAPI_TASK_MAX_SDATA ];
    double         ddata[ KAAPI_TASK_MAX_DDATA ];
    float          fdata[ KAAPI_TASK_MAX_FDATA ];
    void*          pdata[ KAAPI_TASK_MAX_PDATA ];
    struct kaapi_task_condition_t* arg_condition;
  };
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_task_t;
/*------- end same as in kaapi_type.h.in !!!!! */

/** Kaapi stack of task definition
*/
typedef struct kaapi_stack_t {
  kaapi_task_t* pc;             /** task counter: next task to execute, 0 if empty stack */
  kaapi_task_t* sp;             /** stack counter: next free task entry */
#if defined(KAAPI_DEBUG)
  kaapi_task_t* end_sp;         /** past the last stack counter: next entry after the last task in stack array */
#endif
  kaapi_task_t* task;           /** stack of tasks */

  char*         sp_data;        /** stack counter for the data: next free data entry */
#if defined(KAAPI_DEBUG)
  char*         end_sp_data;    /** past the last stack counter: next entry after the last task in stack array */
#endif
  char*         data;           /** stack of data with the same scope than task */
} kaapi_stack_t;

/** Kaapi frame definition
*/
typedef struct kaapi_frame_t {
    kaapi_task_t* pc;
    kaapi_task_t* sp;
    char*         sp_data;
} kaapi_frame_t;

/** Condition for task
*/
typedef struct kaapi_task_condition_t {
  kaapi_task_body_t  save_body;            /** C function that represent the body to execute*/
  void*              save_arg_condition;   /** used to restore correct task information */
} kaapi_task_condition_t;


#endif /* _KAAPI_PRIVATE_STRUCTURE_H */
