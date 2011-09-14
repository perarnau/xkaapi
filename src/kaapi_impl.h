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
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
** clement.pernet@imag.fr
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

/* Mark that we compile source of the library.
   Only used to avoid to include public definitition of some types.
*/
#define KAAPI_COMPILE_SOURCE 1

#include "config.h"
#include "kaapi.h"
#include "kaapi_error.h"
#include <string.h>

#include "kaapi_defs.h"
#include "kaapi_allocator.h"

/* Maximal number of recursive calls used to store the stack of frames.
   The value indicates the maximal number of frames that can be pushed
   into the stackframe for each thread.
   
   If an assertion is thrown at runtime, and if this macro appears then
   it is necessary to increase the maximal number of frames in a stack.
*/
#define KAAPI_MAX_RECCALL 256

/* Define if ready list is used
   This flag activates :
   - the use of readythread during work stealing: a thread that signal 
   a task to becomes ready while the associated thread is suspended move
   the thread to a readylist. The ready thread is never stolen and should
   only be used in order to reduce latency to retreive work (typically
   at the end of a steal operation).
   - if a task activates a suspended thread (e.g. bcast tasks) then activated
   thread is put into the readylist of the processor that executes the task.
   The threads in ready list may be stolen by other processors.
*/
#define KAAPI_USE_READYLIST 1

/** Current implementation relies on isoaddress allocation to avoid
    communication during synchronization at the end of partitionning.
*/
//#define KAAPI_ADDRSPACE_ISOADDRESS 1

/* Flags to define method to manage concurrency between victim and thieves
   - CAS: based on atomic modify update
   - THE: based on Dijkstra like protocol to ensure mutual exclusion
   - SEQ: only used to test performances penalty with comparizon of ideal seq. impl.
   Code using SEQ execution method cannot runs with several threads.
*/
#define KAAPI_CAS_METHOD 0
#define KAAPI_THE_METHOD 1
#define KAAPI_SEQ_METHOD 2

/* Selection of the method to manage concurrency between victim/thief 
   to steal task:
*/
#ifndef KAAPI_USE_EXECTASK_METHOD
#define KAAPI_USE_EXECTASK_METHOD KAAPI_CAS_METHOD
#endif


#ifdef __GNU__
#  define likely(x)      __builtin_expect(!!(x), 1)
#  define unlikely(x)    __builtin_expect(!!(x), 0)
#else
#  define likely(x)      (x)
#  define unlikely(x)    (x)
#endif

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
#  ifndef EWOULDBLOCK
#    define EWOULDBLOCK     EAGAIN
#  endif 
#endif 

#include "kaapi_memory.h"


/** This is the new version on top of X-Kaapi
*/
extern const char* get_kaapi_version(void);


/* ========================================================================= */
/** Flag to move all threads into suspend state
*/
extern volatile int kaapi_suspendflag;

/* ========================================================================= */
/** Counter of thread into the suspended state
*/
extern kaapi_atomic_t kaapi_suspendedthreads;


/* ================== Library initialization/terminaison ======================= */
/** Initialize the machine level runtime.
    Return 0 in case of success. Else an error code.
*/
extern int kaapi_mt_init(void);

/** Finalize the machine level runtime.
    Return 0 in case of success. Else an error code.
*/
extern int kaapi_mt_finalize(void);

/** Suspend all threads except the main threads.
    Should be called by the main thread !
*/
extern void kaapi_mt_suspend_threads(void);

/** Call by the threads to be put into suspended state
*/
extern void kaapi_mt_suspend_self( struct kaapi_processor_t* kproc );

/** Resume all threads except the main threads.
*/
extern void kaapi_mt_resume_threads(void);

/** initialize suspend/resume sub-functionnalities 
*/
extern void kaapi_mt_suspendresume_init(void);


/** Initialize hw topo.
    Based on hwloc library.
    Return 0 in case of success else an error code
*/
extern int kaapi_hw_init(void);

/** Initialization of the NUMA affinity workqueue
*/
extern int kaapi_sched_affinity_initialize(void);

/** Destroy
*/
extern void kaapi_sched_affinity_destroy(void);

/* Fwd declaration 
*/
struct kaapi_listrequest_t;
struct kaapi_procinfo_list_t;

/* Fwd declaration
*/
struct kaapi_tasklist_t;
struct kaapi_comlink_t;
struct kaapi_taskdescr_t;

/* ============================= Processor list ============================ */

/* ========================================================================== */
/** kaapi_mt_register_procs
    register the kprocs for mt architecture.
*/
extern int kaapi_mt_register_procs(struct kaapi_procinfo_list_t*);

/* ========================================================================== */
/** kaapi_cuda_register_procs
    register the kprocs for cuda architecture.
*/
#if defined(KAAPI_USE_CUDA)
extern int kaapi_cuda_register_procs(struct kaapi_procinfo_list_t*);
#endif

/* ========================================================================== */
/** free list
*/
extern void kaapi_procinfo_list_free(struct kaapi_procinfo_list_t*);


/* ============================= Task routines ============================ */
/*
*/
#include "kaapi_task.h"


/* ============================= Hash table for WS ============================ */
/* must be include after definition of:
    kaapi_gd_t
    struct kaapi_version_t*
    kaapi_pair_ptrint_t
    struct kaapi_metadata_info_t*
    struct kaapi_taskdescr_t*
*/
#include "kaapi_hashmap.h"

/* ============================= A VICTIM ============================ */
/** \ingroup WS
    This data structure should contains all necessary informations to post a request to a selected node.
    It should be extended in case of remote work stealing.
*/
typedef struct kaapi_victim_t {
  struct kaapi_processor_t* kproc; /** the victim processor */
  uint16_t                  level; /** level in the hierarchy of the source k-processor to reach kproc */
} kaapi_victim_t;


/** Flag to ask generation of a new victim or to report an error
*/
typedef enum kaapi_selecvictim_flag_t {
  KAAPI_SELECT_VICTIM,       /* ask to the selector to return a new victim */
  KAAPI_STEAL_SUCCESS,       /* indicate that previous steal was a success */
  KAAPI_STEAL_FAILED,        /* indicate that previous steal has failed (no work) */   
  KAAPI_STEAL_ERROR          /* indicate that previous steal encounter an error */   
} kaapi_selecvictim_flag_t;


/** \ingroup WS
    Select a victim for next steal request
    \param kproc [IN] the kaapi_processor_t that want to emit a request
    \param victim [OUT] the selection of the victim
    \param victim [IN] a flag to give feedback about the steal operation
    \retval 0 in case of success 
    \retval EINTR in case of detection of the termination 
    \retval else error code    
*/
typedef int (*kaapi_selectvictim_fnc_t)( struct kaapi_processor_t*, struct kaapi_victim_t*, kaapi_selecvictim_flag_t flag );

/** \ingroup WS
    Emit a steal request
    \param kproc [IN] the kaapi_processor_t that want to emit a request
    \retval the stolen thread
*/
typedef struct kaapi_thread_context_t* (*kaapi_emitsteal_fnc_t)(struct kaapi_processor_t*);


/* =======vvvvvvvvvvvvvvvvvv===================== Default parameters ============================ */

/**
*/
enum kaapi_memory_type_t {
  KAAPI_MEM_MACHINE   = 0,
  KAAPI_MEM_NODE      = 1,
  KAAPI_MEM_CACHE     = 2
};

enum kaapi_memory_id_t {
  KAAPI_MEMORY_ID_NODE      = 0,
  KAAPI_MEMORY_ID_CACHE     = 1,
  KAAPI_MAX_MEMORY_ID       = 4  /* reserve 2 and 3 */
};

/** cpuset: at most 128 differents ressources
*/
typedef uint64_t kaapi_cpuset_t[2];

struct kaapi_taskdescr_t;

/**
*/
typedef struct kaapi_affinityset_t {
    kaapi_cpuset_t                 who;       /* who is in this set */
    size_t                         mem_size;
    int                            os_index;  /* numa node id or ??? */
    int                            ncpu;
    short                          type;      /* see kaapi_memory_t */
    struct kaapi_affinity_queue_t* queue;     /* yes ! */ 
} kaapi_affinityset_t;

/**
*/
typedef struct kaapi_hierarchy_one_level_t {
  unsigned short                count;           /* number of kaapi_affinityset_t at this level */
  kaapi_affinityset_t*          affinity; 
} kaapi_hierarchy_one_level_t;

/** Memory hierarchy of the local machine
    * memory.depth: depth of the hierarchy
    * memory.levels[i].affinity.ncpu: number of cpu that share this memory at level i
    * memory.levels[i].affinity.who: cpu set of which PU is contains by memory k at level i
    * memory.levels[i].affinity.mem_size: size of the k memory  at level i
*/
typedef struct kaapi_hierarchy_t {
  unsigned short               depth;
  kaapi_hierarchy_one_level_t* levels;
} kaapi_hierarchy_t;


/** Definition of parameters for the runtime system
*/
typedef struct kaapi_rtparam_t {
  size_t                   stacksize;                       /* default stack size */
  unsigned int             syscpucount;                     /* number of physical cpus of the system */
  unsigned int             cpucount;                        /* number of physical cpu used for execution */
  kaapi_selectvictim_fnc_t wsselect;                        /* default method to select a victim */
  kaapi_emitsteal_fnc_t	   emitsteal;
  unsigned int		       use_affinity;                    /* use cpu affinity */
  int                      display_perfcounter;             /* set to 1 iff KAAPI_DISPLAY_PERF */
  uint64_t                 startuptime;                     /* time at the end of kaapi_init */
  int                      alarmperiod;                     /* period for alarm */

  struct kaapi_procinfo_list_t* kproc_list;                 /* list of kprocessors to initialized */
  kaapi_cpuset_t           usedcpu;                         /* cpuset of used physical ressources */
  kaapi_hierarchy_t        memory;                          /* memory hierarchy */
  unsigned int*	           kid2cpu;                        /* mapping: kid->phys cpu  */
  unsigned int*  	       cpu2kid;                        /* mapping: phys cpu -> kid */
} kaapi_rtparam_t;

extern kaapi_rtparam_t kaapi_default_param;


/* ============================= REQUEST ============================ */
/** Private status of request
    \ingroup WS
*/
enum kaapi_reply_status_t {
  KAAPI_REQUEST_S_POSTED = 0,
  KAAPI_REPLY_S_NOK      = 1,
  KAAPI_REPLY_S_TASK     = 2,
  KAAPI_REPLY_S_TASK_FMT = 3,
  KAAPI_REPLY_S_THREAD   = 4,
  KAAPI_REPLY_S_ERROR    = 5
};



/* ============================= Format for task/data structure ============================ */
#include "kaapi_format.h"

/* ============================= Simple C API for network ============================ */
#include "kaapi_network.h"


/* ============================= The structure for handling suspendended thread ============================ */
/** Forward reference to data structure are defined in kaapi_machine.h
*/
struct kaapi_wsqueuectxt_cell_t;


/* ============================= The thread context data structure ============================ */
/** The thread context data structure
    This data structure should be extend in case where the C-stack is required to be suspended and resumed.
    This data structure is always at position ((kaapi_thread_context_t*)stackaddr) - 1 of stack at address
    stackaddr.
    It was made opaque to the user API because we do not want to expose the way we execute stack in the
    user code.
    
    WARNING: sfp should be the first field of the data structure in order to be able to recover in the public
    API sfp (<=> kaapi_thread_t*) from the kaapi_thread_context_t pointer stored in kaapi_current_thread_key.
*/
typedef struct kaapi_thread_context_t {
  kaapi_frame_t*        volatile sfp;            /** pointer to the current frame (in stackframe) */
  kaapi_frame_t*                 esfp;           /** first frame until to execute all frame  */
  struct kaapi_processor_t*      proc;           /** access to the running processor */
  void*                          pad;            /** a padding */
  kaapi_frame_t                  stackframe[KAAPI_MAX_RECCALL];  /** for execution, see kaapi_thread_execframe */

  /* execution state for stack of task */
#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
  kaapi_task_t*         volatile pc      __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the task the thief wants to steal */
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */
  kaapi_task_t*         volatile thiefpc;        /** pointer to the task the thief wants to steal */
#endif  

#if !defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  kaapi_threadgroup_t            thgrp;          /** the current thread group, used to push task */
#endif

  /* the way to execute task inside a thread, if ==0 uses kaapi_thread_execframe */
  kaapi_threadgroup_t            the_thgrp;      /* not null iff execframe != kaapi_thread_execframe */
  int                            unstealable;    /* !=0 -> cannot be stolen */
  int                            partid;         /* used by static scheduling to identify the thread in the group */
  kaapi_big_hashmap_t*           kversion_hm;    /* used by static scheduling */
  
  struct kaapi_thread_context_t* _next;          /** to be linkable either in proc->lfree or proc->lready */
  struct kaapi_thread_context_t* _prev;          /** to be linkable either in proc->lfree or proc->lready */

#if defined(KAAPI_USE_CUDA)
  kaapi_atomic_t                 lock;           /** */ 
#endif
  kaapi_address_space_id_t       asid;           /* the address where is the thread */
  kaapi_cpuset_t                 affinity;       /* bit i == 1 -> can run on procid i */

  void*                          alloc_ptr;      /** pointer really allocated */
  uint32_t                       size;           /** size of the data structure allocated */
  kaapi_task_t*                  task;           /** bottom of the stack of task */

  struct kaapi_wsqueuectxt_cell_t* wcs;          /** point to the cell in the suspended list, iff thread is suspended */

  /* statically allocated reply */
  kaapi_reply_t			         static_reply;
  /* enough space to store a stealcontext that begins at static_reply->udata+static_reply->offset */
  char	                         sc_data[sizeof(kaapi_stealcontext_t)-sizeof(kaapi_stealheader_t)];

  /* warning: reply is variable sized
     so do not add members from here
   */
  uint64_t                       data[1];        /** begin of stack of data */ 
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_thread_context_t;

/* helper function */
#define kaapi_stack2threadcontext(stack)         ( ((kaapi_thread_context_t*)stack)-1 )
#define kaapi_threadcontext2stack(thread)        ( (kaapi_stack_t*)((thread)+1) )
#define kaapi_threadcontext2thread(thread)       ( (kaapi_thread_t*)((thread)->sfp) )

/** \ingroup TASK
*/
static inline kaapi_task_t* _kaapi_thread_toptask( kaapi_thread_context_t* thread ) 
{ return kaapi_thread_toptask( kaapi_threadcontext2thread(thread) ); }

/** \ingroup TASK
*/
static inline int _kaapi_thread_pushtask( kaapi_thread_context_t* thread )
{ return kaapi_thread_pushtask( kaapi_threadcontext2thread(thread) ); }

/** \ingroup TASK
*/
static inline void* _kaapi_thread_pushdata( kaapi_thread_context_t* thread, uint32_t count)
{ return kaapi_thread_pushdata( kaapi_threadcontext2thread(thread), count ); }

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
inline static void kaapi_task_lock_adaptive_steal(kaapi_stealcontext_t* sc)
{
  while (1)
  {
    if ((KAAPI_ATOMIC_READ(&sc->thieves.list.lock) == 0) && KAAPI_ATOMIC_CAS(&sc->thieves.list.lock, 0, 1))
      break ;
    kaapi_slowdown_cpu();
  }
}

inline static void kaapi_task_unlock_adaptive_steal(kaapi_stealcontext_t* sc)
{
  KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
}

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
#else
#  warning "NOT IMPLEMENTED"
#endif

/** \ingroup TASK
    The function kaapi_frame_isempty() will return non-zero value iff the frame is empty. Otherwise return 0.
    \param stack IN the pointer to the kaapi_stack_t data structure. 
    \retval !=0 if the stack is empty
    \retval 0 if the stack is not empty or argument is an invalid stack pointer
*/
static inline int kaapi_frame_isempty(volatile kaapi_frame_t* frame)
{ return (frame->pc <= frame->sp); }

/** \ingroup TASK
    The function kaapi_stack_bottom() will return the top task.
    The bottom task is the first pushed task into the stack.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_thread_bottomtask(kaapi_thread_context_t* thread) 
{
  kaapi_assert_debug( thread != 0 );
  return thread->task;
}

/* ========== Here include machine specific function: only next definitions should depend on machine =========== */
/** Here include all machine dependent functions and types
*/
#include "kaapi_machine.h"
/* ========== MACHINE DEPEND DATA STRUCTURE =========== */

/* ============================= Commun function for server side (no public) ============================ */
/** lighter than kaapi_thread_clear and used during the steal emit request function
*/
static inline int kaapi_thread_reset(kaapi_thread_context_t* th )
{
  th->sfp        = th->stackframe;
  th->esfp       = th->stackframe;
  th->sfp->sp    = th->sfp->pc  = th->task; /* empty frame */
  th->sfp->sp_data = (char*)&th->data;     /* empty frame */
  th->affinity[0] = ~0UL;
  th->affinity[1] = ~0UL;
  th->unstealable= 0;
  return 0;
}


/**
*/
extern const char* kaapi_cpuset2string( int nproc, kaapi_cpuset_t* affinity );


/**
*/
static inline void kaapi_cpuset_clear(kaapi_cpuset_t* affinity )
{
  (*affinity)[0] = 0;
  (*affinity)[1] = 0;
}


/**
*/
static inline void kaapi_cpuset_full(kaapi_cpuset_t* affinity )
{
  (*affinity)[0] = ~0UL;
  (*affinity)[1] = ~0UL;
}


/**
*/
static inline int kaapi_cpuset_intersect(kaapi_cpuset_t* s1, kaapi_cpuset_t* s2)
{
  return (((*s1)[0] & (*s2)[0]) != 0) || (((*s1)[1] & (*s2)[1]) != 0);
}

/**
*/
static inline int kaapi_cpuset_empty(kaapi_cpuset_t* affinity)
{
  return ((*affinity)[0] == 0) && ((*affinity)[1] == 0);
}

/**
*/
static inline int kaapi_cpuset_set(kaapi_cpuset_t* affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_cpuset_t)*8) );
  if (kid <64)
    (*affinity)[0] |= ((uint64_t)1)<<kid;
  else
    (*affinity)[1] |= ((uint64_t)1)<< (kid-64);
  return 0;
}

/**
*/
static inline int kaapi_cpuset_copy(kaapi_cpuset_t* dest, kaapi_cpuset_t* src )
{
  (*dest)[0] = (*src)[0];
  (*dest)[1] = (*src)[1];
  return 0;
}


/** Return non 0 iff th as affinity with kid
*/
static inline int kaapi_cpuset_has(kaapi_cpuset_t* affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_cpuset_t)*8) );
  if (kid <64)
    return ( (*affinity)[0] & ((uint64_t)1)<< (uint64_t)kid) != (uint64_t)0;
  else
    return ( (*affinity)[1] & ((uint64_t)1)<< (uint64_t)(kid-64)) != (uint64_t)0;
}

/** Return *dest &= mask
*/
static inline void kaapi_cpuset_and(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] &= (*mask)[0];
  (*dest)[1] &= (*mask)[1];
}

/** Return *dest |= mask
*/
static inline void kaapi_cpuset_or(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] |= (*mask)[0];
  (*dest)[1] |= (*mask)[1];
}

/** Return *dest &= ~mask
*/
static inline void kaapi_cpuset_notand(kaapi_cpuset_t* dest, kaapi_cpuset_t* mask )
{
  (*dest)[0] ^= (*mask)[0];
  (*dest)[1] ^= (*mask)[1];
}

/**
*/
static inline int kaapi_sched_suspendlist_empty(kaapi_processor_t* kproc)
{
  if (kproc->lsuspend.head ==0) return 1;
  return 0;
}

/** Note on scheduler lock:
  KAAPI_SCHED_LOCK_CAS -> lock state == 1 iff lock is taken, else 0
  KAAPI_SCHED_LOCK_CAS not defined: see 
    Sewell, P., Sarkar, S., Owens, S., Nardelli, F. Z., and Myreen, M. O. 2010. 
    x86-TSO: a rigorous and usable programmer's model for x86 multiprocessors. 
    Commun. ACM 53, 7 (Jul. 2010), 89-97. 
    DOI= http://doi.acm.org/10.1145/1785414.1785443
*/
static inline int kaapi_sched_initlock( kaapi_atomic_t* lock )
{
  return kaapi_atomic_initlock(lock);
}

static inline int kaapi_sched_trylock( kaapi_atomic_t* lock )
{
  return kaapi_atomic_trylock(lock);
}

/** 
*/
static inline int kaapi_sched_lock( kaapi_atomic_t* lock )
{
  return kaapi_atomic_lock(lock);
}


/**
*/
static inline int kaapi_sched_lock_spin( kaapi_atomic_t* lock, int spincount )
{
  return kaapi_atomic_lock_spin(lock, spincount);
}


/**
*/
static inline int kaapi_sched_unlock( kaapi_atomic_t* lock )
{
  return kaapi_atomic_unlock(lock);
}

static inline void kaapi_sched_waitlock(kaapi_atomic_t* lock)
{
  return kaapi_atomic_waitlock(lock);
}

static inline int kaapi_sched_islocked( kaapi_atomic_t* lock )
{
  return kaapi_atomic_islocked(lock);
}

/** steal/pop (no distinction) a thread to thief with kid
    If the owner call this method then it should protect 
    itself against thieves by using sched_lock & sched_unlock on the kproc.
*/
kaapi_thread_context_t* kaapi_sched_stealready(kaapi_processor_t*, kaapi_processor_id_t);

/** push a new thread into a ready list
*/
void kaapi_sched_pushready(kaapi_processor_t*, kaapi_thread_context_t*);

/** initialize the ready list 
*/
static inline void kaapi_sched_initready(kaapi_processor_t* kproc)
{
  kproc->lready._front = NULL;
  kproc->lready._back = NULL;
}

/** is the ready list empty 
*/
static inline int kaapi_sched_readyempty(kaapi_processor_t* kproc)
{
  return kproc->lready._front == NULL;
}



/** Affinity queue:
    - the affinity queues is attached to a certain level in the memory hierarchy, more
    generally it is attached to an identifier.
    - Several threads may push and pop into the queue.
    - Several threads are considered to the owner of the queue if they have affinity
    with it.
    - The owners push and pop in a LIFO maner (in the head of the queue)
    - The thieves push and pop in the LIFO maner (in the tail of the queue)
    - The owners and the thieves push/pop in the FIFO maner
*/
typedef struct kaapi_affinity_queue_t {
  kaapi_atomic_t                     lock;         /* to serialize operation */
  struct kaapi_taskdescr_t* volatile head;         /* owned by the owner */
  struct kaapi_taskdescr_t* volatile tail;         /* owner by the thief */
  kaapi_allocator_t                  allocator;    /* where to allocate task descriptor and other data structure */
} kaapi_affinity_queue_t;


/** Policy to convert a binding to a mapping (a bitmap) of kaapi_cpuset.
    flag ==0 if task is a dfg task.
*/
extern int kaapi_sched_affinity_binding2mapping(
    kaapi_cpuset_t*              mapping, 
    const kaapi_task_binding_t*  binding,
    const struct kaapi_format_t* task_fmt,
    const struct kaapi_task_t*   task,
    int                          flag
);


/** Return the workqueue that match the mapping
*/
extern kaapi_affinity_queue_t* kaapi_sched_affinity_lookup_queue(
    kaapi_cpuset_t* mapping
);

/**
*/
extern kaapi_affinity_queue_t* kaapi_sched_affinity_lookup_numa_queue(
  int numanodeid
);

/*
*/
extern kaapi_affinity_queue_t* kaapi_sched_affinity_random_queue( kaapi_processor_t* kproc );

/**
*/
extern struct kaapi_taskdescr_t* kaapi_sched_affinity_allocate_td_dfg( 
    kaapi_affinity_queue_t* queue, 
    kaapi_thread_context_t* thread, 
    struct kaapi_task_t*    task, 
    const kaapi_format_t*   task_fmt, 
    unsigned int            war_param
);

/**
*/
extern int kaapi_sched_affinity_owner_pushtask
(
    kaapi_affinity_queue_t* queue,
    struct kaapi_taskdescr_t* td
);

/**
*/
extern struct kaapi_taskdescr_t* kaapi_sched_affinity_owner_poptask
(
  kaapi_affinity_queue_t* queue
);

/**
*/
extern int kaapi_sched_affinity_thief_pushtask
(
    kaapi_affinity_queue_t* queue,
    struct kaapi_taskdescr_t* td
);

/**
*/
extern struct kaapi_taskdescr_t* kaapi_sched_affinity_thief_poptask
(
  kaapi_affinity_queue_t* queue
);



/**
*/
extern int kaapi_thread_clear( kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_thread_print( FILE* file, kaapi_thread_context_t* thread );

/** \ingroup TASK
    The function kaapi_thread_execframe() execute all the tasks in the thread' stack following
    the RFO order in the closures of the frame [frame_sp,..,sp[
    If successful, the kaapi_thread_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval 0 the execution of the stack frame is completed
*/
extern int kaapi_thread_execframe( kaapi_thread_context_t* thread );

/** \ingroup TASK
    The function kaapi_threadgroup_execframe() execute all the tasks in the thread' stack following
    using the list of ready tasks.
    If successful, the kaapi_threadgroup_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
    \retval EINTR the execution of the stack is interupt and the thread is detached to the kprocessor.
*/
extern int kaapi_threadgroup_execframe( kaapi_thread_context_t* thread );

/** Useful
*/
extern kaapi_processor_t* kaapi_get_current_processor(void);

/** \ingroup WS
    Select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag );

/** \ingroup WS
    Select a victim for next steal request using workload then uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_workload_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag );

/** \ingroup WS
    First steal is 0 then select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand_first0( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag );

/** \ingroup WS
    Select victim using the memory hierarchy
*/
extern int kaapi_sched_select_victim_hierarchy( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag );

/** \ingroup WS
    Enter in the infinite loop of trying to steal work.
    Never return from this function...
    If proc is null pointer, then the function allocate a new kaapi_processor_t and 
    assigns it to the current processor.
    This method may be called by 'host' current thread in order to become an executor thread.
    The method returns only in case of terminaison.
*/
extern void kaapi_sched_idle ( kaapi_processor_t* proc );

/** \ingroup WS
    Suspend the current context due to unsatisfied condition and do stealing until the condition becomes true.
    \retval 0 in case of success
    \retval EINTR in case of termination detection
    \TODO reprendre specs
*/
extern int kaapi_sched_suspend ( kaapi_processor_t* kproc );

/** \ingroup WS
    Synchronize the current control flow until all the task in the current frame have been executed.
    \param thread [IN/OUT] the thread that stores the current frame
    \retval 0 in case of success
    \retval !=0 in case of no recoverable error
*/
extern int kaapi_sched_sync_(kaapi_thread_context_t* thread);

/** \ingroup WS
    The method starts a work stealing operation and return until a sucessfull steal
    operation or 0 if no work may be found.
    The kprocessor kproc should have its stack ready to receive a work after a steal request.
    If the stack returned is not 0, either it is equal to the stack of the processor or it may
    be different. In the first case, some work has been insert on the top of the stack. On the
    second case, a whole stack has been stolen. It is to the responsability of the caller
    to treat the both case.
    \retval 0 in case of not stolen work 
    \retval a pointer to a stack that is the result of one workstealing operation.
*/
extern int kaapi_sched_stealprocessor ( 
  kaapi_processor_t*            kproc, 
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
);

/** \ingroup WS
    This method tries to steal work from the tasks of a stack passed in argument.
    The method iterates through all the tasks in the stack until it found a ready task
    or until the request count reaches 0.
    The current implementation is cooperative or concurrent depending of configuration flag.
    only exported for kaapi_stealpoint.
    \param stack the victim stack
    \param task the current running task (cooperative) or 0 (concurrent)
    \retval the number of positive replies to the thieves
*/
extern int kaapi_sched_stealstack  ( 
  struct kaapi_thread_context_t* thread, 
  kaapi_task_t* curr, 
  struct kaapi_listrequest_t* lrequests, 
  struct kaapi_listrequest_iterator_t* lrrange
);

/** \ingroup WS
    \retval 0 if no context could be wakeup
    \retval else a context to wakeup
    \TODO faire specs ici
*/
extern kaapi_thread_context_t* kaapi_sched_wakeup ( 
  kaapi_processor_t* kproc, 
  kaapi_processor_id_t kproc_thiefid, 
  struct kaapi_thread_context_t* cond_thread,
  kaapi_task_t* cond_task
);


/** \ingroup WS
    The method starts a work stealing operation and return the result of one steal request
    The kprocessor kproc should have its stack ready to receive a work after a steal request.
    If the stack returned is not 0, either it is equal to the stack of the processor or it may
    be different. In the first case, some work has been insert on the top of the stack. On the
    second case, a whole stack has been stolen. It is to the responsability of the caller
    to treat the both case.
    \retval 0 in case failure of stealing something
    \retval a pointer to a stack that is the result of one workstealing operation.
*/
extern kaapi_thread_context_t* kaapi_sched_emitsteal ( kaapi_processor_t* kproc );

/** \ingroup HWS
    Hierarchical workstealing routine
    \retval 0 in case failure of stealing something
    \retval a pointer to a stack that is the result of one workstealing operation.
*/
extern kaapi_thread_context_t* kaapi_hws_emitsteal ( kaapi_processor_t* kproc );


/** \ingroup WS
    Advance polling of request for the current running thread.
    If this method is called from an other running thread than proc,
    the behavious is unexpected.
    \param proc should be the current running thread
*/
extern int kaapi_sched_advance ( kaapi_processor_t* proc );


/** \ingroup WS
    Splitter for DFG task
*/
extern int kaapi_task_splitter_dfg(
  kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task, 
  const kaapi_format_t*         task_fmt,
  unsigned int                  war_param, 
  unsigned int                  cw_param, 
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
);

/** \ingroup TASK
    Splitter for a single DFG
*/
extern void kaapi_task_splitter_dfg_single
(
  kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task, 
  const kaapi_format_t*         task_fmt,
  unsigned int                  war_param, 
  unsigned int                  cw_param, 
  kaapi_request_t*		        request
);

/** \ingroup WS
    Wrapper arround the user level Splitter for Adaptive task
*/
extern int kaapi_task_splitter_adapt( 
    kaapi_thread_context_t*       thread, 
    kaapi_task_t*                 task,
    kaapi_task_splitter_t         splitter,
    void*                         argsplitter,
    kaapi_listrequest_t*          lrequests, 
    kaapi_listrequest_iterator_t* lrrange
);


/** \ingroup WS
    Splitter arround tasklist stealing
*/
extern int kaapi_task_splitter_readylist( 
  kaapi_thread_context_t*       thread, 
  struct kaapi_tasklist_t*      tasklist,
  struct kaapi_taskdescr_t**    task_beg,
  struct kaapi_taskdescr_t**    task_end,
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange,
  size_t                        countreq
);

/** \ingroup ADAPTIVE
     Disable steal on stealcontext and wait not more thief is stealing.
 */
static inline void kaapi_steal_disable_sync(kaapi_stealcontext_t* stc)
{
  stc->splitter    = 0;
  stc->argsplitter = 0;
  kaapi_mem_barrier();

  /* synchronize on the kproc lock */
  kaapi_sched_waitlock(&kaapi_get_current_processor()->lock);
}


/**
*/
extern void kaapi_synchronize_steal(kaapi_stealcontext_t*);


/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/

/** Destroy a request
    A posted request could not be destroyed until a reply has been made
*/
#define kaapi_request_destroy( kpsr ) 

static inline kaapi_processor_id_t kaapi_request_getthiefid(kaapi_request_t* r)
{ return (kaapi_processor_id_t) r->kid; }

static inline kaapi_reply_t* kaapi_request_getreply(kaapi_request_t* r)
{ return r->reply; }

/** Return the request status
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
static inline uint64_t kaapi_reply_status( kaapi_reply_t* ksr ) 
{ return ksr->status; }

/** Return true iff the request has been posted
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REPLY_S_ERROR steal request has failed to be posted because the victim refused request
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
static inline int kaapi_reply_test( kaapi_reply_t* ksr )
{ return kaapi_reply_status(ksr) != KAAPI_REQUEST_S_POSTED; }

/** Return true iff the request is a success steal
  \param pksr kaapi_reply_t
*/
static inline int kaapi_reply_ok( kaapi_reply_t* ksr )
{ return kaapi_reply_status(ksr) != KAAPI_REPLY_S_NOK; }

/** Return the data associated with the reply
  \param pksr kaapi_reply_t
*/
static inline kaapi_reply_t* kaapi_replysync_data( kaapi_reply_t* reply ) 
{ 
  kaapi_readmem_barrier();
  return reply;
}

/** Args for tasksteal
*/
typedef struct kaapi_tasksteal_arg_t {
  kaapi_thread_context_t* origin_thread;     /* stack where task was stolen */
  kaapi_task_t*           origin_task;       /* the stolen task into origin_stack */
  const kaapi_format_t*   origin_fmt;        /* the format of the stolen taskx */
  unsigned int            war_param;         /* bit i=1 iff it is a w mode with war dependency */
  unsigned int            cw_param;          /* bit i=1 iff it is a cw mode */
  void*                   copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;

/** Args for taskstealready
*/
typedef struct kaapi_taskstealready_arg_t {
  struct kaapi_tasklist_t*   origin_tasklist; /* the original task list */
  struct kaapi_taskdescr_t** origin_td_beg;   /* range of stolen task into origin_tasklist */
  struct kaapi_taskdescr_t** origin_td_end;   /* range of stolen task into origin_tasklist */
} kaapi_taskstealready_arg_t;

/** User task + args for kaapi_adapt_body. 
    This area represents the part to be sent by the combinator during the remote
    write of the adaptive reply.
    It is store into the field reply->udata of the kaapi_reply_t data structure.
*/
#define KAAPI_REPLY_USER_DATA_SIZE_MAX (KAAPI_REPLY_DATA_SIZE_MAX-sizeof(kaapi_adaptive_thief_body_t))
typedef struct kaapi_taskadaptive_user_taskarg_t {
  /* user defined body */
  kaapi_adaptive_thief_body_t ubody;
  /* user defined args for the body */
  unsigned char               udata[KAAPI_REPLY_USER_DATA_SIZE_MAX];
} kaapi_taskadaptive_user_taskarg_t;


#include "kaapi_tasklist.h"
#include "kaapi_partition.h"
#include "kaapi_event.h"
#include "kaapi_trace.h"


/** Call only on thread in list of suspended threads.
*/
static inline int kaapi_thread_isready( kaapi_thread_context_t* thread )
{
  /* if ready list: use it as state of the thread */
  kaapi_tasklist_t* tl = thread->sfp->tasklist;
  if (tl !=0)
  {
    if ( kaapi_tasklist_isempty(tl) && (KAAPI_ATOMIC_READ(&tl->count_thief) == 0))
      return 1; 
    return 0;
  }

  return kaapi_task_state_isready( kaapi_task_getstate(thread->sfp->pc) );
}

/* Signal handler to dump the state of the internal kprocessors
   This signal handler is attached to SIGALARM when KAAPI_DUMP_PERIOD env. var. is defined.
*/
extern void _kaapi_signal_dump_state(int);

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_IMPL_H */
