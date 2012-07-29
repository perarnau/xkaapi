/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#define TASKLIST_REPLY_ONETD        1

/* Task list priority: order between min max */
#define KAAPI_TASKLIST_NUM_PRIORITY (KAAPI_TASK_MAX_PRIORITY-KAAPI_TASK_MIN_PRIORITY+1)
#define KAAPI_TASKLIST_MAX_PRIORITY KAAPI_TASK_MAX_PRIORITY
#define KAAPI_TASKLIST_MIN_PRIORITY KAAPI_TASK_MIN_PRIORITY

#include "config.h"
#include "kaapi.h"
#include "kaapi_error.h"
#include "kaapi_defs.h"
#include "kaapi_atomic.h"
#include "kaapi_cpuset.h"
#include "kaapi_allocator.h"
#include "kaapi_hashmap.h"
#include "kaapi_memory.h"
#include "kaapi_task.h"
#include "kaapi_format.h"
#include "kaapi_event.h"

#include <string.h>

#include "kaapi_defs.h"



/** Current implementation relies on isoaddress allocation to avoid
    communication during synchronization at the end of partitionning.
*/
//#define KAAPI_ADDRSPACE_ISOADDRESS 1


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

/** Post request to suspend all threads except the main thread.
    Should be called by the main thread !
*/
extern void kaapi_mt_suspend_threads_post(void);

/** Wait for all other threads to be suspended.
    Should be called by the main thread after kaapi_mt_suspend_threads_post
*/
extern void kaapi_mt_suspend_threads_wait(void);

/** Suspend all threads except the main thread.
    Should be called by the main thread !
*/
static inline void kaapi_mt_suspend_threads(void)
{
  kaapi_mt_suspend_threads_post();
  kaapi_mt_suspend_threads_wait();
}

/** Wait for all other threads to be suspended.
    Should be called by the main thread after kaapi_mt_suspend_threads_post
*/
extern void kaapi_mt_suspend_threads_wait(void);

/** Call by the threads to be put into suspended state
*/
extern void kaapi_mt_suspend_self( struct kaapi_processor_t* kproc );

/** Resume all threads except the main threads.
*/
extern void kaapi_mt_resume_threads(void);

/** initialize suspend/resume sub-functionnalities 
*/
extern void kaapi_mt_suspendresume_init(void);

/**
*/
extern void kaapi_mt_begin_parallel(void);

/**
*/
extern void kaapi_mt_end_parallel(void);



/** Initialize hw topo.
    Based on hwloc library.
    Return 0 in case of success else an error code
*/
extern int kaapi_hw_init(void);

/* Fwd declaration 
*/
struct kaapi_listrequest_t;
struct kaapi_procinfo_list_t;

/* Fwd declaration
*/
struct kaapi_tasklist_t;
struct kaapi_comlink_t;
struct kaapi_taskdescr_t;
struct kaapi_big_hashmap_t;
struct kaapi_version_t;
struct kaapi_metadata_info_t;
struct kaapi_mem_data_t;

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


/* ========================== Work stealing protocol========================= */
/* Request status 
*/
typedef enum kaapi_request_status_t {
  KAAPI_REQUEST_S_POSTED   = 0,
  KAAPI_REQUEST_S_NOK      = 1,
  KAAPI_REQUEST_S_OK       = 2,
  KAAPI_REQUEST_S_ERROR    = 3
} kaapi_request_status_t;


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
typedef kaapi_request_status_t (*kaapi_emitsteal_fnc_t)(struct kaapi_processor_t*);

/** \ingroup WS
    Called to initialize the emit a steal context
    \param kproc [IN] the kaapi_processor_t that want to emit a request
    \retval the stolen thread
*/
typedef int (*kaapi_emitsteal_init_t)(struct kaapi_processor_t*);

/* \ingroup AFF
 * Return a processor with more affinity to a task.
 */
typedef struct kaapi_processor_t* (*kaapi_affinity_fnc_t)(struct kaapi_processor_t*,struct kaapi_taskdescr_t*);

/* \ingroup AFF
 * Return the td of the list pointed by head (iterates through prev) that statisfies the criteria 
 */
typedef struct kaapi_taskdescr_t* (*kaapi_steal_by_affinity_fnc_t)(const struct kaapi_processor_t*, struct kaapi_taskdescr_t*);

/* \ingroup SCHED
 * Return a priority from ct path value
 */
typedef uint8_t (*kaapi_ct2prio_fnc_t)(uint64_t, struct kaapi_taskdescr_t*);

/* =======vvvvvvvvvvvvvvvvvv===================== Memory type ============================ */

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

struct kaapi_taskdescr_t;

/**
*/
typedef struct kaapi_affinityset_t {
    kaapi_cpuset_t                 who;       /* cpu id in this set */
    size_t                         mem_size;  /* memory size of this set */
    int                            os_index;  /* numa node id or ??? */
    int                            ncpu;      /* number of cpu */
    short                          type;      /* see kaapi_memory_t */
} kaapi_affinityset_t;

/**
*/
typedef struct kaapi_hierarchy_one_level_t {
  unsigned short           count;       /* number of kaapi_affinityset_t at this level */
  kaapi_affinityset_t*     affinity; 
  kaapi_hws_levelid_t      levelid;
  char*	                   name;
} kaapi_hierarchy_one_level_t;

/** Memory hierarchy of the local machine
    * memory.depth: depth of the hierarchy
    * memory.levels[i].affinity.ncpu: number of cpu that share this memory at level i
    * memory.levels[i].affinity.who: cpu set of which PU is contains by memory k at level i
    * memory.levels[i].affinity.mem_size: size of the k memory  at level i
*/
typedef struct kaapi_hierarchy_t {
  unsigned short               depth;
  unsigned short               numalevel;
  kaapi_hierarchy_one_level_t* levels;
} kaapi_hierarchy_t;


/** Definition of parameters for the runtime system
*/
typedef struct kaapi_rtparam_t {
  size_t                        stacksize;           /* default stack size */
  size_t                        stacksize_master;    /* stack size for the master thread */
  unsigned int                  syscpucount;         /* number of physical cpus of the system */
  unsigned int                  cpucount;            /* number of physical cpu used for execution */
  unsigned int                  gpucount;            /* number of physical gpu used for execution */
  kaapi_selectvictim_fnc_t      wsselect;            /* default method to select a victim */
  kaapi_emitsteal_fnc_t	        emitsteal;
  kaapi_emitsteal_init_t        emitsteal_initctxt;  /* call to initialize the emitsteal ctxt */
  kaapi_affinity_fnc_t	        affinity;	         	 /* call to pick a processor */
  kaapi_steal_by_affinity_fnc_t steal_by_affinity;   /* call to pick a TD in a list during steal operation */
  unsigned int		              use_affinity;        /* use cpu affinity */
  kaapi_ct2prio_fnc_t		        ctpriority;          /* use critical path priorities, if 0 no */
  int                           display_perfcounter; /* set to 1 iff KAAPI_DISPLAY_PERF */
#if defined(KAAPI_USE_CUPTI)
   uint64_t		                  cudastartuptime;
#endif
#if defined(KAAPI_USE_CUDA)
   uint64_t		                  cudawindowsize;
   unsigned int		              cudapeertopeer;
#endif
  uint64_t                      startuptime;         /* time at the end of kaapi_init */
  int                           alarmperiod;         /* period for alarm */
  uint64_t                      eventmask;           /* event mask */

  struct kaapi_procinfo_list_t* kproc_list;          /* list of kprocessors to initialized */
  kaapi_cpuset_t                usedcpu;             /* cpuset of used physical ressources */
  kaapi_hierarchy_t             memory;              /* memory hierarchy */
  unsigned int*	                kid2cpu;             /* mapping: kid->phys cpu  */
  unsigned int*  	              cpu2kid;             /* mapping: phys cpu -> kid */
} kaapi_rtparam_t;

extern kaapi_rtparam_t kaapi_default_param;


/* ============================= REQUEST ============================ */
/** Private status of request
    \ingroup WS
    
    The protocol to steal work is very simple:
    1- the thief that want to post a request should provide:
      - an status to indicate the completion of the request (kaapi_atomic_t)
      - a frame where to store theft tasks
    2. The post method initialize fields, do a write barrier and set the status to
    KAAPI_REQUEST_S_POSTED
    3. A victim thread that replies must fill the kaapi_tasksteal_arg_t and
    call kaapi_request_replytask with the status of the steal request: 
    either KAAPI_REQUEST_S_OK in case of success, else KAAPI_REQUEST_S_NOK.
    If the victim want to change kind of reply, it can change the body of the
    task. The pre-allocated arguments for the task is currently of size 
    sizeof(kaapi_tasksteal_arg_t).
*/


/* ============================= Simple C API for network ============================ */
#include "kaapi_network.h"


/*============= The structure for handling suspendended thread ======================== */
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
  kaapi_stack_t                  stack;
  kaapi_frame_t                  stackframe[KAAPI_MAX_RECCALL] __attribute__((aligned (KAAPI_CACHE_LINE)));

#if !defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  kaapi_threadgroup_t            thgrp;          /** the current thread group, used to push task */
#endif

  /* the way to execute task inside a thread, if ==0 uses kaapi_thread_execframe */
  kaapi_threadgroup_t            the_thgrp;      /* not null iff execframe != kaapi_thread_execframe */
  int                            unstealable;    /* !=0 -> cannot be stolen */
  int                            partid;         /* used by static scheduling to identify the thread in the group */
  struct kaapi_big_hashmap_t*    kversion_hm;    /* used by static scheduling */
  
  struct kaapi_thread_context_t* _next;          /** to be linkable */
  struct kaapi_thread_context_t* _prev;          /** to be linkable */

//#if defined(KAAPI_USE_CUDA)
#if 0
#warning "lock already defined in stack or in kproc"
  kaapi_atomic_t                 lock;           /** */ 
#endif
  kaapi_address_space_id_t       asid;           /* the address where is the thread */
  kaapi_cpuset_t                 affinity;       /* bit i == 1 -> can run on procid i */

  void*                          alloc_ptr;      /** pointer really allocated */
  uint32_t                       size;           /** size of the data structure allocated */
  uint32_t                       stack_size;     /** The stack size attribut */

  struct kaapi_wsqueuectxt_cell_t* wcs;          /** point to the cell in the suspended list, iff thread is suspended */

#if 0 // DEPRECATED_ATTRIBUTE
  /* enough space to store a stealcontext that begins at static_reply->udata+static_reply->offset */
  char	                         sc_data[sizeof(kaapi_stealcontext_t)-sizeof(kaapi_stealheader_t)];

#endif
  char                           data[1] __attribute__((aligned (KAAPI_CACHE_LINE)));  /** begin of stack of data */ 
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_thread_context_t;

/* helper function */
#define kaapi_threadcontext2thread(thread)       ( (thread)->stack.sfp )



/* ========================================================================= */
/* */
extern uint64_t kaapi_perf_thread_delayinstate(struct kaapi_processor_t* kproc);


/* ========== Here include machine specific function: only next definitions should depend on machine =========== */
/** Here include all machine dependent functions and types
*/
#include "kaapi_machine.h"
/* ========== MACHINE DEPEND DATA STRUCTURE =========== */


#include "kaapi_tasklist.h"
#include "kaapi_sched.h"


/**
*/
static inline int kaapi_sched_suspendlist_empty(kaapi_processor_t* kproc)
{
  if (kproc->lsuspend.head ==0) return 1;
  return 0;
}

/**
*/
static inline int kaapi_sched_initlock( kaapi_lock_t* lock )
{
  return kaapi_atomic_initlock(lock);
}

static inline int kaapi_sched_trylock( kaapi_lock_t* lock )
{
  return kaapi_atomic_trylock(lock);
}

/** 
*/
static inline int kaapi_sched_lock( kaapi_lock_t* lock )
{
  return kaapi_atomic_lock(lock);
}

/**
*/
static inline int kaapi_sched_unlock( kaapi_lock_t* lock )
{
  return kaapi_atomic_unlock(lock);
}

static inline void kaapi_sched_waitlock( kaapi_lock_t* lock)
{
  kaapi_atomic_waitlock(lock);
}


static inline
kaapi_thread_t* kaapi_thread_push_frame_( kaapi_thread_context_t* thread )
{
  kaapi_frame_t* fp = (kaapi_frame_t*)thread->stack.sfp;
  /* save the top frame */
  fp[1].sp_data   = fp->sp_data;
  fp[1].pc        = fp->sp;
  fp[1].sp        = fp->sp;
  /* push a new frame */
  return (kaapi_thread_t*)++thread->stack.sfp;
}


static inline
kaapi_thread_t*  kaapi_thread_pop_frame_( kaapi_thread_context_t* thread )
{
  kaapi_thread_t* retval = (kaapi_thread_t*)--thread->stack.sfp;
  kaapi_synchronize_steal_thread( thread );
  return retval;  
}

/**
*/
extern int kaapi_thread_clear( kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_thread_print( FILE* file, kaapi_thread_context_t* thread );


/** Useful
*/
extern kaapi_processor_t* kaapi_get_current_processor(void);

/** \ingroup WS
    Select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/** \ingroup WS
    Select a victim for next steal request using workload then uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_workload_rand( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/** \ingroup WS
    First steal is 0 then select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand_first0( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/** \ingroup WS
    Select victim using the memory hierarchy
*/
extern int kaapi_sched_select_victim_hierarchy( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/* experimental: */
extern int kaapi_sched_select_victim_pws( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/* experimental: */
extern int kaapi_sched_select_victim_hwsn( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

/* experimental: */
extern int kaapi_sched_select_victim_aff( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
);

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
    \retval KAAPI_REQUEST_S_NOK in case of failure of stealing something
    \retval KAAPI_REQUEST_S_OK in case of success of the steal operation
*/
extern kaapi_request_status_t kaapi_sched_flat_emitsteal ( kaapi_processor_t* kproc );

/** \ingroup WS
    The method initialize the information required for the flat emitsteal function.
    \retval 0 in case success
    \retval an error code
*/
extern int kaapi_sched_flat_emitsteal_init(kaapi_processor_t*);

/** \ingroup WS
    Advance polling of request for the current running thread.
    If this method is called from an other running thread than proc,
    the behavious is unexpected.
    \param proc should be the current running thread
*/
extern int kaapi_sched_advance ( kaapi_processor_t* proc );


/** \ingroup ADAPTIVE
     Disable steal on stealcontext and wait not more thief is stealing.
 */
__attribute__((deprecated))
static inline void kaapi_steal_disable_sync(kaapi_stealcontext_t* stc)
{
#if 0
  stc->splitter    = 0;
  stc->argsplitter = 0;
  kaapi_mem_barrier();

  /* synchronize on the kproc lock */
  kaapi_sched_waitlock(&kaapi_get_current_processor()->lock);
#endif
}

/** \ingroup WS
*/
extern uint8_t kaapi_ctpath2prio_max( uint64_t tinfinity, struct kaapi_taskdescr_t* td);

/** \ingroup WS
*/
extern uint8_t kaapi_ctpath2prio_linear( uint64_t tinfinity, struct kaapi_taskdescr_t* td);

/** \ingroup WS
*/
extern uint8_t kaapi_ctpath2prio_identity( uint64_t tinfinity, struct kaapi_taskdescr_t* td);

static inline int _kaapi_workqueue_lock( 
    kaapi_workqueue_t* kwq
)
{
  return kaapi_atomic_lock(kwq->lock);
}


/** Unlock the workqueue
*/
static inline int _kaapi_workqueue_unlock( 
    kaapi_workqueue_t* kwq
)
{
  return kaapi_atomic_unlock(kwq->lock);
}



/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/

/** Destroy a request
    A posted request could not be destroyed until a reply has been made
*/
#define kaapi_request_destroy( kpsr ) 

static inline kaapi_processor_id_t kaapi_request_getthiefid(kaapi_request_t* r)
{ return (kaapi_processor_id_t) r->ident; }

/** Return the request status
  \param pksr kaapi_request_t
  \retval KAAPI_REQUEST_S_OK sucessfull steal operation
  \retval KAAPI_REQUEST_S_NOK steal request has failed
  \retval KAAPI_REQUEST_S_ERROR error during steal request processing
*/
static inline int kaapi_request_status_get( kaapi_atomic_t* status ) 
{ return KAAPI_ATOMIC_READ(status); }

/** Return the request status
  \param pksr kaapi_request_t
  \retval KAAPI_REQUEST_S_OK sucessfull steal operation
  \retval KAAPI_REQUEST_S_NOK steal request has failed
  \retval KAAPI_REQUEST_S_ERROR error during steal request processing
*/
static inline int kaapi_request_status( kaapi_request_t* kr ) 
{ return kaapi_request_status_get(kr->status); }

/** Return true iff the request has a reply
  \param kr kaapi_request_t
  \retval KAAPI_REQUEST_S_OK sucessfull steal operation
  \retval KAAPI_REQUEST_S_NOK steal request has failed
  \retval KAAPI_REQUEST_S_ERROR steal request has failed to be posted because the victim refused request
*/
static inline int kaapi_request_status_test( kaapi_atomic_t* status )
{ return KAAPI_ATOMIC_READ(status) != KAAPI_REQUEST_S_POSTED; }

/** Return true iff the request has been posted
  \param kr kaapi_request_t
  \retval KAAPI_REQUEST_S_OK sucessfull steal operation
  \retval KAAPI_REQUEST_S_NOK steal request has failed
  \retval KAAPI_REQUEST_S_ERROR steal request has failed to be posted because the victim refused request
*/
static inline int kaapi_request_test( kaapi_request_t* kr )
{ return kaapi_request_status_test(kr->status); }

/** Return true iff the request is a success steal
  \param kr kaapi_request_t
*/
static inline int kaapi_request_ok( kaapi_request_t* kr )
{ return kaapi_request_status(kr) != KAAPI_REQUEST_S_NOK; }

/** Return the data associated with the reply
  \param kr kaapi_request_t
*/
static inline void kaapi_request_syncdata( kaapi_request_t* kr ) 
{ 
  kaapi_readmem_barrier();
}


/* ======================== Perf counter interface: machine dependent ========================*/
#include "kaapi_trace.h"

/** Collect and display trace */
extern void kaapi_collect_trace(void);


/* */
extern void kaapi_set_workload( kaapi_processor_t*, unsigned long workload );

/* */
extern void kaapi_set_self_workload( unsigned long workload );



#include "kaapi_partition.h"
#include "kaapi_event_recorder.h"

/** \ingroup TASK
    Reply to a steal request.
    Return !=0 if the request cannot be replied.
*/
static inline int kaapi_request_replytask
( 
  kaapi_request_t*        request, 
  kaapi_request_status_t  value
)
{
  kaapi_atomic_t* const status = request->status;
#if defined(KAAPI_USE_PERFCOUNTER)
  {
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    KAAPI_IFUSE_TRACE(kproc,
      kaapi_writemem_barrier();
      KAAPI_EVENT_PUSH3(kproc, 0, KAAPI_EVT_SEND_REPLY, 
                        request->victim, request->ident, request->serial );
    );
  }
#endif
  request->status = 0;
  if (value == KAAPI_REQUEST_S_OK)
  {
    /* even if tasks will be preempted, reply ok. On remote side, the processor will abort
       preempted tasks
    */
    KAAPI_ATOMIC_WRITE_BARRIER(status, KAAPI_REQUEST_S_OK);
  }
  else
  {
    /* failed to steal: avoid unnecessary memory barrier */
    KAAPI_ATOMIC_WRITE(status, KAAPI_REQUEST_S_NOK);
  }
  return 0;
}



/*** TO BE MOVED INTO kaapi_task.h once tasklist.h could be included... */
/** Call only on thread in list of suspended threads.
*/
static inline int kaapi_stack_isready( kaapi_stack_t* stack )
{
  /* if ready list: use it as state of the thread */
  kaapi_frame_t* fp = stack->sfp;
  struct kaapi_tasklist_t* tl = fp->tasklist;
  if (tl !=0)
  {
    if (tl->master ==0) 
      return (KAAPI_ATOMIC_READ(&tl->cnt_exec) == tl->total_tasks);
    return 1; /* TODO voir */
  }

  return kaapi_task_isready(fp->pc);
}

/** DEPRECATED
*/
static inline int kaapi_thread_isready( kaapi_thread_context_t* thread )
{ return kaapi_stack_isready(&thread->stack); }


/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PUBLIC INTERFACE ........................................*/

/* Signal handler to dump the state of the internal kprocessors
   This signal handler is attached to SIGALARM when KAAPI_DUMP_PERIOD env. var. is defined.
*/
extern void _kaapi_signal_dump_state(int);

/* Signal handler attached to:
    - SIGINT
    - SIGQUIT
    - SIGABRT
    - SIGTERM
    - SIGSTOP
  when the library is configured with --with-perfcounter in order to flush some counters.
*/
extern void _kaapi_signal_dump_counters(int);

/* Signal handler to print the backtrace
   This signal handler is attached to:
    - SIGABRT
    - SIGTERM
    - SIGSEGV
    - SIGFPE 
    - SIGILL
    If the library is configured with --with-perfcounter, then the function call _kaapi_signal_dump_counters.
*/
extern void _kaapi_signal_dump_backtrace(int);

#if defined(__cplusplus)
}
#endif

#endif
