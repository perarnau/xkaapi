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

/* Maximal number of recursive call used to store the stack of frames.
   This is used to use a stack of frame.
   Some optimization may have been done by avoiding using this structure.
*/
#define KAAPI_MAX_RECCALL 1024

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


/** Highest level, more trace generated */
#define KAAPI_LOG_LEVEL 10

#if defined(KAAPI_DEBUG)
#  define kaapi_assert_debug_m(cond, msg) \
      { int __kaapi_cond = cond; \
        if (!__kaapi_cond) \
        { \
          printf("[%s]: LINE: %u FILE: %s, ", msg, __LINE__, __FILE__);\
          abort();\
        }\
      }
#  define KAAPI_LOG(l, fmt, ...) \
      do { if (l<= KAAPI_LOG_LEVEL) { printf("%i:"fmt, kaapi_get_current_processor()->kid, ##__VA_ARGS__); fflush(0); } } while (0)

#else
#  define kaapi_assert_debug_m(cond, msg)
#  define KAAPI_LOG(l, fmt, ...) 
#endif /* defined(KAAPI_DEBUG)*/

#define kaapi_assert_m(cond, msg) \
      { \
        if (!(cond)) \
        { \
          printf("[%s]: \n\tLINE: %u FILE: %s, ", msg, __LINE__, __FILE__);\
          abort();\
        }\
      }


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


/** This is the new version on top of X-Kaapi
*/
extern const char* get_kaapi_version(void);

/** Global hash table of all formats: body -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_bybody[256];

/** Global hash table of all formats: fmtid -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_byfmtid[256];


/* ================== Library initialization/terminaison ======================= */
/** Initialize the machine level runtime.
    Return 0 in case of success. Else an error code.
*/
extern int kaapi_mt_init(void);

/** Finalize the runtime.
    Return 0 in case of success. Else an error code.
*/
extern int kaapi_mt_finalize(void);

/* Fwd declaration 
*/
struct kaapi_listrequest_t;
struct kaapi_procinfo_list;

/* ============================= Processor list ============================ */

/* ========================================================================== */
/** kaapi_mt_register_procs
    register the kprocs for mt architecture.
*/
extern int kaapi_mt_register_procs(struct kaapi_procinfo_list*);

/* ========================================================================== */
/** kaapi_cuda_register_procs
    register the kprocs for cuda architecture.
*/
#if defined(KAAPI_USE_CUDA)
extern int kaapi_cuda_register_procs(struct kaapi_procinfo_list*);
#endif


/* ============================= A VICTIM ============================ */
/** \ingroup WS
    This data structure should contains all necessary informations to post a request to a selected node.
    It should be extended in case of remote work stealing.
*/
typedef struct kaapi_victim_t {
  struct kaapi_processor_t* kproc; /** the victim processor */
  kaapi_uint16_t            level; /** level in the hierarchy of the source k-processor to reach kproc */
} kaapi_victim_t;

/** \ingroup WS
    Select a victim for next steal request
    \param kproc [IN] the kaapi_processor_t that want to emit a request
    \param victim [OUT] the selection of the victim
    \retval 0 in case of success 
    \retval EINTR in case of detection of the termination 
    \retval else error code
    
*/
typedef int (*kaapi_selectvictim_fnc_t)( struct kaapi_processor_t*, struct kaapi_victim_t* );


/* ============================= Default parameters ============================ */
/** Initialise default formats
*/
extern void kaapi_init_basicformat(void);

/** Setup KAAPI parameter from
    1/ the command line option
    2/ form the environment variable
    3/ default values
*/
extern int kaapi_setup_param( int argc, char** argv );
    
/** Definition of parameters for the runtime system
*/
typedef struct kaapi_rtparam_t {
  size_t                   stacksize;              /* default stack size */
  unsigned int             syscpucount;            /* number of physical cpus of the system */
  unsigned int             cpucount;               /* number of physical cpu used for execution */
  kaapi_selectvictim_fnc_t wsselect;               /* default method to select a victim */
  unsigned int		         use_affinity;           /* use cpu affinity */
  unsigned int		         kid_to_cpu[KAAPI_MAX_PROCESSOR]; /* mapping: kid->phys cpu  ?*/
  int                      display_perfcounter;    /* set to 1 iff KAAPI_DISPLAY_PERF */
  kaapi_uint64_t           startuptime;            /* time at the end of kaapi_init */
} kaapi_rtparam_t;

extern kaapi_rtparam_t kaapi_default_param;


/* ============================= REQUEST ============================ */
/** Private status of request
    \ingroup WS
*/
enum kaapi_reply_status_t {
  KAAPI_REQUEST_S_POSTED = 0,
  KAAPI_REPLY_S_NOK,
  KAAPI_REPLY_S_TASK,
  KAAPI_REPLY_S_TASK_FMT,
  KAAPI_REPLY_S_THREAD,
  KAAPI_REPLY_S_ERROR
};


/* ============================= Format for task ============================ */
/** \ingroup TASK
    Kaapi task format
    The format should be 1/ declared 2/ register before any use in task.
    The format object is only used in order to interpret stack of task.    
*/
typedef struct kaapi_format_t {
  kaapi_format_id_t          fmtid;                                   /* identifier of the format */
  short                      isinit;                                  /* ==1 iff initialize */
  const char*                name;                                    /* debug information */
  
  /* case of format for a structure or for a task */
  kaapi_uint32_t             size;                                    /* sizeof the object */  
  void                       (*cstor)( void* dest);
  void                       (*dstor)( void* dest);
  void                       (*cstorcopy)( void* dest, const void* src);
  void                       (*copy)( void* dest, const void* src);
  void                       (*assign)( void* dest, const void* src);
  void                       (*print)( FILE* file, const void* src);

  /* only if it is a format of a task  */
  kaapi_task_body_t          default_body;                            /* iff a task used on current node */
  kaapi_task_body_t          entrypoint[KAAPI_PROC_TYPE_MAX];         /* maximum architecture considered in the configuration */
  int                        count_params;                            /* number of parameters */
  kaapi_access_mode_t        *mode_params;                            /* only consider value with mask 0xF0 */
  kaapi_offset_t             *off_params;                             /* access to the i-th parameter: a value or a shared */
  struct kaapi_format_t*     *fmt_params;                             /* format for each params */
  kaapi_uint32_t             *size_params;                            /* sizeof of each params */

  struct kaapi_format_t      *next_bybody;                            /* link in hash table */
  struct kaapi_format_t      *next_byfmtid;                           /* link in hash table */

  size_t (*get_param_size)(const struct kaapi_format_t*, unsigned int, const void*);
  
  /* only for Monotonic bound format */
  int    (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                      const void* value, const struct kaapi_format_t* fmtvalue );
} kaapi_format_t;


/* ============================= Helper for bloc allocation of individual entries ============================ */
/*
*/
#define KAAPI_BLOCENTRIES_SIZE 32

/*
*/
#define KAAPI_DECLARE_BLOCENTRIES(NAME, TYPE) \
typedef struct NAME {\
  TYPE         data[KAAPI_BLOCENTRIES_SIZE]; \
  int          pos;  /* next free in data */\
  struct NAME* next; /* link list of bloc */\
} NAME


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
  kaapi_frame_t*                 stackframe;     /** for execution, see kaapi_thread_execframe */

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
  kaapi_task_t*         volatile pc      __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the task the thief wants to steal */
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */
  kaapi_task_t*         volatile thiefpc;        /** pointer to the task the thief wants to steal */
#endif
#if !defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  kaapi_threadgroup_t            thgrp;          /** the current thread group, used to push task */
#endif

  int                            unstealable;    /* !=0 -> cannot be stolen */
  int                            partid;         /* used by static scheduling to identify the thread in the group */

  struct kaapi_thread_context_t* _next;          /** to be linkable either in proc->lfree or proc->lready */
  struct kaapi_thread_context_t* _prev;          /** to be linkable either in proc->lfree or proc->lready */

  kaapi_atomic_t                 lock __attribute__((aligned (KAAPI_CACHE_LINE)));           /** ??? */ 
  kaapi_affinity_t               affinity;       /* bit i == 1 -> can run on procid i */

  void*                          alloc_ptr;      /** pointer really allocated */
  kaapi_uint32_t                 size;           /** size of the data structure allocated */
  kaapi_task_t*                  task;           /** bottom of the stack of task */

  struct kaapi_wsqueuectxt_cell_t* wcs;          /** point to the cell in the suspended list, iff thread is suspended */

  /* statically allocated reply */
  kaapi_reply_t			              static_reply;
  /* enough space to store a stealcontext that begins at static_reply->udata+static_reply->offset */
  char	                          sc_data[sizeof(kaapi_stealcontext_t)-sizeof(kaapi_stealheader_t)];

  /* warning: reply is variable sized
     so do not add members from here
   */
  kaapi_uint64_t                 data[1];        /** begin of stack of data */ 
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_thread_context_t;

/* helper function */
#define kaapi_stack2threadcontext(stack)         ( ((kaapi_thread_context_t*)stack)-1 )
#define kaapi_threadcontext2stack(thread)        ( (kaapi_stack_t*)((thread)+1) )
#define kaapi_threadcontext2thread(thread)       ( (kaapi_thread_t*)((thread)->sfp) )

/* ===================== Default internal task body ==================================== */
/** Body of the nop task 
    \ingroup TASK
*/
extern void kaapi_nop_body( void*, kaapi_thread_t*);

/** Body of the startup task 
    \ingroup TASK
*/
extern void kaapi_taskstartup_body( void*, kaapi_thread_t*);

/** Body of the task that mark a task to suspend execution
    \ingroup TASK
*/
extern void kaapi_suspend_body( void*, kaapi_thread_t*);

/** Body of the task that mark a task as under execution
    \ingroup TASK
*/
extern void kaapi_exec_body( void*, kaapi_thread_t*);

/** Body of task steal created on thief stack to execute a task
    \ingroup TASK
*/
extern void kaapi_tasksteal_body( void*, kaapi_thread_t* );

/** Write result after a steal 
    \ingroup TASK
*/
extern void kaapi_taskwrite_body( void*, kaapi_thread_t* );

/** Merge result after a steal
    \ingroup TASK
*/
extern void kaapi_aftersteal_body( void*, kaapi_thread_t* );

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_adapt_body( void*, kaapi_thread_t* );


/* ============================= Implementation method ============================ */
/** Note: a body is a pointer to a function. We assume that a body <=> void* and has
    64 bits on 64 bits architecture.
    The 4 highest bits are used to store the state of the task :
    - 0000 : the task has been pushed on the stack (<=> a user pointer function has never high bits == 11)
    - 0100 : the task has been execute either by the owner / either by a thief
    - 1000 : the task has been steal by a thief for execution
    - 1100 : the task has been theft by a thief for execution and it was executed, the body is "aftersteal body"
    
    Other reserved bits are :
    - 0010 : the task is terminated. This state if only put for debugging.
*/
#if (SIZEOF_VOIDP == 4)
#warning "This code assume that 4 higher bits is available on any function pointer. It was not verify of this configuration"
#  define KAAPI_MASK_BODY_TERM    (0x1UL << 28)
#  define KAAPI_MASK_BODY_PREEMPT (0x2UL << 28) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2UL << 28)
#  define KAAPI_MASK_BODY_EXEC    (0x4UL << 28)
#  define KAAPI_MASK_BODY_STEAL   (0x8UL << 28)
#  define KAAPI_MASK_BODY_STATE   (0xEUL << 28)
#  define KAAPI_MASK_BODY         (0xFUL << 28)
#  define KAAPI_MASK_BODY_SHIFTR   28UL
#  define KAAPI_TASK_ATOMIC_OR(a, v) KAAPI_ATOMIC_OR_ORIG(a, v)

#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)a, (kaapi_uint32_t)o, (kaapi_uint32_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR_ORIG( (kaapi_atomic_t*)a, (kaapi_uint32_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND_ORIG( (kaapi_atomic_t*)a, (kaapi_uint32_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic_t*)a, (kaapi_uint32_t)v)

#elif (SIZEOF_VOIDP == 8)
#  define KAAPI_MASK_BODY_TERM    (0x1UL << 60UL)
#  define KAAPI_MASK_BODY_PREEMPT (0x2UL << 60UL) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2UL << 60UL)
#  define KAAPI_MASK_BODY_EXEC    (0x4UL << 60UL)
#  define KAAPI_MASK_BODY_STEAL   (0x8UL << 60UL)
#  define KAAPI_MASK_BODY_STATE   (0xEUL << 60UL)
#  define KAAPI_MASK_BODY         (0xFUL << 60UL)
#  define KAAPI_MASK_BODY_SHIFTR   60UL
#  define KAAPI_TASK_ATOMIC_OR(a, v) KAAPI_ATOMIC_OR64_ORIG(a, v)

#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS64( (kaapi_atomic64_t*)(a), (kaapi_uint64_t)o, (kaapi_uint64_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR64_ORIG( (kaapi_atomic64_t*)(a), (kaapi_uint64_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND64_ORIG( (kaapi_atomic64_t*)(a), (kaapi_uint64_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic64_t*)a, (kaapi_uint64_t)v)

#else
#  error "No implementation for pointer to function with size greather than 8 bytes. Please contact the authors."
#endif

/** \ingroup TASK
*/
//@{
#define kaapi_task_getstate(task)\
      (task)->u.state

#define kaapi_task_setstate(task, value)\
      (task)->u.state = (value)

#define kaapi_task_setstate_barrier(task, value)\
      { kaapi_writemem_barrier(); (task)->u.state = (value); }

#define kaapi_task_state_get(state)\
      (((state) & KAAPI_MASK_BODY) >> KAAPI_MASK_BODY_SHIFTR)

#define kaapi_task_state_issteal(state)       \
      ((state) & KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_isexec(state)        \
      ((state) & KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_isterm(state)        \
      ((state) & KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_isaftersteal(state)  \
      ((state) & KAAPI_MASK_BODY_AFTER)

#define kaapi_task_state_ispreempted(state)  \
      ((state) & KAAPI_MASK_BODY_PREEMPT)

#define kaapi_task_state_isspecial(state)     \
      ((state) & KAAPI_MASK_BODY)

#define kaapi_task_state_isnormal(state)     \
      (((state) & KAAPI_MASK_BODY) ==0)

/* this macro should only be called on a theft task to determine if it is ready */
#define kaapi_task_state_isready(state)       \
      (((state) & (KAAPI_MASK_BODY_AFTER|KAAPI_MASK_BODY_TERM)) !=0)

#define kaapi_task_state_isstealable(state)   \
      (((state) & (KAAPI_MASK_BODY_STEAL|KAAPI_MASK_BODY_EXEC)) ==0)

#define kaapi_task_state_setsteal(state)      \
    ((state) | KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_setexec(state)       \
    ((state) | KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_setterm(state)       \
    ((state) | KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_setafter(state)       \
    ((state) | KAAPI_MASK_BODY_AFTER)
    
#define kaapi_task_body2state(body)           \
    ((kaapi_uintptr_t)body)

#define kaapi_task_state2body(state)           \
    ((kaapi_task_body_t)(state))

#define kaapi_task_state2int(state)            \
    ((int)(state >> KAAPI_MASK_BODY_SHIFTR))

/** \ingroup TASK
    Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->u.body = body;
}

/** \ingroup TASK
    Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(kaapi_task_t* task)
{
  return kaapi_task_state2body( task->u.state & ~KAAPI_MASK_BODY );
}
//@}

/** \ingroup TASK
    The function kaapi_task_body_isstealable() will return non-zero value iff the task body may be stolen.
    All user tasks are stealable.
    \param body IN a task body
*/
inline static int kaapi_task_body_isstealable(kaapi_task_body_t body)
{ 
  kaapi_uintptr_t state  = (kaapi_uintptr_t)body;
  body = kaapi_task_state2body(state);
  return kaapi_task_state_isstealable(state)
      && (body != kaapi_taskstartup_body) 
      && (body != kaapi_nop_body)
      && (body != kaapi_tasksteal_body) 
      && (body != kaapi_taskwrite_body)
      && (body != kaapi_taskfinalize_body) 
      && (body != kaapi_adapt_body)
      ;
}

/** \ingroup TASK
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    All previous internal task body are not stealable. All user task are stealable.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isstealable(const kaapi_task_t* task)
{ 
  return kaapi_task_body_isstealable(task->u.body);
}

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
static inline void* _kaapi_thread_pushdata( kaapi_thread_context_t* thread, kaapi_uint32_t count)
{ return kaapi_thread_pushdata( kaapi_threadcontext2thread(thread), count ); }

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
/** Atomically: OR of the task state with the value in 'state' and return the previous value.
*/
static inline kaapi_uintptr_t kaapi_task_andstate( kaapi_task_t* task, kaapi_uintptr_t state )
{
  kaapi_uintptr_t retval = KAAPI_ATOMIC_ANDPTR_ORIG(&task->u.state, state);
  return retval;
}

static inline kaapi_uintptr_t kaapi_task_orstate( kaapi_task_t* task, kaapi_uintptr_t state )
{
#if defined(__i386__)||defined(__x86_64)
  /* WARNING: here we assume that the locked instruction do a writememory barrier */
#else
  kaapi_writemem_barrier();
#endif
  kaapi_uintptr_t retval = KAAPI_ATOMIC_ORPTR_ORIG(&task->u.state, state);
  return retval;
}

static inline int kaapi_task_teststate( kaapi_task_t* task, kaapi_uintptr_t state )
{
  /* assume a mem barrier has been done */
  return (task->u.state & state) !=0;
}

/** \ingroup TASK
   adaptive steal mode locking.
   Task state is encoded in body TAES bits.
   in the case of an adaptive task, we have:
   TAESbits | task_adapt_body, where
   (T)erm = 0
   (A)fter = 0
   (E)xec = 1
   (S)teal = ?
   Thus we assume lock_steal() is called once
   the adaptive task has started execution,
   which is a valid assumption (otherwise it
   would not have any work to split and we
   would not be here).
   Then the TAES bits state holds true for the
   whole execution of the adaptive task and allows
   us to implement steal locking based on the
   Steal bit and an atomic or operation.
   Steal = 1 means locked.
 */

inline static void kaapi_task_lock_adaptive_steal(kaapi_stealcontext_t* sc)
{
#if 0
  /* does not work, unlock may overwrite the
     STEAL state set by the splitter, making
     the synchro protocol fail.
   */

  const uintptr_t locked_state =
    KAAPI_MASK_BODY_STEAL |
    KAAPI_MASK_BODY_EXEC |
    (uintptr_t)kaapi_adapt_body;

  while (1)
  {
    /* the previous state was not locked, we won */
    if (kaapi_task_orstate(task, locked_state) != locked_state)
      break ;

    kaapi_slowdown_cpu();
  }
#else
  while (1)
  {
    if ((KAAPI_ATOMIC_READ(&sc->thieves.list.lock) == 0) && KAAPI_ATOMIC_CAS(&sc->thieves.list.lock, 0, 1))
      break ;
    kaapi_slowdown_cpu();
  }
#endif
}

inline static void kaapi_task_unlock_adaptive_steal(kaapi_stealcontext_t* sc)
{
#if 0 /* not working, cf. above comment */
  const uintptr_t unlocked_state =
    KAAPI_MASK_BODY_EXEC | (uintptr_t)kaapi_adapt_body;
  kaapi_task_setstate_barrier(task, unlocked_state);
#else
  KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
#endif
}

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
#else
#  warning "NOT IMPLEMENTED"
#endif

/** Should be only use in debug mode
    - other bodies should be added
*/
#if !defined(KAAPI_NDEBUG)
static inline int kaapi_isvalid_body( kaapi_task_body_t body)
{
  return 
    (kaapi_format_resolvebybody( body ) != 0) 
      || (body == kaapi_taskmain_body)
      || (body == kaapi_tasksteal_body)
      || (body == kaapi_taskwrite_body)
      || (body == kaapi_aftersteal_body)
      || (body == kaapi_nop_body)
  ;
}
#else
static inline int kaapi_isvalid_body( kaapi_task_body_t body)
{
  return 1;
}
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


/* ========================================================================= */
/* */
extern kaapi_uint64_t kaapi_perf_thread_delayinstate(struct kaapi_processor_t* kproc);



/* ========== Here include machine specific function: only next definitions should depend on machine =========== */
/** Here include all machine dependent functions and types
*/
#include "kaapi_machine.h"
/* ========== MACHINE DEPEND DATA STRUCTURE =========== */



/* ========================================================================== */
/** Compute a hash value from a string
*/
extern kaapi_uint32_t kaapi_hash_value_len(const char * data, int len);

/*
*/
extern kaapi_uint32_t kaapi_hash_value(const char * data);

/** Hash value for pointer.
    Used for data flow dependencies
*/
static inline kaapi_uint32_t kaapi_hash_ulong(kaapi_uint64_t ptr)
{
  /* */
  kaapi_uint64_t val = ptr >> 3;
  val = (val & 0xFFFF) ^ (val>>32);
  return (kaapi_uint32_t)val;
}


/**
 * Compression 64 -> 7 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^7 + X^3 + 1
 */
static inline kaapi_uint32_t kaapi_hash_ulong7(kaapi_uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000080) v ^= 0x00000009;
    return (kaapi_uint32_t) (v&0x0000007F);
}


/**
 * Compression 64 -> 6 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^6 + X + 1
 */
static inline kaapi_uint32_t kaapi_hash_ulong6(kaapi_uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000040) v ^= 0x00000003;
    if (v & 0x00000080) v ^= 0x00000006;
    return (kaapi_uint32_t) (v&0x0000003F);
}

/**
 * Compression 64 -> 5 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^5 + X^2 + 1
 */
static inline kaapi_uint32_t kaapi_hash_ulong5(kaapi_uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000020) v ^= 0x00000005;
    if (v & 0x00000040) v ^= 0x0000000A;
    if (v & 0x00000080) v ^= 0x00000014;
    return (kaapi_uint32_t) (v&0x0000001F);
}


/* ======================== Dependencies resolution function ========================*/
/** \ingroup DFG
*/
typedef struct kaapi_gd_t {
  kaapi_access_mode_t         last_mode;    /* last access mode to the data */
  void*                       last_version; /* last verion of the data, 0 if not ready */
} kaapi_gd_t;

/* fwd decl
*/
struct kaapi_version_t;


/* ============================= Hash table for WS ============================ */
/*
*/
typedef struct kaapi_hashentries_t {
  union { /* depending of the kind of hash table... */
    kaapi_gd_t                value;
    struct kaapi_version_t*   dfginfo;     /* list of tasks to wakeup at the end */
  } u;
  void*                       key;
  struct kaapi_hashentries_t* next; 
} kaapi_hashentries_t;

KAAPI_DECLARE_BLOCENTRIES(kaapi_hashentries_bloc_t, kaapi_hashentries_t);


/* Hashmap default size.
   Warning in kapai_hashmap_t, entry_map type should have a size that is
   equal to KAAPI_HASHMAP_SIZE.
*/
#define KAAPI_HASHMAP_SIZE 32

/*
*/
typedef struct kaapi_hashmap_t {
  kaapi_hashentries_t* entries[KAAPI_HASHMAP_SIZE];
  kaapi_hashentries_bloc_t* currentbloc;
  kaapi_hashentries_bloc_t* allallocatedbloc;
  kaapi_uint32_t entry_map;                 /* type size must match KAAPI_HASHMAP_SIZE */
} kaapi_hashmap_t;


/*
*/
extern int kaapi_hashmap_init( kaapi_hashmap_t* khm, kaapi_hashentries_bloc_t* initbloc );

/*
*/
extern int kaapi_hashmap_clear( kaapi_hashmap_t* khm );

/*
*/
extern int kaapi_hashmap_destroy( kaapi_hashmap_t* khm );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_findinsert( kaapi_hashmap_t* khm, void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_find( kaapi_hashmap_t* khm, void* ptr );

/*
*/
extern kaapi_hashentries_t* kaapi_hashmap_insert( kaapi_hashmap_t* khm, void* ptr );

/*
*/
extern kaapi_hashentries_t* get_hashmap_entry( kaapi_hashmap_t* khm, kaapi_uint32_t key);

/*
*/
extern void set_hashmap_entry( kaapi_hashmap_t* khm, kaapi_uint32_t key, kaapi_hashentries_t* entries);


/* ============================= Commun function for server side (no public) ============================ */
/** lighter than kaapi_thread_clear and used during the steal emit request function
*/
static inline int kaapi_thread_reset(kaapi_thread_context_t* th )
{
  th->sfp        = th->stackframe;
  th->esfp       = th->stackframe;
  th->sfp->sp    = th->sfp->pc  = th->task; /* empty frame */
  th->sfp->sp_data = (char*)&th->data;     /* empty frame */
  th->affinity   = ~0UL;
  th->unstealable= 0;
  return 0;
}

static inline int kaapi_thread_clearaffinity(kaapi_thread_context_t* th )
{
  th->affinity = 0;
  return 0;
}

/**
*/
static inline int kaapi_thread_setaffinity(kaapi_thread_context_t* th, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_affinity_t)*8) );
  th->affinity |= ((kaapi_affinity_t)1)<<kid;
  return 0;
}

/** Return non 0 iff th as affinity with kid
*/
static inline int kaapi_thread_hasaffinity(unsigned long affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_affinity_t)*8) );
  return (int)(affinity & ((kaapi_affinity_t)1)<<kid);
}

/**
*/
static inline int kaapi_sched_suspendlist_empty(kaapi_processor_t* kproc)
{
  if (kproc->lsuspend.head ==0) return 1;
  return 0;
}

/** Call only on thread that has the top task theft.
*/
static inline int kaapi_thread_isready( kaapi_thread_context_t* thread )
{
  kaapi_assert_debug( kaapi_task_state_issteal(thread->sfp->pc->u.state) );
  return kaapi_task_state_isready(thread->sfp->pc->u.state);
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
#if defined(KAAPI_SCHED_LOCK_CAS)
  KAAPI_ATOMIC_WRITE(lock,0);
#else
  KAAPI_ATOMIC_WRITE(lock,1);
#endif
  return 0;
}

static inline int kaapi_sched_trylock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  /* implicit barrier in KAAPI_ATOMIC_CAS if lock is taken */
  ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
  kaapi_assert_debug( !ok || (ok && KAAPI_ATOMIC_READ(lock) == 1) );
  return ok;
#else
  if (KAAPI_ATOMIC_DECR(lock) ==0) 
  {
    return 1;
  }
  return 0;
#endif
}

/** 
*/
static inline int kaapi_sched_lock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  do {
    ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
    if (ok) break;
    kaapi_slowdown_cpu();
  } while (1);
  /* implicit barrier in KAAPI_ATOMIC_CAS */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) != 0 );
#else
acquire:
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
  while (KAAPI_ATOMIC_READ(lock) <=0)
    kaapi_slowdown_cpu(); 
  goto acquire;
#endif
  return 0;
}


/**
*/
static inline int kaapi_sched_lock_spin( kaapi_atomic_t* lock, int spincount )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  int ok;
  do {
    ok = (KAAPI_ATOMIC_READ(lock) ==0) && KAAPI_ATOMIC_CAS(lock, 0, 1);
    if (ok) break;
    kaapi_slowdown_cpu();
  } while (1);
  /* implicit barrier in KAAPI_ATOMIC_CAS */
  kaapi_assert_debug( KAAPI_ATOMIC_READ(lock) != 0 );
#else
  int i;
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
  for (i=0; (KAAPI_ATOMIC_READ(lock) <=0) && (i<spincount); ++i)
    kaapi_slowdown_cpu();
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
#endif
  return 0;
}


/**
*/
static inline int kaapi_sched_unlock( kaapi_atomic_t* lock )
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  kaapi_assert_debug( (unsigned)KAAPI_ATOMIC_READ(lock) == (unsigned)(1) );
  /* mplicit barrier in KAAPI_ATOMIC_WRITE_BARRIER */
  KAAPI_ATOMIC_WRITE_BARRIER(lock, 0);
#else
  KAAPI_ATOMIC_WRITE_BARRIER(lock, 1);
#endif
  return 0;
}

static inline void kaapi_sched_waitlock(kaapi_atomic_t* lock)
{
  /* wait until reaches the unlocked state */

#if defined(KAAPI_SCHED_LOCK_CAS)
  while (KAAPI_ATOMIC_READ(lock))
#else
  while (KAAPI_ATOMIC_READ(lock) == 0)
#endif
    kaapi_slowdown_cpu();
}

/** steal/pop (no distinction) a thread to thief with kid
    If the owner call this method then it should protect 
    itself against thieves by using sched_lock & sched_unlock
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

/**
*/
extern int kaapi_thread_clear( kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_thread_print( FILE* file, kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_task_print( FILE* file, kaapi_task_t* task );

/** \ingroup TASK
    The function kaapi_thread_execframe() execute all the tasks in the thread' stack following
    the RFO order in the closures of the frame [frame_sp,..,sp[
    If successful, the kaapi_thread_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
*/
extern int kaapi_thread_execframe( kaapi_thread_context_t* thread );

/** Useful
*/
extern kaapi_processor_t* kaapi_get_current_processor(void);

/** \ingroup WS
    Select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Select a victim for next steal request using workload then uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_workload_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    First steal is 0 then select a victim for next steal request using uniform random selection over all cores.
*/
extern int kaapi_sched_select_victim_rand_first0( kaapi_processor_t* kproc, kaapi_victim_t* victim);

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
int kaapi_sched_suspend ( kaapi_processor_t* kproc );

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
  kaapi_processor_t* kproc, 
  kaapi_listrequest_t* lrequests, 
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
  struct kaapi_thread_context_t* cond 
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
  unsigned int                  war_param, 
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
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
static inline unsigned long kaapi_reply_status( kaapi_reply_t* ksr ) 
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
  kaapi_format_t*         origin_fmt;        /* set by tasksteal the stolen task into origin_stack */
  unsigned int            war_param;         /* bit i=1 iff it is a w mode with war dependency */
  void*                   copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;


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




/* ======================== Perf counter interface: machine dependent ========================*/
/* for perf_regs access: SHOULD BE 0 and 1 
   All counters have both USER and SYS definition (sys == program that execute the scheduler).
   * KAAPI_PERF_ID_T1 is considered as the T1 (computation time) in the user space
   and as TSCHED, the scheduling time if SYS space. In workstealing litterature it is also named Tidle.
   [ In Kaapi, TIDLE is the time where the thread (kprocessor) is not scheduled on hardware... ]
*/
#define KAAPI_PERF_USER_STATE       0
#define KAAPI_PERF_SCHEDULE_STATE   1

/* return a reference to the idp-th performance counter of the k-processor in the current set of counters */
#define KAAPI_PERF_REG(kproc, idp) ((kproc)->curr_perf_regs[(idp)])

/* return a reference to the idp-th USER performance counter of the k-processor */
#define KAAPI_PERF_REG_USR(kproc, idp) ((kproc)->perf_regs[KAAPI_PERF_USER_STATE][(idp)])

/* return a reference to the idp-th USER performance counter of the k-processor */
#define KAAPI_PERF_REG_SYS(kproc, idp) ((kproc)->perf_regs[KAAPI_PERF_SCHEDULE_STATE][(idp)])

/* return the sum of the idp-th USER and SYS performance counters */
#define KAAPI_PERF_REG_READALL(kproc, idp) (KAAPI_PERF_REG_SYS(kproc, idp)+KAAPI_PERF_REG_USR(kproc, idp))

/* internal */
extern void kaapi_perf_global_init(void);

/* */
extern void kaapi_perf_global_fini(void);

/* */
extern void kaapi_perf_thread_init ( kaapi_processor_t* kproc, int isuser );
/* */
extern void kaapi_perf_thread_fini ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_start ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_stop ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser );
/* */
extern int kaapi_perf_thread_state(kaapi_processor_t* kproc);
/* */
extern kaapi_uint64_t kaapi_perf_thread_delayinstate(kaapi_processor_t* kproc);

/* */
extern void kaapi_set_workload( kaapi_processor_t*, kaapi_uint32_t workload );

/* */
extern void kaapi_set_self_workload( kaapi_uint32_t workload );

#include "kaapi_staticsched.h"

/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PUBLIC INTERFACE ........................................*/

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_IMPL_H */
