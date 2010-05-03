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
** theo.trouillon@imag.fr
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

/**/
extern double t_finalize;

/* mark that we compile source of the library */
#define KAAPI_COMPILE_SOURCE 1

#include "config.h"
#include "kaapi.h"
#include "kaapi_error.h"
#include <string.h>

#include "kaapi_defs.h"

/* Maximal number of recursive call used to store the stack of frames
*/
#define KAAPI_MAX_RECCALL 1024

/* Flags to define method to manage concurrency between victim and thieves
   - STEALCAS: based on compare & swap method
   - STEALTHE: based on Dijkstra like protocol to ensure mutual exclusion
*/
#define KAAPI_STEALCAS_METHOD 0
#define KAAPI_STEALTHE_METHOD 1

/* Selection of the method to manage concurrency between victim/thief 
   to steal task:
*/
#ifndef KAAPI_USE_STEALTASK_METHOD
#define KAAPI_USE_STEALTASK_METHOD KAAPI_STEALCAS_METHOD
#endif


/* Selection of the method to steal into frame:
*/
#ifndef KAAPI_USE_STEALFRAME_METHOD
#define KAAPI_USE_STEALFRAME_METHOD KAAPI_STEALTHE_METHOD
#endif

/* Verification of correct choice of values */
#if (KAAPI_USE_STEALFRAME_METHOD !=KAAPI_STEALCAS_METHOD) && (KAAPI_USE_STEALFRAME_METHOD !=KAAPI_STEALTHE_METHOD)
#error "Bad definition of value for steal frame method"
#endif

#if (KAAPI_USE_STEALTASK_METHOD !=KAAPI_STEALCAS_METHOD) && (KAAPI_USE_STEALTASK_METHOD !=KAAPI_STEALTHE_METHOD)
#error "Bad definition of value for steal frame method"
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
#endif

/**
*/
#  define kaapi_assert_m(cond, msg) \
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


// This is the new version on top of X-Kaapi
extern const char* get_kaapi_version(void);

/** Global hash table of all formats: body -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_bybody[256];

/** Global hash table of all formats: fmtid -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_byfmtid[256];


/* Fwd declaration 
*/
struct kaapi_processor_t;
struct kaapi_listrequest_t;



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
/** Initialise default format
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
  unsigned int		         kid_to_cpu[KAAPI_MAX_PROCESSOR];
  int                      display_perfcounter;    /* set to 1 iff KAAPI_DISPLAY_PERF */
  kaapi_uint64_t           startuptime;            /* time at the end of kaapi_init */
} kaapi_rtparam_t;

extern kaapi_rtparam_t kaapi_default_param;



/* ============================= REQUEST ============================ */
/** Private status of request
    \ingroup WS
*/
enum kaapi_request_status_t {
  KAAPI_REQUEST_S_EMPTY   = 0,
  KAAPI_REQUEST_S_POSTED  = 1,
  KAAPI_REQUEST_S_SUCCESS = 2,
  KAAPI_REQUEST_S_FAIL    = 3,
  KAAPI_REQUEST_S_ERROR   = 4,
  KAAPI_REQUEST_S_QUIT    = 5
};



/* ============================= Format for task ============================ */
/*
*/
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
  kaapi_task_body_t          entrypoint[KAAPI_MAX_ARCHITECTURE];      /* maximum architecture considered in the configuration */
  int                        count_params;                            /* number of parameters */
  kaapi_access_mode_t        *mode_params;                            /* only consider value with mask 0xF0 */
  kaapi_offset_t             *off_params;                             /* access to the i-th parameter: a value or a shared */
  struct kaapi_format_t*     *fmt_params;                             /* format for each params */
  kaapi_uint32_t             *size_params;                            /* sizeof of each params */

  struct kaapi_format_t      *next_bybody;                            /* link in hash table */
  struct kaapi_format_t      *next_byfmtid;                           /* link in hash table */
  
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



/* ============================= The stack data structure ============================ */
/** Kaapi stack of tasks definition
   \ingroup TASK
   The stack store list of tasks as well as a stack of data.
   Both sizes are fixed at initialization of the stack object.
   The stack is truly a stack when used in conjonction with frame.
   A frame capture the state (pc, sp, sp_data) of the stack in order
   to restore it. The implementation also used kaapi_retn_body in order 
   to postpone the restore operation after a set of tasks (see kaapi_stack_taskexecall).

   Before and after the execution of a task, the state of the computation is only
   defined by the stack state (pc, sp, sp_data and the content of the stack). Not that
   kaapi_stack_execframe and other funcitons to execute tasks may cached internal state (pc). 
   The C-stack doesnot need to be saved in that case.
   
   \TODO save also the C-stack if we try to suspend execution during a task execution
   \TODO a better separation between the thread context and the stack it self
   
   Warning this stack structure is just after the internal kaapi_threadcontext_t structure
   which is opaque to the API.
*/
typedef struct kaapi_stack_t {
  struct kaapi_task_t*      task;           /** pointer to the first pushed task */
  char*                     data;           /** stack of data with the same scope than task */
  int                       sticky;         /** 1 iff the stack could not be steal else by a context swap */ 

  volatile int              hasrequest __attribute__((aligned (KAAPI_CACHE_LINE)));     /** points to the k-processor structure */
  volatile int              haspreempt;     /** !=0 if preemption is requested */
  kaapi_request_t*          requests;       /** points to the requests set in the processor structure */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_stack_t;



/* ============================= The thread context data structure ============================ */
/** The thread context data structure
    This data structure should be extend in case where the C-stack is required to be suspended and resumed.
    This data structure is always at position ((kaapi_thread_context_t*)stackaddr) - 1 of stack at address
    stackaddr.
    It was made opaque to the user API because we do not want to expose the way we execute stack in the
    user code.
*/
typedef struct kaapi_thread_context_t {
  kaapi_frame_t*        volatile sfp;            /** pointer to the current frame (in stackframe) */
  kaapi_frame_t*                 esfp;           /** first frame until to execute all frame  */
  int                            errcode;        /** set by task execution to signal incorrect execution */
  struct kaapi_processor_t*      proc;           /** access to the running processor */
  kaapi_frame_t*                 stackframe;     /** for execution, see kaapi_stack_execframe */
  struct kaapi_thread_context_t* _next;          /** to be stackable */

#if (KAAPI_USE_STEALFRAME_METHOD == KAAPI_STEALTHE_METHOD)
  kaapi_frame_t*        volatile thieffp __attribute__((aligned (KAAPI_CACHE_LINE))); /** pointer to the thief frame where to steal */
#endif
#if (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)
  kaapi_task_t*         volatile thiefpc;        /** pointer to the task the thief wants to steal */
#endif
  kaapi_atomic_t                 lock;           /** */ 

  void*                          alloc_ptr;      /** pointer really allocated */
  kaapi_uint32_t                 size;           /** size of the data structure allocated */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_thread_context_t;

/* helper function */
#define kaapi_stack2threadcontext(stack)         ( ((kaapi_thread_context_t*)stack)-1 )
#define kaapi_threadcontext2stack(thread)        ( (kaapi_stack_t*)((thread)+1) )
#define kaapi_threadcontext2thread(thread)       ( (kaapi_thread_t*)((thread)->sfp))



/* ============================= The structure for adaptive algorithm ============================ */
/** 
*/
struct kaapi_taskadaptive_result_t;

/** \ingroup ADAPT
    Extent data structure for adaptive task.
    This data structure is attached to any adaptative tasks.
*/
typedef struct kaapi_taskadaptive_t {
  kaapi_stealcontext_t                sc;              /* user visible part of the data structure &sc == kaapi_stealcontext_t* */

  kaapi_atomic_t                      lock;            /* required for access to list */
  struct kaapi_taskadaptive_result_t* head __attribute__((aligned(KAAPI_CACHE_LINE))); /* head of the LIFO order of result */
  struct kaapi_taskadaptive_result_t* tail __attribute__((aligned(KAAPI_CACHE_LINE))); /* tail of the LIFO order of result */
  kaapi_atomic_t                      thievescount __attribute__((aligned(KAAPI_CACHE_LINE)));     /* #thieves of the owner of this structure.... */
  struct kaapi_taskadaptive_t*        origin_master;    /* who to report global end at the end of computation, 0 iff first master task */
  kaapi_task_splitter_t               save_splitter;   /* for steal_[begin|end]critical section */
  void*                               save_argsplitter;/* idem */
  kaapi_frame_t                       frame;
} kaapi_taskadaptive_t;


/** \ingroup ADAPT
    Data structure that allows to store results of child tasks of an adaptive task.
    This data structure is stored... in the victim heap and serve as communication 
    media between victim and thief.
*/
typedef struct kaapi_taskadaptive_result_t {
  /* same as public part of the structure in kaapi.h */
  void*                               data;             /* the data produced by the thief */
  size_t                              size_data;        /* size of data */
  void* volatile                      arg_from_victim;  /* arg from the victim after preemption of one victim */
  void* volatile                      arg_from_thief;   /* arg of the thief passed at the preemption point */
  int volatile                        req_preempt;

  /* Private part of the structure */
  volatile int                        thief_term;       /* */
  struct kaapi_taskadaptive_t*        master;           /* who to signal at the end of computation, 0 iff master task */
  int                                 flag;             /* where is allocated data */

  struct kaapi_taskadaptive_result_t* rhead;            /* double linked list of thieves of this thief */
  struct kaapi_taskadaptive_result_t* rtail;            /* */

  struct kaapi_taskadaptive_result_t* next;             /* link fields in kaapi_taskadaptive_t */
  struct kaapi_taskadaptive_result_t* prev;             /* */

  void*				      addr_tofree;	/* the non aligned malloc()ed addr */
  
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_taskadaptive_result_t;

#define KAAPI_RESULT_DATAUSR    0x01
#define KAAPI_RESULT_DATARTS    0x02



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

/** Body of the task that do signal to a task after steal op
    \ingroup TASK
*/
extern void kaapi_tasksig_body( void*, kaapi_thread_t*);

/** Merge result after a steal
    \ingroup TASK
*/
extern void kaapi_aftersteal_body( void*, kaapi_thread_t* );

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_adapt_body( void*, kaapi_thread_t* );


/* ============================= Implementation method ============================ */

/** \ingroup TASK
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    All previous internal task body are not stealable. All user task are stealable.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isstealable(const kaapi_task_t* task)
{ 
  return (task->body != kaapi_taskstartup_body) && (task->body != kaapi_nop_body)
      && (task->body != kaapi_suspend_body) && (task->body != kaapi_exec_body) && (task->body != kaapi_aftersteal_body) 
      && (task->body != kaapi_tasksteal_body) && (task->body != kaapi_taskwrite_body) && (task->body != kaapi_tasksig_body)
      && (task->body != kaapi_taskfinalize_body) && (task->body != kaapi_taskreturn_body) && (task->body != kaapi_adapt_body)
      ;
}


/** \ingroup TASK
    Set the extra body of the task
*/
static inline void kaapi_task_setextrabody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->ebody = body;
}

/** \ingroup TASK
    Get the extra body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getextrabody(kaapi_task_t* task)
{
  return task->ebody;
}

/** \ingroup TASK
*/
static inline kaapi_task_t* _kaapi_thread_toptask( kaapi_thread_context_t* thread ) 
{
  return kaapi_thread_toptask( kaapi_threadcontext2thread(thread) );
}


/** \ingroup TASK
*/
static inline int _kaapi_thread_pushtask( kaapi_thread_context_t* thread )
{
  return kaapi_thread_pushtask( kaapi_threadcontext2thread(thread) );
}


/** \ingroup TASK
*/
static inline void* _kaapi_thread_pushdata( kaapi_thread_context_t* thread, kaapi_uint32_t count)
{
  return kaapi_thread_pushdata( kaapi_threadcontext2thread(thread), count );
}


#if 0
/** \ingroup TASK
    The function kaapi_thread_save_frame() saves the current frame of a stack into
    the frame data structure.
    If successful, the kaapi_thread_save_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack IN a pointer to the kaapi_stack_t data structure.
    \param frame OUT a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int _kaapi_thread_save_frame( kaapi_thread_context_t* thread, kaapi_frame_t* frame)
{
  kaapi_assert_debug( (thread !=0) && (frame !=0) );
  frame->pc       = thread->sfp->pc;
  frame->sp       = thread->sfp->sp;
  frame->sp_data  = thread->sfp->sp_data;
  return 0;  
}

/** \ingroup TASK
    The function kaapi_thread_restore_frame() restores the frame context of a stack into
    the stack data structure.
    If successful, the kaapi_thread_restore_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \param frame IN a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int _kaapi_thread_restore_frame( kaapi_thread_context_t* thread, const kaapi_frame_t* frame)
{
  kaapi_assert_debug( (thread !=0) && (frame !=0) );
  thread->sfp->sp       = frame->sp;
  thread->sfp->pc       = frame->pc;
  thread->sfp->sp_data  = frame->sp_data;
  return 0;  
}

#endif

#if 0
/** \ingroup TASK
    The function kaapi_task_haslocality() will return non-zero value iff the task has locality constraints.
    In this case, the field locality my be read to resolved locality constraints.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_haslocality(const kaapi_task_t* task)
{ return (task->flag & KAAPI_TASK_LOCALITY); }

/** \ingroup TASK
    The function kaapi_task_isadaptive() will return non-zero value iff the task is an adaptive task.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isadaptive(const kaapi_task_t* task)
{
  return (task->body == kaapi_adapt_body); 
}
#endif


/** \ingroup TASK
    The function kaapi_stack_init() initializes the stack using the buffer passed in parameter. 
    The buffer must point to a memory region with at least count bytes allocated.
    If successful, the kaapi_stack_init() function will return zero and the buffer should
    never be used again.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t to initialize.
    \param size  IN the size in bytes of the buffer for the tasks.
    \param buffer INOUT the buffer to used to store the stack of tasks.
    \retval EINVAL invalid argument: bad stack pointer or count is not enough to store at least one task or buffer is 0.
*/
extern int kaapi_stack_init( kaapi_stack_t* stack, kaapi_uint32_t size, void* buffer );


/** \ingroup TASK
    The function kaapi_stack_clear() clears the stack.
    If successful, the kaapi_stack_clear() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t to clear.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_stack_clear( kaapi_stack_t* stack );


/** \ingroup TASK
    The function kaapi_frame_isempty() will return non-zero value iff the frame is empty. Otherwise return 0.
    \param stack IN the pointer to the kaapi_stack_t data structure. 
    \retval !=0 if the stack is empty
    \retval 0 if the stack is not empty or argument is an invalid stack pointer
*/
static inline int kaapi_frame_isempty(volatile kaapi_frame_t* frame)
{
  return (frame->pc <= frame->sp);
}


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
  return kaapi_threadcontext2stack(thread)->task;
}


/* ========================================================================= */
/* Shared object and access mode                                             */
/** \ingroup DFG
*/
typedef struct kaapi_gd_t {
  kaapi_access_mode_t last_mode;    /* last access mode to the data */
  void*               last_version; /* last verion of the data, 0 if not ready */
} kaapi_gd_t;



extern struct kaapi_processor_t* kaapi_get_current_processor(void);
typedef kaapi_uint32_t kaapi_processor_id_t;
extern kaapi_processor_id_t kaapi_get_current_kid(void);

/** Initialize a request
    \param kpsr a pointer to a kaapi_steal_request_t
*/
static inline void kaapi_request_init( struct kaapi_processor_t* kproc, kaapi_request_t* pkr )
{
  pkr->status = KAAPI_REQUEST_S_EMPTY; 
  pkr->flag   = 0; 
  pkr->reply  = 0;
  pkr->thread = 0; 
  pkr->mthread= 0; 
  pkr->proc   = kproc;
#if 0
  fprintf(stdout,"%i kproc clear request @req=%p\n", kaapi_get_current_kid(), (void*)pkr );
  fflush(stdout);
#endif
}


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
extern kaapi_uint32_t kaapi_hash_value(const char * data);


/* ============================= Hash table for WS ============================ */
/*
*/
typedef struct kaapi_counters_list {
    kaapi_atomic_t*              reader_counter; 
    kaapi_task_t*                waiting_task;
    struct kaapi_counters_list*  next;           //next reader counter
} kaapi_counters_list;

/*
*/
typedef struct kaapi_deps_t {
  kaapi_task_t*               last_writer;
  kaapi_thread_t*             last_writer_thread;
} kaapi_deps_t;

/*
*/
typedef struct kaapi_hashentries_t {
  kaapi_gd_t                  value;
  kaapi_deps_t*		      datas;  /* list of task to wakeup at the end */
  void*                       key;
  struct kaapi_hashentries_t* next; 
} kaapi_hashentries_t;

KAAPI_DECLARE_BLOCENTRIES(kaapi_hashentries_bloc_t, kaapi_hashentries_t);


#define KAAPI_HASHMAP_SIZE 128
/*
*/
typedef struct kaapi_hashmap_t {
  kaapi_hashentries_t* entries[KAAPI_HASHMAP_SIZE];
  kaapi_hashentries_bloc_t* currentbloc;
  kaapi_hashentries_bloc_t* allallocatedbloc;
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






/* ============================= Commun function for server side (no public) ============================ */
/**
*/
static inline int kaapi_sched_suspendlist_empty(kaapi_processor_t* kproc)
{
  if (kproc->lsuspend.head ==0) return 1;
  return 0;
}

/**
*/
extern int kaapi_thread_clear( kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_stack_print  ( FILE* file, kaapi_thread_context_t* thread );

/** Useful
*/
extern int kaapi_task_print( FILE* file, kaapi_task_t* task, kaapi_task_bodyid_t taskid );

/** \ingroup TASK
    The function kaapi_stack_execframe() execute all the tasks in the thread' stack following
    the RFO order in the closures of the frame [frame_sp,..,sp[
    If successful, the kaapi_stack_execframe() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
*/
extern int kaapi_stack_execframe( kaapi_thread_context_t* thread );

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
    Select a victim for next steal request using random selection level by level. Each time the method
    try to steal at level i, it first try to steal at level 0 until i.
    The idea of the algorithm is the following. Initial values are: level=0, toplevel=0
       1- the method do random selection at level and increment toplevel
       2- it increment level, if level >= toplevel then level=0, toplevel++
       3- if toplevel > maximal level then level=0, toplevel=0
*/
extern int kaapi_sched_select_victim_rand_incr( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Only do rando ws on the first level of the hierarchy. Assume that all cores are connected
    together using the first level hierarchy information.
*/
extern int kaapi_sched_select_victim_rand_first( kaapi_processor_t* kproc, kaapi_victim_t* victim);

/** \ingroup WS
    Helper function for some of the above random selection of victim
*/
extern int kaapi_select_victim_rand_atlevel( kaapi_processor_t* kproc, int level, kaapi_victim_t* victim );


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
extern int kaapi_sched_stealprocessor ( kaapi_processor_t* kproc );


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
extern int kaapi_sched_stealstack  ( struct kaapi_thread_context_t* thread, kaapi_task_t* curr, int count, kaapi_request_t* request );


/** \ingroup WS
    \retval 0 if no context could be wakeup
    \retval else a context to wakeup
    \TODO faire specs ici
*/
extern kaapi_thread_context_t* kaapi_sched_wakeup ( kaapi_processor_t* kproc );


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
extern int kaapi_task_splitter_dfg(kaapi_thread_context_t* thread, kaapi_task_t* task, int count, struct kaapi_request_t* array);

/** \ingroup WS
    Wrapper arround the user level Splitter for Adaptive task
*/
extern int kaapi_task_splitter_adapt( 
    kaapi_thread_context_t* thread, 
    kaapi_task_t* task,
    kaapi_task_splitter_t splitter,
    void* argsplitter,
    int count, 
    struct kaapi_request_t* array
);


/** \ingroup ADAPTIVE
     Disable steal on stealcontext and wait not more thief is stealing.
     Return 0 in case of success else return an error code.
 */
static inline int kaapi_steal_disable_sync(kaapi_stealcontext_t* stc)
{
  stc->splitter    = 0;
  stc->argsplitter = 0;
  kaapi_mem_barrier();

  while (KAAPI_ATOMIC_READ(&stc->is_there_thief) !=0)
    ;
  return 0;
}


/** \ingroup ADAPTIVE
    free a result previously allocate with kaapi_allocate_thief_result
    \param ktr IN the result to free
 */
extern void kaapi_free_thief_result(struct kaapi_taskadaptive_result_t* ktr);


/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/** \ingroup ADAPTIVE
    Reply a value to a steal request. If retval is !=0 it means that the request
    has successfully adapt to steal work. Else 0.
    While it reply to a request, the function DO NOT decrement the request count on the stack.
    This function is machine dependent.
*/
extern int _kaapi_request_reply( 
  kaapi_request_t*        request, 
  kaapi_thread_context_t* retval, 
  int                     isok
);

/** Destroy a request
    A posted request could not be destroyed until a reply has been made
*/
#define kaapi_request_destroy( kpsr ) 


/** Wait the end of request and return the error code
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_ERROR steal request has failed to be posted because the victim refused request
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
extern int kaapi_reply_wait( kaapi_reply_t* ksr );


/** Return true iff the request has been posted
  \param pksr kaapi_request_t
*/
static inline int kaapi_request_test( kaapi_request_t* kpsr )
{ return (kpsr->status == KAAPI_REQUEST_S_POSTED); }


/** Return true iff the request has been processed
  \param pksr kaapi_reply_t
*/
static inline int kaapi_reply_test( kaapi_reply_t* kpsr )
{ return (kpsr->status != KAAPI_REQUEST_S_POSTED); }


/** Return true iff the request is a success steal
  \param pksr kaapi_reply_t
*/
static inline int kaapi_reply_ok( kaapi_reply_t* kpsr )
{ return (kpsr->status == KAAPI_REQUEST_S_SUCCESS); }


/** Return the request status
  \param pksr kaapi_reply_t
  \retval KAAPI_REQUEST_S_SUCCESS sucessfull steal operation
  \retval KAAPI_REQUEST_S_FAIL steal request has failed
  \retval KAAPI_REQUEST_S_QUIT process should terminate
*/
static inline int kaapi_request_status( kaapi_reply_t* reply ) 
{ return reply->status; }


/** Return the data associated with the reply
  \param pksr kaapi_reply_t
*/
static inline kaapi_thread_context_t* kaapi_request_data( kaapi_reply_t* reply ) 
{ 
  kaapi_readmem_barrier();
  return reply->data; 
}


/** Args for tasksteal
*/
typedef struct kaapi_tasksteal_arg_t {
  kaapi_thread_context_t* origin_thread;     /* stack where task was stolen */
  kaapi_task_t*           origin_task;       /* the stolen task into origin_stack */
  kaapi_format_t*         origin_fmt;        /* set by tasksteal the stolen task into origin_stack */
  void*                   copy_task_args;    /* set by tasksteal a copy of the task args */
} kaapi_tasksteal_arg_t;


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


/**
 */
extern void kaapi_set_workload( kaapi_uint32_t workload );

/* ======================== Dependencies resolution function ========================*/

/*
typedef struct kaapi_dependenciessignal_arg_t {
    kaapi_task_body_t       real_body; //Real body to execute
    void*                   real_datas;
    kaapi_task_splitter_t   real_splitter;
    kaapi_format_t*         real_format;
    counters_list *         readers_list; //counters to decrement
} kaapi_dependenciessignal_arg_t;

void kaapi_dependenciessignal_body( kaapi_task_t* task, kaapi_stack_t* stack );
*/

/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PUBLIC INTERFACE ........................................*/

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_IMPL_H */
