/*
** kaapi.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
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
#ifndef _KAAPI_H
#define _KAAPI_H 1

#ifndef _KAAPI_DISABLE_WARNINGS
# if defined(__cplusplus)
#  if !defined(__GXX_EXPERIMENTAL_CXX0X__)
#   warning kaapi.h use variadic macros
#   warning you should try a compiler supporting the upcomming C++ standard
#   warning (with g++, try -std=c++0x or -std=gnu++0x for example)
#  endif
# else
#  if !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
#   warning kaapi.h use C99 constructions (such as variadic macros)
#   warning you should use a -std=c99 with gcc for example
#  endif
# endif
#endif


#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include "kaapi_error.h"

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef __GNUC__
#  define KAAPI_MAX_DATA_ALIGNMENT (__alignof__(void*))
#else
#  define KAAPI_MAX_DATA_ALIGNMENT 8
#endif

/* Kaapi name for stdint typedefs.
 */
typedef uintptr_t kaapi_uintptr_t;
typedef uint8_t   kaapi_uint8_t;
typedef uint16_t  kaapi_uint16_t;
typedef uint32_t  kaapi_uint32_t;
typedef uint64_t  kaapi_uint64_t;

typedef intptr_t  kaapi_intptr_t;
typedef int8_t    kaapi_int8_t;
typedef int16_t   kaapi_int16_t;
typedef int32_t   kaapi_int32_t;
typedef int64_t   kaapi_int64_t;

/** Atomic type
*/
typedef struct kaapi_atomic_t {
#if defined(__APPLE__)
  volatile kaapi_int32_t  _counter;
#else
  volatile kaapi_uint32_t _counter;
#endif
} kaapi_atomic_t;
typedef kaapi_atomic_t kaapi_atomic32_t;


typedef struct kaapi_atomic64_t {
#if defined(__APPLE__)
  volatile kaapi_int64_t  _counter;
#else
  volatile kaapi_uint64_t _counter;
#endif
} kaapi_atomic64_t;


#if defined(__i386__)||defined(__x86_64)
     /* WARNING here Compiler fence to keep operations. Note that on X86 no reorder of write ops
        so, we do not need extra hardware fence operation.
     */
#    define kaapi_writemem_barrier_api()  __asm__ __volatile__("" : : : "memory" )
#else
#    error "Fence operation should must put here. See the memory consistency of the hardware you use"
#endif

/* ========================================================================== */
struct kaapi_task_t;
struct kaapi_stack_t;
struct kaapi_thread_t;
struct kaapi_thread_context_t;
struct kaapi_stealcontext_t;


/* ========================================================================== */
/** Task body
    \ingroup TASK
    See internal doc in order to have better documentation of invariant between the task and the thread.
*/
typedef void (*kaapi_task_body_t)(void* /*task arg*/, struct kaapi_thread_t* /* thread */);
/* do not separate representation of the body and its identifier (should be format identifier) */
typedef kaapi_task_body_t kaapi_task_bodyid_t;

/** Define the cache line size. 
*/
#define KAAPI_CACHE_LINE 64

/** Maximal number of architecture
    Current naming is:
    - KAAPI_PROC_TYPE_CPU:   core of multicore machine
    - KAAPI_PROC_TYPE_GPU:   core of GPU card (Nvidia GPU)
    - KAAPI_PROC_TYPE_MPSOC: core of a MPSoC chip
*/
#define KAAPI_MAX_ARCHITECTURE 3

#define KAAPI_PROC_TYPE_CPU     0x0
#define KAAPI_PROC_TYPE_GPU     0x1
#define KAAPI_PROC_TYPE_MPSOC   0x2
#define KAAPI_PROC_TYPE_DEFAULT KAAPI_PROC_TYPE_CPU


/* ========================================================================== */
/** \ingroup WS
    Get the workstealing concurrency number, i.e. the number of kernel
    activities to execute the user level thread. 
    This function is machine dependent.
    \retval the number of active threads to steal tasks
 */
extern int kaapi_getconcurrency (void);

/** \ingroup WS
    Set the workstealing concurrency number, i.e. the number of kernel
    activities to execute the user level thread.
    If successful, the kaapi_setconcurrency() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \retval ENOSYS if the function is not available on a given architecture (e.g. MPSoC)
    \retval EINVAL if no memory ressource is available
    \retval ENOMEM if no memory ressource is available
    \retval EAGAIN if the system laked the necessary ressources to create another thread
    on return, the concurrency number may has been set to a different number than requested.
 */
extern int kaapi_setconcurrency (unsigned int concurrency);


/* ========================================================================== */
/** kaapi_advance.
    The function kaapi_advance() makes progress of steal requests in order to 
    help the runtim to be more reactive.
*/
extern int kaapi_advance(void);


/* ========================================================================== */
/** kaapi_get_elapsedtime
    The function kaapi_get_elapsedtime() will return the elapsed time in second
    since an epoch.
*/
extern double kaapi_get_elapsedtime(void);

/** kaapi_get_elapsedns
    The function kaapi_get_elapsedtime() will return the elapsed time since an epoch
    in nano second unit.
*/
extern kaapi_uint64_t kaapi_get_elapsedns(void);


/* ========================================================================= */
/* Shared object and access mode                                             */
/* ========================================================================= */
/** Kaapi access mode mask
    \ingroup DFG
*/
#define KAAPI_ACCESS_MASK_RIGHT_MODE   0x1f   /* 5 bits, ie bit 0, 1, 2, 3, 4, including P mode */
#define KAAPI_ACCESS_MASK_MODE         0xf    /* without P mode */
#define KAAPI_ACCESS_MASK_MODE_P       0x10   /* only P mode */

#define KAAPI_ACCESS_MASK_MEMORY       0x20   /* memory location for the data:  */
#define KAAPI_ACCESS_MEMORY_STACK      0x00   /* data is in the Kaapi stack */
#define KAAPI_ACCESS_MEMORY_HEAP       0x20   /* data is in the heap */

/*@{*/
typedef enum kaapi_access_mode_t {
  KAAPI_ACCESS_MODE_VOID= 0,        /* 0000 0000 : */
  KAAPI_ACCESS_MODE_V   = 1,        /* 0000 0001 : */
  KAAPI_ACCESS_MODE_R   = 2,        /* 0000 0010 : */
  KAAPI_ACCESS_MODE_W   = 4,        /* 0000 0100 : */
  KAAPI_ACCESS_MODE_CW  = 8,        /* 0000 1100 : */
  KAAPI_ACCESS_MODE_P   = 8,        /* 0001 0000 : */
  KAAPI_ACCESS_MODE_RW  = KAAPI_ACCESS_MODE_R|KAAPI_ACCESS_MODE_W
} kaapi_access_mode_t;
/*@}*/


/** Kaapi macro on access mode
    \ingroup DFG
*/
/*@{*/
#define KAAPI_ACCESS_GET_MODE( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE )

#define KAAPI_ACCESS_IS_READ( m ) \
  ((m) & KAAPI_ACCESS_MODE_R)

#define KAAPI_ACCESS_IS_WRITE( m ) \
  ((m) & KAAPI_ACCESS_MODE_W)

#define KAAPI_ACCESS_IS_POSTPONED( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_P)

/* W and CW */
#define KAAPI_ACCESS_IS_ONLYWRITE( m ) \
  (KAAPI_ACCESS_IS_WRITE(m) && !KAAPI_ACCESS_IS_READ(m))

#define KAAPI_ACCESS_IS_READWRITE( m ) \
  ( ((m) & KAAPI_ACCESS_MASK_RIGHT_MODE) == (KAAPI_ACCESS_MODE_W|KAAPI_ACCESS_MODE_R))

/** Return true if two modes are concurrents
    a == b and a or b is R or CW
    or a or b is postponed.
*/
#define KAAPI_ACCESS_IS_CONCURRENT(a,b) ((((a)==(b)) && (((b) & 2) !=0)) || ((a|b) & KAAPI_ACCESS_MODE_P))
/*@}*/



/* ========================================================================= */
/* Format of a task                                                          */
/* ========================================================================= */
/** \ingroup DFG
     Format identifier of data structure or task
*/
typedef kaapi_uint32_t kaapi_format_id_t;

/** \ingroup DFG
     Offset to access to parameter of a task
*/
typedef kaapi_uint32_t kaapi_offset_t;


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

/** predefined format 
*/
/*@{*/
extern kaapi_format_t kaapi_char_format;
extern kaapi_format_t kaapi_short_format;
extern kaapi_format_t kaapi_int_format;
extern kaapi_format_t kaapi_long_format;
extern kaapi_format_t kaapi_longlong_format;
extern kaapi_format_t kaapi_uchar_format;
extern kaapi_format_t kaapi_ushort_format;
extern kaapi_format_t kaapi_uint_format;
extern kaapi_format_t kaapi_ulong_format;
extern kaapi_format_t kaapi_ulonglong_format;
extern kaapi_format_t kaapi_float_format;
extern kaapi_format_t kaapi_double_format;
/*@}*/


/* ========================================================================= */
/* Task and stack interface                                                  */
/* ========================================================================= */
struct kaapi_processor_t;

/* Stack identifier */
typedef kaapi_uint32_t kaapi_stack_id_t;

/** \ingroup WS
    Reply structure to return value after a steal request
*/
typedef struct kaapi_reply_t {
  volatile kaapi_uint16_t        status;          /* reply status */
  struct kaapi_thread_context_t* data;            /* output data */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_reply_t;


/** \ingroup WS
    Server side of a request send by a processor.
    This data structure is pass in parameter of the splitter function.
*/
typedef struct kaapi_request_t {
  kaapi_uint16_t                 status;         /* server status */
  kaapi_uint16_t                 flag;           /* partial steal of task | processed during the execution of the runing task */
  struct kaapi_reply_t*          reply;          /* caller status */
  struct kaapi_thread_t*         thread;         /* internal thread pointer where to store result of the steal operation */
  struct kaapi_thread_context_t* mthread;        /* internal thread pointer where to store result of the steal operation */
  struct kaapi_processor_t*      proc;           /* owner of the request */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_request_t;


/** Kaapi Thread context
    This is the public view of the stack of frame contains in kaapi_thread_context_t
    We only expose the field to push task or data.
*/
typedef struct kaapi_thread_t {
    struct kaapi_task_t* pc;
    struct kaapi_task_t* sp;
    char*                sp_data;
} kaapi_thread_t;


/** Kaapi frame definition
   \ingroup TASK
   Same structure as kaapi_thread_t but we keep different type names to avoid automatic conversion.
*/
typedef struct kaapi_frame_t {
    struct kaapi_task_t* pc;
    struct kaapi_task_t* sp;
    char*                sp_data;
} kaapi_frame_t;



#if !defined(KAAPI_COMPILE_SOURCE)
typedef struct kaapi_thread_context_t {
  kaapi_thread_t*                 truc;           /** pointer to the current frame (in stackframe) */
} kaapi_thread_context_t;
#endif


/* ========================================================================= */
/** Kaapi task definition
    \ingroup TASK
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  kaapi_task_bodyid_t   body;      /** task body  */
  kaapi_task_bodyid_t   volatile ebody;     /** extra task body  */
  void*                 sp;        /** data stack pointer of the data frame for the task  */
  void*                 pad;       /** padding  */
} kaapi_task_t ;


/* ========================================================================= */
/** Task splitter
    \ingroup TASK
    A splitter should always return the number of work returns to the list of requests.
*/
typedef int (*kaapi_task_splitter_t)(
  struct kaapi_stealcontext_t* /*stc */, 
  int /*count*/, 
  struct kaapi_request_t* /*array*/, void* /*userarg*/);

/** Task reducer
    \ingroup TASK
*/
typedef int (*kaapi_task_reducer_t) (
#ifdef __cplusplus
 kaapi_stealcontext_t* /* stc */,
 void* arg_thief, ...
#endif
);

/** Steal context for adaptive algorithm
    \ingroup ADAPT
*/
typedef struct kaapi_stealcontext_t {
  kaapi_task_t*                  ownertask;
  struct kaapi_thread_context_t* ctxtthread;
  kaapi_thread_t*                thread;
  kaapi_task_splitter_t          splitter;
  void*                          argsplitter;

  volatile int                   hasrequest;
  kaapi_request_t*               requests;

  volatile int                   haspreempt;
  void* volatile                 arg_from_victim;    /* arg from the victim after preemption of one victim */
  void* volatile                 current_thief_work; /* work of the thief after preemption */
} kaapi_stealcontext_t;



/* ========================================================================= */
/** \ingroup DFG
    Kaapi access, public
*/
typedef struct kaapi_access_t {
  void*                  data;    /* global data */
  void*                  version; /* used to set the data to access (R/W/RW/CW) if steal, used to store output after steal */
//  struct kaapi_access_t* next;    /* not necessary used */
} kaapi_access_t;

#define kaapi_data(type, a)\
  ((type*)(a)->data)

#define KAAPI_DATA(type, a)\
  ((type*)a.data)


/* ========================================================================= */
/* Interface                                                                 */
/* ========================================================================= */

/** \ingroup TASK
    The function kaapi_access_init() initialize an access from a user defined pointer
    \param access INOUT a pointer to the kaapi_access_t data structure to initialize
    \param value INOUT a pointer to the user data
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_access_init(kaapi_access_t* access, void* value )
{
  access->data = value;
#if !defined(KAAPI_NDEBUG)
  access->version = 0;
#endif  
  return;
}


/** \ingroup TASK
    Return a pointer to parameter of the task (void*) pointer
*/
static inline void* kaapi_task_getargs(kaapi_task_t* task) 
{
  return task->sp;
}

/** \ingroup TASK
    Return a reference to parameter of the task (type*) pointer
*/
#define kaapi_task_getargst(task,type) ((type*)kaapi_task_getargs(task))

/** \ingroup TASK
    Set the pointer to parameter of the task (void*) pointer
*/
static inline void* kaapi_task_setargs(kaapi_task_t* task, void* arg) 
{
  return task->sp = arg;
}

/** \ingroup TASK
    Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->body = body;
}

/** \ingroup TASK
    Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(kaapi_task_t* task)
{
  return task->body;
}

/** The main task arguments
    \ingroup TASK
*/
typedef struct kaapi_taskmain_arg_t {
  int    argc;
  char** argv;
  void (*mainentry)(int, char**);
} kaapi_taskmain_arg_t;

/** The main task
    \ingroup TASK
*/
extern void kaapi_taskmain_body( void*, kaapi_thread_t* );


/** \ingroup TASK
    Return pointer to the self stack
*/
extern kaapi_thread_t* kaapi_self_thread (void);


/** \ingroup TASK
    The function kaapi_thread_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_thread_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_thread_t data structure where to push data
    \retval a pointer to the next task to push or 0.
*/
static inline void* kaapi_thread_pushdata( kaapi_thread_t* thread, kaapi_uint32_t count)
{
  void* retval;
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug( (char*)thread->sp_data+count <= (char*)thread->sp );
  retval = thread->sp_data;
  thread->sp_data += count;
  return retval;
}


/** \ingroup TASK
    The function kaapi_thread_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_thread_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param frame INOUT a pointer to the kaapi_frame_t data structure where to push data
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_thread_allocateshareddata(kaapi_access_t* access, kaapi_thread_t* thread, kaapi_uint32_t count)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug( (char*)thread->sp_data+count <= (char*)thread->sp );
  access->data = thread->sp_data;
  access->version = 0;
  thread->sp_data += count;
  return;
}

/** \ingroup TASK
    The function kaapi_stack_top() will return the top task.
    The top task is not part of the stack, it will be the next pushed task.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_thread_toptask( kaapi_thread_t* thread) 
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp >= (char*)thread->sp_data);
  return thread->sp;
}


/** \ingroup TASK
    The function kaapi_stack_push() pushes the top task into the stack.
    If successful, the kaapi_stack_push() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_thread_pushtask(kaapi_thread_t* thread)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp >= (char*)thread->sp_data);

  kaapi_writemem_barrier_api();

  --thread->sp;
  return 0;
}


#define kaapi_task_initdfg( task, taskbody, arg ) \
  do { \
    (task)->sp       = (arg);\
    (task)->body     = (task)->ebody = taskbody;\
  } while (0)


/** \ingroup TASK
    Initialize a task with given flag for adaptive attribut
*/
static inline int kaapi_task_init( kaapi_task_t* task, kaapi_task_bodyid_t taskbody, void* arg ) 
{
  kaapi_task_initdfg( task, taskbody, arg );
  return 0;
}


/** \ingroup TASK
    The function kaapi_thread_save_frame() saves the current frame of a stack into
    the frame data structure.
    If successful, the kaapi_thread_save_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack IN a pointer to the kaapi_stack_t data structure.
    \param frame OUT a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
extern int kaapi_thread_save_frame( kaapi_thread_t*, kaapi_frame_t*);


/** \ingroup TASK
    The function kaapi_thread_restore_frame() restores the frame context of a stack into
    the stack data structure.
    If successful, the kaapi_thread_restore_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \param frame IN a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
extern int kaapi_thread_restore_frame( kaapi_thread_t*, const kaapi_frame_t*);


/** \ingroup TASK
    The function kaapi_sched_sync() execute all childs tasks of the current running task.
    If successful, the kaapi_sched_sync() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
extern int kaapi_sched_sync( void );


/* ========================================================================= */
/* API for adaptive algorithm                                                */
/* ========================================================================= */

/** \ingroup WS
    Return a new stealcontext to be used with other function of the adaptive API.
*/
kaapi_stealcontext_t* kaapi_thread_pushstealcontext( kaapi_thread_t* thread );


/** \ingroup WS
    Helper to expose to many part of the internal API.
    Return 1 iff its remains works...
*/
extern int kaapi_sched_stealstack_helper( kaapi_stealcontext_t* stc );


/** \ingroup ADAPTIVE
    Test if the current execution should process steal request into the task.
    This function also poll for other requests on the thread of control,
    it may invoke processing of streal request of previous pushed tasks.
    \retval !=0 if they are a steal request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_stealpoint_isactive( kaapi_stealcontext_t* stc )
{
  if (stc->hasrequest) 
  {
    /* \TODO: ici appel systematique a kaapi_sched_stealprocessor dans le cas ou la seule tache
       est la tache 'task' afin de retourner vite pour le traitement au niveau applicatif.
       
       Dans le cas concurrent, on ne passe jamais par la (appel direct de kaapi_stealprocessor).
       Dans le cas cooperatif, le thread courant se vol lui meme puis repond
    */
    return kaapi_sched_stealstack_helper( stc );
  }
  return 0;
}


/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then call the splitter function with given arguments.
    \retval 0 \TODO code de retour
*/
#define kaapi_stealpoint( stc, splitter, ...) \
   (kaapi_stealpoint_isactive(stc) ? (splitter)( stc, (stc)->hasrequest, (stc)->requests, ##__VA_ARGS__), 1 : 0 )
    
    
/** \ingroup ADAPTIVE
    Return true iff the request correctly posted
    \param pksr kaapi_request_t
*/
#define kaapi_request_ok( kpsr )\
  ((kpsr)->status == 1 /*== KAAPI_REQUEST_S_POSTED*/)

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then pass arg_victim argument to the victim and return !=0 value
    \retval !=0 if it exists a prending preempt request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_preemptpoint_isactive( kaapi_stealcontext_t* stc )
{
  return stc->haspreempt;
}

/** \ingroup ADAPTIVE
    Helper function to pass argument between the victim and the thief.
    On return the victim argument may be read.
*/
extern int kaapi_preemptpoint_before_reducer_call( kaapi_stealcontext_t* stc, void* arg_for_victim, int size );
extern int kaapi_preemptpoint_after_reducer_call ( kaapi_stealcontext_t* stc, int reducer_retval );


static inline int kaapi_is_null(void* p)
{
  /* checking for null on a macro param
     where param is an address makes g++
     complain about the never nullness
     of the arg. so we use this function
     to check for null pointers.
   */
  return p == NULL;
}

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request to the current task
    and if it is true then pass arg_victim argument to the victim and call the reducer function with incomming victim argument
    for the thief. Extra arguments are added at the end of the parameter when calling reducer function.
    The reducer function is assumed to be of the form:
      (*reducer)(kaapi_stack_t*, kaapi_task_t*, void* arg_from_victim, ...)
    Where ... must match the list of extra parameter.
    \retval !=0 if a prending preempt request(s) has been processed onto the given task.
    \retval 0 else
*/
#define kaapi_preemptpoint( stc, reducer, arg_for_victim, size_arg_victim, ...)\
  ( kaapi_preemptpoint_isactive(stc) ? \
        kaapi_preemptpoint_before_reducer_call(stc, arg_for_victim, size_arg_victim),\
        kaapi_preemptpoint_after_reducer_call( stc, \
        ( kaapi_is_null((void*)reducer) ? 0: ((int(*)())(reducer))( stc, stc->arg_from_victim, ##__VA_ARGS__))) \
    : \
        0\
  )


/* Helper for kaapi_preempt_nextthief
   Return 1 iff a thief as been preempted.
*/
extern int kaapi_preempt_nextthief_helper_head( kaapi_stealcontext_t* stc, void* arg_to_thief );

/* Helper for kaapi_preempt_nextthief_reverse
   Return 1 iff a thief as been preempted.
*/
extern int kaapi_preempt_nextthief_helper_tail( kaapi_stealcontext_t* stc, void* arg_to_thief );


/** \ingroup ADAPTIVE
   Try to preempt next thief in the reverse order defined by the order of the steal request reply, i.e.
   preempt first the last thief.
   Return true iff some work have been preempted and should be processed locally.
   If no more thief can been preempted, then the return value of the function kaapi_preempt_nextthief() is 0.
   If it exists a thief, then the call to kaapi_preempt_nextthief() will return the
   value the call to reducer function.
   
   reducer function should has the following signature:
      int (*)( stc, void* thief_work, ... )
   where ... is the same arguments as passed to kaapi_preempt_nextthief.
*/
#define kaapi_preempt_thief_head( stc, arg_to_thief, reducer, ... ) \
(									\
 kaapi_preempt_nextthief_helper_head(stc, arg_to_thief) ?		\
 (									\
  kaapi_is_null((void*)reducer) ? 0 :					\
  ((kaapi_task_reducer_t)reducer)(stc, stc->current_thief_work, ##__VA_ARGS__) \
 ) : 0									\
)


/** \ingroup ADAPTIVE
   Try to preempt next thief in the order defined by the order of the steal request reply.
   Note that this function is called kaapi_preempt_nextthief_reverse for historical reason
   because the order defined by kaapi_preempt_nextthief was the natural order.
   Return true iff some work have been preempted and should be processed locally.
   If no more thief can been preempted, then the return value of the function kaapi_preempt_nextthief() is 0.
   If it exists a thief, then the call to kaapi_preempt_nextthief() will return the
   value the call to reducer function.
   
   reducer function should has the following signature:
      int (*)( kaapi_stack_t* stack, kaapi_task_t* task, void* thief_work, ... )
   where ... is the same arguments as passed to kaapi_preempt_nextthief.
*/
#define kaapi_preempt_nextthief_tail( stc, arg_to_thief, reducer, ... ) \
(									\
 kaapi_preempt_nextthief_helper_tail(stc, arg_to_thief) ?		\
 (									\
  kaapi_is_null((void*)reducer) ? 0 :					\
  ((kaapi_task_reducer_t)reducer)(stc, stc->current_thief_work, ##__VA_ARGS__) \
 ) : 0									\
)


/** \ingroup ADAPTIVE
    Reply a value to a steal request. If retval is !=0 it means that the request
    has successfully adapt to steal work. Else 0.
    This function could not be called in concurrence with other calls:
    - if the adaptive algorithm matches correctly the standard structure (beginsteal/endsteal) etc..
    the scheduler ensure that only one thief is able to call the method.
    \param stack INOUT the stack of the victim that has been used to replies to the request
    \param task IN the stolen task
    \param request INOUT data structure used to replied by the thief
    \param thief_stack INOUT the output stack that will be used to the thief
    \param size IN the size in bytes to store the result
    \param retval IN the result of the steal request 0 iff failed else success
*/
extern int kaapi_request_reply(
    kaapi_stealcontext_t*          stc,
    kaapi_request_t*               request, 
    int size, int retval,
    int inser_head
);

static inline int kaapi_request_reply_head(
    kaapi_stealcontext_t*          stc,
    kaapi_request_t*               request, 
    int size, int retval 
)
{ return kaapi_request_reply(stc, request, size, retval, 1 ); }

/** \ingroup ADAPTIVE
*/
static inline int kaapi_request_reply_tail(
    kaapi_stealcontext_t*          stc,
    kaapi_request_t*               request, 
    int size, int retval 
)
{ return kaapi_request_reply(stc, request, size, retval, 0 ); }


/** \ingroup ADAPTIVE
*/
static inline int kaapi_request_reply_failed(     
    kaapi_stealcontext_t*          stc,
    kaapi_request_t*               request
)
{ return kaapi_request_reply( stc, request, 0, 0, 1 ); }


/** \ingroup ADAPTIVE
    Set an splitter to be called in concurrence with the execution of the next instruction
    if a steal request is sent to the task.
    \retval EINVAL in case of error (task not adaptive kind)
    \retval 0 else
*/
static inline int kaapi_stealbegin(
    kaapi_stealcontext_t* stc,
    kaapi_task_splitter_t splitter, void* arg_tasksplitter)
{
  stc->argsplitter = arg_tasksplitter;
  kaapi_writemem_barrier_api();
  stc->splitter    = splitter;
  return 0;
}

/** \ingroup ADAPTIVE
    Erase the previously splitter action and avoid concurrent steal on return.
    \retval 0 in case of success
    \retval !=0 in case of error code
*/
extern int kaapi_stealend(kaapi_stealcontext_t* stc);

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_taskfinalize_body( void*, kaapi_thread_t* );

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
    This method should be called in a frame with scope included by the frame where was done the call 
    to kaapi_thread_pushstealcontext that creates the context.
*/
static inline int kaapi_finalize_steal( kaapi_stealcontext_t* stc )
{
  kaapi_task_t* task = kaapi_thread_toptask(stc->thread);
  kaapi_task_init( task, kaapi_taskfinalize_body, stc );
  kaapi_thread_pushtask(stc->thread);
  return 0;
}

/** Body of the task in charge of finalize of thief of adaptive task
    \ingroup TASK
*/
extern void kaapi_taskreturn_body( void*, kaapi_thread_t* );

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
*/
static inline int kaapi_return_steal( kaapi_stealcontext_t* stc, void* retval, int size )
{
/*
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp;
  kaapi_assert( (size ==0) || size <= ta->result_size );


  ta->local_result_data = retval;
  ta->local_result_size = size;  
*/
  kaapi_task_t* task = kaapi_thread_toptask(stc->thread);
  kaapi_task_init( task, kaapi_taskreturn_body, stc );
  kaapi_thread_pushtask(stc->thread);
  return 0;
}

/** \ingroup PERF
    performace counters
*/
#if defined(KAAPI_USE_PAPIPERFCOUNTER)
# include <papi.h>
#endif

#define KAAPI_PERF_ID_USER_POS (31)
#define KAAPI_PERF_ID_USER_MASK (1 << KAAPI_PERF_ID_USER_POS)

#define KAAPI_PERF_ID(U, I) (KAAPI_PERF_ID_ ## I | (U) << KAAPI_PERF_ID_USER_POS)
#define KAAPI_PERF_ID_USER(I) KAAPI_PERF_ID(1, I)
#define KAAPI_PERF_ID_PRIV(I) KAAPI_PERF_ID(0, I)

#define KAAPI_PERF_ID_TASKS         0  /* count number of executed tasks */
#define KAAPI_PERF_ID_STEALREQOK    1  /* count number of successful steal requests */
#define KAAPI_PERF_ID_STEALREQ      2  /* count number of steal requests */
#define KAAPI_PERF_ID_STEALOP       3  /* count number of steal operation to reply to requests */
#define KAAPI_PERF_ID_SUSPEND       4  /* count number of suspend */
#define KAAPI_PERF_ID_TIDLE         5  /* nano second of idle time */
#define KAAPI_PERF_ID_TPREEMPT      6  /* nano second of preempt time */

#define KAAPI_PERF_ID_ENDSOFTWARE   7  /* mark end of software counters */

#define KAAPI_PERF_ID_PAPI_BASE    (KAAPI_PERF_ID_ENDSOFTWARE)
#define KAAPI_PERF_ID_PAPI_0       (KAAPI_PERF_ID_PAPI_BASE + 0)
#define KAAPI_PERF_ID_PAPI_1       (KAAPI_PERF_ID_PAPI_BASE + 1)
#define KAAPI_PERF_ID_PAPI_2       (KAAPI_PERF_ID_PAPI_BASE + 2)

#define KAAPI_PERF_ID_PAPI_MAX     (KAAPI_PERF_ID_PAPI_2 - KAAPI_PERF_ID_PAPI_BASE + 1)
#define KAAPI_PERF_ID_MAX          (KAAPI_PERF_ID_PAPI_2 + 1)
#define KAAPI_PERF_ID_ALL           KAAPI_PERF_ID_MAX

#define KAAPI_PERF_USR_COUNTER      0x1  /* for kaapi_perf_accum_counters or kaapi_perf_read_counters*/
#define KAAPI_PERF_SYS_COUNTER      0x2  /* could be ored */

/* Counter type
*/
typedef long long kaapi_perf_counter_t;
/* Value
*/
typedef unsigned int kaapi_perf_id_t;
typedef struct kaapi_perf_idset_t
{
  unsigned int count;
  unsigned char idmap[KAAPI_PERF_ID_MAX];
} kaapi_perf_idset_t;

#define KAAPI_PERF_IDSET_SINGLETON(I) {1, {I}}

#define KAAPI_PERF_IDSET_TASKS ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_TASKS)
#define KAAPI_PERF_IDSET_STEALREQOK ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_STEALREQOK)
#define KAAPI_PERF_IDSET_STEALREQ ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_STEALREQ)
#define KAAPI_PERF_IDSET_STEALOP ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_STEALOP)
#define KAAPI_PERF_IDSET_SUSPEND ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_SUSPEND)
#define KAAPI_PERF_IDSET_TIDLE ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_TIDLE)
#define KAAPI_PERF_IDSET_TPREEMPT ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_TPREEMPT)
#define KAAPI_PERF_IDSET_PAPI_0 ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_PAPI_0)
#define KAAPI_PERF_IDSET_PAPI_1 ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_PAPI_1)
#define KAAPI_PERF_IDSET_PAPI_2 ((kaapi_perf_idset_t*)(uintptr_t)KAAPI_PERF_ID_PAPI_2)

/* idset */
extern void kaapi_perf_idset_zero(kaapi_perf_idset_t*);
extern void kaapi_perf_idset_add(kaapi_perf_idset_t*, kaapi_perf_id_t);

/* global access */
extern void _kaapi_perf_accum_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter);
extern void _kaapi_perf_read_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter);

/* global access, user functions */
static inline void kaapi_perf_accum_counters(const kaapi_perf_idset_t* idset, kaapi_perf_counter_t* counter)
{ _kaapi_perf_accum_counters(idset, KAAPI_PERF_USR_COUNTER, counter); }

static inline void kaapi_perf_read_counters(const kaapi_perf_idset_t* idset, kaapi_perf_counter_t* counter)
{ _kaapi_perf_read_counters(idset, KAAPI_PERF_USR_COUNTER, counter); }

/* local access, internal functions */
extern void _kaapi_perf_read_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter);
extern void _kaapi_perf_accum_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* accum);

/* local access, user functions */
static inline void kaapi_perf_read_register(const kaapi_perf_idset_t* idset, kaapi_perf_counter_t* counter)
{ _kaapi_perf_read_register(idset, KAAPI_PERF_USR_COUNTER, counter); }

static inline void kaapi_perf_accum_register(const kaapi_perf_idset_t* idset, kaapi_perf_counter_t* accum)
{ _kaapi_perf_accum_register(idset, KAAPI_PERF_USR_COUNTER, accum); }

/* utility */
extern const char* kaapi_perf_id_to_name(kaapi_perf_id_t);
extern size_t kaapi_perf_counter_num(void);


/* ========================================================================= */
/* Format declaration & registration                                         */
/* ========================================================================= */
/** \ingroup TASK
    Register a format
*/
extern kaapi_format_id_t kaapi_format_register( 
        kaapi_format_t*             fmt,
        const char*                 name
);

/** \ingroup TASK
    Register a task format 
*/
extern kaapi_format_id_t kaapi_format_taskregister( 
        kaapi_format_t*           (*fmt_fnc)(void),
        kaapi_task_body_t           body,
        const char*                 name,
        size_t                      size,
        int                         count,
        const kaapi_access_mode_t   mode_param[],
        const kaapi_offset_t        offset_param[],
        const kaapi_format_t*       fmt_params[]
);

/** \ingroup TASK
    Register a task body into its format
*/
extern kaapi_format_id_t kaapi_format_taskregister_body( 
        kaapi_format_t*             fmt,
        kaapi_task_body_t           body,
        int                         archi
);

/** \ingroup TASK
    Register a data structure format
*/
extern kaapi_format_id_t kaapi_format_structregister( 
        kaapi_format_t*           (*fmt_fnc)(void),
        const char*                 name,
        size_t                      size,
        void                       (*cstor)( void* ),
        void                       (*dstor)( void* ),
        void                       (*cstorcopy)( void*, const void*),
        void                       (*copy)( void*, const void*),
        void                       (*assign)( void*, const void*),
        void                       (*print)( FILE* file, const void* src)
);

/** \ingroup TASK
    Resolve a format data structure from the body of a task
*/
extern kaapi_format_t* kaapi_format_resolvebybody(kaapi_task_bodyid_t key);

/** \ingroup TASK
    Resolve a format data structure from the format identifier
*/
extern kaapi_format_t* kaapi_format_resolvebyfmit(kaapi_format_id_t key);

#define KAAPI_REGISTER_TASKFORMAT( formatobject, name, fnc_body, ... ) \
  static inline kaapi_format_t* formatobject(void) \
  {\
    static kaapi_format_t formatobject##_object;\
    return &formatobject##_object;\
  }\
  static inline void __attribute__ ((constructor)) __kaapi_register_format_##formatobject (void)\
  { \
    static int isinit = 0;\
    if (isinit) return;\
    isinit = 1;\
    kaapi_format_taskregister( &formatobject, fnc_body, name, ##__VA_ARGS__);\
  }


#define KAAPI_REGISTER_STRUCTFORMAT( formatobject, name, size, cstor, dstor, cstorcopy, copy, assign ) \
  static inline kaapi_format_t* fnc_formatobject(void) \
  {\
    static kaapi_format_t formatobject##_object;\
    return &formatobject##_object;\
  }\
  static inline void __attribute__ ((constructor)) __kaapi_register_format_##formatobject (void)\
  { \
    static int isinit = 0;\
    if (isinit) return;\
    isinit = 1;\
    kaapi_format_structregister( &formatobject, name, size, cstor, dstor, cstorcopy, copy, assign );\
  }


/**
 */
extern void _kaapi_dummy(void*);

/* ========================================================================= */
/* Initialization / destruction functions
 */
extern void __attribute__ ((constructor)) kaapi_init(void);

extern void __attribute__ ((destructor)) kaapi_fini(void);

#if !defined(KAAPI_COMPILE_SOURCE)

/** To force reference to kaapi_init.c in order to link against kaapi_init and kaapi_fini
 */
static void __attribute__((unused)) __kaapi_dumy_dummy(void)
{
  _kaapi_dummy(NULL);
}
#endif


#ifdef __cplusplus
}
#endif

#endif /* _KAAPI_H */
