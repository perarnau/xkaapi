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


/* ========================================================================== */
struct kaapi_task_t;
struct kaapi_stack_t;
struct kaapi_thread_t;
struct kaapi_thread_context_t;

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


/**
 */
extern void kaapi_set_workload( kaapi_uint32_t workload );


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
extern kaapi_format_t kaapi_shared_format;
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
  struct kaapi_thread_context_t* thread;         /* thread of the thief where to store result of the steal operation */
  struct kaapi_processor_t*      proc;           /* owner of the request */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_request_t;


/** Task splitter
    \ingroup TASK
    A splitter should always return the number of work returns to the list of requests.
*/
typedef int (*kaapi_task_splitter_t)(
  struct kaapi_stack_t* /*stack */, 
  struct kaapi_task_t* /* task */, 
  int /*count*/, 
  struct kaapi_request_t* /*array*/, void* /*userarg*/);


/** Task reducer
    \ingroup TASK
*/
typedef int (*kaapi_task_reducer_t) (
#ifdef __cplusplus
 struct kaapi_stack_t* /*stack */, struct kaapi_task_t* /* task */, 
 void* arg_thief, ...
#endif
);


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
   \ingroup STACK
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
  kaapi_task_bodyid_t   ebody;     /** extra task body  */
  void*                 sp;        /** data stack pointer of the data frame for the task  */
  void*                 pad;       /** padding  */
} kaapi_task_t ;


struct kaapi_taskadaptive_result_t;

/** \ingroup ADAPT
    Extent data structure for adaptive task.
    This data structure is attached to any adaptative tasks.
*/
typedef struct kaapi_taskadaptive_t {
  void*                               user_sp;         /* user argument */
  kaapi_task_splitter_t               splitter;        /* C function that represent the body to split a task, interest only if isadaptive*/
  void*                               argsplitter;     /* arg for splitter */
  kaapi_atomic_t                      thievescount;    /* required for the finalization of the victim */
  struct kaapi_taskadaptive_result_t* head;            /* head of the LIFO order of result */
  struct kaapi_taskadaptive_result_t* tail;            /* tail of the LIFO order of result */

  struct kaapi_taskadaptive_result_t* current_thief;   /* points to the current kaapi_taskadaptive_result_t to preemption */

  struct kaapi_taskadaptive_t*        mastertask;      /* who to signal at the end of computation, 0 iff master task */
  struct kaapi_taskadaptive_result_t* result;          /* points on kaapi_taskadaptive_result_t to copy args in preemption or finalization
                                                          null iff thief has been already preempted
                                                       */
  int                                 result_size;     /* for debug copy of result->size_data to avoid remote read in finalize */
  int                                 local_result_size; /* size of result to be copied in kaapi_taskfinalize */
  void*                               local_result_data; /* data of result to be copied int kaapi_taskfinalize */
  void*                               arg_from_victim; /* arg received by the victim in case of preemption */
} kaapi_taskadaptive_t;


/** \ingroup ADAPT
    Data structure that allows to store results of child tasks of an adaptive task.
    This data structure is stored... in the victim heap and serve as communication 
    media between victim and thief.
*/
typedef struct kaapi_taskadaptive_result_t {
  volatile int*                       signal;           /* signal of preemption pointer on the thief stack haspreempt */
  volatile int                        req_preempt;      /* */
  volatile int                        thief_term;       /* */
  int                                 flag;             /* state of the result */
  struct kaapi_taskadaptive_result_t* rhead;            /* next result of the next thief */
  struct kaapi_taskadaptive_result_t* rtail;            /* next result of the next thief */
  void**                              parg_from_victim; /* point to arg_from_victim in thief kaapi_taskadaptive_t */
  struct kaapi_taskadaptive_result_t* next;             /* link field to the previous spawned thief */
  struct kaapi_taskadaptive_result_t* prev;             /* link field to the next spawned thief */
  int                                 size_data;        /* size of data */
  double                              data[1];
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_taskadaptive_result_t;

#define KAAPI_RESULT_INSTACK   0x01
#define KAAPI_RESULT_INHEAP    0x02



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
    Return the state of a task
*/
#define kaapi_task_getstate(task) ((task)->flag & KAAPI_TASK_MASK_STATE)

/** \ingroup TASK
    Set the state of the task
*/
#define kaapi_task_setstate(task, s) ((task)->flag = ((task)->flag & ~KAAPI_TASK_MASK_STATE) | (s))

/** \ingroup TASK
    Return the flags of the task
*/
#define kaapi_task_getflags(task) ((task)->flag & KAAPI_TASK_MASK_FLAGS)

/** \ingroup TASK
    Set the flags of the task
*/
#define kaapi_task_setflags(task, f) ((task)->flag |= f & KAAPI_TASK_MASK_FLAGS)

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
    Set the pointer to parameter of the task (void*) pointer
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->body = body;
}

/** \ingroup TASK
    Set the pointer to parameter of the task (void*) pointer
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(kaapi_task_t* task)
{
  return task->body;
}


static inline void kaapi_task_setextrabody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->ebody = body;
}

/** \ingroup TASK
    Set the pointer to parameter of the task (void*) pointer
*/
static inline kaapi_task_bodyid_t kaapi_task_getextrabody(kaapi_task_t* task)
{
  return task->ebody;
}


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
extern void kaapi_taskfinalize_body( void*, kaapi_thread_t* );

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_adapt_body( void*, kaapi_thread_t* );

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
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    All previous internal task body are not stealable. All user task are stealable.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isstealable(const kaapi_task_t* task)
{ 
  return (task->body != kaapi_taskstartup_body) && (task->body != kaapi_nop_body)
      && (task->body != kaapi_suspend_body) && (task->body != kaapi_exec_body) && (task->body != kaapi_aftersteal_body) 
      && (task->body != kaapi_tasksteal_body) && (task->body != kaapi_taskwrite_body) && (task->body != kaapi_tasksig_body)
      && (task->body != kaapi_taskfinalize_body) && (task->body != kaapi_adapt_body)
      ;
}


/** \ingroup STACK
    Return pointer to the self stack
*/
extern kaapi_thread_t* kaapi_self_thread (void);


/** \ingroup STACK
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

/** \ingroup STACK
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


/** \ingroup STACK
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
#if defined(__i386__)||defined(__x86_64)
  /* WARNING here Compiler fence to keep operations. Note that on X86 no reorder of write ops
     so, we do not need extra hardware fence operation.
  */
  __asm__ __volatile__("" : : : "memory" );
#else
#error "Fence operation should must put here. See the memory consistency of the hardware you use"
#endif
  --thread->sp;
  return 0;
}


/** \ingroup TASK
    Initialize a task with given flag for adaptive attribut or task constraints.
*/
#if 0
static inline int kaapi_task_initadaptive( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_task_bodyid_t taskbody, void* arg, kaapi_uint32_t flag ) 
{
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*) kaapi_thread_pushdata( stack, sizeof(kaapi_taskadaptive_t) );
  kaapi_assert_debug( ta !=0 );
  ta->user_sp               = arg;
  ta->splitter              = 0;
  ta->argsplitter           = 0;
  ta->thievescount._counter = 0;
  ta->head                  = 0;
  ta->tail                  = 0;
  ta->result                = 0;
  ta->mastertask            = 0;
  ta->arg_from_victim       = 0;
  task->sp                  = ta;
  task->flag                = flag | KAAPI_TASK_ADAPTIVE;
  task->body                = taskbody;
  return 0;
}
#endif

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


/** \ingroup STACK
    The function kaapi_thread_save_frame() saves the current frame of a stack into
    the frame data structure.
    If successful, the kaapi_thread_save_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack IN a pointer to the kaapi_stack_t data structure.
    \param frame OUT a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
extern int kaapi_thread_save_frame( kaapi_thread_t*, kaapi_frame_t*);


/** \ingroup STACK
    The function kaapi_thread_restore_frame() restores the frame context of a stack into
    the stack data structure.
    If successful, the kaapi_thread_restore_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \param frame IN a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
extern int kaapi_thread_restore_frame( kaapi_thread_t*, const kaapi_frame_t*);


/** \ingroup STACK
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

/*
*/
extern int kaapi_sched_stealstack_helper( struct kaapi_thread_context_t* thread, kaapi_task_t* curr );

/** \ingroup ADAPTIVE
    Test if the current execution should process steal request into the task.
    This function also poll for other requests on the thread of control,
    it may invoke processing of streal request of previous pushed tasks.
    \retval !=0 if they are a steal request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_stealpoint_isactive( struct kaapi_thread_context_t* thread, kaapi_task_t* task )
{
#if 0
  int count = stack->hasrequest;
  if (count) 
  {
    /* \TODO: ici appel systematique a kaapi_sched_stealprocessor dans le cas ou la seule tache
       est la tache 'task' afin de retourner vite pour le traitement au niveau applicatif.
       
       Dans le cas concurrent, on ne passe jamais par la (appel direct de kaapi_stealprocessor).
       Dans le cas cooperatif, le thread courant se vol lui meme puis repond
    */
    count = stack->hasrequest -= kaapi_sched_stealstack_helper(stack, task );
    return (count ==0 ? 0 : 1);
  }
#endif
  return 0;
}


/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then call the splitter function with given arguments.
    \retval 0 \TODO code de retour
*/
#define kaapi_stealpoint( stack, task, splitter, ...) \
   (kaapi_stealpoint_isactive(stack, task) ? \
     (splitter)( stack, task, (stack)->hasrequest, (stack)->requests, ##__VA_ARGS__) :\
     0\
   ), (stack)->hasrequest =0 
    
    
/** \ingroup ADAPTIVE
    Return true iff the request correctly posted
    \param pksr kaapi_request_t
*/
#define kaapi_request_ok( kpsr )\
  ((kpsr)->status == 1 /*KAAPI_REQUEST_S_POSTED*/)

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then pass arg_victim argument to the victim and return !=0 value
    \retval !=0 if it exists a prending preempt request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_preemptpoint_isactive( kaapi_thread_t* thread, kaapi_task_t* task __attribute__((unused)) )
{
#if 0
  int retval = stack->haspreempt;
#if defined(TODO)
  kaapi_assert_debug( !(task->flag & KAAPI_TASK_ADAPTIVE) || !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) );
#endif
  return retval;
#endif
  return 0;
}

/** \ingroup ADAPTIVE
    Helper function to pass argument between the victim and the thief.
    On return the victim argument may be read.
*/
extern int kaapi_preemptpoint_before_reducer_call( kaapi_thread_t* thread, kaapi_task_t* task, void* arg_for_victim, int size );
extern int kaapi_preemptpoint_after_reducer_call ( kaapi_thread_t* thread, kaapi_task_t* task, int reducer_retval );


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
#define kaapi_preemptpoint( stack, task, reducer, arg_for_victim, size_arg_victim, ...)\
  ( kaapi_preemptpoint_isactive(stack, task) ? \
        kaapi_preemptpoint_before_reducer_call(stack, task, arg_for_victim, size_arg_victim),\
        kaapi_preemptpoint_after_reducer_call( stack, task, \
        ( kaapi_is_null((void*)reducer) ? 0: ((int(*)())(reducer))( stack, task, ((kaapi_taskadaptive_t*)(task)->sp)->arg_from_victim, ##__VA_ARGS__))) \
    : \
        0\
  )


/* Helper for kaapi_preempt_nextthief
   Return 1 iff a thief as been preempted.
*/
extern int kaapi_preempt_nextthief_helper( kaapi_thread_t* thread, kaapi_task_t* task, void* arg_to_thief );

/** \ingroup ADAPTIVE
   Try to preempt next thief in the reverse order defined by the order of the steal request reply, i.e.
   preempt first the last thief.
   Return true iff some work have been preempted and should be processed locally.
   If no more thief can been preempted, then the return value of the function kaapi_preempt_nextthief() is 0.
   If it exists a thief, then the call to kaapi_preempt_nextthief() will return the
   value the call to reducer function.
   
   reducer function should has the following signature:
      int (*)( kaapi_stack_t* stack, kaapi_task_t* task, void* thief_work, ... )
   where ... is the same arguments as passed to kaapi_preempt_nextthief.
*/
#define kaapi_preempt_nextthief( stack, task, arg_to_thief, reducer, ... ) \
(									\
 kaapi_preempt_nextthief_helper(stack, task, arg_to_thief) ?		\
 (									\
  kaapi_is_null((void*)reducer) ? 0 :					\
  ((kaapi_task_reducer_t)reducer)(stack, task, ((kaapi_taskadaptive_t*)task->sp)->current_thief->data, ##__VA_ARGS__) \
 ) : 0									\
)


/* Helper for kaapi_preempt_nextthief_reverse
   Return 1 iff a thief as been preempted.
*/
extern int kaapi_preempt_nextthief_reverse_helper( kaapi_thread_t* thread, kaapi_task_t* task, void* arg_to_thief );

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
#define kaapi_preempt_nextthief_reverse( stack, task, arg_to_thief, reducer, ... ) \
(									\
 kaapi_preempt_nextthief_reverse_helper(stack, task, arg_to_thief) ?		\
 (									\
  kaapi_is_null((void*)reducer) ? 0 :					\
  ((kaapi_task_reducer_t)reducer)(stack, task, ((kaapi_taskadaptive_t*)task->sp)->current_thief->data, ##__VA_ARGS__) \
 ) : 0									\
)


/** \ingroup ADAPTIVE
    Reply a value to a steal request. If retval is !=0 it means that the request
    has successfully adapt to steal work. Else 0.
    While it reply to a request, the function decrement the request count on the stack.
    This function is machine dependent.
    \param stack INOUT the stack of the victim that has been used to replies to the request
    \param task IN the stolen task
    \param request INOUT data structure used to replied by the thief
    \param thief_stack INOUT the output stack that will be used to the thief
    \param size IN the size in bytes to store the result
    \param retval IN the result of the steal request 0 iff failed else success
*/
extern int kaapi_request_reply( 
    kaapi_request_t*               request, 
    kaapi_task_t*                  task, 
    struct kaapi_thread_context_t* thief_thread, 
    int size, int retval 
);

extern int kaapi_request_reply_head(
    kaapi_request_t*               request, 
    kaapi_task_t*                  task, 
    struct kaapi_thread_context_t* thief_thread, 
    int size, int retval 
);

extern int kaapi_request_reply_tail(
    kaapi_request_t*               request, 
    kaapi_task_t*                  task, 
    struct kaapi_thread_context_t* thief_thread, 
    int size, int retval 
);

/** \ingroup ADAPTIVE
    Specialization to reply failed to a processor
    \param stack INOUT the stack of the victim that has been used to replies to the request
    \param task IN the stolen task
    \param request INOUT data structure used to replied by the thief
*/
static inline int kaapi_request_reply_failed(     
    kaapi_request_t* request
)
{ return kaapi_request_reply( request, 0, 0, 0, 0 ); }


/** \ingroup ADAPTIVE
    Set an splitter to be called in concurrence with the execution of the next instruction
    if a steal request is sent to the task.
    \retval EINVAL in case of error (task not adaptive kind)
    \retval 0 else
*/
static inline int kaapi_stealbegin(
    kaapi_thread_t* thread __attribute__((unused)), 
    kaapi_task_t* task, kaapi_task_splitter_t splitter, void* arg_tasksplitter)
{
/*  if (!kaapi_task_isadaptive(task)) return EINVAL; */
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp;
  ta->splitter    = splitter;
  ta->argsplitter = arg_tasksplitter;
  return 0;
}

/** \ingroup ADAPTIVE
    Erase the previously splitter action and avoid concurrent steal on return.
    \retval 0 in case of success
    \retval !=0 in case of error code
*/
extern int kaapi_stealend(kaapi_thread_t* thread, kaapi_task_t* task);

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
*/
static inline int kaapi_finalize_steal( kaapi_thread_t* thread, kaapi_task_t* task )
{
#if 0
/*  if (kaapi_task_isadaptive(task) && !(task->flag & KAAPI_TASK_ADAPT_NOSYNC))*/
  {
    kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
    kaapi_task_t* task = kaapi_stack_toptask(stack);
    kaapi_task_init( task, kaapi_taskfinalize_body, ta ); /*, KAAPI_TASK_DFG|KAAPI_TASK_STICKY ); */
    kaapi_stack_pushtask(stack);
  }
#endif
  return 0;
}

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
*/
static inline int kaapi_return_steal( kaapi_thread_t* thread, kaapi_task_t* task, void* retval, int size )
{
#if 0
/*  if (kaapi_task_isadaptive(task) && !(task->flag & KAAPI_TASK_ADAPT_NOSYNC))*/
  {
    kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
    kaapi_assert( (size ==0) || size <= ta->result_size );
    ta->local_result_data = retval;
    ta->local_result_size = size;
    kaapi_task_t* task = kaapi_stack_toptask(stack);
    kaapi_task_init( task, kaapi_taskfinalize_body, ta ); /*, KAAPI_TASK_DFG|KAAPI_TASK_STICKY );*/
    kaapi_stack_pushtask(stack);
  }
#endif
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
        kaapi_task_bodyid_t         bodyid,
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

#define KAAPI_REGISTER_TASKFORMAT( formatobject, name, fnc_body_id, fnc_body, ... ) \
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
    kaapi_format_taskregister( &formatobject, (kaapi_task_bodyid_t)fnc_body_id, fnc_body, name, ##__VA_ARGS__);\
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
