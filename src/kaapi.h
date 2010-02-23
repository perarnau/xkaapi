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

#if defined(KAAPI_CONCURRENT_WS)
#  ifdef __APPLE__
#    include <libkern/OSAtomic.h>
#  endif
#endif


#include <stdint.h>
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

#define KAAPI_PROC_TYPE_NO    0x0
#define KAAPI_PROC_TYPE_CPU   0x1
#define KAAPI_PROC_TYPE_GPU   0x2
#define KAAPI_PROC_TYPE_MPSOC 0x3

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
    The function kaapi_advance() makes progress of steal requests
*/
extern int kaapi_advance(void);

/* ========================================================================== */
/** kaapi_get_elapsedtime
    The function kaapi_get_elapsedtime() will return the elapsed time since an epoch.
*/
extern double kaapi_get_elapsedtime(void);


/* ========================================================================== */
/** COmpute a hash value from a string
*/
extern kaapi_uint32_t kaapi_hash_value_len(const char * data, int len);
extern kaapi_uint32_t kaapi_hash_value(const char * data);

/* ========================================================================= */
/* Shared object and access mode                                             */
/* ========================================================================= */
/** Kaapi Access mode
    \ingroup DFG
*/
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

/** Kaapi access mode mask
    \ingroup DFG
*/
#define KAAPI_ACCESS_MASK_RIGHT_MODE   0x1f   /* 5 bits, ie bit 0, 1, 2, 3, 4, including P mode */
#define KAAPI_ACCESS_MASK_MODE         0xf    /* without P mode */
#define KAAPI_ACCESS_MASK_MODE_P       0x10   /* only P mode */

#define KAAPI_ACCESS_MASK_MEMORY       0x20   /* memory location for the data:  */
#define KAAPI_ACCESS_MEMORY_STACK      0x00   /* data is in the Kaapi stack */
#define KAAPI_ACCESS_MEMORY_HEAP       0x20   /* data is in the heap */


/** Bits for task flag field.
   \ingroup TASK 
   DEFAULT flags is for normal task that can be stolen and executed every where.
    - KAAPI_TASK_STICKY: if set, the task could not be theft else the task can (default).
    - KAAPI_TASK_ADAPTIVE: if set, the task is an adaptative task that could be stolen or preempted.
    - KAAPI_TASK_LOCALITY: if set, the task as locality constraint defined in locality data field.
    - KAAPI_TASK_SYNC: if set, the task does engender synchronisation, victim should stop on a stolen task
    before continuing the fast execution using DFG schedule.
*/
/*@{*/
#define KAAPI_TASK_MASK_FLAGS 0xf  /* 4 bits 0xf ie bits 0, 1, 2, 3 to type of the task */
#define KAAPI_TASK_SYNC       0x1   /* 000 0001 */
#define KAAPI_TASK_STICKY     0x2   /* 000 0010 */
#define KAAPI_TASK_ADAPTIVE   0x4   /* 000 0100 */ 
#define KAAPI_TASK_LOCALITY   0x8   /* 000 1000 */
#define KAAPI_TASK_DFG        KAAPI_TASK_SYNC
#define KAAPI_TASK_RFO        0     /* such a task should have KAAPI_TASK_MASK_READY set */

/** Bits for the task state
   \ingroup TASK 
*/
#define KAAPI_TASK_MASK_STATE 0x70  /* 3 bits 0x70 ie bits 4, 5, 6 to encode the state of the task the task */
typedef enum { 
  KAAPI_TASK_S_INIT  =        0x00, /* 0000 0000 */
  KAAPI_TASK_S_EXEC  =        0x10, /* 0001 0000 */
  KAAPI_TASK_S_STEAL =        0x20, /* 0010 0000 */
  KAAPI_TASK_S_WAIT  =        0x30, /* 0011 0000 */
  KAAPI_TASK_S_TERM  =        0x40  /* 0100 0000 */
} kaapi_task_state_t;
#define KAAPI_TASK_MASK_READY 0x80  /* 1000 0000 */ /* 1 bit 0x80 ie bit 7 to encode if the task is marked as ready */

/** Bits for the proc type
   \ingroup TASK 
*/
#define KAAPI_TASK_MASK_PROC  0x700 /* 3 bits 0x70 ie bits 8, 9, 10 to encode the processor type of the task */
#define KAAPI_TASK_PROC_CPU   0x100 /* 0001 0000 0000 */
#define KAAPI_TASK_PROC_GPU   0x200 /* 0010 0000 0000 */
#define KAAPI_TASK_PROC_MPSOC 0x400 /* 0100 0000 0000 */

/** Bits for attribut of adaptive task
    \ingroup ADAPT
*/
#define KAAPI_TASK_ADAPT_MASK_ATTR 0x7000 /* 3 bits: bit 12, 13, 14 */
#define KAAPI_TASK_ADAPT_DEFAULT   0x0000 /* 000 0000 0000 0000: default: preemption & sync of child adaptive tasks */
#define KAAPI_TASK_ADAPT_NOPREEMPT 0x1000 /* 001 0000 0000 0000: no preemption of child adaptive tasks */
#define KAAPI_TASK_ADAPT_NOSYNC    0x2000 /* 010 0000 0000 0000: no synchronisation of child adaptive tasks */
/*@}*/




/* ========================================================================== */
struct kaapi_task_t;
struct kaapi_stack_t;
/** Task body
    \ingroup TASK
*/
typedef void (*kaapi_task_body_t)(struct kaapi_task_t* /*task*/, struct kaapi_stack_t* /* stack */);

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
    Client side of the request
*/
typedef struct kaapi_reply_t {
  volatile kaapi_uint16_t  status;          /* reply status */
  struct kaapi_stack_t*    data;            /* output data */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_reply_t;


/** \ingroup WS
    This value should not collapse with KAAPI_TASK_ADAPT_NOPREEMPT and KAAPI_TASK_ADAPT_NOSYNC values
    because the are also store in the same bit field.
*/
/*@{*/
#define KAAPI_REQUEST_FLAG_PARTIALSTEAL  0x1    /* Set iff steal of adaptive task */
#define KAAPI_REQUEST_FLAG_APPLILEVEL    0x2    /* Set if the scope of result in case of adaptive steal is the running addaptive task */
/*@}*/

/** \ingroup WS
    Server side of a request send by a processor.
    This data structure is pass in parameter of the splitter function.
*/
typedef struct kaapi_request_t {
  kaapi_uint16_t           status;         /* server status */
  kaapi_uint16_t           flag;           /* partial steal of task | processed during the execution of the runing task */
  struct kaapi_reply_t*    reply;          /* caller status */
  struct kaapi_stack_t*    stack;          /* stack of the thief where to store result of the steal operation */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_request_t;


/** Task splitter
    \ingroup TASK
    A splitter should always return the number of work returns to the list of requests.
*/
typedef int (*kaapi_task_splitter_t)(struct kaapi_stack_t* /*stack */, struct kaapi_task_t* /* task */, int /*count*/, struct kaapi_request_t* /*array*/, void* /*userarg*/);


/** Task reducer
    \ingroup TASK
*/
typedef int (*kaapi_task_reducer_t) (
#ifdef __cplusplus
 struct kaapi_stack_t* /*stack */, struct kaapi_task_t* /* task */, 
 void* arg_thief, ...
#endif
);

/** Kaapi stack of tasks definition
   \ingroup STACK
   The stack store list of tasks as well as a stack of data.
   Both sizes are fixed at initialization of the stack object.
   The stack is truly a stack when used in conjonction with frame.
   A frame capture the state (pc, sp, sp_data) of the stack in order
   to restore it. The implementation also used kaapi_retn_body in order 
   to postpone the restore operation after a set of tasks (see kaapi_stack_taskexecall).

   Before and after the execution of a task, the state of the computation is only
   defined by the stack state (pc, sp, sp_data and the content of the stack). Not that
   kaapi_stack_execall and other funcitons to execute tasks may cached internal state (pc). 
   The C-stack doesnot need to be saved in that case.
   
   \TODO save also the C-stack if we try to suspend execution during a task execution
*/
typedef struct kaapi_stack_t {
  volatile int              hasrequest;     /** points to the k-processor structure */
  volatile int              haspreempt;     /** !=0 if preemption is requested */
  struct kaapi_task_t      *pc;             /** task counter: next task to execute, 0 if empty stack */
  struct kaapi_task_t      *sp;             /** stack counter: next free task entry */
  struct kaapi_task_t*      task;           /** pointer to the first pushed task */

  char*                     sp_data;        /** stack counter for the data: next free data entry */
  char*                     data;           /** stack of data with the same scope than task */


  kaapi_request_t          *requests;       /** points to the requests set in the processor structure */
  kaapi_uint32_t            size;           /** size of the data structure */
  struct kaapi_stack_t*     _next;          /** to be stackable */
  struct kaapi_processor_t* _proc;          /** (internal) access to the attached processor */
} kaapi_stack_t;


/** Kaapi frame definition
   \ingroup STACK
*/
typedef struct kaapi_frame_t {
    struct kaapi_task_t* pc;
    struct kaapi_task_t* sp;
    char*                sp_data;
} kaapi_frame_t;


/* ========================================================================= */
/* What is a task ?                                                          */
/* ========================================================================= */
/** Kaapi task definition
    \ingroup TASK
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  kaapi_uint32_t        flag;      /** flags: after a padding on 64 bit architecture !!!  */
  kaapi_task_body_t     body;      /** C function that represent the body to execute */
  void*                 sp;        /** data stack pointer of the data frame for the task  */
  kaapi_format_t*       format;    /** format, 0 if not def !!!  */
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
/* Shared object and access mode                                             */
/* ========================================================================= */
/** \ingroup DFG
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

#define KAAPI_ACCESS_IS_ONLYWRITE( m ) \
  (KAAPI_ACCESS_IS_WRITE(m) && !KAAPI_ACCESS_IS_READ(m))

#define KAAPI_ACCESS_IS_READWRITE( m ) \
  ((m) == (KAAPI_ACCESS_MODE_W|KAAPI_ACCESS_MODE_R))

/** Return true if two modes are concurrents
    a == b and a or b is R or CW
    or a or b is postponed.
*/
#define KAAPI_ACCESS_IS_CONCURRENT(a,b) ((((a)==(b)) && (((b) & 2) !=0)) || ((a|b) & KAAPI_ACCESS_MODE_P))
/*@}*/


/** \ingroup DFG
*/
typedef struct kaapi_gd_t {
  kaapi_access_mode_t last_mode;    /* last access mode to the data */
  void*               last_version; /* last verion of the data, 0 if not ready */
}  __attribute__((aligned(KAAPI_MAX_DATA_ALIGNMENT))) kaapi_gd_t;


/** \ingroup DFG
    Kaapi access, public
*/
typedef struct kaapi_access_t {
  void* data;                    /* global data */
  void* version;                 /* used for WS, 0 if not yet hashed */
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
    Return the proc type of the task
*/
#define kaapi_task_proctype(task) ((task)->flag & KAAPI_TASK_MASK_PROC)


/** \ingroup TASK
    Return a pointer to parameter of the task (void*) pointer
*/
static inline void* kaapi_task_getargs(kaapi_task_t* task) 
{
  if (task->flag & KAAPI_TASK_ADAPTIVE) 
    return ((kaapi_taskadaptive_t*)task->sp)->user_sp;
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
  if (task->flag & KAAPI_TASK_ADAPTIVE) 
    return ((kaapi_taskadaptive_t*)task->sp)->user_sp = arg;
  return task->sp = arg;
}

/** \ingroup TASK
    Set the pointer to parameter of the task (void*) pointer
*/
static inline kaapi_task_body_t kaapi_task_setbody(kaapi_task_t* task, kaapi_task_body_t body )
{
  return task->body = body;
}


/** Body of the startup task 
    \ingroup TASK
*/
extern void kaapi_taskstartup_body( kaapi_task_t*, kaapi_stack_t*);

/** Body of the nop task 
    \ingroup TASK
*/
extern void kaapi_nop_body( kaapi_task_t*, kaapi_stack_t*);

/** Body of the task that restore the frame pointer 
    \ingroup TASK
*/
extern void kaapi_retn_body( kaapi_task_t*, kaapi_stack_t*);

/** Body of the task that mark a task to suspend execution
    \ingroup TASK
*/
extern void kaapi_suspend_body( kaapi_task_t*, kaapi_stack_t*);

/** Body of the task that do signal to a task after steal op
    \ingroup TASK
*/
extern void kaapi_tasksig_body( kaapi_task_t* task, kaapi_stack_t* stack);

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_taskfinalize_body( kaapi_task_t* task, kaapi_stack_t* stack );


/** \ingroup TASK
    The function kaapi_task_isstealable() will return non-zero value iff the task may be stolen.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isstealable(const kaapi_task_t* task)
{ return !(task->flag & KAAPI_TASK_STICKY); }

/** \ingroup TASK
    The function kaapi_task_setstealable() will return non-zero value iff the task' flag has been set.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_setstealable(kaapi_task_t* task)
{ return task->flag |= KAAPI_TASK_STICKY; }

/** \ingroup TASK
    The function kaapi_task_unsetstealable() will return non-zero value iff the task' flag has been unset.
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_unsetstealable(kaapi_task_t* task)
{ return task->flag &= ~KAAPI_TASK_STICKY; }

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
{ return (task->flag & KAAPI_TASK_ADAPTIVE); }

/** \ingroup TASK
    The function kaapi_task_issync() will return non-zero value iff the such stolen task will introduce data dependency
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_issync(const kaapi_task_t* task)
{ return (task->flag & KAAPI_TASK_SYNC); }

/** \ingroup TASK
    The function kaapi_task_isready() will return non-zero value iff the task is maked as ready
    \param task IN a pointer to the kaapi_task_t to test.
*/
inline static int kaapi_task_isready(const kaapi_task_t* task)
{ return (task->flag & KAAPI_TASK_MASK_READY); }


/** \ingroup STACK
    Return pointer to the self stack
*/
extern kaapi_stack_t* kaapi_self_stack (void);


/** \ingroup STACK
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

/** \ingroup STACK
    The function kaapi_stack_clear() clears the stack.
    If successful, the kaapi_stack_clear() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t to clear.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_stack_clear( kaapi_stack_t* stack );

/** \ingroup STACK
    The function kaapi_stack_isempty() will return non-zero value iff the stack is empty. Otherwise return 0.
    If the argument is a bad pointer then the function kaapi_stack_isempty returns a non value as if the stack was empty.
    \param stack IN the pointer to the kaapi_stack_t data structure. 
    \retval !=0 if the stack is empty
    \retval 0 if the stack is not empty or argument is an invalid stack pointer
*/
static inline int kaapi_stack_isempty(const kaapi_stack_t* stack)
{
  return (stack ==0) || (stack->pc <= stack->sp);
}

/** \ingroup STACK
    The function kaapi_stack_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_stack_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline void* kaapi_stack_pushdata(kaapi_stack_t* stack, kaapi_uint32_t count)
{
  void* retval;
#if defined(KAAPI_DEBUG)
  if (stack ==0) return 0;
  kaapi_assert_debug( (char*)stack->sp_data+count >= (char*)stack->sp );
#endif
  retval = stack->sp_data;
  stack->sp_data += count;
  return retval;
}


/** \ingroup STACK
    The function kaapi_stack_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_stack_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_access_t kaapi_stack_pushshareddata(kaapi_stack_t* stack, kaapi_uint32_t count)
{
  kaapi_access_t retval;
#if defined(KAAPI_DEBUG)
  if (stack ==0) { retval.data = 0; return retval; }
  kaapi_assert_debug( (char*)stack->sp_data+count <= (char*)stack->sp );
#endif
  retval.data = stack->sp_data;
#if defined(KAAPI_DEBUG)
  retval.version = 0;
#endif  
  stack->sp_data += count;
  return retval;
}

/** \ingroup STACK
    The function kaapi_stack_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_stack_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_stack_allocateshareddata(kaapi_access_t* access, kaapi_stack_t* stack, kaapi_uint32_t count)
{
#if defined(KAAPI_DEBUG)
  if (stack ==0) { access->data = 0; return; }
  kaapi_assert_debug( (char*)stack->sp_data+count <= (char*)stack->sp );
#endif
  access->data = stack->sp_data;
#if defined(KAAPI_DEBUG)
  access->version = 0;
#endif  
  stack->sp_data += count;
  return;
}


/** \ingroup STACK
    The function kaapi_access_init() initialize an access from a user defined pointer
    \param access INOUT a pointer to the kaapi_access_t data structure to initialize
    \param value INOUT a pointer to the user data
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_access_init(kaapi_access_t* access, void* value )
{
  access->data = value;
#if defined(KAAPI_DEBUG)
  access->version = 0;
#endif  
  return;
}


#define kaapi_stack_topdata(stack) \
    (stack)->sp_data

/** \ingroup STACK
    The function kaapi_stack_bottom() will return the top task.
    The bottom task is the first pushed task into the stack.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_stack_bottomtask(kaapi_stack_t* stack) 
{
#if defined(KAAPI_DEBUG)
  if (stack ==0) return 0;
#endif
  if (stack->pc <= stack->sp) return 0;
  return stack->task;
}

/** \ingroup STACK
    The function kaapi_stack_top() will return the top task.
    The top task is not part of the stack, it will be the next pushed task.
    If successful, the kaapi_stack_top() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_stack_toptask(kaapi_stack_t* stack) 
{
#if defined(KAAPI_DEBUG)
  if (stack ==0) return 0;
  kaapi_assert_debug((char*)stack->sp >= (char*)stack->sp_data);
#endif
  return stack->sp;
}

/** \ingroup STACK
    The function kaapi_stack_push() pushes the top task into the stack.
    If successful, the kaapi_stack_push() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_stack_pushtask(kaapi_stack_t* stack)
{
#if defined(KAAPI_DEBUG)
  if (stack ==0) return EINVAL;
  kaapi_assert_debug((char*)stack->sp >= (char*)stack->sp_data);
#endif
#if defined(KAAPI_CONCURRENT_WS)
#ifdef __APPLE__
  OSMemoryBarrier();
#else 
  __sync_synchronize();
#endif
#endif
  --stack->sp;
  return 0;
}


/** \ingroup TASK
    Initialize a task with given flag for adaptive attribut or task constraints.
*/
static inline int kaapi_task_initadaptive( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_task_body_t taskbody, void* arg, kaapi_uint32_t flag ) 
{
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*) kaapi_stack_pushdata( stack, sizeof(kaapi_taskadaptive_t) );
#if defined(KAAPI_DEBUG)
  task->format = 0;
#endif
  task->flag   = flag | KAAPI_TASK_ADAPTIVE;
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
  task->body                = taskbody;
  return 0;
}

#if defined(KAAPI_DEBUG)
#  define kaapi_task_format_debug(task) \
    (task)->format   = 0
#else
/* bug: should be set to 0 in case of badly initialize task.... */
#  define kaapi_task_format_debug(task) \
    (task)->format   = 0
#endif

#define kaapi_task_initdfg( stack, task, taskbody, arg ) \
  do { \
    (task)->body     = (taskbody);\
    (task)->sp       = (arg);\
    (task)->flag     = KAAPI_TASK_DFG;\
    (task)->format   = 0;\
  } while (0)


/** \ingroup TASK
    Initialize a task with given flag for adaptive attribut
*/
static inline int kaapi_task_init( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_task_body_t taskbody, void* arg, kaapi_uint32_t flag ) 
{
  if (flag & KAAPI_TASK_ADAPTIVE)
    kaapi_task_initadaptive(stack, task, taskbody, arg, flag); /* here only flag & KAAPI_TASK_ADAPT_MASK_ATTR */
  else {
    kaapi_assert_debug(flag & KAAPI_TASK_DFG);  /* if no ADAPT, must be DFG. Could be both     */
    kaapi_task_initdfg(stack, task, taskbody, arg );
  }
  if (flag & KAAPI_TASK_DFG)
    task->flag |= flag & (KAAPI_TASK_MASK_FLAGS|KAAPI_TASK_MASK_PROC);
    
  return 0;
}


/** \ingroup STACK
    The function kaapi_stack_save_frame() saves the current frame of a stack into
    the frame data structure.
    If successful, the kaapi_stack_save_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack IN a pointer to the kaapi_stack_t data structure.
    \param frame OUT a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int kaapi_stack_save_frame( kaapi_stack_t* stack, kaapi_frame_t* frame)
{
#if defined(KAAPI_DEBUG)
  if ((stack ==0) || (frame ==0)) {
    abort();
    return EINVAL;
  }
#endif
  frame->pc      = stack->pc;
  frame->sp      = stack->sp;
  frame->sp_data = stack->sp_data;
  return 0;  
}

/** \ingroup STACK
    The function kaapi_stack_restore_frame() restores the frame context of a stack into
    the stack data structure.
    If successful, the kaapi_stack_restore_frame() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \param frame IN a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int kaapi_stack_restore_frame( kaapi_stack_t* stack, const kaapi_frame_t* frame)
{
#if defined(KAAPI_DEBUG)
  if ((stack ==0) || (frame ==0)) {
    abort();
    return EINVAL;
  }
#endif
  stack->pc      = frame->pc;
  stack->sp      = frame->sp;
  stack->sp_data = frame->sp_data;
  return 0;  
}

/** \ingroup STACK
    The function kaapi_stack_pushretn() push a task in charge of restoring the stack with the frame given in parameter.
    If successful, the kaapi_stack_pushretn() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \param frame IN a pointer to the kaapi_frame_t data structure.
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int kaapi_stack_pushretn( kaapi_stack_t* stack, const kaapi_frame_t* frame)
{
  kaapi_task_t* retn;
  kaapi_frame_t* arg_retn;
  retn = kaapi_stack_toptask(stack);
  retn->flag  = KAAPI_TASK_STICKY;
  retn->body  = &kaapi_retn_body;
  kaapi_task_format_debug( retn );
  arg_retn = (kaapi_frame_t*)kaapi_stack_pushdata(stack, sizeof(kaapi_frame_t));
  retn->sp = (void*)arg_retn;
  *arg_retn = *frame;
  kaapi_stack_pushtask(stack);
  return 0;
}

/** \ingroup STACK
    The function kaapi_stack_execchild() execute the given and all childs task.
    If successful, the kaapi_stack_execchild() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer
    \retval EWOULDBLOCK the execution of the task will block the control flow.
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
extern int kaapi_stack_execchild(kaapi_stack_t* stack, kaapi_task_t* task);

/** \ingroup STACK
    The function kaapi_stack_execall() execute all the tasks in the stack following
    the RFO order.
    If successful, the kaapi_stack_execall() function will return zero and the stack is empty.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
    \retval EWOULDBLOCK the execution of the stack will block the control flow.
*/
extern int kaapi_stack_execall(kaapi_stack_t* stack);

/** \ingroup STACK
    The function kaapi_sched_sync() execute all tasks from pc stack pointer and all their child tasks.
    If successful, the kaapi_sched_sync() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
extern int kaapi_sched_sync(kaapi_stack_t* stack);

/** \ingroup WS
    Try to steal work from tasks in the stack, else call splitter of the task. 
    Deprecated
*/
#if 0
extern int kaapi_sched_stealtask( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_task_splitter_t splitter );
#endif

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
extern int kaapi_sched_stealstack  ( kaapi_stack_t* stack, kaapi_task_t* task );

/** \ingroup ADAPTIVE
    Test if the current execution should process steal request into the task.
    This function also poll for other requests on the thread of control,
    it may invoke processing of streal request of previous pushed tasks.
    \retval !=0 if they are a steal request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_stealpoint_isactive( kaapi_stack_t* stack, kaapi_task_t* task )
{
  int count = stack->hasrequest;
  if (count) 
  {
    /* \TODO: ici appel systematique a kaapi_sched_stealprocessor dans le cas ou la seule tache
       est la tache 'task' afin de retourner vite pour le traitement au niveau applicatif.
       
       Dans le cas concurrent, on ne passe jamais par la (appel direct de kaapi_stealprocessor).
       Dans le cas cooperatif, le thread courant se vol lui meme puis repond
    */
    count = stack->hasrequest -= kaapi_sched_stealstack(stack, task);
    return (count ==0 ? 0 : 1);
  }
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
static inline int kaapi_preemptpoint_isactive( kaapi_stack_t* stack, kaapi_task_t* task )
{
  int retval = stack->haspreempt;
  kaapi_assert_debug( !(task->flag & KAAPI_TASK_ADAPTIVE) || !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) );
  return retval;
}

/** \ingroup ADAPTIVE
    Helper function to pass argument between the victim and the thief.
    On return the victim argument may be read.
*/
extern int kaapi_preemptpoint_before_reducer_call( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_for_victim, int size );
extern int kaapi_preemptpoint_after_reducer_call( kaapi_stack_t* stack, kaapi_task_t* task, int reducer_retval );


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
extern int kaapi_preempt_nextthief_helper( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_to_thief );

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
extern int kaapi_preempt_nextthief_reverse_helper( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_to_thief );

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
    kaapi_stack_t* stack, 
    kaapi_task_t* task, 
    kaapi_request_t* request, 
    kaapi_stack_t* thief_stack, 
    int size, int retval 
);

extern int kaapi_request_reply_head
(
    kaapi_stack_t* stack, 
    kaapi_task_t* task, 
    kaapi_request_t* request, 
    kaapi_stack_t* thief_stack, 
    int size, int retval 
);

extern int kaapi_request_reply_tail
(
    kaapi_stack_t* stack, 
    kaapi_task_t* task, 
    kaapi_request_t* request, 
    kaapi_stack_t* thief_stack, 
    int size, int retval 
);

/** \ingroup ADAPTIVE
    Specialization to reply failed to a processor
    \param stack INOUT the stack of the victim that has been used to replies to the request
    \param task IN the stolen task
    \param request INOUT data structure used to replied by the thief
*/
static inline int kaapi_request_reply_failed(     
    kaapi_stack_t* stack, 
    kaapi_task_t* task, 
    kaapi_request_t* request
)
{ return kaapi_request_reply( stack, task, request, 0, 0, 0 ); }


/** \ingroup ADAPTIVE
    Set an splitter to be called in concurrence with the execution of the next instruction
    if a steal request is sent to the task.
    \retval EINVAL in case of error (task not adaptive kind)
    \retval 0 else
*/
static inline int kaapi_stealbegin(kaapi_stack_t* stack, kaapi_task_t* task, kaapi_task_splitter_t splitter, void* arg_tasksplitter)
{
  if (!kaapi_task_isadaptive(task)) return EINVAL;
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
extern int kaapi_stealend(kaapi_stack_t* stack, kaapi_task_t* task);

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
*/
static inline int kaapi_finalize_steal( kaapi_stack_t* stack, kaapi_task_t* task )
{
  if (kaapi_task_isadaptive(task) && !(task->flag & KAAPI_TASK_ADAPT_NOSYNC))
  {
    kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
    kaapi_task_t* task = kaapi_stack_toptask(stack);
    kaapi_task_init( stack, task, &kaapi_taskfinalize_body, ta, KAAPI_TASK_DFG|KAAPI_TASK_STICKY );
    kaapi_stack_pushtask(stack);
  }
  return 0;
}

/** \ingroup ADAPTIVE
    Push the task that, on execution will wait the terminaison of the previous 
    adaptive task 'task' and all the thieves.
    The local result, if not null will be pushed after the end of execution of all local tasks.
*/
static inline int kaapi_return_steal( kaapi_stack_t* stack, kaapi_task_t* task, void* retval, int size )
{
  if (kaapi_task_isadaptive(task) && !(task->flag & KAAPI_TASK_ADAPT_NOSYNC))
  {
    kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
    kaapi_assert( (size ==0) || size <= ta->result_size );
    ta->local_result_data = retval;
    ta->local_result_size = size;
    kaapi_task_t* task = kaapi_stack_toptask(stack);
    kaapi_task_init( stack, task, &kaapi_taskfinalize_body, ta, KAAPI_TASK_DFG|KAAPI_TASK_STICKY );
    kaapi_stack_pushtask(stack);
  }
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
/* Format declaration                                                        */
/* ========================================================================= */
/** \ingroup TASK
    Register the task's format data structure
*/
extern kaapi_format_id_t kaapi_format_register( 
        kaapi_format_t*             fmt,
        const char*                 name
);

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
extern kaapi_format_t* kaapi_format_resolvebybody(kaapi_task_body_t key);

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
    kaapi_format_taskregister( &formatobject, (fnc_body), name, ##__VA_ARGS__);\
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
