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
#ifndef _KAAPI_H
#define _KAAPI_H 1
#define KAAPI_H _KAAPI_H
/*!
    @header kaapi
    @abstract   This is the public header for XKaapi
    @discussion XKaapi
*/
#if defined (_WIN32)
#  include <windows.h>
#  include <winnt.h>
#endif


#if !defined(__SIZEOF_POINTER__)
#  if defined(__LP64__) || defined(__x86_64__)
#    define __SIZEOF_POINTER__ 8
#  elif defined(__i386__) || defined(__ppc__)
#    define __SIZEOF_POINTER__ 4
#  else
#    error KAAPI needs __SIZEOF_* macros. Use a recent version of gcc
#  endif
#endif

#if ((__SIZEOF_POINTER__ != 4) && (__SIZEOF_POINTER__ != 8)) 
#  error KAAPI cannot be compiled on this architecture due to strange size for __SIZEOF_POINTER__
#endif

#include <stdint.h>
#include <stdio.h> /* why ? */
#include <stdlib.h>
#include <errno.h>
#if !defined (_WIN32)
#  include <alloca.h>
#endif

#include "kaapi_error.h"
#include <limits.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__linux__)
#  define KAAPI_HAVE_COMPILER_TLS_SUPPORT 1
#elif defined(__APPLE__)
#endif

#ifdef __GNUC__
#  define KAAPI_MAX_DATA_ALIGNMENT (__alignof__(void*))
#else
#  define KAAPI_MAX_DATA_ALIGNMENT 8
#endif

/* Kaapi types.
 */
typedef uint32_t kaapi_globalid_t;
typedef uint32_t kaapi_processor_id_t;
typedef uint64_t kaapi_affinity_t;
typedef uint32_t kaapi_format_id_t;
typedef uint32_t kaapi_offset_t;
typedef uint64_t kaapi_gpustream_t;

/** Reducor to reduce value with cw access mode */
typedef void (*kaapi_reducor_t)(void* /*result*/, const void* /*value*/);

/** Redinit: build a neutral element for the reduction law */
typedef void (*kaapi_redinit_t)(void* /*result*/);

/* Fwd decl
*/
struct kaapi_task_t;
struct kaapi_thread_context_t;
struct kaapi_stealcontext_t;
struct kaapi_taskadaptive_result_t;
struct kaapi_format_t;
struct kaapi_processor_t;
struct kaapi_request_t;
struct kaapi_reply_t;
struct kaapi_tasklist_t;
  
/* =========================== atomic support ====================================== */
#include "kaapi_atomic.h"


/* ========================================================================== */
/* Main function: initialization of the library; terminaison and abort        
   In case of normal terminaison, all internal objects are (I hope so !) deleted.
   The abort function is used in order to try to flush most of the internal buffer.
   kaapi_init should be called before any other kaapi function.
   \param flag [IN] if !=0 then start execution in parallel, else only the main thread is started
   \retval 0 in case of sucsess
   \retval EALREADY if already called
*/
extern int kaapi_init(int flag, int* argc, char*** argv);

/* Kaapi finalization. 
   After call to this functions all other kaapi function calls may not success.
   \retval 0 in case of sucsess
   \retval EALREADY if already called
*/
extern int kaapi_finalize(void);


/** Declare the beginning of a parallel region
    \param schedflag flag to drive the scheduling of this parallel region
*/
extern void kaapi_begin_parallel(int schedflag);

typedef enum {
  KAAPI_SCHEDFLAG_DEFAULT = 0,
  KAAPI_SCHEDFLAG_NOWAIT  = 0x1,
  KAAPI_SCHEDFLAG_STATIC  = 0x2
} kaapi_schedflag_t;

/** Declare the end of a parallel region
    \param flag == 0 then an implicit sync is inserted before existing the region.
    \param flag == 1 then implicit sync is not inserted.
*/
extern void kaapi_end_parallel(int flag);

/* Get the current processor kid. 
   \retval the current processor id
*/
extern unsigned int kaapi_get_self_kid(void);

/* Abort 
*/
extern void kaapi_abort(void);

/* ========================== utilities ====================================== */
extern void* kaapi_malloc_align( unsigned int align_size, size_t size, void** addr_tofree);

static inline void* _kaapi_align_ptr_for_alloca(void* ptr, uintptr_t align)
{
  kaapi_assert_debug( (align !=0) && ((align ==64) || (align ==32) || (align ==16) \
                                   || (align == 8) || (align == 4) || (align == 2) || (align == 1)) );\
  if (align <8) return ptr;
  --align;
  if ( (((uintptr_t)ptr) & align) !=0U)
    ptr = (void*)((((uintptr_t)ptr) + align ) & ~align);
  kaapi_assert_debug( (((uintptr_t)ptr) & align) == 0U);
  return ptr;
}

#define kaapi_alloca_align( align, size) _kaapi_align_ptr_for_alloca( alloca(size + (align <8 ? 0 : align -1) ), align )


/** Define the cache line size. 
*/
#define KAAPI_CACHE_LINE 64


/** Processor type
*/
#define KAAPI_PROC_TYPE_HOST    0x1
#define KAAPI_PROC_TYPE_CUDA    0x2
#define KAAPI_PROC_TYPE_MPSOC   0x3
#define KAAPI_PROC_TYPE_MAX     0x4
#define KAAPI_PROC_TYPE_CPU     KAAPI_PROC_TYPE_HOST
#define KAAPI_PROC_TYPE_GPU     KAAPI_PROC_TYPE_CUDA
#define KAAPI_PROC_TYPE_DEFAULT KAAPI_PROC_TYPE_HOST


/* ========================================================================== */
/** \ingroup WS
    Get the workstealing concurrency number, i.e. the number of kernel
    activities to execute the user level thread. 
    This function is machine dependent.
    \retval the number of active threads to steal tasks
 */
extern int kaapi_getconcurrency (void);

/** \ingroup WS
    Set the workstealing conccurency by instanciating kprocs according
    to the registration of each subsystem (ie. mt, cuda...)
    If successful, the kaapi_setconcurrency() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \retval ENOSYS if the function is not available on a given architecture (e.g. MPSoC)
    \retval EINVAL if no memory ressource is available
    \retval ENOMEM if no memory ressource is available
    \retval EAGAIN if the system laked the necessary ressources to create another thread
    on return, the concurrency number may has been set to a different number than requested.
 */
extern int kaapi_setconcurrency(void);


/* ========================================================================== */
/** kaapi_get_elapsedtime
    The function kaapi_get_elapsedtime() will return the elapsed time in second
    since an epoch.
    Default (generic) function is based on system clock (gettimeofday).
*/
extern double kaapi_get_elapsedtime(void);

/** kaapi_get_elapsedns
    The function kaapi_get_elapsedtime() will return the elapsed time since an epoch
    in nano second unit.
*/
extern uint64_t kaapi_get_elapsedns(void);


/* ========================================================================= */
/* Shared object and access mode                                             */
/* ========================================================================= */
/** Kaapi access mode mask
    \ingroup DFG
*/
#define KAAPI_ACCESS_MASK_RIGHT_MODE   0x1f   /* 5 bits, ie bit 0, 1, 2, 3, 4, including P mode */
#define KAAPI_ACCESS_MASK_MODE         0xf    /* without P mode */
#define KAAPI_ACCESS_MASK_MODE_P       0x10   /* only P mode */
#define KAAPI_ACCESS_MASK_MODE_F       0x20   /* only Fifo mode */

#define KAAPI_ACCESS_MASK_MEMORY       0x20   /* memory location for the data:  */
#define KAAPI_ACCESS_MEMORY_STACK      0x00   /* data is in the Kaapi stack */
#define KAAPI_ACCESS_MEMORY_HEAP       0x20   /* data is in the heap */

/*@{*/
typedef enum kaapi_access_mode_t {
  KAAPI_ACCESS_MODE_VOID= 0,        /* 0000 0000 : */
  KAAPI_ACCESS_MODE_V   = 1,        /* 0000 0001 : */
  KAAPI_ACCESS_MODE_R   = 2,        /* 0000 0010 : */
  KAAPI_ACCESS_MODE_W   = 4,        /* 0000 0100 : */
  KAAPI_ACCESS_MODE_CW  = 8,        /* 0000 1000 : */
  KAAPI_ACCESS_MODE_P   = 16,       /* 0001 0000 : */
  KAAPI_ACCESS_MODE_F   = 32,       /* 0010 0000 : only valid with _W or _R */
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

#define KAAPI_ACCESS_IS_CUMULWRITE( m ) \
  ((m) & KAAPI_ACCESS_MODE_CW)

#define KAAPI_ACCESS_IS_POSTPONED( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_P)

#define KAAPI_ACCESS_IS_FIFO( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_F)

/* W and CW */
#define KAAPI_ACCESS_IS_ONLYWRITE( m ) \
  (KAAPI_ACCESS_IS_WRITE(m) && !KAAPI_ACCESS_IS_READ(m))

#define KAAPI_ACCESS_IS_READWRITE( m ) \
  ( ((m) & KAAPI_ACCESS_MASK_RIGHT_MODE) == (KAAPI_ACCESS_MODE_W|KAAPI_ACCESS_MODE_R))

/** Return true if two modes are concurrents
    a == b and a or b is R or CW
    or a or b is postponed.
*/
#define KAAPI_ACCESS_IS_CONCURRENT(a,b) ((((a)==(b)) && (((b) == KAAPI_ACCESS_MODE_R)||((b)==KAAPI_ACCESS_MODE_CW))) || ((a|b) & KAAPI_ACCESS_MODE_P))
/*@}*/



/* ========================================================================= */
/* Format of a task                                                          */
/* ========================================================================= */
/** predefined format 
*/
/*@{*/
extern struct kaapi_format_t* kaapi_char_format;
extern struct kaapi_format_t* kaapi_short_format;
extern struct kaapi_format_t* kaapi_int_format;
extern struct kaapi_format_t* kaapi_long_format;
extern struct kaapi_format_t* kaapi_longlong_format;
extern struct kaapi_format_t* kaapi_uchar_format;
extern struct kaapi_format_t* kaapi_ushort_format;
extern struct kaapi_format_t* kaapi_uint_format;
extern struct kaapi_format_t* kaapi_ulong_format;
extern struct kaapi_format_t* kaapi_ulonglong_format;
extern struct kaapi_format_t* kaapi_float_format;
extern struct kaapi_format_t* kaapi_double_format;
extern struct kaapi_format_t* kaapi_longdouble_format;
extern struct kaapi_format_t* kaapi_voidp_format;
/*@}*/


/* ========================================================================= */
/* Task and stack interface                                                  */
/* ========================================================================= */

/** Kaapi frame definition
   \ingroup TASK
   Same structure as kaapi_thread_t but we keep different type names to avoid automatic conversion.
*/
typedef struct kaapi_frame_t {
    struct kaapi_task_t*     pc;
    struct kaapi_task_t*     sp;
    char*                    sp_data;
    struct kaapi_tasklist_t* tasklist;  /* Not null -> list of ready task, see static_sched.h */
} kaapi_frame_t;


typedef kaapi_frame_t kaapi_thread_t;


/* ========================================================================== */
/** Task body
    \ingroup TASK
    See internal doc in order to have better documentation of invariant between the task and the thread.
*/
typedef void (*kaapi_task_body_t)(void* /*task arg*/, kaapi_thread_t* /* thread or stream */);
typedef void (*kaapi_task_vararg_body_t)(void* /*task arg*/,  kaapi_thread_t* /* thread or stream */, ...);
typedef kaapi_task_body_t kaapi_task_bodyid_t;

#if !defined(KAAPI_COMPILE_SOURCE)
typedef struct kaapi_thread_context_t {
  kaapi_thread_t*                 truc;    /** pointer to the current frame (in stackframe) */
} kaapi_thread_context_t;
#endif


#if !defined(KAAPI_COMPILE_SOURCE)
struct kaapi_threadgrouprep_t {
  /* public part */
  kaapi_thread_t**           threads;      /* array on top frame of each threadctxt, array[-1] = mainthread */
  int32_t                    grpid;        /* group identifier */
  int32_t                    group_size;   /* number of threads in the group */
};
#else
struct kaapi_threadgrouprep_t;
#endif
typedef struct kaapi_threadgrouprep_t* kaapi_threadgroup_t;

/* ========================================================================= */
/* Task binding
   \ingroup TASK
   Definition of owner compute rules constraints to map task where is located
   a data.
   The data may be specify in 2 exclusive forms:
   - with respect to one physical address
   - with respecto to one or several pointer parameters of a task
 */
typedef enum {
  KAAPI_BINDING_ANY = 0,       /* bound to any core */
  KAAPI_BINDING_OCR_ADDR,      /* bound to one address of u.ocr_addr */
  KAAPI_BINDING_OCR_PARAM      /* bound to one or several parameter of the task (bitmap in u.ocr_param)*/
/* missing: hierarchical level, eg: L3, NUMA, MACHINE, CLUSTER... */
} kaapi_task_binding_type_t;


typedef struct kaapi_task_binding
{
  kaapi_task_binding_type_t type;
} kaapi_task_binding_t;


/* ========================================================================= */
/** Kaapi task definition
    \ingroup TASK
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  uintptr_t             state;     /** state of the task */
  kaapi_task_body_t     body;      /** task body  */
  void*                 sp;        /** data stack pointer of the data frame for the task  */
  struct kaapi_task_t*  reserved;  /** reserved field for internal usage */
//TO ADD AFTER  kaapi_task_binding_t  binding;   /** binding information or 0  */
} kaapi_task_t __attribute__((aligned(8))); /* should be aligned on 64 bits boundary on Intel & Opteron */



/* ========================================================================= */
/** Task splitter
    \ingroup TASK
    A splitter should always return the number of work returns to the list of requests.
*/
typedef int (*kaapi_task_splitter_t)(
  struct kaapi_stealcontext_t* /*stc */, 
  int /*count*/, 
  struct kaapi_request_t* /*array*/, 
  void* /*userarg*/);

/** Reducer called on the victim side
    \ingroup TASK
*/
typedef int (*kaapi_victim_reducer_t)
(struct kaapi_stealcontext_t*, void* arg_thief, void*, size_t, void*);

/** Reducer called on the thief side
    \ingroup TASK
*/
typedef int (*kaapi_thief_reducer_t)
(struct kaapi_taskadaptive_result_t*, void* arg_from_victim, void*);


/** \ingroup ADAPT
    Adaptive stealing header. The header is the
    part visible by the remote write during the
    reply. Thus we separate it from the remaining
    context to avoid duplicating fields.
*/
typedef struct kaapi_stealheader_t
{
  /* steal method modifiers */
  uint32_t flag; 

  /* (topmost) master stealcontext */
  struct kaapi_stealcontext_t* msc;

  /* thief result, independent of preemption */
  struct kaapi_taskadaptive_result_t* ktr;

} kaapi_stealheader_t;


/** Adaptive stealing context
    \ingroup ADAPT
*/
typedef struct kaapi_stealcontext_t {
  /* header seen by both combinator and local */
  kaapi_stealheader_t            header;

  /* reference to the preempt flag in the thread's reply_t data structure */
  volatile uint64_t* preempt __attribute__((aligned));

  /* splitter context */
  kaapi_task_splitter_t volatile splitter;
  void* volatile                 argsplitter;
  
  /* needed for steal sync protocol */
  kaapi_task_t*		             ownertask;

  kaapi_task_splitter_t          save_splitter;
  void*                          save_argsplitter;
  
  void*                          data_victim;       /* pointer on the thief side to store args from the victim */
  size_t                         sz_data_victim;

  /* initial saved frame */
  kaapi_frame_t                  frame;

  /* thieves related context, 2 cases */
  union
  {
    /* 0) an atomic counter if preemption disabled */
    kaapi_atomic_t count;

    /* 1) a thief list if preemption enabled */
    struct
    {
      kaapi_atomic_t lock;

      struct kaapi_taskadaptive_result_t* volatile head
      __attribute__((aligned(KAAPI_CACHE_LINE)));

      struct kaapi_taskadaptive_result_t* volatile tail
      __attribute__((aligned(KAAPI_CACHE_LINE)));
    } list;
  } thieves;

} kaapi_stealcontext_t;


/** \ingroup ADAPT
    Extra body with kaapi_stealcontext_t as extra arg.
*/
typedef void (*kaapi_adaptive_thief_body_t)(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


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

  volatile uint64_t*	              status;	          /* reply status pointer */
  volatile uint64_t*	              preempt;          /* preemption pointer */

#if defined(KAAPI_COMPILE_SOURCE)
  kaapi_task_t			              state;		/* ktr state represented by a task */

#define KAAPI_RESULT_DATAUSR          0x01
#define KAAPI_RESULT_DATARTS          0x02
  int                                 flag;             /* where is allocated data */

  struct kaapi_taskadaptive_result_t* rhead;            /* double linked list of thieves of this thief */
  struct kaapi_taskadaptive_result_t* rtail;            /* */

  struct kaapi_taskadaptive_result_t* next;             /* link fields in kaapi_taskadaptive_t */
  struct kaapi_taskadaptive_result_t* prev;             /* */

  void*				                  addr_tofree;	    /* the non aligned malloc()ed addr */
#endif /* defined(KAAPI_COMPILE_SOURCE) */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_taskadaptive_result_t;


/** \ingroup ADAPT
    Get the adaptive result data from stealcontext ktr
*/
static inline void* kaapi_adaptive_result_data(kaapi_stealcontext_t* sc)
{
  kaapi_assert_debug(sc->header.ktr);
  return sc->header.ktr->data;
}


/** \ingroup WS
    Reply data structure used to return work after a steal request.
    When work is stolen from a victim, work could be stored in the memory data. 
    The runtime ensure that at least KAAPI_REPLY_DATA_SIZE_MAX byte is available.
    Depending on the work stealing protocol, more data may be available.
    
    After the data has been write to memory, the status is set to one of
    the kaapi_reply_status_t value indicating the success in stealing, the
    failure or an error.

    Thread kinds of objects may be pass in the reply data structure:
    - a task with its body (pointer to function)
    - a format of a task (its format id)
    - a thread (a pointer to a thread)
    
    On return to a successfull theft request of a task or task format, the caller will invoke:
    1/ decode the format if it is a format and store the body into u.s_task.body
    2/ push a task <body,sp> = <u.s_task.body, fubar+offset> into its frame
    3/ execute all the tasks into the frame
*/
typedef struct kaapi_reply_t {

  /* every thread has a status and a preempt words used for remote
     communication. 
     A pointer on this word is used both on the victim side to
     send preemption signal. The thief test preemption on this flag
   */
  volatile uint64_t status;
  volatile uint64_t preempt;

  /* private, since sc is private and sizeof differs */
#if defined(KAAPI_COMPILE_SOURCE)
  uint16_t offset;    /* offset in udata of the task arg */
  union
  {
    struct /* task body */ 
    {
      kaapi_task_bodyid_t	body;
    } s_task;

    struct /* formated body */
    {
      kaapi_format_id_t		fmt;
    } s_taskfmt;

    /* thread */
    struct kaapi_thread_context_t* s_thread;
  } u;
#define KAAPI_REPLY_DATA_SIZE_MAX (32 * KAAPI_CACHE_LINE)
  unsigned char udata[KAAPI_REPLY_DATA_SIZE_MAX]; /* task data */
#endif /* private */

} __attribute__((aligned(KAAPI_CACHE_LINE))) kaapi_reply_t;



/** \ingroup WS
    Server side of a request send by a processor.
    This opaque data structure is pass in parameter of the splitter function.
*/
typedef struct kaapi_request_t {
  volatile int                  status;         /* request status */
  kaapi_processor_id_t          ident;          /* system wide id of the remote queue */
  kaapi_task_t*                 thief_task;     /* placeholder to store the solen task */
  struct kaapi_tasksteal_arg_t* thief_sp;       /* placeholder to store thief task sp  */
  kaapi_taskadaptive_result_t*  ktr;            /* only used in adaptive interface to avoid  */
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_request_t;


/* ========================================================================= */
/** \ingroup DFG
    Kaapi access, public
*/
typedef struct kaapi_access_t {
  void*                  data;    /* global data */
  void*                  version; /* used to set the data to access (R/W/RW/CW) if steal, used to store output after steal */
} kaapi_access_t;

#define kaapi_data(type, a)\
  ((type*)((kaapi_access_t*)(a))->data)


/* ========================================================================= */
/* Interface                                                                 */
/* ========================================================================= */
/** \ingroup TASK
    The function kaapi_access_init() initialize an access from a user defined pointer
    \param access INOUT a pointer to the kaapi_access_t data structure to initialize
    \param value INOUT a pointer to the user data
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_access_init(kaapi_access_t* a, void* value )
{
  a->data = value;
#if !defined(KAAPI_NDEBUG)
  a->version = 0;
#endif  
  return;
}


/** \ingroup TASK
    Return a pointer to parameter of the task (void*) pointer
*/
static inline void* kaapi_task_getargs( const kaapi_task_t* task) 
{
  return (void*)task->sp;
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
    Return pointer to the self stack
    Only work if sfp is the first field of kaapi_thread_context_t
*/
#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  extern __thread kaapi_thread_t**    kaapi_current_thread_key;
  extern __thread kaapi_threadgroup_t kaapi_current_threadgroup_key;
#  define kaapi_self_thread() \
     (*kaapi_current_thread_key)
#  define kaapi_self_threadgroup() \
     kaapi_current_threadgroup_key
#  define kaapi_set_threadgroup( thgrp) \
     kaapi_current_threadgroup_key = thgrp

#else
  extern kaapi_thread_t* kaapi_self_thread (void);
  extern kaapi_threadgroup_t kaapi_self_threadgroup(void);
  extern void kaapi_set_threadgroup(kaapi_threadgroup_t thgrp);
#endif

/** \ingroup TASK
    The function kaapi_thread_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_thread_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_thread_t data structure where to push data
    \retval a pointer to the next task to push or 0.
*/
static inline void* kaapi_thread_pushdata( kaapi_thread_t* thread, uint32_t count)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug( (char*)thread->sp_data+count <= (char*)thread->sp );
  {
    void* const retval = thread->sp_data;
    thread->sp_data += count;
    return retval;
  }
}


/** \ingroup TASK
    same as kaapi_thread_pushdata, but with alignment constraints.
    note the alignment must be a power of 2 and not 0
    \param align the alignment size, in BYTES
*/
static inline void* kaapi_thread_pushdata_align(kaapi_thread_t* thread, uint32_t count, uintptr_t align)
{
  kaapi_assert_debug( (align !=0) && ((align == 8) || (align == 4) || (align == 2)));
  const uintptr_t mask = align - (uintptr_t)1;

  if ((uintptr_t)thread->sp_data & mask)
    thread->sp_data = (char*)((uintptr_t)(thread->sp_data + align) & ~mask);

  return kaapi_thread_pushdata(thread, count);
}

static inline void* kaapi_alloca( kaapi_thread_t* thread, uint32_t count)
{ return kaapi_thread_pushdata(thread, count); }

/** \ingroup TASK
    The function kaapi_thread_pushdata() will return the pointer to the next top data.
    The top data is not yet into the stack.
    If successful, the kaapi_thread_pushdata() function will return a pointer to the next data to push.
    Otherwise, an 0 is returned to indicate the error.
    \param frame INOUT a pointer to the kaapi_frame_t data structure where to push data
    \retval a pointer to the next task to push or 0.
*/
static inline void kaapi_thread_allocateshareddata(kaapi_access_t* a, kaapi_thread_t* thread, uint32_t count)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug( (char*)thread->sp_data+count <= (char*)thread->sp );
  a->data = thread->sp_data;
#if !defined(KAAPI_NDEBUG)
  a->version = 0;
#endif
  thread->sp_data += count;
  return;
}

/** \ingroup TASK
    The function kaapi_thread_toptask() will return the top task.
    The top task is not part of the stack, it will be the next pushed task.
    If successful, the kaapi_thread_toptask() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_thread_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_thread_toptask( kaapi_thread_t* thread) 
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp >= (char*)thread->sp_data);
  return thread->sp;
}


/** \ingroup TASK
    The function kaapi_thread_pushtask() pushes the top task into the stack.
    If successful, the kaapi_thread_pushtask() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_thread_pushtask(kaapi_thread_t* thread)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp >= (char*)thread->sp_data);

  /* not need on X86 archi: write are ordered.
   */
#if !(defined(__x86_64) || defined(__i386__))
  kaapi_writemem_barrier();
#endif
  --thread->sp;
  return 0;
}


/** \ingroup TASK
    The function kaapi_thread_pushtask_withpartitionid() pushes the top task into a stack
    attached to the partitionid pid.
    Check and compute dependencies for the top task to be pushed into the tid-th partition.
    On return the task is pushed into a partition.
    If successful, the kaapi_thread_pushtask_withpartitionid() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_thread_pushtask_withpartitionid(kaapi_thread_t* thread, int pid);

/** \ingroup TASK
    The function kaapi_thread_pushtask_withocr() pushes the top task into a stack
    attached to the same partition id than the (numa)node that stores ptr.
    If successful, the kaapi_thread_pushtask_withocr() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_thread_pushtask_withocr(kaapi_thread_t* thread, const void* ptr)
{
  kaapi_task_binding_t* attribut = 
   (kaapi_task_binding_t*)kaapi_thread_pushdata( thread, sizeof(kaapi_task_binding_t) );
  attribut->type = KAAPI_BINDING_OCR_ADDR;
//TODO  attribut->u.ocr_addr.addr = (uintptr_t)ptr;
  kaapi_thread_pushtask(thread);
  return 0;
}


/** \ingroup TASK
    Task initialization routines
*/
static inline void kaapi_task_initdfg
  (kaapi_task_t* task, kaapi_task_body_t body, void* arg)
{
  task->body  = body;
  task->sp    = arg;
  task->state = 0;
  task->thief = 0;
//  task->binding.type = KAAPI_BINDING_ANY;
}

static inline int kaapi_task_init
  ( kaapi_task_t* task, kaapi_task_bodyid_t taskbody, void* arg ) 
{
  kaapi_task_initdfg(task, taskbody, arg);
  return 0;
}


/** \ingroup TASK
    The function clear the contents of a frame
    \retval EINVAL invalid argument: bad pointer.
*/
static inline int kaapi_frame_clear( kaapi_frame_t* fp)
{
  fp->pc = fp->sp = 0;
  fp->sp_data = 0;
  fp->tasklist = 0;
  return 0;
}

/** \ingroup TASK
    The function kaapi_thread_push_frame() saves the current frame of a stack 
    and push a new frame.
    If successful, the kaapi_thread_push_frame() return the new kaapi_thread_t
    where to push task.
    Otherwise, return 0.
*/
extern kaapi_thread_t* kaapi_thread_push_frame(void);

/** \ingroup TASK
    The function kaapi_thread_pop_frame() pop the current frame of a stack
    and restore the previous saved frame.
    If successful, the kaapi_thread_pop_frame() return the new kaapi_thread_t
    where to push task.
    Otherwise, return 0.
*/
extern kaapi_thread_t* kaapi_thread_pop_frame(void);


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
    The function kaapi_thread_set_unstealable() changes the stealable state of 
    the current executing thread. If the argument is not 0, then the state of the
    current thread changes to unstealable state. If the argument is 0, then the
    statte changes to 'stealable' state, allowing thief to steal the current thread.
*/
extern void kaapi_thread_set_unstealable(unsigned int);

/** \ingroup TASK
    The function kaapi_sched_sync() execute all childs tasks of the current running task.
    If successful, the kaapi_sched_sync() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINTR the control flow has received a KAAPI interrupt.
*/
extern int kaapi_sched_sync( void );

/** Change rerepresentation of the tasks inside the current frame in order
    to build a tasklist that contains ready tasks and tasks that will
    be activated by ready tasks.
    The execution of the frame should then be only be considered using
    kaapi_thread_execframe_tasklist
    \retval EINVAL invalid current thread
    \retval ENOENT if current thread does not has any tasks
    \retval 0 in case of success
*/
extern int kaapi_sched_computereadylist( void );

/** Clear the tasklist of the current frame that has been previously 
    computed by 'kaapi_sched_computereadylist'
    \retval EINVAL invalid current thread
    \retval 0 in case of success
*/
extern int kaapi_sched_clearreadylist( void );

/* ========================================================================= */
/* API for adaptive algorithm                                                */
/* ========================================================================= */
/** \ingroup WS
    Allows the current thread of execution to create tasks on demand.
    A thread that begin an adaptive section, allows thiefs to call a splitter
    function with an arguments in order to push tasks for thiefs.
    If not thief is idle, then no tasks will be created and there is no cost
    due to creations of tasks.

    This is the way Kaapi implements adaptive algorithms.
    
    The current thread of execution (and the current executing task) should
    mark the section of code where tasks are created on demand by using 
    the instructions kaapi_task_begin_adaptive and kaapi_task_end_adaptive. 
    Between this two instructions, a splitter may be invoked with its arguments:
    - in concurrence with the local execution thread iff flag = KAAPI_SC_CONCURRENT
    - with cooperation with the local execution thread iff flag =KAAPI_SC_COOPERATIVE.
    In the case of the cooperative execution, the code should test presence of request
    using the instruction kaapi_stealpoint.
*/
typedef enum kaapi_stealcontext_flag {
  KAAPI_SC_CONCURRENT    = 0x1,
  KAAPI_SC_COOPERATIVE   = 0x2,
  KAAPI_SC_PREEMPTION    = 0x4,
  KAAPI_SC_NOPREEMPTION  = 0x8,
  KAAPI_SC_INIT          = 0x10,   /* 1 == iff initilized (for lazy init) */
  KAAPI_SC_AGGREGATE	 = 0x20,
  KAAPI_SC_HWS_SPLITTER	 = 0x40,
  
  KAAPI_SC_DEFAULT = KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION
} kaapi_stealcontext_flag;

/** Begin adaptive section of code
*/
kaapi_stealcontext_t* kaapi_task_begin_adaptive( 
  kaapi_thread_t*       thread,
  int                   flag,
  kaapi_task_splitter_t splitter,
  void*                 argsplitter
);

/** \ingroup ADAPTIVE
    Mark the end of the adaptive section of code.
    After the call to this function, all thieves have finish to compute in parallel,
    and memory location produced in concurrency may be read by the calling thread.
*/
extern void kaapi_task_end_adaptive( kaapi_stealcontext_t* stc );

/** \ingroup ADAPTIVE
    Allocate the return data structure for the thief. This structure should be passed
    by the user to the replied tasks. The user may pass an application buffer (data) that
    contains at least 'size' bytes. Else if the size is >0 then the runtime allocates
    a memory region that contains at least size byte.
    The return value is an opaque pointer that should be pass to kaapi_request_reply.
    If the user reply 'success' to the thief then both this structure and the associated data 
    will be hold by the runtime system and deallocate automatically. 
    Else if the user finally decide to reply 'failed' then it should explicitly deallocate it 
    by a call to kaapi_deallocate_thief_result.
    \param sc INOUT the steal context
    \param size IN the size in bytes of the memory to store the result of a thief
    \param data IN a user defined data or 0 to means to be allocated by the runtime
*/
extern struct kaapi_taskadaptive_result_t* kaapi_allocate_thief_result(
    struct kaapi_request_t* kreq, int size, void* data
);

/** \ingroup ADAPTIVE
    Dellocate the return data structure already allocated to the thief. 
    The call never deallocate a application level buffer passed to kaapi_allocate_thief_result.
    The user should only call this function in case of aborting the return to the thief
    by replying 'failed'. In case of reply 'success' this function is automatically called
    by the runtime.
    \param result INOUT the pointer to the opaque data structure returned by kaapi_allocate_thief_result.
*/
extern int kaapi_deallocate_thief_result( struct kaapi_taskadaptive_result_t* result );

#define KAAPI_REQUEST_REPLY_HEAD 0x0
#define KAAPI_REQUEST_REPLY_TAIL 0x1

/** \ingroup ADAPTIVE
*/
extern int kaapi_request_reply(
  kaapi_stealcontext_t* sc, 
  kaapi_request_t*      req, 
  int                   headtail_flag
);

/** \ingroup ADAPTIVE
    Initialize an adaptive task to be executed by a thief.
    In case of sucess, the function returns the pointer of a memory region where to store 
    arguments for the entrypoint. Once pushed, the task is executed by the thief with
    first argument the pointer return the call to kaapi_reply_init_adaptive_task.
    \param sc the stealcontext
    \param req the request emitted by a thief
    \param body the entry point of the task to execute
    \param size the user data size of the task arguments
    \param result that taskadaptive_result used for signalisation.
*/
extern void* kaapi_reply_init_adaptive_task (
    struct kaapi_stealcontext_t*        sc,
    kaapi_request_t*                    req,
    kaapi_task_body_t                   body,
    size_t				                      size,
    struct kaapi_taskadaptive_result_t* result
);

/** \ingroup ADAPTIVE
*/
extern void kaapi_request_reply_failed(kaapi_request_t*);

/** \ingroup ADAPTIVE
    push the task associated with an adaptive request
*/
static inline void kaapi_reply_pushhead_adaptive_task(kaapi_stealcontext_t* sc, kaapi_request_t* req)
{
  /* sc the stolen stealcontext */
  kaapi_request_reply(sc, req, KAAPI_REQUEST_REPLY_HEAD);
}

/** \ingroup ADAPTIVE
    push the task associated with an adaptive request
*/
static inline void kaapi_reply_pushtail_adaptive_task(kaapi_stealcontext_t* sc, kaapi_request_t* req)
{
  /* sc the stolen stealcontext */
  kaapi_request_reply(sc, req, KAAPI_REQUEST_REPLY_TAIL);
}

/** \ingroup ADAPTIVE
    push the task associated with an adaptive request
*/
static inline void kaapi_reply_push_adaptive_task(kaapi_stealcontext_t* sc, kaapi_request_t* r)
{
  kaapi_reply_pushhead_adaptive_task(sc, r);
}

/** \ingroup ADAPTIVE
    Set an splitter to be called in concurrence with the execution of the next instruction
    if a steal request is sent to the task.
    The old splitter may be saved using getsplitter calls.
    \retval EINVAL in case of error (task not adaptive kind)
    \retval 0 else
*/
static inline int kaapi_steal_setsplitter(
    kaapi_stealcontext_t* stc,
    kaapi_task_splitter_t splitter, void* arg_tasksplitter)
{
  stc->argsplitter = 0;
  stc->splitter    = 0;
  kaapi_writemem_barrier();
  stc->argsplitter = arg_tasksplitter;
  stc->splitter    = splitter;
  return 0;
}

#if 0 /* revoir ici, ces fonctions sont importantes pour le cooperatif */
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
  /* \TODO: ici appel systematique a kaapi_sched_stealprocessor dans le cas ou la seule tache
     est la tache 'task' afin de retourner vite pour le traitement au niveau applicatif.
     
     Dans le cas concurrent, on ne passe jamais par la (appel direct de kaapi_stealprocessor).
     Dans le cas cooperatif, le thread courant se vol lui meme puis repond
  */
  return kaapi_sched_stealstack_helper( stc );
}


#if !defined(__cplusplus)
/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then call the splitter function with given arguments.
    \retval 0 \TODO code de retour
*/
#define kaapi_stealpoint( stc, splitter, ...) \
   (kaapi_stealpoint_isactive(stc) ? (splitter)( stc, (stc)->hasrequest, (stc)->requests, ##__VA_ARGS__), 1 : 0 )

#endif

#endif /* if 0 */


/* Return the thief result of the next thief from the head of the list to preempt or 0 if no thief may be preempted
*/
extern struct kaapi_taskadaptive_result_t* kaapi_get_thief_head( kaapi_stealcontext_t* stc );

/* Return the thief result of the next thief from the tail of the list to preempt or 0 if no thief may be preempted
*/
extern struct kaapi_taskadaptive_result_t* kaapi_get_thief_tail( kaapi_stealcontext_t* stc );

struct kaapi_taskadaptive_result_t* kaapi_get_next_thief( struct kaapi_taskadaptive_result_t* pos );

struct kaapi_taskadaptive_result_t* kaapi_get_prev_thief( struct kaapi_taskadaptive_result_t* pos );



/** Preempt a thief.
    Send a preemption request to the thief with result data structure ktr.
    And pass extra arguments (arg_to_thief).
    The call to the function returns 1 iff:
    - ktr has been preempted or finished
    - ktr has been replaced by the thieves of ktr into the list of stc
*/
extern int kaapi_preempt_thief_helper( 
  kaapi_stealcontext_t*               stc, 
  struct kaapi_taskadaptive_result_t* ktr, 
  void*                               arg_to_thief 
);

/** \ingroup ADAPTIVE
   Post a preemption request to thief. Do not wait preemption occurs.
   Return 0 iff some work have been preempted and should be processed locally.
   If the thief has already finished its computation bfore sending the signal,
   then the return value is ECHILD.
*/
extern int kaapi_preemptasync_thief( 
  kaapi_stealcontext_t*               stc, 
  struct kaapi_taskadaptive_result_t* ktr, 
  void*                               arg_to_thief 
);

/** The thief should have been preempted using preempasync_thief
    Returns 0 when the thief has reply to its preemption flag
*/
extern int kaapi_preemptasync_waitthief
( 
 kaapi_stealcontext_t*               sc,
 struct kaapi_taskadaptive_result_t* ktr
);

/** \ingroup ADAPTIVE
    Remove the thief ktr form the list of stc iff it is has finished its computation and returns 0.
    Else returns EBUSY.
*/
extern int kaapi_remove_finishedthief( 
  kaapi_stealcontext_t*               stc, 
  struct kaapi_taskadaptive_result_t* ktr
);


/** \ingroup ADAPTIVE
   Try to preempt the thief referenced by tr. Wait either preemption occurs or the end of the thief.
   Once the thief has received the preempt and send back result to the victim who preempt it, then
   the function reducer is called.
   Return value is the return value of the reducer function or 0 if no reducer is given.
      
   The reducer function should has the following signature:
      int (*)( stc, void* thief_arg, void* thief_result, size_t thief_ressize, ... )
   where ... is the same extra arguments passed to kaapi_preempt_nextthief.
*/

static inline int kaapi_preempt_thief
(kaapi_stealcontext_t* sc,
 kaapi_taskadaptive_result_t* ktr,
 void* thief_arg,
 kaapi_victim_reducer_t reducer,
 void* reducer_arg)
{
  int res = 0;

  if (kaapi_preempt_thief_helper(sc, ktr, thief_arg))
  {
    if (reducer)
    {
      res = reducer
	(sc, ktr->arg_from_thief, ktr->data, ktr->size_data, reducer_arg);
    }

    kaapi_deallocate_thief_result(ktr);
  }

  return res;
}

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then pass arg_victim argument to the victim and return !=0 value
    \retval !=0 if it exists a prending preempt request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_preemptpoint_isactive(const kaapi_stealcontext_t* ksc)
{
  kaapi_assert_debug(ksc->preempt != 0);
  return *ksc->preempt == 1;
}


/** \ingroup ADAPTIVE
    Helper function to pass argument between the victim and the thief.
    On return the victim argument may be read.
*/
extern int kaapi_preemptpoint_before_reducer_call( 
    kaapi_stealcontext_t* stc,
    void* arg_for_victim, 
    void* result_data, 
    size_t result_size
);
extern int kaapi_preemptpoint_after_reducer_call ( 
    kaapi_stealcontext_t* stc
);

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request to the current task
    and if it is true then pass arg_victim argument to the victim and call the reducer function with incomming victim argument
    for the thief. Extra arguments are added at the end of the parameter when calling reducer function.
    The reducer function is assumed to be of the form:
      (*reducer)(kaapi_taskadaptive_result_t*, void* arg_from_victim, ...)
    Where ... must match the list of extra parameter.
    \retval !=0 if a prending preempt request(s) has been processed onto the given task.
    \retval 0 else
*/

static inline int kaapi_preemptpoint
(kaapi_stealcontext_t* sc,
 kaapi_thief_reducer_t reducer,
 void* victim_arg,
 void* result_data,
 size_t result_size,
 void* reducer_arg)
{
  int res = 0;

  if (kaapi_preemptpoint_isactive(sc)) /* unlikely */
  {
    res = 1; /* tell the thief about reduction */

    kaapi_preemptpoint_before_reducer_call
      (sc, victim_arg, result_data, result_size);
    if (reducer)
      reducer(sc->header.ktr, sc->header.ktr->arg_from_victim, reducer_arg);

    kaapi_preemptpoint_after_reducer_call(sc);
  }

  return res;
}

/** Begin critical section with respect to steal operation
    \ingroup TASK
*/
extern int kaapi_steal_begincritical( kaapi_stealcontext_t* sc );

/** End critical section with respect to steal operation
    \ingroup TASK
*/
extern int kaapi_steal_endcritical( kaapi_stealcontext_t* sc );

/** Same as kaapi_steal_endcritical but stealing left disabled
    \ingroup TASK
\THIERRY    
*/
extern int kaapi_steal_endcritical_disabled( kaapi_stealcontext_t* sc );

/** \ingroup ADAPTIVE
    Signal end of a thief, required to be call if kaapi_steal_finalize is not call in order
    to ensure end of the computation.
*/
extern int kaapi_steal_thiefreturn( kaapi_stealcontext_t* stc );


/* ========================================================================= */
/* API for graph partitioning                                                */
/* ========================================================================= */
/** Type of allowed memory view for the memory interface:
    - 1D array (base, size)
      simple contiguous 1D array
    - 2D array (base, size[2], lda)
      assume a row major storage of the memory : the 2D array has
      size[0] rows of size[1] rowwidth. lda is used to pass from
      one row to the next one.
    The base (kaapi_pointer_t) is not part of the view description
    
*/
#define KAAPI_MEMORY_VIEW_1D 1
#define KAAPI_MEMORY_VIEW_2D 2  /* assume row major */
#define KAAPI_MEMORY_VIEW_3D 3
typedef struct kaapi_memory_view_t {
  int    type;
  size_t size[2];
  size_t lda;
  size_t wordsize;
} kaapi_memory_view_t;


/** Identifier to an address space id
*/
typedef uint64_t  kaapi_address_space_id_t;



/** Returns the numa node (>=0 value) that stores address addr
    Return -1 in case of failure and set errno to the error code.
    Possible error is:
    * ENOENT: function not available on this configuration
    * EINVAL: invalid argument
*/
extern int kaapi_numa_get_page_node(const void* addr);

/** Bind pages of an array [addr, size) on all numa nodes using
    a bloc cyclic strategy of bloc size equals to blocsize. 
    addr should be aligned to page boundary. 
    blocsize should be multiple of the page size.
    Returns 0 in case of success
    Returns -1 in case of failure and set errno to the error code.
    Possible error is:
    * ENOENT: function not available on this configuration
    * EINVAL: invalid argument blocsize
    * EFAULT: invalid address addr
*/
extern int kaapi_numa_bind_bloc1dcyclic
 (const void* addr, size_t size, size_t blocsize);


/** Bind pages from address between addr and addr+size on the numa node node
    In case of success, the function returns 0.
    Else it returns -1 and set errno to the error code.
    See mbind to interpret error code on linux system other than:
    * ENOENT: function not available on this configuration
*/
extern int kaapi_numa_bind(const void* addr, size_t size, int node);


/** Type of pointer for all address spaces.
    The pointer encode both the pointer (field ptr) and the location of the address space
    in asid.
    Pointer arithmetic is allowed on this type on the ptr field.
*/
typedef struct kaapi_pointer_t { 
  kaapi_address_space_id_t asid;
  uintptr_t                ptr;
} kaapi_pointer_t;

static inline void* __kaapi_pointer2void(kaapi_pointer_t p)
{ return (void*)p.ptr; }


/** Data shared between address space and task
    Such data structure is referenced through the pointer arguments of tasks using a handle.
    All tasks, after construction of tasklist, have effective parameter a handle to a data
    in place of pointer to a data: Before creation of a tasklist data structure, a task
    has a direct access through a pointer to object in the shared address space of the process.
    After tasklist creation, each pointer parameter is replaced by a pointer to a kaapi_data_t
    that points to the data and the view of the data.
    
    The data also stores a pointer to the meta data information for fast lookup.
    Warning: The ptr should be the first field of the data structure.
*/
typedef struct kaapi_data_t {
  kaapi_pointer_t               ptr;                /* address of data */
  kaapi_memory_view_t           view;               /* view of data */
  struct kaapi_metadata_info_t* mdi;                /* if not null, pointer to the meta data */
} kaapi_data_t;


/** Handle to data.
    See comments in kaapi_data_t structure.
*/
typedef kaapi_data_t* kaapi_handle_t;


/** Synchronize all shared memory in the local address space to the up-to-date value.
*/
extern int kaapi_memory_synchronize(void);

/** Create a thread group with size threads. 
    Mapping function should be set at creation step. 
    For each thread tid of the group, the function mapping is called with:
      mapping(ctxt_mapping, nodecount, tid) -> (gid, proctype)
    in order to defines the site gid and the kind of architecture that will executes the thread tid.
    The value ctxt_mapping is a user defined data structure that can be used to hold
    parameter for computing the mapping.
    If mapping ==0, then the default runtime mapping if a 1-block cyclic distribution scheme (round robin).
    Return 0 in case of success or the error code.
    
A REVOIR: quid si (gid) n'a pas une architecture cible
*/
extern int kaapi_threadgroup_create(kaapi_threadgroup_t* thgrp, int size, 
  kaapi_address_space_id_t (*mapping)(void*, int /*nodecount*/, int /*tid*/),
  void* ctxt_mapping
);


/**
*/
#define KAAPI_THGRP_DEFAULT_FLAG  0
#define KAAPI_THGRP_SAVE_FLAG     0x1
extern int kaapi_threadgroup_begin_partition(kaapi_threadgroup_t thgrp, int flag );


/**
*/
void kaapi_threadgroup_force_archtype(kaapi_threadgroup_t group, unsigned int part, unsigned int type);

/**
*/
void kaapi_threadgroup_force_kasid(kaapi_threadgroup_t group, unsigned int part, unsigned int arch, unsigned int user);

/**
*/
extern int kaapi_threadgroup_set_iteration_step(kaapi_threadgroup_t thgrp, int maxstep );

/** Check and compute dependencies for task 'task' to be pushed into the tid-th partition.
    On return the task is pushed into the partition if it is local for the execution.
    \return ESRCH if the task is not pushed because the tid-th partition is not local
    \return 0 in case of success
    \return other value due to error
*/
extern int kaapi_threadgroup_computedependencies(kaapi_threadgroup_t thgrp, int tid, kaapi_task_t* task);


#if !defined(KAAPI_COMPILE_SOURCE)
/**
*/
static inline kaapi_thread_t* kaapi_threadgroup_thread( kaapi_threadgroup_t thgrp, int tid ) 
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_assert_debug( (tid>=-1) && (tid<thgrp->group_size) );
  kaapi_thread_t* thread = thgrp->threads[tid];
  return thread;
}
#endif /* !defined(KAAPI_COMPILE_SOURCE) */


/**
*/
extern int kaapi_threadgroup_end_partition(kaapi_threadgroup_t thgrp );

/**
*/
extern int kaapi_threadgroup_begin_execute(kaapi_threadgroup_t thgrp );

/**
*/
extern int kaapi_threadgroup_end_execute(kaapi_threadgroup_t thgrp );


/** Memory synchronization with copies to the original memory
*/
extern int kaapi_threadgroup_synchronize(kaapi_threadgroup_t thgrp );

#if 0
/** Memory synchronization with copies to the original memory for a
    specific set of data
*/
extern int kaapi_threadgroup_synchronize(kaapi_threadgroup_t thgrp );
#endif

/**
    \retval 0 in case of success
    \retval EBUSY if threads are already attached to the group
*/
extern int kaapi_threadgroup_destroy(kaapi_threadgroup_t thgrp );

/**
*/
extern int kaapi_threadgroup_print(FILE* file, kaapi_threadgroup_t thgrp );

/**
*/
extern int kaapi_threadgroup_save(kaapi_threadgroup_t thgrp );

/**
*/
extern int kaapi_threadgroup_restore(kaapi_threadgroup_t thgrp );



/* ========================================================================= */
/* Standard task body                                                        */
/* ========================================================================= */
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

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
extern void kaapi_taskfinalize_body( void*, kaapi_thread_t* );


/** Scheduler information pass by the runtime to task forked
    with set static attribut
    - array nkproc have the same dimension of number of static
    proc types. nkproc[i] == number of proc type i (i+1== KAAPI_PROC_TYPE_HOST|GPU|MPSOC)
    used for static scheduling.
    - bitmap may be also pass here.
*/
typedef struct kaapi_staticschedinfo_t {
  int16_t  nkproc[KAAPI_PROC_TYPE_MAX]; 
} kaapi_staticschedinfo_t;

/** Body of the task in charge of finalize of adaptive task
    \ingroup TASK
*/
typedef struct kaapi_staticschedtask_arg_t {
  void*                    sub_sp;    /* encapsulated task */
  kaapi_task_vararg_body_t sub_body;  /* encapsulated body */
  intptr_t                 key;
  kaapi_staticschedinfo_t  schedinfo; /* number of partition */
} kaapi_staticschedtask_arg_t;

extern void kaapi_staticschedtask_body( void*, kaapi_thread_t* );

/* ========================================================================= */
/* THE workqueue to be used with adaptive interface                          */
/* ========================================================================= */
/** work work_queue_t: the main important data structure.
    It steal/pop are managed by a Dijkstra like protocol.
    The threads that want to steal serialize their access
    through a lock.
    The workqueue can only be used within adaptive interface.
    The following assumptions should be true:
    - the current thread of control should be an kaapi thread in order to ensure
    correctness of the implementation.
    - the owner of the queue can call _init, _set, _pop, _size and _isempty
    - the function _steal must be call in the context of the splitter (see adaptive interface)

    Note about the implementation.
    - The internal lock used in case of conflic is the kaapi_processor lock.
    - data field required to be correctly aligned in order to ensure atomicity of read/write. 
     Put them on two separate lines of cache (assume == 64bytes) due to different access by 
     concurrent threads. Currently only IA32 & x86-64.
     An assertion is put inside the constructor to verify that this field are correctly aligned.
*/
typedef long kaapi_workqueue_index_t;

typedef struct {
  volatile kaapi_workqueue_index_t beg __attribute__((aligned(64)));
  volatile kaapi_workqueue_index_t end __attribute__((aligned(64)));
  kaapi_atomic_t*                  lock;
} kaapi_workqueue_t;


/** Initialize the workqueue to be an empty (null) range workqueue.
    Do memory barrier before updating the queue.
    Attach the workqueue to the current kprocessor, ie the lock to ensure consistent concurrent operation
    is the lock of the current kprocessor.
*/
extern int kaapi_workqueue_init( 
    kaapi_workqueue_t* kwq, 
    kaapi_workqueue_index_t b, 
    kaapi_workqueue_index_t e 
);

/** Initialize the workqueue to be an empty (null) range workqueue.
    Do memory barrier before updating the queue.
    Explicit specification of the lock to used to ensure consistent concurrent operations.
*/
extern int kaapi_workqueue_init_with_lock( 
    kaapi_workqueue_t* kwq, 
    kaapi_workqueue_index_t b, 
    kaapi_workqueue_index_t e, 
    kaapi_atomic_t* thelock 
);

/** destroy 
*/
static inline int kaapi_workqueue_destroy( kaapi_workqueue_t* kwq )
{
  return 0;
}

/** This function set new bounds for the workqueue.
    The only garantee is that is a concurrent thread tries
    to access to the size of the queue while a thread set the workqueue, 
    then the concurrent thread will see the size of the queue before the call to set
    or it will see a nul size queue.
    \retval 0 in case of success
    \retval ESRCH if the current thread is not a kaapi thread.
*/
extern int kaapi_workqueue_set( kaapi_workqueue_t* kwq, kaapi_workqueue_index_t b, kaapi_workqueue_index_t e);

/**
*/
static inline kaapi_workqueue_index_t kaapi_workqueue_range_begin( kaapi_workqueue_t* kwq )
{
  return kwq->beg;
}

/**
*/
static inline kaapi_workqueue_index_t kaapi_workqueue_range_end( kaapi_workqueue_t* kwq )
{
  return kwq->end;
}

/**
*/
static inline kaapi_workqueue_index_t kaapi_workqueue_size( kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_t size = kwq->end - kwq->beg;
  return (size <0 ? 0 : size);
}

/**
*/
static inline unsigned int kaapi_workqueue_isempty( kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_t size = kwq->end - kwq->beg;
  return size <= 0;
}


/** This function should be called by the current kaapi thread that own the workqueue.
    The function push work into the workqueue.
    Assuming that before the call, the workqueue is [beg,end).
    After the successful call to the function the workqueu becomes [newbeg,end).
    newbeg is assumed to be less than beg. Else it is a pop operation, 
    see kaapi_workqueue_pop.
    Return 0 in case of success 
    Return EINVAL if invalid arguments
*/
static inline int kaapi_workqueue_push(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t newbeg
)
{
  if ( kwq->beg  > newbeg )
  {
    kaapi_mem_barrier();
    kwq->beg = newbeg;
    return 0;
  }
  return EINVAL;
}


/** Helper function called in case of conflict.
    Return EBUSY is the queue is empty.
    Return EINVAL if invalid arguments
    Return ESRCH if the current thread is not a kaapi thread.
*/
extern int kaapi_workqueue_slowpop(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t* beg,
  kaapi_workqueue_index_t* end,
  kaapi_workqueue_index_t  size
);


/** This function should be called by the current kaapi thread that own the workqueue.
    Return 0 in case of success 
    Return EBUSY is the queue is empty
    Return EINVAL if invalid arguments
    Return ESRCH if the current thread is not a kaapi thread.
*/
static inline int kaapi_workqueue_pop(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t* beg,
  kaapi_workqueue_index_t* end,
  kaapi_workqueue_index_t max_size
)
{
  kaapi_workqueue_index_t loc_beg;
  kaapi_assert_debug( max_size >0 );
  loc_beg = kwq->beg + max_size;
  kwq->beg = loc_beg;
  kaapi_mem_barrier();

  if (loc_beg < kwq->end)
  {
    /* no conflict */
    *end = loc_beg;
    *beg = *end - max_size;
    return 0;
  }

  /* conflict */
  loc_beg -= max_size;
  kwq->beg = loc_beg;
  return kaapi_workqueue_slowpop(kwq, beg, end, max_size);
}


/** This function should only be called into a splitter to ensure correctness
    the lock of the victim kprocessor is assumed to be locked to handle conflict.
    Return 0 in case of success 
    Return ERANGE if the queue is empty or less than requested size.
 */
static inline int kaapi_workqueue_steal(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t* beg,
  kaapi_workqueue_index_t* end,
  kaapi_workqueue_index_t size
)
{
  kaapi_workqueue_index_t loc_end;

  kaapi_assert_debug( 0 < size );

  /* disable gcc warning */
  *beg = 0;
  *end = 0;

  loc_end = kwq->end - size;
  kwq->end = loc_end;
  kaapi_mem_barrier();

  if (loc_end < kwq->beg)
  {
    kwq->end = loc_end+size;
    return ERANGE; /* false */
  }

  *beg = loc_end;
  *end = *beg + size;
  
  return 0; /* true */
}  


/** kaapi exported splitters
 */

typedef struct kaapi_splitter_context
{
  kaapi_workqueue_t wq;
  kaapi_task_body_t body;
  size_t ktr_size;
  size_t data_size;
  unsigned char data[1];
} kaapi_splitter_context_t;

int kaapi_splitter_default
(struct kaapi_stealcontext_t*, int, struct kaapi_request_t*, void*);


/* ========================================================================= */
/* Perf counter                                                              */
/* ========================================================================= */
/** \ingroup PERF
    performace counters
*/
#define KAAPI_PERF_ID_USER_POS (31)
#define KAAPI_PERF_ID_USER_MASK (1 << KAAPI_PERF_ID_USER_POS)

#define KAAPI_PERF_ID(U, I) (KAAPI_PERF_ID_ ## I | (U) << KAAPI_PERF_ID_USER_POS)
#define KAAPI_PERF_ID_USER(I) KAAPI_PERF_ID(1, I)
#define KAAPI_PERF_ID_PRIV(I) KAAPI_PERF_ID(0, I)

#define KAAPI_PERF_ID_TASKS         0  /* count number of executed tasks */
#define KAAPI_PERF_ID_STEALREQOK    1  /* count number of successful steal requests */
#define KAAPI_PERF_ID_STEALREQ      2  /* count number of steal requests */
#define KAAPI_PERF_ID_STEALOP       3  /* count number of steal operation to reply to requests */
#define KAAPI_PERF_ID_SUSPEND       4  /* count number of suspended thread */
#define KAAPI_PERF_ID_T1            5  /* nano second of compte time */
/*#define KAAPI_PERF_ID_TIDLE         6  / * nano second of idle time */ 
#define KAAPI_PERF_ID_TPREEMPT      7  /* nano second of preempt time */
#define KAAPI_PERF_ID_ALLOCTHREAD   8  /* count number of allocated thread */
#define KAAPI_PERF_ID_FREETHREAD    9  /* count number of free thread */
#define KAAPI_PERF_ID_QUEUETHREAD   10 /* count the maximal number of thread in queue */

#define KAAPI_PERF_ID_ENDSOFTWARE   11 /* mark end of software counters */

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
typedef int64_t kaapi_perf_counter_t;

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

/** return the size of the view
*/
static inline size_t kaapi_memory_view_size( const kaapi_memory_view_t* kmv )
{
  switch (kmv->type) 
  {
    case KAAPI_MEMORY_VIEW_1D: return kmv->size[0]*kmv->wordsize;
    case KAAPI_MEMORY_VIEW_2D: return kmv->size[0]*kmv->size[1]*kmv->wordsize;
    default:
      kaapi_assert(0);
      break;
  }
  return 0;
}

/** Return non null value iff the view is contiguous
*/
static inline int kaapi_memory_view_iscontiguous( const kaapi_memory_view_t* kmv )
{
  switch (kmv->type) {
    case KAAPI_MEMORY_VIEW_1D: return 1;
    case KAAPI_MEMORY_VIEW_2D: return  kmv->lda == kmv->size[1]; /* row major storage */
    default:
      break;
  } 
  return 0;
}

static inline kaapi_memory_view_t kaapi_memory_view_make1d(size_t size, size_t wordsize)
{
  kaapi_memory_view_t retval;
  retval.type     = KAAPI_MEMORY_VIEW_1D;
  retval.size[0]  = size;
  retval.wordsize = wordsize;
  retval.size[1] = 0;
  retval.lda = 0;
  return retval;
}

static inline kaapi_memory_view_t kaapi_memory_view_make2d(size_t n, size_t m, size_t lda, size_t wordsize)
{
  kaapi_memory_view_t retval;
  retval.type     = KAAPI_MEMORY_VIEW_2D;
  retval.size[0]  = n;
  retval.size[1]  = m;
  retval.lda      = lda;
  retval.wordsize = wordsize;
  return retval;
}


/** \ingroup TASK
    Allocate a new format data
*/
extern struct kaapi_format_t* kaapi_format_allocate(void);

/** \ingroup TASK
    Register a format
*/
extern kaapi_format_id_t kaapi_format_register( 
        struct kaapi_format_t*      fmt,
        const char*                 name
);

/** \ingroup TASK
    Register a task format with static definition.
    \param fmt the kaapi_format_t object
    \param body the default body (if !=0) of the function to execute
    \param name the name of the format
    \param size the size in byte of the task arguments
    \param count the number of parameters
    \param mode_param, an array of size count given the access mode for each param
    \param offset_param, an array of size count given the offset of the data from the pointer to the argument of the task
    \param offset_version, an array of size count given the offset of the version (if any)
    \param offset_cwflag, an array of size count given the offset of the cw special flag (if any)
    \param fmt_param, an array of size count given the format of each param
    \param size_param, an array of size count given the size of each parameter.
*/
extern kaapi_format_id_t kaapi_format_taskregister_static( 
    struct kaapi_format_t*      fmt,
    kaapi_task_body_t           body,
    kaapi_task_body_t           bodywh,
    const char*                 name,
    size_t                      size,
    int                         count,
    const kaapi_access_mode_t   mode_param[],
    const kaapi_offset_t        offset_param[],
    const kaapi_offset_t        offset_version[],
    const struct kaapi_format_t*fmt_param[],
    const kaapi_memory_view_t   view_param[],
    const kaapi_reducor_t       reducor_param[],
    const kaapi_redinit_t       redinit_param[],
    const kaapi_task_binding_t* task_binding
);

/** \ingroup TASK
    Register a task format with dynamic definition
*/
extern kaapi_format_id_t kaapi_format_taskregister_func( 
    struct kaapi_format_t*        fmt, 
    kaapi_task_body_t             body,
    kaapi_task_body_t             bodywh,
    const char*                   name,
    size_t                        size,
    size_t                      (*get_count_params)(const struct kaapi_format_t*, const void*),
    kaapi_access_mode_t         (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*),
    void*                       (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*),
    kaapi_access_t              (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*),
    void                        (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*),
    const struct kaapi_format_t*(*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*),
    kaapi_memory_view_t         (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*),
    void                        (*set_view_param)  (const struct kaapi_format_t*, unsigned int, void*, const kaapi_memory_view_t*),
    void                        (*reducor )        (const struct kaapi_format_t*, unsigned int, void*, const void*),
    void                        (*redinit )        (const struct kaapi_format_t*, unsigned int, const void* sp, void* ),
    void                        (*get_task_binding)(const struct kaapi_format_t*, const void*, kaapi_task_binding_t*)
);

/** \ingroup TASK
    format accessor
*/
extern kaapi_access_mode_t kaapi_fmt_get_mode_param
(const struct kaapi_format_t* f, size_t i, const void* p);

/** \ingroup TASK
    format accessor
*/
extern void kaapi_fmt_set_access_param
(const struct kaapi_format_t* f, size_t i, void* p, const kaapi_access_t* a);

/** \ingroup TASK
    format accessor
*/
extern void* kaapi_fmt_get_off_param
(const struct kaapi_format_t* f, size_t i, const void* p);

/** \ingroup TASK
    Register a task format 
*/
extern void kaapi_format_set_task_body
(struct kaapi_format_t*, unsigned int, kaapi_task_body_t);

/** \ingroup TASK
    Register a task body into its format.
    A task may have multiple implementation this function specifies in 'archi'
    the target archicture for the body.
    
    Internally, in case of caching task, it may be necessary to call the
    entry of a task with parameter that does not directly points to a data,
    but to a handle that points to the data. 
    This body (bodywh) is automatically generated in the C++ interface.
*/
extern kaapi_task_body_t kaapi_format_taskregister_body( 
        struct kaapi_format_t*      fmt,
        kaapi_task_body_t           body,
        kaapi_task_body_t           bodywh, /* or 0 */
        int                         archi
);

/** \ingroup TASK
    Register a data structure format
*/
extern kaapi_format_id_t kaapi_format_structregister( 
        struct kaapi_format_t*    (*fmt_fnc)(void),
        const char*                 name,
        size_t                      size,
        void                       (*cstor)( void* ),
        void                       (*dstor)( void* ),
        void                       (*cstorcopy)( void*, const void*),
        void                       (*copy)( void*, const void*),
        void                       (*assign)( void*, const kaapi_memory_view_t*, const void*, const kaapi_memory_view_t*),
        void                       (*print)( FILE* file, const void* src)
);

/** \ingroup TASK
    Resolve a format data structure from the body of a task
*/
extern struct kaapi_format_t* kaapi_format_resolvebybody(kaapi_task_bodyid_t key);

/** \ingroup TASK
    Resolve a format data structure from the format identifier
*/
extern struct kaapi_format_t* kaapi_format_resolvebyfmit(kaapi_format_id_t key);

#define KAAPI_REGISTER_TASKFORMAT( formatobject, name, fnc_body, ... ) \
  static inline struct kaapi_format_t* formatobject(void) \
  {\
    static struct kaapi_format_t* formatobject##_object =0;\
    if (formatobject##_object==0) formatobject##_object = kaapi_format_allocate();\
    return formatobject##_object;\
  }\
  static inline void __attribute__ ((constructor)) __kaapi_register_format_##formatobject (void)\
  { \
    static int isinit = 0;\
    if (isinit) return;\
    isinit = 1;\
    kaapi_format_taskregister_static( formatobject(), fnc_body, 0, name, ##__VA_ARGS__,  0 /* for reduction operators not supported in C */, 0);\
  }


#define KAAPI_REGISTER_STRUCTFORMAT( formatobject, name, size, cstor, dstor, cstorcopy, copy, assign ) \
  static inline struct kaapi_format_t* fnc_formatobject(void) \
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


/* ========================================================================== */
/** \ingroup HWS
    initialize hierarchical workstealing. must be called only once globally.
    \retval -1 on error, 0 on success
 */
extern int kaapi_hws_init_global(void);
extern int kaapi_hws_fini_global(void);

/* ========================================================================== */
/** \ingroup HWS
    initialize hierarchical workstealing. must be called only once per kproc.
    \retval -1 on error, 0 on success
 */
extern int kaapi_hws_init_perproc(struct kaapi_processor_t*);
extern int kaapi_hws_fini_perproc(struct kaapi_processor_t*);

/* ========================================================================== */
/** \ingroup HWS
    hierarchy level identifiers and masks
 */

typedef enum kaapi_hws_levelid
{
  KAAPI_HWS_LEVELID_LO = -1,

  KAAPI_HWS_LEVELID_L3 = 0,
  KAAPI_HWS_LEVELID_NUMA,
  KAAPI_HWS_LEVELID_SOCKET,
  KAAPI_HWS_LEVELID_MACHINE,
  KAAPI_HWS_LEVELID_FLAT,
  KAAPI_HWS_LEVELID_MAX,

  KAAPI_HWS_LEVELID_FIRST = 0

} kaapi_hws_levelid_t;

typedef enum kaapi_hws_levelmask
{
  KAAPI_HWS_LEVELMASK_L3 = 1 << KAAPI_HWS_LEVELID_L3,
  KAAPI_HWS_LEVELMASK_NUMA = 1 << KAAPI_HWS_LEVELID_NUMA,
  KAAPI_HWS_LEVELMASK_SOCKET = 1 << KAAPI_HWS_LEVELID_SOCKET,
  KAAPI_HWS_LEVELMASK_MACHINE = 1 << KAAPI_HWS_LEVELID_MACHINE,
  KAAPI_HWS_LEVELMASK_FLAT = 1 << KAAPI_HWS_LEVELID_FLAT,

  KAAPI_HWS_LEVELMASK_ALL =
  KAAPI_HWS_LEVELMASK_L3 |
  KAAPI_HWS_LEVELMASK_NUMA |
  KAAPI_HWS_LEVELMASK_SOCKET |
  KAAPI_HWS_LEVELMASK_MACHINE |
  KAAPI_HWS_LEVELMASK_FLAT,

  KAAPI_HWS_LEVELMASK_INVALID = 0

} kaapi_hws_levelmask_t;

/* ========================================================================== */
/** \ingroup HWS
    push a task at a given hierarchy level
    \retval -1 on error, 0 on success
 */
extern int kaapi_hws_pushtask(kaapi_task_body_t, void*, kaapi_hws_levelid_t);

static inline int kaapi_hws_pushtask_flat(kaapi_task_body_t body, void* data)
{
  return kaapi_hws_pushtask(body, data, KAAPI_HWS_LEVELID_FLAT);
}

static inline int kaapi_hws_pushtask_machine(kaapi_task_body_t body, void* data)
{
  return kaapi_hws_pushtask(body, data, KAAPI_HWS_LEVELID_MACHINE);
}

static inline int kaapi_hws_pushtask_numa(kaapi_task_body_t body, void* data)
{
  return kaapi_hws_pushtask(body, data, KAAPI_HWS_LEVELID_NUMA);
}
/* ========================================================================== */
/** \ingroup HWS
    retrieve the memory node id for the given request
    \retval the node id
 */
extern unsigned int kaapi_hws_get_request_nodeid(const struct kaapi_request_t*);

/* ========================================================================== */
/** \ingroup HWS
    get the node count at the given hierarchy level
    \retval the node count
 */
extern unsigned int kaapi_hws_get_node_count(kaapi_hws_levelid_t);

/* ========================================================================== */
/** \ingroup HWS
    get the leaf count at the given hierarchy level
    \retval the leaf count
 */
extern unsigned int kaapi_hws_get_leaf_count(kaapi_hws_levelid_t);

/* ========================================================================== */
/** \ingroup HWS
    get splitter info related to HWS
    \retval -1 if this is not a HWS splitter. 0 otherwise.
 */
extern int kaapi_hws_get_splitter_info
(kaapi_stealcontext_t*, kaapi_hws_levelid_t*);
extern void kaapi_hws_clear_splitter_info
(kaapi_stealcontext_t*);

/* ========================================================================== */
/** \ingroup HWS
    adaptive task related routines
 */
extern void* kaapi_hws_init_adaptive_task
(
 kaapi_stealcontext_t*, kaapi_request_t*,
 kaapi_task_body_t, size_t,
 kaapi_task_splitter_t, kaapi_taskadaptive_result_t*
);

extern void kaapi_hws_reply_adaptive_task
(kaapi_stealcontext_t*, kaapi_request_t*);

extern void kaapi_hws_end_adaptive(kaapi_stealcontext_t* sc);

/* ========================================================================== */
/** \ingroup HWS
    wait until all the tasks pushed in the hierarchy queues are done
 */
extern void kaapi_hws_sched_sync(void);
extern void kaapi_hws_sched_sync_once(void);

/* ========================================================================== */
/** \ingroup HWS
    print counters
 */
void kaapi_hws_print_counters(void);


#ifdef __cplusplus
}
#endif

#endif /* _KAAPI_H */
