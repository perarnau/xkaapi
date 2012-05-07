/*
** kaapi.h
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

//#define KAAPI_VERBOSE 1

#define KAAPI_NO_CPU	    1	/* no CPU task executed */
//#define KAAPI_CUDA_NO_H2D   1	/* host to device copies disabled */
//#define KAAPI_CUDA_NO_D2H   1	/* device to host copies disabled */

#define __KAAPI__ 1
#define __KAAPI_MINOR__ 2

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
struct kaapi_taskadaptive_result_t;
struct kaapi_format_t;
struct kaapi_processor_t;
struct kaapi_request_t;
struct kaapi_reply_t;
struct kaapi_tasklist_t;
struct kaapi_listrequest_iterator_t;
struct kaapi_listrequest_t;

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
   \retval the current processor kid
*/
extern unsigned int kaapi_get_self_kid(void);

/* Get the current processor id. 
   \retval the current processor id
*/
extern unsigned int kaapi_get_self_cpu_id(void);

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
/** \ingroup HWS
    hierarchy level identifiers and masks
 */

/** TG: to describe here.
    - what about the order / value ? Do they impact the implementation ?
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


/** TG: to describe here
*/
typedef enum kaapi_hws_levelmask
{
  KAAPI_HWS_LEVELMASK_L3      = 1 << KAAPI_HWS_LEVELID_L3,
  KAAPI_HWS_LEVELMASK_NUMA    = 1 << KAAPI_HWS_LEVELID_NUMA,
  KAAPI_HWS_LEVELMASK_SOCKET  = 1 << KAAPI_HWS_LEVELID_SOCKET,
  KAAPI_HWS_LEVELMASK_MACHINE = 1 << KAAPI_HWS_LEVELID_MACHINE,
  KAAPI_HWS_LEVELMASK_FLAT    = 1 << KAAPI_HWS_LEVELID_FLAT,

  KAAPI_HWS_LEVELMASK_ALL     =
      KAAPI_HWS_LEVELMASK_L3 |
      KAAPI_HWS_LEVELMASK_NUMA |
      KAAPI_HWS_LEVELMASK_SOCKET |
      KAAPI_HWS_LEVELMASK_MACHINE |
      KAAPI_HWS_LEVELMASK_FLAT,

  KAAPI_HWS_LEVELMASK_INVALID = 0

} kaapi_hws_levelmask_t;


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
  KAAPI_ACCESS_MODE_S   = 64,       /* 0100 0000 : for Quark support: scratch mode */
  KAAPI_ACCESS_MODE_RW  = KAAPI_ACCESS_MODE_R|KAAPI_ACCESS_MODE_W,
  KAAPI_ACCESS_MODE_SCRATCH = KAAPI_ACCESS_MODE_S|KAAPI_ACCESS_MODE_V
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
/** Task priority
*/
#define KAAPI_TASK_MAX_PRIORITY     1
#define KAAPI_TASK_MIN_PRIORITY     0

#define KAAPI_TASK_STATE_SIGNALED   0x20   /* mask: extra flag to set the task as signaled for preemption */

/** Flag for task.
    \ingroup TASK
    Note that this is only for user level construction.
    Some values are exclusives (e.g. COOPERATIVE and CONCURRENT),
    and are only represented by one bit.
*/
typedef enum kaapi_task_flag_t {
  KAAPI_TASK_UNSTEALABLE      = 0x01,
  KAAPI_TASK_SPLITTABLE       = 0x02,
  KAAPI_TASK_S_CONCURRENT     = 0x04,
  KAAPI_TASK_S_COOPERATIVE    = 0x08,
  KAAPI_TASK_S_PREEMPTION     = 0x10,
  KAAPI_TASK_S_NOPREEMPTION   = 0x20,
  KAAPI_TASK_S_INITIALSPLIT   = 0x40,
  
  KAAPI_TASK_S_DEFAULT = 0
} kaapi_task_flag_t;


/** Kaapi task definition
    \ingroup TASK
    A Kaapi task is the basic unit of computation. It has a constant size including some task's specific values.
    Variable size task has to store pointer to the memory where found extra data.
    The body field is the pointer to the function to execute. The special value 0 correspond to a nop instruction.
*/
typedef struct kaapi_task_t {
  kaapi_task_body_t             body;      /** task body  */
  void*                         sp;        /** data stack pointer of the data frame for the task  */
  union { /* should be of size of uintptr */
    struct {
      kaapi_atomic8_t           state;     /** state of the task */
      uint8_t                   priority;  /** of the task */
      uint8_t                   ocr;       /** of the task */
      uint8_t                   flag;      /** scheduling information */
      /* ... */                            /** some bits are available on 64bits LP machine */
    } s;
    uintptr_t                   dummy;     /* to clear previous fields in one write */
  } u; 
  void* volatile                reserved;  /** reserved field for internal usage */
} kaapi_task_t __attribute__((aligned(8))); /* should be aligned on 64 bits boundary on Intel & Opteron */


static inline void kaapi_task_set_ocr_index(kaapi_task_t* task, uint8_t ith)
{ 
  kaapi_assert_debug( ith <255 );
  task->u.s.ocr = 1U+ith; 
}

static inline void kaapi_task_set_priority(kaapi_task_t* task, uint8_t prio)
{ 
  kaapi_assert_debug( prio <= KAAPI_TASK_MAX_PRIORITY );
  task->u.s.priority = prio; 
}


/* ========================================================================= */
/** Task splitter
    \ingroup TASK
    Deprecated type. This interface will be removed.
*/
struct kaapi_stealcontext_t;
typedef int (*kaapi_task_splitter_t)(
  struct kaapi_stealcontext_t* /*stc */, 
  int                          /*count*/, 
  struct kaapi_request_t*      /*array*/, 
  void*                        /*userarg*/
);

/* New type for splitter:
   - called in order to split a running or init task
   - return value is 0 in case of success call to the splitter.
   - return value is ECHILD if no work can be split again.
   This value mean also that all futures calls will failed to split
   work because the work set is empty (forever).
*/
typedef int (*kaapi_adaptivetask_splitter_t)(
  struct kaapi_task_t*                 /* task */,
  void*                                /* user arg */,
  struct kaapi_listrequest_t*          /* list of requests */, 
  struct kaapi_listrequest_iterator_t* /* iterator over the list*/
);


extern int kaapi_api_listrequest_iterator_count(
  struct kaapi_listrequest_iterator_t* lrrange
);
extern struct kaapi_request_t* kaapi_api_listrequest_iterator_get( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
);
extern struct kaapi_request_t* kaapi_api_listrequest_iterator_next( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
);
extern int kaapi_listrequest_api_iterator_empty( 
  struct kaapi_listrequest_iterator_t* lrrange
);

#if !defined(KAAPI_COMPILE_SOURCE)
static inline int kaapi_listrequest_iterator_count(
  struct kaapi_listrequest_iterator_t* lrrange
)
{ return kaapi_api_listrequest_iterator_count(lrrange); }

static inline struct kaapi_request_t* kaapi_listrequest_iterator_get( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
)
{ return kaapi_api_listrequest_iterator_get(lrequests, lrrange); }

static inline struct kaapi_request_t* kaapi_listrequest_iterator_next( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
)
{ return kaapi_api_listrequest_iterator_next( lrequests, lrrange ); }

static inline int kaapi_listrequest_iterator_empty( 
  struct kaapi_listrequest_iterator_t* lrrange
)
{ return kaapi_listrequest_api_iterator_empty( lrrange ); }
#endif

/** Reducer called on the victim side
    \ingroup TASK
*/
typedef int (*kaapi_victim_reducer_t)(
  struct kaapi_stealcontext_t*, 
  void*  /*arg_thief */, 
  void*  /* ?? */, 
  size_t /* ?? */, 
  void*  /* */
);

/** Reducer called on the thief side
    \ingroup TASK
*/
typedef int (*kaapi_thief_reducer_t)(
  struct kaapi_taskadaptive_result_t*, 
  void* arg_from_victim, 
  void*
);


/** \ingroup WS
    Request send by a processor.
    This data structure is pass in parameter of the splitter function.
    On return, the processor that emits the request will retreive task(s) in the frame.
*/
typedef struct kaapi_request_t {
  kaapi_atomic_t*               status;         /* request status */
  uintptr_t                     ident;          /* system wide id of the queue */
  kaapi_frame_t                 frame;          /* where to store theft tasks/data */
#if defined(KAAPI_USE_PERFCOUNTER)
  uintptr_t                     victim;         /* victim */
  uintptr_t                     serial;         /* serial number */
#endif
} __attribute__((aligned (KAAPI_CACHE_LINE))) kaapi_request_t;


/* ========================================================================= */
/** \ingroup DFG
    Kaapi access, public
*/
typedef struct kaapi_access_t {
  void*    data;    /* global data */
  void*    version; /* used to set the data to access (R/W/RW/CW) if steal, used to store output after steal */
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
    \param thread INOUT a pointer to the kaapi_thread_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_thread_toptask( kaapi_thread_t* thread) 
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp >= (char*)thread->sp_data);
  return thread->sp;
}

/** \ingroup TASK
    The function kaapi_thread_nexttask() is be used to return 
    the next top task of a task that is not yet pushed.
    This feature is required in order to prepare several tasks before pushing them
    and made them visible to other threads.
    If successful, the kaapi_thread_nexttask() function will return a pointer 
    to the next of next task to push.
    Otherwise, an 0 is returned to indicate the error.
    
    Warning: the function can only be called to compute the position of a task given
    a reference task that is not yet pushed. Once tasks are pushed, the function cannot
    be called.
    
    \param thread INOUT a pointer to the kaapi_thread_t data structure.
    \param task IN a pointer to a task
    \retval a pointer to the next task to push after task or 0.
*/
static inline kaapi_task_t* kaapi_thread_nexttask( kaapi_thread_t* thread, kaapi_task_t* task )
{ 
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)task >= (char*)thread->sp_data);
  return task-1;
}

/** \ingroup TASK
    The function kaapi_thread_pushtask() pushes the top task into the stack.
    If successful, the kaapi_thread_pushtask() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_thread_pushtask(kaapi_thread_t* thread)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)thread->sp > (char*)thread->sp_data);

  /* not need on X86 archi: write are ordered.
   */
#if !(defined(__x86_64) || defined(__i386__))
  kaapi_writemem_barrier();
#endif
  --thread->sp;
  return 0;
}

/** \ingroup TASK
    The function kaapi_thread_pushtask_adaptive() pushes the top task into the stack
    and set its flag to be splittable.
    If successful, the kaapi_thread_pushtask_adaptive() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_thread_pushtask_adaptive(
  kaapi_thread_t* thread,
  kaapi_adaptivetask_splitter_t user_splitter
);

/** \ingroup TASK
    The function kaapi_thread_pushtask() pushes the top task into the stack.
    If successful, the kaapi_thread_pushtask() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
static inline int kaapi_thread_push_packedtasks(kaapi_thread_t* thread, int count)
{
  kaapi_assert_debug( thread !=0 );
  kaapi_assert_debug((char*)(thread->sp-count) > (char*)thread->sp_data);

  /* not need on X86 archi: write are ordered.
   */
#if !(defined(__x86_64) || defined(__i386__))
  kaapi_writemem_barrier();
#endif
  thread->sp -= count;
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
    The function kaapi_thread_distribute_task() pushes the top task into the mailbox of processor kid.
    The task must be ready and does not have false dependencies (RAW or WAW) nor CW accesses.
    If the kid correspond to the current kprocessor then the call to kaapi_thread_distribute_task()
    is equivalent to the call of kaapi_thread_push().
    If successful, the kaapi_thread_distribute_task() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param stack INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_thread_distribute_task (
  kaapi_thread_t* thread,
  kaapi_processor_id_t kid
);

/** \ingroup TASK
    Task initialization routines
*/
static inline void kaapi_task_init
  ( kaapi_task_t* task, kaapi_task_bodyid_t body, void* arg ) 
{
  task->body      = body;
  task->sp        = arg;
  task->u.dummy   = 0;
  task->reserved  = 0;
}

__attribute__((deprecated))
static inline void kaapi_task_initdfg 
  (kaapi_task_t* task, kaapi_task_body_t body, void* arg)
{
  kaapi_task_init( task, body, arg );
}

/** \ingroup TASK
    Task initialization routines
*/
static inline void kaapi_task_init_withstate
  (kaapi_task_t* task, kaapi_task_body_t body, void* arg, uintptr_t state)
{
  task->body      = body;
  task->sp        = arg;
  task->u.dummy   = 0;
  KAAPI_ATOMIC_WRITE(&task->u.s.state, state);
  task->reserved = 0;
}

/** \ingroup TASK
    Task initialization routines
*/
extern void kaapi_task_init_with_flag
  (kaapi_task_t* task, kaapi_task_body_t body, void* arg, kaapi_task_flag_t flag);



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
typedef enum kaapi_stealcontext_flag_t {
  KAAPI_SC_CONCURRENT    = KAAPI_TASK_S_CONCURRENT,
  KAAPI_SC_COOPERATIVE   = KAAPI_TASK_S_COOPERATIVE,
  KAAPI_SC_PREEMPTION    = KAAPI_TASK_S_PREEMPTION,
  KAAPI_SC_NOPREEMPTION  = KAAPI_TASK_S_NOPREEMPTION,
  KAAPI_SC_INITIALSPLIT  = KAAPI_TASK_S_INITIALSPLIT,
  
  KAAPI_SC_DEFAULT = KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION
} kaapi_stealcontext_flag_t;

/** Begin adaptive section of code
*/
void* kaapi_task_begin_adaptive( 
  kaapi_thread_t*               thread,
  int                           flag,
  kaapi_adaptivetask_splitter_t splitter,
  void*                         argsplitter
);


/** \ingroup ADAPTIVE
    Mark the end of the adaptive section of code.
    After the call to this function, the runtime pushed task to wait completion
    of all thieves. This function is non blocking instruction.
    The caller that want to wait for real completion must call kaapi_sched_sync
    or derivative function.
    Atfer sync, all memory location produced in concurrency may be read 
    by the calling thread. Before synchronization, the runtime does not guarantee
    anything.
    \param context [IN] should be the context returned by kaapi_task_begin_adaptive
    \retval EAGAIN iff at least one thief was not preempted.
    \retval 0 iff all the thieves haved finished.
*/
extern int kaapi_task_end_adaptive( 
    kaapi_thread_t* thread,
    void*           context 
);

/** \ingroup ADAPTIVE
    The function kaapi_request_taskarg() return the pointer to the thief task arguments.
    If successful, the kaapi_request_taskarg() function will return a valid pointer.
    Otherwise, an 0 is returned to indicate the error.
    \param request INOUT a pointer to the kaapi_request_t data structure
    \retval a valid pointer to a memory region with at least size bytes or 0.
*/
static inline void* kaapi_request_pushdata( kaapi_request_t* request, uint32_t size)
{ return kaapi_thread_pushdata(&request->frame, size); }

/** \ingroup ADAPTIVE
    Macro that cast returned value of kaapi_request_taskarg to a typed pointer.
    If successful, the kaapi_request_taskargt() will return a valid pointer.
    Otherwise, an 0 is returned to indicate the error.
    \param request INOUT a pointer to the kaapi_request_t data structure
    \retval a valid pointer to a memory region with at least sizeof(type) bytes or 0.
*/
#define kaapi_request_taskargt(request,type) ((type*)kaapi_request_taskarg(request, sizeof(type)))

/** \ingroup ADAPTIVE
    The function kaapi_request_pushtask() pushes the top task into the request's frame.
    If successful, the kaapi_request_pushtask() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param thread INOUT a pointer to the kaapi_stack_t data structure.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_request_pushtask(kaapi_request_t* request, kaapi_task_t* victim_task);

/** \ingroup ADAPTIVE
    The function kaapi_request_toptask() will return the task to reply.
    Even if the function name means that several task may be pushed, the current implementation
    only accept ONE pushed task per request.
    If successful, the kaapi_thread_toptask() function will return a pointer to the next task to push.
    Otherwise, an 0 is returned to indicate the error.
    \param request INOUT a pointer to the kaapi_request_t data structure.
    \retval a pointer to the next task to push or 0.
*/
static inline kaapi_task_t* kaapi_request_toptask( kaapi_request_t* request )
{ return kaapi_thread_toptask(&request->frame); }

/** \ingroup ADAPTIVE
    The function kaapi_request_pushtask_adaptive() pushes the top adaptive task into the thief stack.
    The pushed task must be an adaptive task.
    If successful, the kaapi_request_pushtask_adaptive() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param request INOUT a pointer to the kaapi_request_t data structure.
    \param victim_task IN a pointer to the victim stack under splitting operation.
    \param headtail_flag IN either KAAPI_REQUEST_REPLY_HEAD or KAAPI_REQUEST_REPLY_TAIL
    \retval EINVAL invalid argument: bad request pointer.
*/
extern int kaapi_request_pushtask_adaptive(
  kaapi_request_t* request, 
  kaapi_task_t* victim_task, 
  kaapi_adaptivetask_splitter_t user_splitter,
  int headtail_flag
);

#define KAAPI_REQUEST_REPLY_HEAD 0x0
#define KAAPI_REQUEST_REPLY_TAIL 0x1

/** \ingroup ADAPTIVE
    push the task associated with an adaptive request
*/
static inline void kaapi_request_pushtask_adaptive_tail(
  kaapi_request_t* request, 
  kaapi_task_t* victim_task,
  kaapi_adaptivetask_splitter_t splitter
)
{
  /* sc the stolen stealcontext */
  kaapi_request_pushtask_adaptive(request, victim_task, splitter, KAAPI_REQUEST_REPLY_TAIL);
}

/** \ingroup ADAPTIVE
    push the task associated with an adaptive request
*/
static inline void kaapi_request_pushtask_adaptive_head(
  kaapi_request_t* request, 
  kaapi_task_t* victim_task,
  kaapi_adaptivetask_splitter_t splitter
)
{
  kaapi_request_pushtask_adaptive(request, victim_task, splitter, KAAPI_REQUEST_REPLY_HEAD);
}

/** \ingroup ADAPTIVE
    The function kaapi_request_committask() mades visible pushed tasks to the thief.
    If successful, the kaapi_request_committask() function will return zero.
    Otherwise, an error number will be returned to indicate the error.
    \param request INOUT a pointer to the kaapi_request_t data structure.
    \param victim_task IN a pointer to the victim stack under splitting operation.
    \param headtail_flag IN either KAAPI_REQUEST_REPLY_HEAD or KAAPI_REQUEST_REPLY_TAIL
    \retval EINVAL invalid argument: bad request pointer.
*/
extern int kaapi_request_committask(kaapi_request_t* request);

/* Return the thief result of the next thief from the head of the list to preempt or 0 if no thief may be preempted
*/
extern struct kaapi_thief_iterator_t* kaapi_thiefiterator_head( kaapi_task_t* adaptivetask );

/* Return the thief result of the next thief from the tail of the list to preempt or 0 if no thief may be preempted
*/
extern struct kaapi_thief_iterator_t* kaapi_thiefiterator_tail( kaapi_task_t* adaptivetask );

extern int kaapi_thiefiterator_equal( struct kaapi_thief_iterator_t* i1, struct kaapi_thief_iterator_t* i2 );

kaapi_task_t* kaapi_thiefiterator_get( struct kaapi_thief_iterator_t* pos );

extern struct kaapi_thief_iterator_t* kaapi_thiefiterator_next( struct kaapi_thief_iterator_t* pos );

extern struct kaapi_thief_iterator_t* kaapi_thiefiterator_prev( struct kaapi_thief_iterator_t* pos );


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
__attribute__((deprecated))
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
__attribute__((deprecated))
extern int kaapi_deallocate_thief_result( struct kaapi_taskadaptive_result_t* result );

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


/** Preempt a thief.
    Send a preemption request to the thief with result data structure ktr.
    And pass extra arguments (arg_to_thief).
    The call to the function returns 1 iff:
    - ktr has been preempted or finished
    - ktr has been replaced by the thieves of ktr into the list of stc
*/
#if 0
extern int kaapi_preempt_thief_helper( 
  kaapi_stealcontext_t*               stc, 
  struct kaapi_taskadaptive_result_t* ktr, 
  void*                               arg_to_thief 
);
#endif // #if 0

/** \ingroup ADAPTIVE
   Post a preemption request to thief. Do not wait preemption occurs.
   Return 0 iff some work have been preempted and should be processed locally.
   If the thief has already finished its computation bfore sending the signal,
   then the return value is ECHILD.
*/
extern int kaapi_preemptasync_thief( 
  struct kaapi_thief_iterator_t*     thief, 
  void*                              arg_to_thief 
);

/** The thief should have been preempted using preempasync_thief
    Returns 0 when the thief has reply to its preemption flag
*/
extern int kaapi_preemptasync_waitthief
( 
  struct kaapi_thief_iterator_t*     thief 
);

/** \ingroup ADAPTIVE
    Remove the thief ktr form the list of stc iff it is has finished its computation and returns 0.
    Else returns EBUSY.
*/
#if 0
extern int kaapi_remove_finishedthief( 
  kaapi_stealcontext_t*               stc, 
  struct kaapi_taskadaptive_result_t* ktr
);
#endif // #if 0


/** \ingroup ADAPTIVE
   Try to preempt the thief referenced by tr. Wait either preemption occurs or the end of the thief.
   Once the thief has received the preempt and send back result to the victim who preempt it, then
   the function reducer is called.
   Return value is the return value of the reducer function or 0 if no reducer is given.
      
   The reducer function should has the following signature:
      int (*)( stc, void* thief_arg, void* thief_result, size_t thief_ressize, ... )
   where ... is the same extra arguments passed to kaapi_preempt_nextthief.
*/
#if 0
__attribute__((deprecated))
static inline int kaapi_preempt_thief(
  kaapi_stealcontext_t* sc,
  struct kaapi_taskadaptive_result_t* ktr,
  void* thief_arg,
  kaapi_victim_reducer_t reducer,
  void* reducer_arg
)
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
  return 0;
}
#endif // #if 0

/** \ingroup ADAPTIVE
    Test if the current execution should process preemt request into the task
    and then pass arg_victim argument to the victim and return !=0 value
    \retval !=0 if it exists a prending preempt request(s) to process onto the given task.
    \retval 0 else
*/
static inline int kaapi_preemptpoint_isactive(const kaapi_task_t* task)
{
  return KAAPI_ATOMIC_READ(&task->u.s.state) & KAAPI_TASK_STATE_SIGNALED; 
}


/** \ingroup ADAPTIVE
    Test if the current execution is preempted.
    \retval !=0 if it exists a prending preempt request(s) on the current thread
    \retval 0 else
*/
extern int kaapi_thread_is_preempted(void);


/** \ingroup ADAPTIVE
    Helper function to pass argument between the victim and the thief.
    On return the victim argument may be read.
*/
#if 0
extern int kaapi_preemptpoint_before_reducer_call( 
    kaapi_stealcontext_t* stc,
    void* arg_for_victim, 
    void* result_data, 
    size_t result_size
);
extern int kaapi_preemptpoint_after_reducer_call ( 
    kaapi_stealcontext_t* stc
);
#endif // #if 0

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
#if 0
__attribute__((deprecated))
static inline int kaapi_preemptpoint
(kaapi_stealcontext_t* sc,
 kaapi_thief_reducer_t reducer,
 void* victim_arg,
 void* result_data,
 size_t result_size,
 void* reducer_arg)
{
  int res = 0;
#if 0

  if (kaapi_preemptpoint_isactive(sc)) /* unlikely */
  {
    res = 1; /* tell the thief about reduction */

    kaapi_preemptpoint_before_reducer_call
      (sc, victim_arg, result_data, result_size);
    if (reducer)
      reducer(sc->header.ktr, sc->header.ktr->arg_from_victim, reducer_arg);

    kaapi_preemptpoint_after_reducer_call(sc);
  }
#endif
  return res;
}
#endif // #if 0

/** Begin critical section with respect to steal operation
    \ingroup TASK
*/
#if 0
__attribute__((deprecated))
extern int kaapi_steal_begincritical( kaapi_stealcontext_t* sc );

/** End critical section with respect to steal operation
    \ingroup TASK
*/
__attribute__((deprecated))
extern int kaapi_steal_endcritical( kaapi_stealcontext_t* sc );

/** Same as kaapi_steal_endcritical but stealing left disabled
    \ingroup TASK
\THIERRY    
*/
__attribute__((deprecated))
extern int kaapi_steal_endcritical_disabled( kaapi_stealcontext_t* sc );

/** \ingroup ADAPTIVE
    Signal end of a thief, required to be call if kaapi_steal_finalize is not call in order
    to ensure end of the computation.
*/
extern int kaapi_steal_thiefreturn( kaapi_stealcontext_t* stc );
#endif // #if 0

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
#if defined(KAAPI_USE_CUDA)
#include <cuda_runtime_api.h>
#endif
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

extern int kaapi_memory_synchronize_pointer( void * );

/* Register memory for Xkaapi optimizations */
extern int kaapi_memory_register( void* ptr, kaapi_memory_view_t view );

/* Register memory for Xkaapi optimizations */
extern void kaapi_memory_unregister( void* ptr );

#if defined(KAAPI_USE_CUDA)
extern int
kaapi_cuda_proc_sync_all( void );
#endif

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

extern void kaapi_staticschedtask_body( void*, kaapi_thread_t*, kaapi_task_t* pc );

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
/* Original type */
typedef long kaapi_workqueue_index_t;

/* For OMP support of ull loop */
typedef unsigned long long kaapi_workqueue_index_ull_t;

typedef struct {
  union {
    struct {
      volatile kaapi_workqueue_index_t end __attribute__((aligned(64)));
      volatile kaapi_workqueue_index_t beg __attribute__((aligned(64))); /* cache line */
    } li;
    struct {
      volatile kaapi_workqueue_index_ull_t end __attribute__((aligned(64)));
      volatile kaapi_workqueue_index_ull_t beg __attribute__((aligned(64))); /* cache line */   
    } ull;
  } rep;
  kaapi_lock_t*                    lock;
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
    Attach the workqueue to the current kprocessor, ie the lock to ensure consistent concurrent operation
    is the lock of the current kprocessor.
*/
extern int kaapi_workqueue_init_ull( 
    kaapi_workqueue_t* kwq, 
    kaapi_workqueue_index_ull_t b, 
    kaapi_workqueue_index_ull_t e 
);

/** Initialize the workqueue to be an empty (null) range workqueue.
    Do memory barrier before updating the queue.
    Explicit specification of the lock to used to ensure consistent concurrent operations.
*/
extern int kaapi_workqueue_init_with_lock( 
    kaapi_workqueue_t* kwq, 
    kaapi_workqueue_index_t b, 
    kaapi_workqueue_index_t e, 
    kaapi_lock_t* thelock 
);

/** Initialize the workqueue to be an empty (null) range workqueue.
    Do memory barrier before updating the queue.
    Explicit specification of the lock to used to ensure consistent concurrent operations.
*/
extern int kaapi_workqueue_init_with_lock_ull( 
    kaapi_workqueue_t* kwq, 
    kaapi_workqueue_index_ull_t b, 
    kaapi_workqueue_index_ull_t e, 
    kaapi_lock_t* thelock 
);

/** destroy 
*/
static inline int kaapi_workqueue_destroy( kaapi_workqueue_t* kwq )
{
  return 0;
}

/** This function set new bounds for the workqueue.
    There is no guarantee on this function with respect to concurrent thieves.
    The caller must ensure atomic update by surrounding the call to
    kaapi_workqueue_reset by kaapi_workqueue_lock/kaapi_workqueue_unlock
    \retval 0 in case of success
    \retval else an error code
*/
static inline int kaapi_workqueue_reset( 
  kaapi_workqueue_t*      kwq, 
  kaapi_workqueue_index_t beg, 
  kaapi_workqueue_index_t end
)
{
  kaapi_assert_debug( beg <= end );
  
  /* may be not thread save ! */
  kwq->rep.li.beg = beg;
  kwq->rep.li.end = end;
  return 0;  
}

/** This function set new bounds for the workqueue.
    There is no guarantee on this function with respect to concurrent thieves.
    The caller must ensure atomic update by surrounding the call to
    kaapi_workqueue_reset by kaapi_workqueue_lock/kaapi_workqueue_unlock
    \retval 0 in case of success
    \retval else an error code
*/
static inline int kaapi_workqueue_reset_ull( 
  kaapi_workqueue_t*          kwq, 
  kaapi_workqueue_index_ull_t beg, 
  kaapi_workqueue_index_ull_t end
)
{
  kaapi_assert_debug( beg <= end );
  
  /* may be not thread save ! */
  kwq->rep.ull.beg = beg;
  kwq->rep.ull.end = end;
  return 0;  
}


/* deprecated: */
__attribute__((deprecated))
extern int kaapi_workqueue_set( 
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t beg, 
  kaapi_workqueue_index_t end
);


/**
*/
static inline kaapi_workqueue_index_t 
  kaapi_workqueue_range_begin( kaapi_workqueue_t* kwq )
{
  return kwq->rep.li.beg;
}

/**
*/
static inline kaapi_workqueue_index_ull_t 
  kaapi_workqueue_range_begin_ull( kaapi_workqueue_t* kwq )
{
  return kwq->rep.ull.beg;
}

/**
*/
static inline kaapi_workqueue_index_t 
  kaapi_workqueue_range_end( kaapi_workqueue_t* kwq )
{
  return kwq->rep.li.end;
}

/**
*/
static inline kaapi_workqueue_index_ull_t 
  kaapi_workqueue_range_end_ull( kaapi_workqueue_t* kwq )
{
  return kwq->rep.li.end;
}

/**
*/
static inline kaapi_workqueue_index_t 
  kaapi_workqueue_size( kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_t size = kwq->rep.li.end - kwq->rep.li.beg;
  return (size <0 ? 0 : size);
}

/**
*/
static inline kaapi_workqueue_index_ull_t 
  kaapi_workqueue_size_ull( kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_ull_t b = kwq->rep.ull.beg;
  kaapi_workqueue_index_ull_t e = kwq->rep.ull.end;
  if (e <= b) return 0;
  return e - b;
}

/**
*/
static inline unsigned int kaapi_workqueue_isempty( const kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_t size = kwq->rep.li.end - kwq->rep.li.beg;
  return size <= 0;
}

/**
*/
static inline unsigned int kaapi_workqueue_isempty_ull( const kaapi_workqueue_t* kwq )
{
  kaapi_workqueue_index_t size = kwq->rep.ull.end - kwq->rep.ull.beg;
  return size <= 0;
}

/** This function should be called by the current kaapi thread that own the workqueue.
    The function pushes work into the workqueue.
    Assuming that before the call, the workqueue is [beg,end).
    After the successful call to the function the workqueu becomes [newbeg,end).
    newbeg is assumed to be less than beg. Else it is a pop operation, 
    see kaapi_workqueue_pop.
    Return 0 in case of success 
    Return EINVAL if invalid arguments
*/
static inline int kaapi_workqueue_push(
  kaapi_workqueue_t*      kwq, 
  kaapi_workqueue_index_t newbeg
)
{
  if ( kwq->rep.li.beg  > newbeg )
  {
    kaapi_mem_barrier();
    kwq->rep.li.beg = newbeg;
    return 0;
  }
  return EINVAL;
}

/** This function should be called by the current kaapi thread that own the workqueue.
    The function pushes work into the workqueue.
    Assuming that before the call, the workqueue is [beg,end).
    After the successful call to the function the workqueu becomes [newbeg,end).
    newbeg is assumed to be less than beg. Else it is a pop operation, 
    see kaapi_workqueue_pop.
    Return 0 in case of success 
    Return EINVAL if invalid arguments
*/
static inline int kaapi_workqueue_push_ull(
  kaapi_workqueue_t*          kwq, 
  kaapi_workqueue_index_ull_t newbeg
)
{
  if ( kwq->rep.ull.beg  > newbeg )
  {
    kaapi_mem_barrier();
    kwq->rep.ull.beg = newbeg;
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

/** Helper function called in case of conflict.
    Return EBUSY is the queue is empty.
    Return EINVAL if invalid arguments
    Return ESRCH if the current thread is not a kaapi thread.
*/
extern int kaapi_workqueue_slowpop_ull(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_ull_t* beg,
  kaapi_workqueue_index_ull_t* end,
  kaapi_workqueue_index_ull_t  size
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
  kaapi_workqueue_index_t loc_init;
  kaapi_assert_debug( max_size >0 );

  loc_beg = kwq->rep.li.beg;
  loc_init = loc_beg;
  loc_beg += max_size;
  kwq->rep.li.beg = loc_beg;
  kaapi_mem_barrier();

  if (loc_beg < kwq->rep.li.end)
  {
    /* no conflict */
    *end = loc_beg;
    *beg = loc_beg - max_size;
    return 0;
  }

  /* conflict */
  kwq->rep.li.beg = loc_init;
  return kaapi_workqueue_slowpop(kwq, beg, end, max_size);
}


/** This function should be called by the current kaapi thread that own the workqueue.
    Return 0 in case of success 
    Return EBUSY is the queue is empty
    Return EINVAL if invalid arguments
    Return ESRCH if the current thread is not a kaapi thread.
*/
static inline int kaapi_workqueue_pop_ull(
  kaapi_workqueue_t*           kwq, 
  kaapi_workqueue_index_ull_t* beg,
  kaapi_workqueue_index_ull_t* end,
  kaapi_workqueue_index_ull_t  max_size
)
{
  kaapi_workqueue_index_ull_t loc_beg;
  kaapi_workqueue_index_ull_t loc_init;

  loc_beg = kwq->rep.ull.beg;
  loc_init = loc_beg;
  loc_beg += max_size;
  kwq->rep.ull.beg = loc_beg;
  kaapi_mem_barrier();

  if (loc_beg < kwq->rep.ull.end)
  {
    /* no conflict */
    *end = loc_beg;
    *beg = loc_beg - max_size;
    return 0;
  }

  /* conflict */
  kwq->rep.ull.beg = loc_init;
  return kaapi_workqueue_slowpop_ull(kwq, beg, end, max_size);
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
  kaapi_workqueue_index_t loc_init;

  kaapi_assert_debug( 0 < size );
  kaapi_assert_debug( kaapi_atomic_assertlocked(kwq->lock) );

  /* disable gcc warning */
  *beg = 0;
  *end = 0;

  loc_end  = kwq->rep.li.end;
  loc_init = loc_end;
  loc_end -= size;
  kwq->rep.li.end = loc_end;
  kaapi_mem_barrier();

  if (loc_end < kwq->rep.li.beg)
  {
    kwq->rep.li.end = loc_init;
    return ERANGE; /* false */
  }

  *beg = loc_end;
  *end = *beg + size;
  
  return 0; /* true */
}  


/** This function should only be called into a splitter to ensure correctness
    the lock of the victim kprocessor is assumed to be locked to handle conflict.
    Return 0 in case of success 
    Return ERANGE if the queue is empty or less than requested size.
 */
static inline int kaapi_workqueue_steal_ull(
  kaapi_workqueue_t*           kwq, 
  kaapi_workqueue_index_ull_t* beg,
  kaapi_workqueue_index_ull_t* end,
  kaapi_workqueue_index_ull_t  size
)
{
  kaapi_workqueue_index_ull_t loc_end;
  kaapi_workqueue_index_ull_t loc_init;

  kaapi_assert_debug( kaapi_atomic_assertlocked(kwq->lock) );

  /* disable gcc warning */
  *beg = 0;
  *end = 0;

  loc_end  = kwq->rep.ull.end;
  if (loc_end < size) 
    return ERANGE;
  loc_init = loc_end;
  loc_end -= size;
  kwq->rep.ull.end = loc_end;
  kaapi_mem_barrier();

  if (loc_end < kwq->rep.ull.beg)
  {
    kwq->rep.ull.end = loc_init;
    return ERANGE; /* false */
  }

  *beg = loc_end;
  *end = *beg + size;
  
  return 0; /* true */
}  


/** Lock the workqueue
*/
extern int kaapi_workqueue_lock( 
    kaapi_workqueue_t* kwq
);

/** Unlock the workqueue
*/
extern int kaapi_workqueue_unlock( 
    kaapi_workqueue_t* kwq
);



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

#define KAAPI_PERF_ID(U, I)  (KAAPI_PERF_ID_ ## I | (U) << KAAPI_PERF_ID_USER_POS)
#define KAAPI_PERF_ID_USER(I) KAAPI_PERF_ID(1, I)
#define KAAPI_PERF_ID_PRIV(I) KAAPI_PERF_ID(0, I)


#define KAAPI_PERF_ID_TASKS         0  /* count number of executed tasks */
#define KAAPI_PERF_ID_STEALREQOK    1  /* count number of successful steal requests */
#define KAAPI_PERF_ID_STEALREQ      2  /* count number of steal requests emitted */
#define KAAPI_PERF_ID_STEALOP       3  /* count number of steal operation to reply to requests */
#define KAAPI_PERF_ID_STEALIN       4  /* count number of receive steal requests */

#define KAAPI_PERF_ID_SUSPEND       5  /* count number of suspended thread */
#define KAAPI_PERF_ID_T1            6  /* nano second of compte time */

#define KAAPI_PERF_ID_TPREEMPT      7  /* nano second of preempt time */
#define KAAPI_PERF_ID_ALLOCTHREAD   8  /* count number of allocated thread */
#define KAAPI_PERF_ID_FREETHREAD    9  /* count number of free thread */
#define KAAPI_PERF_ID_QUEUETHREAD   10 /* count the maximal number of thread in queue */
#define KAAPI_PERF_ID_TASKLISTCALC  11 /* tick to compute task lists in ns */

#define KAAPI_PERF_ID_ENDSOFTWARE   14 /* mark end of software counters */

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

/* */
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
  retval.size[1]  = 0;
  retval.lda      = 0;
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
//DEPRECATED
  void*                       (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*),
  kaapi_access_t              (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*),
  void                        (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*),
  const struct kaapi_format_t*(*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*),
  kaapi_memory_view_t         (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*),
  void                        (*set_view_param)  (const struct kaapi_format_t*, unsigned int, void*, const kaapi_memory_view_t*),
  void                        (*reducor )        (const struct kaapi_format_t*, unsigned int, void*, const void*),
  void                        (*redinit )        (const struct kaapi_format_t*, unsigned int, const void* sp, void* ),
  void                        (*get_task_binding)(const struct kaapi_format_t*, const void*, kaapi_task_binding_t*),
  kaapi_adaptivetask_splitter_t	(*get_splitter)(const struct kaapi_format_t*, const void*)
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
DEPRECATED
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
    kaapi_format_taskregister_static( formatobject(), fnc_body, 0, name, ##__VA_ARGS__,  0, 0 /* for reduction operators not supported in C */,0);\
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


#ifdef __cplusplus
}
#endif

#endif /* _KAAPI_H */
