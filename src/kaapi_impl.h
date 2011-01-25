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
#include "kaapi_memory.h"
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

#  define KAAPI_DEBUG_INST(inst) inst
#else
#  define kaapi_assert_debug_m(cond, msg)
#  define KAAPI_LOG(l, fmt, ...) 
#  define KAAPI_DEBUG_INST(inst)
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

/** Finalize the machine level runtime.
    Return 0 in case of success. Else an error code.
*/
extern int kaapi_mt_finalize(void);

/** Initialize hw topo.
    Based on hwloc library.
    Return 0 in case of success else an error code
*/
extern int kaapi_hw_init();

/* Fwd declaration 
*/
struct kaapi_listrequest_t;
struct kaapi_procinfo_list_t;

/* Fwd declaration
*/
struct kaapi_tasklist_t;
struct kaapi_comlink_t;

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


/* =======vvvvvvvvvvvvvvvvvv===================== Default parameters ============================ */
/** Initialise default formats
*/
extern void kaapi_init_basicformat(void);

/**
*/
enum kaapi_memory_type_t {
  KAAPI_MEM_NODE,
  KAAPI_MEM_CACHE
};

/**
*/
typedef uint64_t kaapi_cpuset_t[2];

/**
*/
typedef struct kaapi_affinityset_t {
    kaapi_cpuset_t        who;
    size_t                ncpu;
    size_t                mem_size;
    short                 type;
} kaapi_affinityset_t;

/**
*/
typedef struct kaapi_hierarchy_one_level_t {
  unsigned short          count;           /* number of kaapi_affinityset_t at this level */
  kaapi_affinityset_t*    affinity;  
} kaapi_hierarchy_one_level_t;

/** Memory hierarchy of the local machine
    * memory.depth: depth of the hierarchy
    * memory.levels[i].count: number of kind of memory at level i
    * memory.levels[i].who[k]: cpu set of which PU is contains by memory k at level i
    * memory.levels[i].size[k]: size of the k memory  at level i
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
  unsigned int		         use_affinity;                    /* use cpu affinity */
  int                      display_perfcounter;             /* set to 1 iff KAAPI_DISPLAY_PERF */
  uint64_t                 startuptime;                     /* time at the end of kaapi_init */

  struct kaapi_procinfo_list_t* kproc_list;                 /* list of kprocessors to initialized */
  kaapi_cpuset_t           usedcpu;                         /* cpuset of used physical ressources */
  kaapi_hierarchy_t        memory;                          /* memory hierarchy */
  unsigned int	           *kid2cpu;                        /* mapping: kid->phys cpu  */
  unsigned int  	         *cpu2kid;                        /* mapping: phys cpu -> kid */
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

typedef enum kaapi_format_flag_t {
  KAAPI_FORMAT_STATIC_FIELD,   /* the format of the task is interpreted using static offset/fmt etc */ 
  KAAPI_FORMAT_DYNAMIC_FIELD      /* the format is interpreted using the function, required for variable args tasks */
} kaapi_format_flag_t;

/** \ingroup TASK
    Kaapi task format
    The format should be 1/ declared 2/ register before any use in task.
    The format object is only used in order to interpret stack of task.    
*/
typedef struct kaapi_format_t {
  kaapi_format_id_t          fmtid;                                   /* identifier of the format */
  short                      isinit;                                  /* ==1 iff initialize */
  const char*                name;                                    /* debug information */
  
  /* flag to indicate how to interpret the following fields */
  kaapi_format_flag_t        flag;
  
  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_STATIC_FIELD */
  uint32_t                   size;                                    /* sizeof the object */  
  void                       (*cstor)( void* dest);
  void                       (*dstor)( void* dest);
  void                       (*cstorcopy)( void* dest, const void* src);
  void                       (*copy)( void* dest, const void* src);
  void                       (*assign)( void* dest, const void* src);
  void                       (*print)( FILE* file, const void* src);

  /* only if it is a format of a task  */
  kaapi_task_body_t          default_body;                            /* iff a task used on current node */
  kaapi_task_body_t          entrypoint[KAAPI_PROC_TYPE_MAX];         /* maximum architecture considered in the configuration */

  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_STATIC_FIELD */
  int                         _count_params;                           /* number of parameters */
  kaapi_access_mode_t        *_mode_params;                            /* only consider value with mask 0xF0 */
  kaapi_offset_t             *_off_params;                             /* access to the i-th parameter: a value or a shared */
  kaapi_offset_t             *_off_versions;                           /* access to the i-th parameter: a value or a shared */
  struct kaapi_format_t*     *_fmt_params;                             /* format for each params */
  kaapi_memory_view_t        *_view_params;                            /* sizeof of each params */
  kaapi_reducor_t            *_reducor_params;                         /* array of reducor in case of cw */

  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_FUNC_FIELD
     - the unsigned int argument is the index of the parameter 
     - the last argument is the pointer to the sp data of the task
  */
  size_t                (*get_count_params)(const struct kaapi_format_t*, const void*);
  kaapi_access_mode_t   (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*);
  void*                 (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*);
  kaapi_access_t        (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*);
  void                  (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*);
  const struct kaapi_format_t*(*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*);
  kaapi_memory_view_t   (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*);
  void                  (*reducor )        (const struct kaapi_format_t*, unsigned int, const void*, void*, const void*);

  /* fields to link the format is the internal tables */
  struct kaapi_format_t      *next_bybody;                            /* link in hash table */
  struct kaapi_format_t      *next_byfmtid;                           /* link in hash table */

  
  /* only for Monotonic bound format */
  int    (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                      const void* value, const struct kaapi_format_t* fmtvalue );
} kaapi_format_t;



/* Helper to interpret the format 
*/
static inline 
size_t                kaapi_format_get_count_params(const struct kaapi_format_t* fmt, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_count_params;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_count_params)(fmt, sp);
}

static inline 
kaapi_access_mode_t   kaapi_format_get_mode_param (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_mode_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_mode_param)(fmt, ith, sp);
}

static inline 
void*                 kaapi_format_get_data_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) == KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_off_params[ith] + (char*)sp;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_off_param)(fmt, ith, sp);
}

static inline 
kaapi_access_t         kaapi_format_get_access_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) != KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) {
    kaapi_access_t retval;
    retval.data    = *(void**)(fmt->_off_params[ith] + (char*)sp);
    retval.version = *(void**)(fmt->_off_versions[ith] + (char*)sp);
    return retval;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_access_param)(fmt, ith, sp);
}

static inline 
void         kaapi_format_set_access_param  (const struct kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_access_t* a)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) != KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) {
    *(void**)(fmt->_off_params[ith] + (char*)sp) = a->data;
    *(void**)(fmt->_off_versions[ith] + (char*)sp) = a->version;
    return;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->set_access_param)(fmt, ith, sp, a);
}

static inline 
const struct kaapi_format_t* kaapi_format_get_fmt_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_fmt_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_fmt_param)(fmt, ith, sp);
}

static inline 
kaapi_memory_view_t kaapi_format_get_view_param (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_view_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_view_param)(fmt, ith, sp);
}


static inline 
void          kaapi_format_reduce_param (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp, void* result, const void* value)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    (*fmt->_reducor_params[ith])(result, value);
    return;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->reducor)(fmt, ith, sp, result, value);
}



/* ============================= Helper for bloc allocation of individual entries ============================ */
/*
*/
#define KAAPI_BLOCENTRIES_SIZE 32
#define KAAPI_BLOCALLOCATOR_SIZE 4096

/*
*/
#define KAAPI_DECLARE_BLOCENTRIES(NAME, TYPE) \
typedef struct NAME {\
  TYPE         data[KAAPI_BLOCENTRIES_SIZE]; \
  uintptr_t    pos;  /* next free in data */\
  struct NAME* next; /* link list of bloc */\
  void*        ptr;  /* memory pointer of allocated bloc */\
} NAME


/* Macro to define a generic bloc allocator of byte.
*/
typedef struct kaapi_allocator_bloc_t {
  double                           data[KAAPI_BLOCALLOCATOR_SIZE/sizeof(double)-sizeof(uintptr_t)-sizeof(struct kaapi_allocator_bloc_t*)];
  uintptr_t                        pos;  /* next free in data */
  struct kaapi_allocator_bloc_t*   next; /* link list of bloc */
} kaapi_allocator_bloc_t;

typedef struct kaapi_allocator_t {
  kaapi_allocator_bloc_t* currentbloc;
  kaapi_allocator_bloc_t* allocatedbloc;
} kaapi_allocator_t;

#define KAAPI_DECLARE_GENBLOCENTRIES(ALLOCATORNAME) \
  typedef kaapi_allocator_t ALLOCATORNAME

/**/
static inline int kaapi_allocator_init( kaapi_allocator_t* va ) 
{
  va->allocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/**/
static inline int kaapi_allocator_destroy( kaapi_allocator_t* va )
{
  while (va->allocatedbloc !=0)
  {
    kaapi_allocator_bloc_t* curr = va->allocatedbloc;
    va->allocatedbloc = curr->next;
    free (curr);
  }
  va->allocatedbloc = 0;
  va->currentbloc = 0;
  return 0;
}

/**/
static inline void* kaapi_allocator_allocate( kaapi_allocator_t* va, size_t size )
{
  void* entry;
  /* round size to double size */
  size = (size+sizeof(double)-1)/sizeof(double);
  const size_t sz_max = KAAPI_BLOCALLOCATOR_SIZE/sizeof(double)-sizeof(uintptr_t)-sizeof(kaapi_allocator_bloc_t*);
  if ((va->currentbloc == 0) || (va->currentbloc->pos + size >= sz_max))
  {
    kaapi_assert_debug( sizeof(kaapi_allocator_bloc_t) > size );
    va->currentbloc = (kaapi_allocator_bloc_t*)malloc( sizeof(kaapi_allocator_bloc_t) );
    va->currentbloc->next = va->allocatedbloc;
    va->allocatedbloc = va->currentbloc;
    va->currentbloc->pos = 0;
  }
  
  entry = &va->currentbloc->data[va->currentbloc->pos];
  va->currentbloc->pos += size;
  KAAPI_DEBUG_INST( memset( entry, 0, size*sizeof(double) ) );
  return entry;
}





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

  /* the way to execute task inside a thread, if ==0 uses kaapi_thread_execframe */
  int (*execframe)( struct kaapi_thread_context_t* thread );
  kaapi_threadgroup_t            the_thgrp;      /* not null iff execframe != kaapi_thread_execframe */
  struct kaapi_tasklist_t*       readytasklist;  /* Not null -> list of ready task, see static_sched.h */
  struct kaapi_comlink_t*        list_send;      /* send and recv list of comsend or comrecv descriptor */
  struct kaapi_comlink_t*        list_recv;
  int                            unstealable;    /* !=0 -> cannot be stolen */
  int                            partid;         /* used by static scheduling to identify the thread in the group */

  struct kaapi_thread_context_t* _next;          /** to be linkable either in proc->lfree or proc->lready */
  struct kaapi_thread_context_t* _prev;          /** to be linkable either in proc->lfree or proc->lready */

#if defined(KAAPI_USE_CUDA)
  kaapi_atomic_t                 lock;           /** */ 
#endif

  kaapi_cpuset_t                 affinity;       /* bit i == 1 -> can run on procid i */

  void*                          alloc_ptr;      /** pointer really allocated */
  uint32_t                       size;           /** size of the data structure allocated */
  kaapi_task_t*                  task;           /** bottom of the stack of task */

  struct kaapi_wsqueuectxt_cell_t* wcs;          /** point to the cell in the suspended list, iff thread is suspended */

  /* statically allocated reply */
  kaapi_reply_t			             static_reply;
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
#  define KAAPI_MASK_BODY_TERM    (0x1)
#  define KAAPI_MASK_BODY_PREEMPT (0x2) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2)
#  define KAAPI_MASK_BODY_EXEC    (0x4)
#  define KAAPI_MASK_BODY_STEAL   (0x8)


#  define KAAPI_TASK_ATOMIC_OR(a, v) KAAPI_ATOMIC_OR_ORIG(a, v)

#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)a, (uint32_t)o, (uint32_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR_ORIG( (kaapi_atomic_t*)a, (uint32_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND_ORIG( (kaapi_atomic_t*)a, (uint32_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic_t*)a, (uint32_t)v)

#elif (SIZEOF_VOIDP == 8)
#  define KAAPI_MASK_BODY_TERM    (0x1UL << 60UL)
#  define KAAPI_MASK_BODY_PREEMPT (0x2UL << 60UL) /* must be different from term */
#  define KAAPI_MASK_BODY_AFTER   (0x2UL << 60UL)
#  define KAAPI_MASK_BODY_EXEC    (0x4UL << 60UL)
#  define KAAPI_MASK_BODY_STEAL   (0x8UL << 60UL)
#  define KAAPI_MASK_BODY         (0xFUL << 60UL)
#  define KAAPI_MASK_BODY_SHIFTR   60UL
#  define KAAPI_TASK_ATOMIC_OR(a, v) KAAPI_ATOMIC_OR64_ORIG(a, v)

#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS64( (kaapi_atomic64_t*)(a), (uint64_t)o, (uint64_t)n )
#  define KAAPI_ATOMIC_ORPTR_ORIG(a, v) \
    KAAPI_ATOMIC_OR64_ORIG( (kaapi_atomic64_t*)(a), (uint64_t)v)
#  define KAAPI_ATOMIC_ANDPTR_ORIG(a, v) \
    KAAPI_ATOMIC_AND64_ORIG( (kaapi_atomic64_t*)(a), (uint64_t)v)
#  define KAAPI_ATOMIC_WRITEPTR_BARRIER(a, v) \
    KAAPI_ATOMIC_WRITE_BARRIER( (kaapi_atomic64_t*)a, (uint64_t)v)

#else
#  error "No implementation for pointer to function with size greather than 8 bytes. Please contact the authors."
#endif

/** \ingroup TASK
*/
/**@{ */

#if (SIZEOF_VOIDP == 4)

static inline uintptr_t kaapi_task_getstate(const kaapi_task_t* task)
{
  return task->state;
}

static inline void kaapi_task_setstate(kaapi_task_t* task, uintptr_t state)
{
  task->state = state;
}

static inline void kaapi_task_setstate_barrier(kaapi_task_t* task, uintptr_t state)
{
  kaapi_writemem_barrier();
  task->state = state;
}

static inline uintptr_t kaapi_task_state_get(uintptr_t state)
{
  return state;
}

static inline unsigned int kaapi_task_state_issteal(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_STEAL;
}

static inline unsigned int kaapi_task_state_isexec(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_EXEC;
}

static inline unsigned int kaapi_task_state_isterm(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_TERM;
}

static inline unsigned int kaapi_task_state_isaftersteal(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_AFTER;
}

static inline unsigned int kaapi_task_state_ispreempted(uintptr_t state)
{
  return state & KAAPI_MASK_BODY_PREEMPT;
}

static inline unsigned int kaapi_task_state_isspecial(uintptr_t state)
{
  return !(state == 0);
}

static inline unsigned int kaapi_task_state_isnormal(uintptr_t state)
{
  return !kaapi_task_state_isspecial(state);
}

static inline unsigned int kaapi_task_state_isready(uintptr_t state)
{
  return (state & (KAAPI_MASK_BODY_AFTER | KAAPI_MASK_BODY_TERM)) != 0;
}

static inline unsigned int kaapi_task_state_isstealable
(uintptr_t state)
{
  return (state & (KAAPI_MASK_BODY_STEAL | KAAPI_MASK_BODY_EXEC)) == 0;
}

static inline unsigned int kaapi_task_state2int(uintptr_t state)
{
  return (unsigned int)state;
}

#define kaapi_task_state_setsteal(__state)	\
    ((__state) | KAAPI_MASK_BODY_STEAL)

#define kaapi_task_state_setexec(__state)       \
    ((__state) | KAAPI_MASK_BODY_EXEC)

#define kaapi_task_state_setterm(__state)       \
    ((__state) | KAAPI_MASK_BODY_TERM)

#define kaapi_task_state_setafter(__state)      \
    ((__state) | KAAPI_MASK_BODY_AFTER)

/** \ingroup TASK
    Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body)
{
  task->state = 0;
  task->body  = body;
}

/** \ingroup TASK
    Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return task->body;
}

#else /* SIZEOF_VOIDP == 8 */

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
      ((((state) & KAAPI_MASK_BODY)==0) || (((state) & (KAAPI_MASK_BODY_AFTER|KAAPI_MASK_BODY_TERM)) !=0))

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
    ((uintptr_t)body)

#define kaapi_task_state2body(state)           \
    ((kaapi_task_body_t)(state))

#define kaapi_task_state2int(state)            \
    ((int)(state >> KAAPI_MASK_BODY_SHIFTR))

/** Set the body of the task
*/
static inline void kaapi_task_setbody(kaapi_task_t* task, kaapi_task_bodyid_t body )
{
  task->u.body = body;
}

/** Get the body of the task
*/
static inline kaapi_task_bodyid_t kaapi_task_getbody(const kaapi_task_t* task)
{
  return kaapi_task_state2body( task->u.state & ~KAAPI_MASK_BODY );
}
/**@} */

#endif /* SIZEOF_VOIDP == 8 */


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
/** Atomically: OR of the task state with the value in 'state' and return the previous value.
*/
static inline uintptr_t kaapi_task_andstate( kaapi_task_t* task, uintptr_t state )
{
  const uintptr_t retval =
#if (SIZEOF_VOIDP == 4)
    KAAPI_ATOMIC_ANDPTR_ORIG(&task->state, state);
#else /* (SIZEOF_VOIDP == 8) */
    KAAPI_ATOMIC_ANDPTR_ORIG(&task->u.state, state);
#endif
  return retval;
}

static inline uintptr_t kaapi_task_orstate( kaapi_task_t* task, uintptr_t state )
{
#if defined(__i386__)||defined(__x86_64)
  /* WARNING: here we assume that the locked instruction do a writememory barrier */
#else
  kaapi_writemem_barrier();
#endif

  const uintptr_t retval =
#if (SIZEOF_VOIDP == 4)
    KAAPI_ATOMIC_ORPTR_ORIG(&task->state, state);
#else
    KAAPI_ATOMIC_ORPTR_ORIG(&task->u.state, state);
#endif

  return retval;
}

static inline int kaapi_task_teststate( kaapi_task_t* task, uintptr_t state )
{
  /* assume a mem barrier has been done */
  return
#if (SIZEOF_VOIDP == 4)
    (task->state & state) != 0;
#else
    (task->u.state & state) != 0;
#endif
}

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
extern uint64_t kaapi_perf_thread_delayinstate(struct kaapi_processor_t* kproc);



/* ========== Here include machine specific function: only next definitions should depend on machine =========== */
/** Here include all machine dependent functions and types
*/
#include "kaapi_machine.h"
/* ========== MACHINE DEPEND DATA STRUCTURE =========== */



/* ========================================================================== */
/** Compute a hash value from a string
*/
extern uint32_t kaapi_hash_value_len(const char * data, size_t len);

/*
*/
extern uint32_t kaapi_hash_value(const char * data);

/** Hash value for pointer.
    Used for data flow dependencies
*/
static inline uint32_t kaapi_hash_ulong(uint64_t ptr)
{
  /* */
  uint64_t val = ptr >> 3;
  val = (val & 0xFFFF) ^ (val>>32);
  return (uint32_t)val;
}


/**
 * Compression 64 -> 7 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^7 + X^3 + 1
 */
static inline uint32_t kaapi_hash_ulong7(uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000080) v ^= 0x00000009;
    return (uint32_t) (v&0x0000007F);
}


/**
 * Compression 64 -> 6 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^6 + X + 1
 */
static inline uint32_t kaapi_hash_ulong6(uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000040) v ^= 0x00000003;
    if (v & 0x00000080) v ^= 0x00000006;
    return (uint32_t) (v&0x0000003F);
}

/**
 * Compression 64 -> 5 bits
 * Sums the 8 bytes modulo 2, then reduces the resulting degree 7 
 * polynomial modulo X^5 + X^2 + 1
 */
static inline uint32_t kaapi_hash_ulong5(uint64_t v)
{
    v ^= (v >> 32);
    v ^= (v >> 16);
    v ^= (v >> 8);
    if (v & 0x00000020) v ^= 0x00000005;
    if (v & 0x00000040) v ^= 0x0000000A;
    if (v & 0x00000080) v ^= 0x00000014;
    return (uint32_t) (v&0x0000001F);
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
    struct kaapi_version_t*   version;     /* for static scheduling */
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
  uint32_t entry_map;                 /* type size must at least KAAPI_HASHMAP_SIZE */
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
extern kaapi_hashentries_t* get_hashmap_entry( kaapi_hashmap_t* khm, uint32_t key);

/*
*/
extern void set_hashmap_entry( kaapi_hashmap_t* khm, uint32_t key, kaapi_hashentries_t* entries);


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
extern const char* kaapi_cpuset2string( int nproc, kaapi_cpuset_t affinity );


/**
*/
static inline void kaapi_cpuset_clear(kaapi_cpuset_t* affinity )
{
  (*affinity)[0] = 0;
  (*affinity)[1] = 0;
}


/**
*/
static inline int kaapi_cpuset_intersect(kaapi_cpuset_t s1, kaapi_cpuset_t s2)
{
  return ((s1[0] & s2[0]) != 0) || ((s1[1] & s2[1]) != 0);
}

/**
*/
static inline int kaapi_cpuset_empty(kaapi_cpuset_t affinity)
{
  return (affinity[0] == 0) && (affinity[1] == 0);
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
static inline int kaapi_cpuset_has(kaapi_cpuset_t affinity, kaapi_processor_id_t kid )
{
  kaapi_assert_debug( (kid >=0) && (kid < sizeof(kaapi_cpuset_t)*8) );
  if (kid <64)
    return ( affinity[0] & ((uint64_t)1)<< (uint64_t)kid) != (uint64_t)0;
  else
    return ( affinity[1] & ((uint64_t)1)<< (uint64_t)(kid-64)) != (uint64_t)0;
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
#if (SIZEOF_VOIDP == 4)
//  kaapi_assert_debug( kaapi_task_state_issteal(thread->sfp->pc->state) );
  return kaapi_task_state_isready(thread->sfp->pc->state);
#else
//  kaapi_assert_debug( kaapi_task_state_issteal(thread->sfp->pc->u.state) );
  return kaapi_task_state_isready(thread->sfp->pc->u.state);
#endif
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
  return 0;
#else
acquire:
  if (KAAPI_ATOMIC_DECR(lock) ==0) return 1;
  while (KAAPI_ATOMIC_READ(lock) <=0)
    kaapi_slowdown_cpu(); 
  goto acquire;
#endif
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
    \retval EINTR the execution of the stack is interupt and the thread is detached to the kprocessor.
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
extern uint64_t kaapi_perf_thread_delayinstate(kaapi_processor_t* kproc);

/* */
extern void kaapi_set_workload( kaapi_processor_t*, uint32_t workload );

/* */
extern void kaapi_set_self_workload( uint32_t workload );


/** \ingroup TASK
    The function kaapi_task_body_isstealable() will return non-zero value iff the task body may be stolen.
    All user tasks are stealable.
    \param body IN a task body
*/
static inline int kaapi_task_body_isstealable(kaapi_task_body_t body)
{ 
  return (body != kaapi_taskstartup_body) 
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
#if (SIZEOF_VOIDP == 4)

  return kaapi_task_state_isstealable(task->state) &&
   kaapi_task_body_isstealable(task->body);

#else /* SIZEOF_VOIDP == 8 */

  const uintptr_t state = (uintptr_t)task->u.body;
  return kaapi_task_state_isstealable(state) &&
    kaapi_task_body_isstealable(kaapi_task_state2body(state));

#endif
}

#include "kaapi_staticsched.h"

/* ======================== MACHINE DEPENDENT FUNCTION THAT SHOULD BE DEFINED ========================*/
/* ........................................ PUBLIC INTERFACE ........................................*/

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_IMPL_H */
