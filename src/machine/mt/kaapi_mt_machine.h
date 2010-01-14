/*
** kaapi_mt_machine.h
** xkaapi
** 
** Created on Tue Mar 31 15:20:42 2009
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
#ifndef _KAAPI_MT_MACHINE_H_
#define _KAAPI_MT_MACHINE_H_ 1

#ifndef _KAAPI_IMPL_H
#  error This file must not be directly included. Use kaapi_impl.h instead
#endif
#include "kaapi_datastructure.h"
#include <stdint.h>
#include <pthread.h>

#if defined(KAAPI_USE_SETJMP)
#  include <setjmp.h>
#elif defined(KAAPI_USE_UCONTEXT)
#  include <ucontext.h>
#endif



/* ============================= Documentation ============================ */
/* This is the multithreaded definition of machine type for X-Kaapi.
   The purpose of the machine.h file is to give all machine dependent functions.

   TODO: implements NUMA features:
          - allocation des processeurs sur les bons bancs memoires
          - creation des threads sur les bons bancs memoires (entre autre pile + internal thread descriptor)
          en 1/ migrant the thread courant sur le processeur cible 2/ creation du thread sur ce processeur
          3/ revenir à la fin sur le processeur main.
*/


/** \ingroup THREAD
   Define the minimum stack size. 
*/
#define KAAPI_STACK_MIN 8192


struct kaapi_processor_t;
typedef kaapi_uint32_t kaapi_processor_id_t;


/* ========================================================================= */
/** Termination detecting barrier
*/
extern kaapi_atomic_t kaapi_term_barrier;

/* ========================================================================= */
/** Termination detecting barrier
*/
extern volatile int kaapi_isterm;

/* ========================================================================= */
/* Functions to manipulate thread context 
   These functions are intended to be portability layer on top of ucontext.h
   interfance or setjmp/longjmp.
   A context, in Kaapi sense, is both the machine register, C-stack, Kaapi-stack
   and table of thread data specific value. A context is represented by kaapi_thread_context_t
   and allows to set flags in order to avoid saving some of the data structure.
   This feature is interesting in the case if only the kaapi stack (and the thread data specific table)
   are required to be saved and restore, without taking care of the C-stack.
   
   The function called always takes the kaapi_thread_context data structure as parameter.
   \ingroup THREAD
 */
/*@{*/

/** The thread context data structure
    This data structure should be extend in case where the C-stack is required to be suspended and resumed.
    The data structure inherits from kaapi_stack_t the stackable field in order to be linked in stack.
*/
typedef kaapi_stack_t kaapi_thread_context_t;

/* list of suspended threadcontext */
typedef struct kaapi_listthreadctxt_t {
  KAAPI_STACK_DECLARE_FIELD(kaapi_thread_context_t);
} kaapi_listthreadctxt_t;

/** \ingroup WS
    Higher level context manipulation.
    This function is machine dependent.
*/
extern int kaapi_makecontext( struct kaapi_processor_t* proc, kaapi_thread_context_t* ctxt, 
                              void (*entrypoint)(void* arg), void* arg 
                            );

/** \ingroup WS
    Higher level context manipulation.
    Assign context onto the running processor proc.
    This function is machine dependent.
*/
extern int kaapi_setcontext( struct kaapi_processor_t* proc, kaapi_thread_context_t * ctxt );

/** \ingroup WS
    Higher level context manipulation.
    Get the context of the running processor proc.
    This function is machine dependent.
*/
extern int kaapi_getcontext( struct kaapi_processor_t* proc, kaapi_thread_context_t * ctxt );
/*@}*/



/* ============================= Kprocessor ============================ */
/** \ingroup WS
*/
typedef struct kaapi_listrequest_t {
  kaapi_atomic_t  count;
  kaapi_request_t requests[KAAPI_MAX_PROCESSOR];
} kaapi_listrequest_t __attribute__((aligned (KAAPI_CACHE_LINE)));


/** \ingroup WS
    This data structure defines a work stealer processor thread.
    The kid is a system wide identifier. In the current version it only contains a local counter value
    from 0 to N-1.
    Each kprocessor stores neighbor information in a hierarchical as defined by the topology.
    When a kprocessor i wants to send a steal request to the kprocessor j, it should 
      - have j in its neighbor set
      - use its requests in the kprocessor j at position kproc_j->request[kproc_i->hindex[level]],
      where level is the lower level set in kprocessor i that contains the kprocessor j.
    
    On each kprocessor i, hkids[l][j] points to the j-th kprocessor at level l.
    
    
    TODO: HIERARCHICAL STRUCTURE IS NOT YET IMPLEMENTED. ONLY FLAT STEAL.
*/
typedef struct kaapi_processor_t {
  kaapi_thread_context_t*  ctxt;                          /* current stack (next version = current active thread) */
  kaapi_processor_id_t     kid;                           /* Kprocessor id */
  kaapi_uint32_t           issteal;                       /* */
  kaapi_uint32_t           hlevel;                        /* number of level for this Kprocessor >0 */
  kaapi_uint16_t*          hindex;                        /* id local identifier of request at each level of the hierarchy, size hlevel */
  kaapi_uint16_t*          hlcount;                       /* count of proc. at each level of the hierarchy, size hlevel */
  kaapi_processor_id_t**   hkids;                         /* ids of all processors at each level of the hierarchy, size hlevel */
  kaapi_reply_t            reply;                         /* use when a request has been emited */
  kaapi_listrequest_t      hlrequests;                    /* all requests attached to each kprocessor ordered by increasing level */

#if defined(KAAPI_CONCURRENT_WS)
  pthread_mutex_t          lock;           
  pthread_cond_t           cond;           
#endif
  void*                    dfgconstraint;                 /* TODO: for DFG constraints evaluation */

  kaapi_listthreadctxt_t   lsuspend;                      /* list of suspended context */
  kaapi_listthreadctxt_t   lfree;                         /* list of free context */
  void*                    fnc_selecarg;                  /* arguments for select victim function, 0 at initialization */
  kaapi_selectvictim_fnc_t fnc_select;                    /* function to select a victim */
  
  /* performance counters */
  kaapi_uint32_t           cnt_tasks;                     /* number of executed tasks */
  kaapi_uint32_t           cnt_stealreqok;                /* number of steal requests replied with success */
  kaapi_uint32_t           cnt_stealreq;                  /* total number of steal requests replied */
  kaapi_uint32_t           cnt_stealop;                   /* number of steal operation: ratio cnt_stealreqok/cnt_stealok avrg number of aggr. */
  kaapi_uint32_t           cnt_suspend;                   /* number of suspend operations*/
  double                   t_idle;                        /* total idle time in second */           
} kaapi_processor_t __attribute__ ((aligned (KAAPI_CACHE_LINE)));

/*
*/
extern int kaapi_processor_init( kaapi_processor_t* kproc );

/*
*/
extern int kaapi_processor_setuphierarchy( kaapi_processor_t* kproc );

/* ........................................ PRIVATE INTERFACE ........................................*/
/** \ingroup STACK
    The function kaapi_context_alloc() allocates in the heap a context with a stack containing 
    at bytes for tasks and bytes for data.
    If successful, the kaapi_context_alloc() function will return a pointer to a kaapi_thread_context_t.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param kproc IN/OUT the kprocessor that make allocation
    \param size_data IN the amount of stack data.
    \retval pointer to the stack 
    \retval 0 if allocation failed
*/
extern kaapi_thread_context_t* kaapi_context_alloc( kaapi_processor_t* kproc );


/** \ingroup STACK
    The function kaapi_context_free() free the context successfuly allocated with kaapi_context_alloc.
    If successful, the kaapi_context_free() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param ctxt INOUT a pointer to the kaapi_thread_context_t to allocate.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_context_free( kaapi_thread_context_t* ctxt );


/** \ingroup WS
    Number of used cores
*/
extern kaapi_uint32_t     kaapi_count_kprocessors;

/** \ingroup WS
    One K-processors per core
*/
extern kaapi_processor_t** kaapi_all_kprocessors;

/** \ingroup WS
    A Kprocessor is a posix thread -> the current kprocessor
*/
extern pthread_key_t kaapi_current_processor_key;

/** \ingroup WS
*/
#define _kaapi_get_current_processor() \
  ((kaapi_processor_t*)pthread_getspecific( kaapi_current_processor_key ))


/** \ingroup WS
    Returns the current stack of tasks
*/
#define _kaapi_self_stack() \
  (_kaapi_get_current_processor()->ctxt)


/* ============================= Hierarchy ============================ */
/** The hierarchy level for this configuration of the library.
    TODO: Should be defined during the configuration of the library.
    The hierarchy level allows to code memory hierarchy.
    The default level is 1, meaning that all processors share a same memory.
    In case of coding other hierarchy level, for instance 2 where subset of processors
    are able to shared common memory (L2 cache) that are faster than main memory.

    Each of the MaxProc_level processor at the hierarchy level has a local index 
    from 0..MaxProc_level-1. The root processor of a given hierarchy level has always
    local index 0.
*/
extern kaapi_uint32_t kaapi_hierarchy_level;

/** \ingroup WS
    Definition of the neighbors type for a given processors at a given level.
*/
typedef struct kaapi_neighbors_t {
  kaapi_uint16_t      count;         /* number of neighbors */
  kaapi_processor_t** kproc;         /* list of neighbor processors, of size count */
  kaapi_uint32_t*     kplid;         /* list of neighbor processor local indexes, of size count */
} kaapi_neighbors_t;


/** \ingroup WS
    Definition of the neighbors for all processors for all level.
    The pointer points to an array of size kaapi_count_kprocessors of arrays of size kaapi_hierarchy_level.
    - kaapi_neighbors[kid][0] returns the neighbors information for the kprocessor with kid at the lowest hierarchy level.
    - kaapi_neighbors[kid][l] returns the neighbors information for the kprocessor with kid at level l.
    - kaapi_neighbors[kid][kaapi_hierarchy_level] returns the neighbors information for the kprocessor with kid 
    at the highest hierarchy level.
*/
extern kaapi_neighbors_t** kaapi_neighbors;


/** \ingroup WS
*/
extern int kaapi_setup_topology(void);




/* ============================= Atomic Function ============================ */
#if (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ > 4) \
|| defined(__INTEL_COMPILER))
/* Note: ICC seems to also support these builtins functions */
#  if defined(__INTEL_COMPILER)
#    warning Using ICC. Please, check if icc really support atomic operations
/* ia64 impl using compare and exchange */
/*#    define KAAPI_CAS(_a, _o, _n) _InterlockedCompareExchange(_a, _n, _o ) */
#  endif

/** TODO: try to use same function without barrier
*/
#  define KAAPI_ATOMIC_CAS(a, o, n) \
    __sync_bool_compare_and_swap( &((a)->_counter), o, n) 

#  define KAAPI_ATOMIC_CAS64(a, o, n) \
    __sync_bool_compare_and_swap( &((a)->_counter), o, n) 

#  define KAAPI_ATOMIC_INCR(a) \
    __sync_add_and_fetch( &((a)->_counter), 1 ) 

#  define KAAPI_ATOMIC_DECR(a) \
    __sync_sub_and_fetch( &((a)->_counter), 1 ) 

#  define KAAPI_ATOMIC_SUB(a, value) \
    __sync_sub_and_fetch( &((a)->_counter), value ) 

#  define KAAPI_ATOMIC_READ(a) \
    ((a)->_counter)

#  define KAAPI_ATOMIC_WRITE(a, value) \
    (a)->_counter = value

#elif defined(KAAPI_USE_APPLE) /* if gcc version on Apple is less than 4.1 */

#  include <libkern/OSAtomic.h>

#  define KAAPI_ATOMIC_CAS(a, o, n) \
    OSAtomicCompareAndSwap32( o, n, &((a)->_counter)) 

#  define KAAPI_ATOMIC_CAS64(a, o, n) \
    OSAtomicCompareAndSwap64( o, n, &((a)->_counter)) 

#  define KAAPI_ATOMIC_INCR(a) \
    OSAtomicIncrement32( &((a)->_counter) ) 

#  define KAAPI_ATOMIC_DECR(a) \
    OSAtomicDecrement32(&((a)->_counter) ) 

#  define KAAPI_ATOMIC_SUB(a, value) \
    OSAtomicAdd32( -value, &((a)->_counter) ) 

#  define KAAPI_ATOMIC_READ(a) \
    ((a)->_counter)

#  define KAAPI_ATOMIC_WRITE(a, value) \
    (a)->_counter = value

#else
#  error "Please add support for atomic operations on this system/architecture"
#endif /* GCC > 4.1 */

#if (SIZEOF_VOIDP == 4)
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)a, o, n )
#else
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS64( (kaapi_atomic64_t*)a, o, n )
#endif


/* ========================================================================== */

/* ============================= Memory Barrier ============================= */

#if defined(KAAPI_USE_APPLE)
#  include <libkern/OSAtomic.h>

static inline void kaapi_writemem_barrier()  
{
  OSMemoryBarrier();
  /* Compiler fence to keep operations from */
  __asm__ __volatile__("" : : : "memory" );
}

static inline void kaapi_readmem_barrier()  
{
  OSMemoryBarrier();
  /* Compiler fence to keep operations from */
  __asm__ __volatile__("" : : : "memory" );
}

/* should be both read & write barrier */
static inline void kaapi_mem_barrier()  
{
  OSMemoryBarrier();
  /* Compiler fence to keep operations from */
  __asm__ __volatile__("" : : : "memory" );
}


#elif defined(KAAPI_USE_LINUX)

#  define kaapi_writemem_barrier() \
    __sync_synchronize()

#  define kaapi_readmem_barrier() \
    __sync_synchronize()

#  define kaapi_mem_barrier() \
   __sync_synchronize()

#else
#  error "Undefined barrier"
#endif /* KAAPI_USE_APPLE */


/* ========================================================================== */
/** Termination dectecting barrier
*/
/**
*/
#define kaapi_barrier_td_init( kpb, value ) \
  KAAPI_ATOMIC_WRITE( kpb, value )

/**
*/
#define kaapi_barrier_td_destroy( kpb )
  
/**
*/
#define kaapi_barrier_td_setactive( kpb, b ) \
  if (b) { KAAPI_ATOMIC_INCR( kpb ); } \
  else KAAPI_ATOMIC_DECR( kpb )

/**
*/
#define kaapi_barrier_td_isterminated( kpb ) \
  (KAAPI_ATOMIC_READ(kpb ) == 0)


/**
*/
#define kaapi_barrier_td_waitterminated( kpb ) \
  while (KAAPI_ATOMIC_READ(kpb ) != 0)


/** Private interface
*/
static inline int kaapi_isterminated(void)
{
  return kaapi_isterm !=0;
}

/** \ingroup WS
*/
static inline int kaapi_listrequest_init( kaapi_listrequest_t* pklr ) 
{
  int i; 
  KAAPI_ATOMIC_WRITE(&pklr->count, 0);
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {  
    kaapi_request_init(&pklr->requests[i]);
  }
  return 0;
}


/** 
*/
#if defined(KAAPI_CONCURRENT_WS)
static inline int kaapi_task_casstate( kaapi_task_t* task, kaapi_uint32_t oldstate, kaapi_uint32_t newstate )
{
  kaapi_uint32_t flag = task->flag;
  kaapi_uint32_t oldflag = (flag & ~KAAPI_TASK_MASK_STATE)|oldstate;
  kaapi_uint32_t newflag = (flag & ~KAAPI_TASK_MASK_STATE)|newstate;
  return KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)&task->flag, oldflag, newflag );
}
#else
static inline int kaapi_task_casstate( kaapi_task_t* task, kaapi_uint32_t oldstate, kaapi_uint32_t newstate )
{
  kaapi_assert_debug( kaapi_task_getstate(task) == oldstate );
  kaapi_task_setstate( task, newstate );
  return 1;
}
#endif

/* ============================= Private functions, machine dependent ============================ */

/** Post a request to a given k-processor
  This method posts a request to victim k-processor. 
  \param src the sender of the request 
  \param hlevel the hierarchy level of the steal
  \param dest the receiver (victim) of the request
  \param return 0 if the request has been successully posted
  \param return !=0 if the request been not been successully posted and the status of the request contains the error code
*/  
static inline int kaapi_request_post( kaapi_processor_t* kproc, kaapi_reply_t* reply, kaapi_victim_t* victim )
{
  kaapi_request_t* req;
  if (kproc ==0) return EINVAL;
  if (victim ==0) return EINVAL;

  req              = &victim->kproc->hlrequests.requests[ kproc->kid ];
  reply->status    = KAAPI_REQUEST_S_POSTED;
  req->reply       = reply;
  req->stack       = kproc->ctxt;
  reply->data      = 0;
  req->status      = KAAPI_REQUEST_S_POSTED;

  kaapi_writemem_barrier();
  
  /* incr without mem. barrier here if the victim see the request status as ok is enough,
     even if the new counter is not yet view
  */
  KAAPI_ATOMIC_INCR( &victim->kproc->hlrequests.count );
  
  return 0;
}

#endif /* _KAAPI_MT_MACHINE_H */
