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

#if defined(KAAPI_USE_NUMA)
#  include <numa.h>
#  include <numaif.h>
#  define KAAPI_KPROCESSOR_ALIGNMENT_SIZE 4096  /* page size */
#else
#  define KAAPI_KPROCESSOR_ALIGNMENT_SIZE KAAPI_CACHE_LINE
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


//struct kaapi_processor_t;
//typedef kaapi_uint32_t kaapi_processor_id_t;


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

/** WS queue of thread context: used
    This data structure should be extend in case where the C-stack is required to be suspended and resumed.
    The data structure inherits from kaapi_stack_t the stackable field in order to be linked in stack.
*/
struct kaapi_wsqueuectxt_cell_t;
typedef struct kaapi_wsqueuectxt_cell_t* kaapi_wsqueuectxt_cell_ptr_t;

/** Cell of the list
*/
typedef struct kaapi_wsqueuectxt_cell_t {
  kaapi_atomic_t               state;  /* 0: in the list, 1: out the list */
  kaapi_thread_context_t*      thread; /* */
  kaapi_wsqueuectxt_cell_ptr_t next;   /* double link list */
  kaapi_wsqueuectxt_cell_ptr_t prev;   /* shared with thief, used to link in free list */
} kaapi_wsqueuectxt_cell_t;


/** Type of bloc of kaapi_liststack_node_t
*/
KAAPI_DECLARE_BLOCENTRIES(kaapi_wsqueuectxt_cellbloc_t, kaapi_wsqueuectxt_cell_t);


/** List of thread context */
typedef struct kaapi_wsqueuectxt_t {
  kaapi_wsqueuectxt_cell_ptr_t  head;
  kaapi_wsqueuectxt_cell_ptr_t  tail;
  kaapi_wsqueuectxt_cell_ptr_t  headfreecell;
  kaapi_wsqueuectxt_cell_ptr_t  tailfreecell;
  kaapi_wsqueuectxt_cellbloc_t* allocatedbloc;
} kaapi_wsqueuectxt_t;

/** lfree data structure,
    kaapi_thread_context_t has both next and prev fields.
*/
KAAPI_QUEUE_DECLARE(kaapi_queuectxt_t, kaapi_thread_context_t);
#define KAAPI_MAXFREECTXT 2

/* clear / is empty / push / pop */
#define kaapi_lfree_clear( kproc )\
  {\
    (kproc)->sizelfree = 0;\
    KAAPI_QUEUE_CLEAR( &(kproc)->lfree );\
  }

#define kaapi_lfree_isempty( kproc )\
    ((kproc)->sizelfree == 0)

#define kaapi_lfree_push( kproc, v )\
  {\
    KAAPI_QUEUE_PUSH_FRONT( &(kproc)->lfree, v );\
    if ((kproc)->sizelfree >= KAAPI_MAXFREECTXT) \
    {\
      kaapi_thread_context_t* ctxt = 0;\
      KAAPI_QUEUE_POP_BACK( &(kproc)->lfree, ctxt ); \
      kaapi_context_free( ctxt );\
    }\
    else\
      ++(kproc)->sizelfree;\
  }

#define kaapi_lfree_pop( kproc, v )\
  {\
    --(kproc)->sizelfree;\
    KAAPI_QUEUE_POP_FRONT( &(kproc)->lfree, v );\
  }

/** lready data structure
*/
typedef struct kaapi_lready
{
  kaapi_thread_context_t* _front;
  kaapi_thread_context_t* _back;
} kaapi_lready_t;

/** push: LIFO order with respect to pop. Only owner may push
*/
static inline int kaapi_wsqueuectxt_isempty( kaapi_wsqueuectxt_t* ls )
{ return (ls->head ==0); }

/**
*/
extern int kaapi_wsqueuectxt_init( kaapi_wsqueuectxt_t* ls );

/**
*/
extern int kaapi_wsqueuectxt_destroy( kaapi_wsqueuectxt_t* ls );

/* Push a ctxt. Must be call by the owner of the queue in case of concurrent execution.
   Return 0 in case of success
   Return ENOMEM if allocation failed
*/
extern int kaapi_wsqueuectxt_push( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t* thread );

#if 0 // not well defined and not yet implemented 
/* Push a ctxt. Can be called by thief processors
   Return 0 in case of success
   Return ENOMEM if allocation failed
*/
extern int kaapi_wsqueuectxt_lockpush( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t* thread );
#endif

/* Pop a ctxt
   Return 0 in case of success
   Return EWOULDBLOCK if list is empty
*/
extern int kaapi_wsqueuectxt_pop( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t** thread );

/* Steal a ctxt
   Return 0 in case of success
   Return EWOULDBLOCK if list is empty
*/
extern int kaapi_wsqueuectxt_steal( kaapi_wsqueuectxt_t* ls, kaapi_thread_context_t** thread );

/* Steal a ctxt on a specific cell
   Return a pointer to the stolen thread in case of success
   Return 0 if the thread was already stolen
*/
kaapi_thread_context_t* kaapi_wsqueuectxt_steal_cell( kaapi_wsqueuectxt_t* ls, kaapi_wsqueuectxt_cell_t* cell );

/** \ingroup WS
    Higher level context manipulation.
    This function is machine dependent.
*/
extern int kaapi_makecontext( struct kaapi_processor_t* proc, kaapi_thread_context_t* thread, 
                              void (*entrypoint)(void* arg), void* arg 
                            );

/** \ingroup WS
    Higher level context manipulation.
    Assign context onto the running processor proc.
    This function is machine dependent.
*/
extern int kaapi_setcontext( struct kaapi_processor_t* proc, kaapi_thread_context_t* thread );

/** \ingroup WS
    Higher level context manipulation.
    Get the context of the running processor proc.
    This function is machine dependent.
*/
extern int kaapi_getcontext( struct kaapi_processor_t* proc, kaapi_thread_context_t* thread );
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
  kaapi_thread_context_t*  thread;                        /* current thread under execution */
  kaapi_processor_id_t     kid;                           /* Kprocessor id */
  kaapi_uint32_t           issteal;                       /* */
  kaapi_uint32_t           hlevel;                        /* number of level for this Kprocessor >0 */
  kaapi_uint16_t*          hindex;                        /* id local identifier of request at each level of the hierarchy, size hlevel */
  kaapi_uint16_t*          hlcount;                       /* count of proc. at each level of the hierarchy, size hlevel */
  kaapi_processor_id_t**   hkids;                         /* ids of all processors at each level of the hierarchy, size hlevel */
  kaapi_reply_t            reply;                         /* use when a request has been emited */
  
  /* cache align */
  kaapi_listrequest_t      hlrequests;                    /* all requests attached to each kprocessor ordered by increasing level */

  /* cache align */
  kaapi_atomic_t           lock                           /* all requests attached to each kprocessor ordered by increasing level */
    __attribute__((aligned(KAAPI_CACHE_LINE)));

  kaapi_wsqueuectxt_t      lsuspend;                      /* list of suspended context */
  kaapi_lready_t	   lready;                        /* list of ready context, concurrent access locked by 'lock' */
  kaapi_thread_context_t*  readythread;                   /* continuation passing parameter to speed up recovery of activity... */

  /* free list */
  kaapi_queuectxt_t        lfree;                         /* queue of free context */
  int                      sizelfree;                     /* size of the queue */

  void*                    fnc_selecarg;                  /* arguments for select victim function, 0 at initialization */
  kaapi_selectvictim_fnc_t fnc_select;                    /* function to select a victim */

  void*                    dfgconstraint;                 /* TODO: for DFG constraints evaluation */

  /* performance register */
  kaapi_perf_counter_t	   perf_regs[2][KAAPI_PERF_ID_MAX];
  kaapi_perf_counter_t*	   curr_perf_regs;                /* either perf_regs[0], either perf_regs[1] */

  int			                 papi_event_set;
  unsigned int		         papi_event_count;
  kaapi_perf_counter_t     start_t[2];                    /* [KAAPI_PERF_SCHEDULE_STATE]= T1 else = Tidle */
   
  double                   t_preempt;                     /* total idle time in second pass in the preemption */           

  /* workload */
  kaapi_atomic_t	         workload;

} kaapi_processor_t __attribute__ ((aligned (KAAPI_KPROCESSOR_ALIGNMENT_SIZE)));

/*
*/
extern int kaapi_processor_init( kaapi_processor_t* kproc );

/*
*/
extern int kaapi_processor_setuphierarchy( kaapi_processor_t* kproc );

#if defined(KAAPI_USE_NUMA)
static inline kaapi_processor_t* kaapi_processor_allocate(void)
{
  kaapi_processor_t* const kproc = (kaapi_processor_t*)numa_alloc_local(sizeof(kaapi_processor_t));
  if (kproc == 0) return 0;

  memset(kproc, 0, sizeof(kaapi_processor_t));

  return kproc;
}

static inline void kaapi_processor_free(kaapi_processor_t* kproc)
{
  numa_free(kproc, sizeof(kaapi_processor_t));
}

#else /* ! KAAPI_USE_NUMA */

static inline kaapi_processor_t* kaapi_processor_allocate(void)
{
  return (kaapi_processor_t*)calloc(sizeof(kaapi_processor_t), 1);
}

static inline void kaapi_processor_free(kaapi_processor_t* kproc)
{
  free(kproc);
}

#endif /* KAAPI_USE_NUMA */



/* ........................................ PRIVATE INTERFACE ........................................*/
/** \ingroup TASK
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


/** \ingroup TASK
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

#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
  extern __thread kaapi_thread_t** kaapi_current_thread_key;
  extern __thread kaapi_threadgroup_t kaapi_current_threadgroup_key;
#endif

extern kaapi_processor_t* kaapi_get_current_processor(void);
extern kaapi_processor_id_t kaapi_get_current_kid(void);

/** \ingroup WS
*/
#define _kaapi_get_current_processor() \
  ((kaapi_processor_t*)pthread_getspecific( kaapi_current_processor_key ))

/** \ingroup WS
    Returns the current thread of tasks
*/
#define _kaapi_self_thread() \
  _kaapi_get_current_processor()->thread


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
#  ifndef KAAPI_ATOMIC_CAS
#    define KAAPI_ATOMIC_CAS(a, o, n) \
      __sync_bool_compare_and_swap( &((a)->_counter), o, n) 
#  endif

#  ifndef KAAPI_ATOMIC_CAS64
#    define KAAPI_ATOMIC_CAS64(a, o, n) \
      __sync_bool_compare_and_swap( &((a)->_counter), o, n) 
#  endif

#  define KAAPI_ATOMIC_AND(a, o) \
    __sync_fetch_and_and( &((a)->_counter), o )

#  ifndef KAAPI_ATOMIC_INCR
#    define KAAPI_ATOMIC_INCR(a) \
      __sync_add_and_fetch( &((a)->_counter), 1 ) 
#  endif

#  ifndef KAAPI_ATOMIC_INCR64
#    define KAAPI_ATOMIC_INCR54(a) \
      __sync_add_and_fetch( &((a)->_counter), 1 ) 
#  endif

#  define KAAPI_ATOMIC_DECR(a) \
    __sync_sub_and_fetch( &((a)->_counter), 1 ) 

#  ifndef KAAPI_ATOMIC_SUB
#    define KAAPI_ATOMIC_SUB(a, value) \
      __sync_sub_and_fetch( &((a)->_counter), value ) 
#  endif      

#  ifndef KAAPI_ATOMIC_SUB64
#    define KAAPI_ATOMIC_SUB64(a, value) \
      __sync_sub_and_fetch( &((a)->_counter), value ) 
#  endif      

#  define KAAPI_ATOMIC_ADD(a, value) \
    __sync_add_and_fetch( &((a)->_counter), value ) 


#elif defined(KAAPI_USE_APPLE) /* if gcc version on Apple is less than 4.1 */

#  include <libkern/OSAtomic.h>

#  ifndef KAAPI_ATOMIC_CAS
#    define KAAPI_ATOMIC_CAS(a, o, n) \
      OSAtomicCompareAndSwap32( o, n, &((a)->_counter)) 
#  endif

#  ifndef KAAPI_ATOMIC_CAS64
#    define KAAPI_ATOMIC_CAS64(a, o, n) \
      OSAtomicCompareAndSwap64( o, n, &((a)->_counter)) 
#  endif

#  define KAAPI_ATOMIC_AND(a, o)			\
    OSAtomicAnd32( &((a)->_counter), o )

#  ifndef KAAPI_ATOMIC_INCR
#    define KAAPI_ATOMIC_INCR(a) \
      OSAtomicIncrement32( &((a)->_counter) ) 
#  endif

#  ifndef KAAPI_ATOMIC_INCR64
#    define KAAPI_ATOMIC_INCR64(a) \
      OSAtomicIncrement64( &((a)->_counter) ) 
#  endif

#  define KAAPI_ATOMIC_DECR(a) \
    OSAtomicDecrement32(&((a)->_counter) ) 

#  ifndef KAAPI_ATOMIC_SUB
#    define KAAPI_ATOMIC_SUB(a, value) \
      OSAtomicAdd32( -value, &((a)->_counter) ) 
#  endif

#  ifndef KAAPI_ATOMIC_SUB64
#    define KAAPI_ATOMIC_SUB64(a, value) \
      OSAtomicAdd64( -value, &((a)->_counter) ) 
#  endif

#  define KAAPI_ATOMIC_ADD(a, value) \
    OSAtomicAdd32( value, &((a)->_counter) ) 

#else
#  error "Please add support for atomic operations on this system/architecture"
#endif /* GCC > 4.1 */

#if (SIZEOF_VOIDP == 4)
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS( (kaapi_atomic_t*)a, (kaapi_uint32_t)o, (kaapi_uint32_t)n )
#else
#  define KAAPI_ATOMIC_CASPTR(a, o, n) \
    KAAPI_ATOMIC_CAS64( (kaapi_atomic64_t*)a, (kaapi_uint64_t)o, (kaapi_uint64_t)n )
#endif


/* ========================================================================== */


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
static inline int kaapi_listrequest_init( kaapi_processor_t* kproc, kaapi_listrequest_t* pklr ) 
{
  int i; 
  KAAPI_ATOMIC_WRITE(&pklr->count, 0);
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {  
    kaapi_request_init(kproc, &pklr->requests[i]);
  }
  return 0;
}

/** 
*/
#if (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALCAS_METHOD)
static inline int kaapi_task_casstate( kaapi_task_t* task, kaapi_task_body_t oldbody, kaapi_task_body_t newbody )
{
  kaapi_atomic_t* kat = (kaapi_atomic_t*)&task->body;
  return KAAPI_ATOMIC_CASPTR( kat, oldbody, newbody );
}
static inline int kaapi_task_cas_extrastate( kaapi_task_t* task, kaapi_task_body_t oldbody, kaapi_task_body_t newbody )
{
  kaapi_atomic_t* kat = (kaapi_atomic_t*)&task->ebody;
  return KAAPI_ATOMIC_CASPTR( kat, oldbody, newbody );
}
#elif (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)
/*
static inline int kaapi_task_casstate( kaapi_task_t* task, kaapi_task_body_t oldbody, kaapi_task_body_t newbody )
{
  kaapi_atomic_t* kat = (kaapi_atomic_t*)&task->body;
  return KAAPI_ATOMIC_CASPTR( kat, oldbody, newbody );
}
static inline int kaapi_task_casstate( kaapi_task_t* task, kaapi_task_body_t oldbody, kaapi_task_body_t newbody )
{
  if (task->body != oldbody ) return 0;
  kaapi_task_setbody(task, newbody );
  return 1;
}
*/
#else
#  warning "NOT IMPLEMENTED"
#endif

/* ============================= Private functions, machine dependent ============================ */
/* */
extern kaapi_uint64_t kaapi_perf_thread_delayinstate(kaapi_processor_t* kproc);

/** Post a request to a given k-processor
  This method posts a request to victim k-processor. 
  \param kproc the sender of the request 
  \param reply where to receive result
  \param dest the receiver (victim) of the request
  \param return 0 if the request has been successully posted
  \param return !=0 if the request been not been successully posted and the status of the request contains the error code
*/
static inline int kaapi_request_post( kaapi_processor_t* kproc, kaapi_reply_t* reply, kaapi_victim_t* victim )
{
  kaapi_request_t* req;
  if (kproc ==0) return EINVAL;
  if (victim ==0) return EINVAL;
  
//  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim->kproc->hlrequests.count) >= 0 );

  req              = &victim->kproc->hlrequests.requests[ kproc->kid ];
  kaapi_assert_debug( req->status == KAAPI_REQUEST_S_EMPTY);
  kaapi_assert_debug( req->proc == victim->kproc);

  reply->status    = KAAPI_REQUEST_S_POSTED;
  req->reply       = reply;
  req->mthread     = kproc->thread;
  req->thread      = kaapi_threadcontext2thread(kproc->thread);
#if defined(KAAPI_USE_PERFCOUNTER)
  req->delay       = kaapi_perf_thread_delayinstate(kproc);
#else  
  req->delay       = 0;
#endif
  reply->data      = 0;

  kaapi_writemem_barrier();
#if 0
  fprintf(stdout,"%i kproc post request to:%p, @req=%p\n", kproc->kid, (void*)victim->kproc, (void*)req );
  fflush(stdout);
#endif
  req->status      = KAAPI_REQUEST_S_POSTED;
  
  /* incr without mem. barrier here if the victim see the request status as ok is enough,
     even if the new counter is not yet view
  */
  KAAPI_ATOMIC_INCR( &victim->kproc->hlrequests.count );
//  kaapi_assert_debug( KAAPI_ATOMIC_READ(&victim->kproc->hlrequests.count) > 0 );
  
  return 0;
}

#endif /* _KAAPI_MT_MACHINE_H */
