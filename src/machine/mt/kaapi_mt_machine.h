/*
** kaapi_mt_machine.h
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
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
#ifndef _KAAPI_MT_MACHINE_H_
#define _KAAPI_MT_MACHINE_H_ 1

#ifndef _KAAPI_IMPL_H
#  error This file must not be directly included. Use kaapi_impl.h instead
#endif
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


#include "kaapi_mt_bitmap.h"

#include "../../tasklist/kaapi_readytasklist.h"

#if defined(KAAPI_USE_CUDA)
#include "../cuda/kaapi_cuda_proc.h"
#endif

#if 0
#include "../../memory/kaapi_mem.h"
#include "../../memory/kaapi_mem_data.h"
#include "../../memory/kaapi_mem_host_map.h"
#endif

/* ========================================================================== */
struct kaapi_procinfo_t;
struct kaapi_event_buffer_t;
struct kaapi_readytasklist_t;

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

/* ========================================================================= */
/** Termination detecting barrier
*/
extern kaapi_atomic_t kaapi_term_barrier;

/* ========================================================================= */
/** Termination flag
*/
extern volatile int kaapi_isterm;

/* ========================================================================= */
/** \ingroup WS
    Number of used cores
*/
extern uint32_t volatile kaapi_count_kprocessors;

/** \ingroup WS
    One K-processors per core
*/
extern struct kaapi_processor_t** kaapi_all_kprocessors;

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
  kaapi_atomic_t               state;      /* 0: in the list, 1: out the list */
  kaapi_cpuset_t               affinity;   /* bit i == 1 -> can run on procid i */
  kaapi_thread_context_t*      thread;     /* */
  kaapi_wsqueuectxt_cell_ptr_t next;       /* double link list */
  kaapi_wsqueuectxt_cell_ptr_t prev;       /* shared with thief, used to link in free list */
} kaapi_wsqueuectxt_cell_t;

#define KAAPI_WSQUEUECELL_INLIST    1  /* a thread has been pushed into the queue */
#define KAAPI_WSQUEUECELL_READY     3  /* see kaapi wakeup, a suspended thread has been put ready */
#define KAAPI_WSQUEUECELL_OUTLIST   4  /* */
#define KAAPI_WSQUEUECELL_STEALLIST 8  /* the thread is under stealing */

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


/** lfree data structure
*/
typedef struct kaapi_lfree
{
  kaapi_thread_context_t* _front;
  kaapi_thread_context_t* _back;
} kaapi_lfree_t;

struct kaapi_taskdescr_t;

/** Mailbox for task, not td
*/
typedef struct kaapi_task_withlink_t {
  struct kaapi_task_withlink_t* volatile next;
  kaapi_task_t                           task;
} kaapi_task_withlink_t;

/** Mail box queue: remote thead may push stealtask into this queue.
*/
typedef  struct {
  kaapi_task_withlink_t*         volatile head;
  kaapi_task_withlink_t*         volatile tail;
}   __attribute__((aligned(KAAPI_CACHE_LINE))) kaapi_mailbox_queue_t;


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



/** \ingroup WS
*/
typedef struct kaapi_listrequest_t {
  kaapi_bitmap_t  bitmap __attribute__((aligned(KAAPI_CACHE_LINE)));  /* bit map of kproc id ! */
} kaapi_listrequest_t __attribute__((aligned (KAAPI_CACHE_LINE)));


/** \ingroup WS
    The global variable that store the list of requests where to reply
*/
extern  kaapi_request_t kaapi_global_requests_list[KAAPI_MAX_PROCESSOR+1];

#define KAAPI_USE_LRI_WITH_SNAPSHOT 1

#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
/* to iterate over list of requests: once an iterator has been captured, 
   the bitmap is reset to 0 into the listrequest.
   All the futur iterations will be done on top of captured bitmap, not
   those associated with the listrequest which can continue to receive requests
*/
typedef struct kaapi_listrequest_iterator_t {
  kaapi_bitmap_value_t bitmap;
  int idcurr;
#if defined(KAAPI_DEBUG)
  kaapi_bitmap_value_t bitmap_t0;
  uintptr_t            count_in;  // count bit captured
  uintptr_t            count_out; // count bit out (iterator_next )
#endif  
} kaapi_listrequest_iterator_t;

#else /* avoid to capture state at the begining of the steal */

/* New implementation to delay the date where bitmap is captured...
   to iterate over list of requests without capture of the request' initiators.
*/
typedef struct kaapi_listrequest_iterator_t {
  kaapi_bitmap_t* bitmap;
  int             idcurr;  /* advance on next, -1 if undefined */
} kaapi_listrequest_iterator_t;
#endif

/* return !=0 iff the range is empty
*/
static inline int kaapi_listrequest_iterator_empty(kaapi_listrequest_iterator_t* lrrange)
{ 
#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
  return kaapi_bitmap_value_empty(&lrrange->bitmap) && (lrrange->idcurr == -1); 
#else
  return kaapi_bitmap_empty(lrrange->bitmap);
#endif
}


/* clear the bit the at given position
*/
static inline void kaapi_listrequest_iterator_unset_at(
  kaapi_listrequest_iterator_t* lrrange, 
  int pos
)
{ 
#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
  kaapi_bitmap_value_unset(&lrrange->bitmap, pos); 
  if (lrrange->idcurr == pos)
  {
    lrrange->idcurr = -1;
    /* same as _iterator_next */
    lrrange->idcurr = kaapi_bitmap_value_first1_and_zero( &lrrange->bitmap )-1;
  }    
#else
  kaapi_bitmap_unset(lrrange->bitmap, pos); 
#endif
}

/* return the number of entries in the range
*/
static inline int kaapi_listrequest_iterator_count(
    kaapi_listrequest_iterator_t* lrrange
)
{ 
#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
  return kaapi_bitmap_value_count(&lrrange->bitmap) + (int)(lrrange->idcurr == -1 ? 0 : 1);
#else
  return kaapi_bitmap_count(lrrange->bitmap);
#endif  
}

/* return the bitmap of all requests */
static inline kaapi_bitmap_value_t kaapi_listrequest_iterator_bitmap(
    kaapi_listrequest_iterator_t* lrrange
)
{ 
  kaapi_bitmap_value_t retval;
#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
  kaapi_bitmap_value_set(&retval, (lrrange->idcurr == -1 ? 0 : lrrange->idcurr) );
  kaapi_bitmap_value_or( &retval, &lrrange->bitmap );
#else
  kaapi_bitmap_value_clear(&retval);
  kaapi_bitmap_value_or( &retval, lrrange->bitmap );
#endif
  return retval;
}

/* get the first request of the range. range iterator should have been initialized by kaapi_listrequest_iterator_init 
*/
static inline kaapi_request_t* kaapi_listrequest_iterator_get( 
  kaapi_listrequest_t* lrequests, kaapi_listrequest_iterator_t* lrrange 
)
{ 
#if defined(KAAPI_USE_LRI_WITH_SNAPSHOT)
  return (lrrange->idcurr == -1 ? 0 : &kaapi_global_requests_list[lrrange->idcurr]);
#else
  return (lrrange->idcurr == -1 ? 0 : &kaapi_global_requests_list[lrrange->idcurr]);
#endif
}

/* get the first request of the range. range iterator should have been initialized by kaapi_listrequest_iterator_init 
*/
static inline kaapi_request_t* kaapi_listrequest_iterator_getkid_andnext( 
  kaapi_listrequest_t* lrequests, kaapi_listrequest_iterator_t* lrrange, int kid 
)
{ 
  kaapi_assert_debug( (kid >=0) && (kid < (int)kaapi_count_kprocessors) );
  if (kid == lrrange->idcurr)
    lrrange->idcurr = kaapi_bitmap_value_first1_and_zero( &lrrange->bitmap )-1;
  else 
    kaapi_bitmap_value_unset( &lrrange->bitmap, kid );
  return &kaapi_global_requests_list[kid]; 
}

/* return the next entry in the request. return 0 if the range is empty.
*/
static inline kaapi_request_t* kaapi_listrequest_iterator_next( 
  kaapi_listrequest_t* lrequests, kaapi_listrequest_iterator_t* lrrange 
)
{
  lrrange->idcurr = kaapi_bitmap_value_first1_and_zero( &lrrange->bitmap )-1;
  kaapi_request_t* retval = (lrrange->idcurr == -1 ? 0 : &kaapi_global_requests_list[lrrange->idcurr]);
  return retval;
} 

/* atomically read the bitmap of the list of requests clear readed bits */
static inline void kaapi_listrequest_iterator_init(
  kaapi_listrequest_t* lrequests, kaapi_listrequest_iterator_t* lrrange
)
{ 
  lrrange->idcurr = -1;
  kaapi_bitmap_swap0( &lrequests->bitmap, &lrrange->bitmap );
#if defined(KAAPI_DEBUG)
  kaapi_bitmap_value_copy( &lrrange->bitmap_t0, &lrrange->bitmap );
  lrrange->count_in  = kaapi_bitmap_value_count(&lrrange->bitmap);
  lrrange->count_out = 0;
#endif
  kaapi_listrequest_iterator_next( lrequests, lrrange );
}

/* clear bitmap and init iterator */
static inline void kaapi_listrequest_iterator_prepare(
  kaapi_listrequest_iterator_t* lrrange
)
{ 
  lrrange->idcurr = -1;
  kaapi_bitmap_value_clear(&lrrange->bitmap);
#if defined(KAAPI_DEBUG)
  kaapi_bitmap_value_clear(&lrrange->bitmap_t0);
  lrrange->count_in  = 0;
  lrrange->count_out = 0;
#endif
}

/* atomically intersect the request bitmap with the
   current one, keeping only the bits in mask. the
   non masked bits are left untouched in target.
 */
static inline void kaapi_listrequest_iterator_update
(
 kaapi_listrequest_t* lrequests,
 kaapi_listrequest_iterator_t* lrrange,
 kaapi_bitmap_value_t* mask
)
{
  /* orig_bitmap = lrequest->bitmap & ~mask;
     lrrange->bitmap |= orig_bitmap & mask;
   */
  kaapi_bitmap_value_t neg_mask;
  kaapi_bitmap_value_t orig_bitmap;

  /* todo: optimize, mask can be stored neged, and can be ored */
  kaapi_bitmap_value_neg(&neg_mask, mask);

  /* atomic read and clear only the masked bits */
  kaapi_bitmap_and(&orig_bitmap, &lrequests->bitmap, &neg_mask);

  /* keep only the masked bits */
  kaapi_bitmap_value_and(&orig_bitmap, mask);

  /* add to the current bitmap */
  kaapi_bitmap_value_or(&lrrange->bitmap, &orig_bitmap);

#if defined(KAAPI_DEBUG)
  kaapi_bitmap_value_or( &lrrange->bitmap_t0, &orig_bitmap );
  lrrange->count_in  += kaapi_bitmap_value_count(&orig_bitmap);
#endif

  /* check if empty before nexting */
  if ((lrrange->idcurr ==-1) && !kaapi_bitmap_value_empty(&lrrange->bitmap))
    kaapi_listrequest_iterator_next( lrequests, lrrange );
}

/* clear the iterator internal bitmap */
static inline void kaapi_listrequest_iterator_clear
  (kaapi_listrequest_iterator_t* lrrange)
{
  kaapi_bitmap_value_clear(&lrrange->bitmap);
}

#if defined(KAAPI_DEBUG)
static inline void kaapi_listrequest_iterator_countreply
  (kaapi_listrequest_iterator_t* lrrange)
{
  ++lrrange->count_out;
}
#endif


/** CPU Memory hierarchy of the local machine
    points to the affiniset defined into the memory hierarchy.
    The array of kids and nkids is a local information that may
    depends on the view of the global map cpu2kid and kid2cpu.
*/
#define ENCORE_UNE_MACRO_DETAILLEE 8
typedef struct kaapi_onelevel_t {
  int                   nkids;   /* number of neighboors that shared memory at this level */
  int                   nsize;   /* allocation size of kids */
  kaapi_processor_id_t* kids;    /* kids[0..nkids-1] == kprocessor id in this hierarchy level */
  int                   nnotself;/* allocation size for notself */
  kaapi_processor_id_t* notself; /* kids[0..nnotself-1] == kid that shared parent set but not self set */
  kaapi_affinityset_t*  set;     /* set[i] set used to by kids[i] */
} kaapi_onelevel_t;

typedef struct kaapi_cpuhierarchy_t {
  unsigned short        depth;
  kaapi_onelevel_t      levels[ENCORE_UNE_MACRO_DETAILLEE];
} kaapi_cpuhierarchy_t;

/** \ingroup WS
    This data structure defines a work stealer processor kernel thread.
    A kprocessor is a container of tasks stored by stack.
    The kid is a system wide identifier. In the current version it only contains a local counter value
    from 0 to N-1.
*/
typedef struct kaapi_processor_t {
  kaapi_thread_context_t*  thread;                    /* current thread under execution */
  kaapi_processor_id_t     kid;                       /* Kprocessor id */

  /* cache align: only the lock */
  kaapi_lock_t             lock            /* all requests attached to each kprocessor ordered by increasing level */
    __attribute__((aligned(KAAPI_CACHE_LINE)));

  int volatile             isidle;                    /* true if kproc is idle (active thread is empty) */

  kaapi_mailbox_queue_t    mailbox;                   /* readylist for task (not td) */
  struct kaapi_readytasklist_t* rtl_remote;           /* readylist of task descriptors (remote push) */
  struct kaapi_readytasklist_t* rtl;
  
  kaapi_wsqueuectxt_t      lsuspend                   /* list of suspended context */
      __attribute__((aligned(KAAPI_CACHE_LINE)));

  /* free list */
  kaapi_lfree_t		         lfree;                     /* queue of free context */
  int                      sizelfree;                 /* size of the queue */
  
  unsigned int             seed;                      /* for the kproc own random generator */
  kaapi_selectvictim_fnc_t fnc_select;                /* function to select a victim */
  uintptr_t                fnc_selecarg[4];           /* arguments for select victim function, 0 at initialization */

  kaapi_emitsteal_fnc_t	   emitsteal;                 /* virtualization of the WS algorithm */
  void*                    emitsteal_ctxt;            /* specific to the WS algorithm */

#if defined(KAAPI_DEBUG)
  volatile uintptr_t       req_version;
  volatile uintptr_t       reply_version;
  volatile uintptr_t       compute_version;
#endif
  
  pthread_mutex_t          suspend_lock;              /* lock used to suspend / resume the threads */
    
  /* hierachical information of other kprocessor */
  int                      cpuid;                     /* os index of the bounded physical cpu */
  int                      numa_nodeid;               /* os index of the bounded physical memory ressource. See  kaapi_memory_id_t */
  kaapi_cpuhierarchy_t     hlevel;                    /* hierarchy */

  /* performance register used if library is configured with. Keep it for binary compatibility */
  kaapi_perf_counter_t	   perf_regs[2][KAAPI_PERF_ID_MAX];
  kaapi_perf_counter_t*	   curr_perf_regs;            /* either perf_regs[0], either perf_regs[1] */
  int	                     papi_event_set;
  unsigned int	           papi_event_count;
  kaapi_perf_counter_t     start_t[2];                /* [KAAPI_PERF_SCHEDULE_STATE]= T1 else = Tidle */
   
  double                   t_preempt;                 /* total idle time in second pass in the preemption */           

  const struct kaapi_procinfo_t* kpi;                 /* proc info */
  
  struct kaapi_event_buffer_t* eventbuffer;           /* event buffer */

#if defined(KAAPI_USE_PERFCOUNTER)
  uintptr_t                serial;                    /* serial number of generated steal */
  kaapi_perf_counter_t     lastcounter;               /* used by libgomp */
#endif

  kaapi_atomic64_t          workload;                 /* workload */
  unsigned int	            proc_type;                /* processor type */
  unsigned int              seed_data;                /* seed for kproc random generator */

  KAAPI_DEBUG_INST(
    struct kaapi_processor_t* victim_kproc;           /* used for debug */
  )

  void*                     data_specific[16];        /* thread data specific or scratch mode */
  size_t                    size_specific[16];        /* size of data specific or scratch mode */

  void*                     libkomp_tls;

  /* cuda */
#if defined(KAAPI_USE_CUDA)
  kaapi_cuda_proc_t         cuda_proc;
#endif

} kaapi_processor_t __attribute__ ((aligned (KAAPI_KPROCESSOR_ALIGNMENT_SIZE)));


/* Initialize an allocated object kaapi_processor_t 
*/
struct kaapi_procinfo;
extern int kaapi_processor_init( 
    kaapi_processor_t* kproc, 
    const struct kaapi_procinfo_t*,
    size_t stacksize
);

/* Destroy an allocated object kaapi_processor_t 
*/
extern int kaapi_processor_destroy(kaapi_processor_t* kproc);



/** Initialize the topology information from each thread
    Update kprocessor data structure only if kaapi_default_param.memory
    has been initialized by the kaapi_hw_init function.
    If hw cannot detect topology nothing is done.
    If KAAPI_CPUSET was not set but the topology is available, then the
    function will get the physical CPU on which the thread is running.
*/
extern int kaapi_processor_computetopo(kaapi_processor_t* kproc);

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

/** Wait until no more theft on the kprocessor
*/
extern void kaapi_synchronize_steal( kaapi_processor_t* );

/** Wait until no more theft on the thread
*/
extern void kaapi_synchronize_steal_thread( kaapi_thread_context_t* );


/* ........................................ PRIVATE INTERFACE ........................................*/
/** \ingroup TASK
    The function kaapi_context_alloc() allocates in the heap a context with a stack containing 
    at bytes for tasks and bytes for data.
    If successful, the kaapi_context_alloc() function will return a pointer to a kaapi_thread_context_t.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param kproc IN/OUT the kprocessor that make allocation
    \param stacksize IN the amount of stack data. If -1 use kaapi_default_param.stacksize 
    \retval pointer to the stack 
    \retval 0 if allocation failed
*/
extern kaapi_thread_context_t* kaapi_context_alloc( 
      kaapi_processor_t* kproc, 
      size_t stacksize );


/** \ingroup TASK
    The function kaapi_context_free() free the context successfuly allocated with kaapi_context_alloc.
    If successful, the kaapi_context_free() function will return zero.  
    Otherwise, an error number will be returned to indicate the error.
    This function is machine dependent.
    \param kproc IN/OUT the kprocessor that make allocation
    \param ctxt INOUT a pointer to the kaapi_thread_context_t to allocate.
    \retval EINVAL invalid argument: bad stack pointer.
*/
extern int kaapi_context_free( 
      kaapi_processor_t* kproc, 
      kaapi_thread_context_t* ctxt );


/* lfree list routines
 */
static inline void kaapi_lfree_clear(struct kaapi_processor_t* kproc)
{
  kproc->sizelfree = 0;
  kproc->lfree._front = NULL;
  kproc->lfree._back = NULL;
}

static inline int kaapi_lfree_isempty(struct kaapi_processor_t* kproc)
{
  return kproc->sizelfree == 0;
}

/* return 1 iff the ctxt is pushed into free list else return 0
*/
static inline int kaapi_lfree_push(
  struct kaapi_processor_t* kproc, 
  kaapi_thread_context_t* ctxt
)
{
  kaapi_lfree_t* const list = &kproc->lfree;

#  define KAAPI_MAXFREECTXT 4
  if (kproc->sizelfree >= KAAPI_MAXFREECTXT)
    return 0;

  /* push the node */
  ctxt->_next = list->_front;
  ctxt->_prev = NULL;
  if (list->_front == NULL)
    list->_back = ctxt;
  else
    list->_front->_prev = ctxt;
  list->_front = ctxt;

  ++kproc->sizelfree;

  return 1;
}

/* Re-design interface between context_free / context_alloc and lfree_push and lfree_pop
*/
static inline kaapi_thread_context_t* kaapi_lfree_pop(struct kaapi_processor_t* kproc)
{
  kaapi_lfree_t* const list = &kproc->lfree;
  kaapi_thread_context_t* const node = list->_front;

  --kproc->sizelfree;

  list->_front = node->_next;
  if (list->_front == NULL)
    list->_back = NULL;
  else
    list->_front->_prev = NULL;

  return node;
}


#if defined(KAAPI_HAVE_COMPILER_TLS_SUPPORT)
extern __thread kaapi_processor_t*      kaapi_current_processor_key;
extern __thread kaapi_threadgroup_t     kaapi_current_threadgroup_key;

/* */
static inline kaapi_processor_t* kaapi_get_current_processor(void)
{ return kaapi_current_processor_key; }

static inline kaapi_thread_context_t* kaapi_self_thread_context(void)
{ return kaapi_current_processor_key->thread;}

#else
extern pthread_key_t kaapi_current_processor_key;

static inline kaapi_processor_t* kaapi_get_current_processor(void)
{ return ((kaapi_processor_t*)pthread_getspecific( kaapi_current_processor_key )); }

static inline kaapi_thread_context_t* kaapi_self_thread_context(void)
{ return kaapi_get_current_processor()->thread;}
#endif

/* */
static inline kaapi_processor_id_t kaapi_get_current_kid(void)
{ return kaapi_get_current_processor()->kid; }

#if 1

#define kaapi_get_current_mem_host_map()      (0)
#define kaapi_processor_get_mem_host_map(a)   (0)

/* TODO remove */
//static inline kaapi_mem_host_map_t*
//kaapi_get_current_mem_host_map(void)
//{ return &kaapi_get_current_processor()->mem_host_map; }

//static inline kaapi_mem_host_map_t*
//kaapi_processor_get_mem_host_map( kaapi_processor_t* kproc )
//{ return &kproc->mem_host_map; }
#endif

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
  kaapi_bitmap_clear(&pklr->bitmap);
  return 0;
}


/* ============================= Private functions, machine dependent ============================ */
/* */
extern uint64_t kaapi_perf_thread_delayinstate(kaapi_processor_t* kproc);

/** Post a request to a given k-processor
  This method posts a request to victim k-processor.
  \param request where to post 
  \param kproc the sender of the request 
  \param reply where to receive result
  \param dest the receiver (victim) of the request
  \param return 0 if the request has been successully posted
  \param return !=0 if the request been not been successully posted and the status of the request contains the error code
*/
static inline kaapi_request_t* kaapi_request_post(
  kaapi_listrequest_t*   l_requests,
  kaapi_request_t*       request,
  kaapi_atomic_t*        status,
  const kaapi_frame_t*   frame 
)
{
  request->frame        = *frame;
  KAAPI_ATOMIC_WRITE(status, KAAPI_REQUEST_S_POSTED);
  request->status       = status;
  kaapi_writemem_barrier();
  kaapi_bitmap_set( &l_requests->bitmap, (int)request->ident );
  return request;
}


/** push: LIFO order with respect to pop. Only owner may push
*/
static inline int kaapi_wsqueuectxt_empty( const kaapi_processor_t* kproc )
{ return (kproc->lsuspend.head ==0); }

/**
*/
extern int kaapi_wsqueuectxt_init( kaapi_wsqueuectxt_t* ls );

/**
*/
extern int kaapi_wsqueuectxt_destroy( kaapi_wsqueuectxt_t* ls );

/** \ingroup WS
   Push a ctxt. Must be call by the owner of the queue in case of concurrent execution.
   Return 0 in case of success
   Return ENOMEM if allocation failed
*/
extern int kaapi_wsqueuectxt_push( kaapi_processor_t* kproc, kaapi_thread_context_t* thread );

/** \ingroup WS
   Steal a ctxt on a specific cell.
   After the thread has been stolen with success, the caller must call kaapi_wsqueuectxt_finish_steal_cell
   in order to allows garbage of cell.
   Return a pointer to the stolen thread in case of success
   Return 0 if the thread was already stolen
*/
extern kaapi_thread_context_t* kaapi_wsqueuectxt_steal_cell( kaapi_wsqueuectxt_cell_t* cell );

/**
*/
static inline void kaapi_wsqueuectxt_finish_steal_cell( kaapi_wsqueuectxt_cell_t* cell )
{
  int ok = KAAPI_ATOMIC_CAS( &cell->state, KAAPI_WSQUEUECELL_STEALLIST, KAAPI_WSQUEUECELL_OUTLIST);
  kaapi_assert( ok );
}

/**
*/
static inline unsigned int kaapi_processor_get_type(const kaapi_processor_t* kproc)
{
  return kproc->proc_type;
}

/** Return true if the kprocess has no work == is begin stealing and suspend list is empty
*/
static inline int kaapi_processor_has_nowork( const kaapi_processor_t* kproc )
{
  return (kproc->isidle !=0) && (kproc->mailbox.head ==0) && kaapi_readytasklist_isempty(kproc->rtl_remote)
    && kaapi_readytasklist_isempty(kproc->rtl)
    && kaapi_wsqueuectxt_empty(kproc);
}

/**
*/
static inline void kaapi_processor_set_workload
(kaapi_processor_t* kproc, unsigned long workload) 
{
  KAAPI_ATOMIC_WRITE(&kproc->workload, workload);
}

/**
*/
static inline int kaapi_processor_get_workload(const kaapi_processor_t* kproc) 
{
  return (int)KAAPI_ATOMIC_READ(&kproc->workload);
}

/**
*/
static inline void kaapi_processor_incr_workload
(kaapi_processor_t* kproc, unsigned long value) 
{
  KAAPI_ATOMIC_ADD(&kproc->workload, value);
}

/**
*/
static inline void kaapi_processor_decr_workload
(kaapi_processor_t* kproc, unsigned long value) 
{
  KAAPI_ATOMIC_SUB(&kproc->workload, value);
}

/**
*/
static inline void kaapi_processor_set_self_workload
(unsigned long workload) 
{
  KAAPI_ATOMIC_WRITE(&kaapi_get_current_processor()->workload, workload);
}

/* Return a pointer to a memory region at least of the given size else return 0.
   The id is a value between 0 and 16-1, where 16 is the maximal number of temporary.
   This function is used to allocate temporary data for task execution.
*/
extern void* _kaapi_gettemporary_data(kaapi_processor_t*kproc, unsigned int id, size_t size);

/* */
extern void kaapi_mt_perf_init(void);
/* */
extern void kaapi_mt_perf_fini(void);
/* */
extern void kaapi_mt_perf_thread_init ( kaapi_processor_t* kproc, int isuser );
/* */
extern void kaapi_mt_perf_thread_fini ( kaapi_processor_t* kproc );
/* */
extern void kaapi_mt_perf_thread_start ( kaapi_processor_t* kproc );
/* */
extern void kaapi_mt_perf_thread_stop ( kaapi_processor_t* kproc );
/* */
extern void kaapi_mt_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser );
/* */
extern uint64_t kaapi_mt_perf_thread_delayinstate(kaapi_processor_t* kproc);
/* */
extern size_t kaapi_mt_perf_counter_num(void);
/* */
extern const char* kaapi_mt_perf_id_to_name(kaapi_perf_id_t id);

#endif /* _KAAPI_MT_MACHINE_H */
