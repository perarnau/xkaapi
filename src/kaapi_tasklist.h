/*
** kaapi_staticsched.h
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com
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
#ifndef _KAAPI_TASKLIST_H_
#define _KAAPI_TASKLIST_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif
#include "config.h"
#include "kaapi_atomic.h"


/* .................................. Implementation notes ......................................*/

/* fwd decl */
struct kaapi_version_t;
struct kaapi_taskdescr_t;
struct kaapi_tasklist_t;


/** Tag for communication
*/
typedef uint64_t kaapi_comtag_t;


/** Activationlink
*/
typedef struct kaapi_activationlink_t {
  struct kaapi_taskdescr_t*      td;     /* the task descriptor to activate */
#if 0
  struct kaapi_tasklist_t*       queue;  /* where to push the task if activated, 0 == local task list */
#endif
  struct kaapi_activationlink_t* next;   /* next task  in the activation list */
} kaapi_activationlink_t;


/** ActivationList
*/
typedef struct kaapi_activationlist_t {
  kaapi_activationlink_t*   front;
  kaapi_activationlink_t*   back;
} kaapi_activationlist_t;


/** Activationlink for recv
*/
typedef struct kaapi_recvactlink_t {
  struct kaapi_taskdescr_t*      td;     /* the task descriptor to activate */
  struct kaapi_tasklist_t*       queue;  /* where to push the task if activated, 0 == local task list */
  struct kaapi_recvactlink_t*    next;   /* next task  in the activation list */
  kaapi_comtag_t                 tag;
} kaapi_recvactlink_t;


/** ActivationList for recv
*/
typedef struct kaapi_recv_list_t {
  kaapi_recvactlink_t*   front;
  kaapi_recvactlink_t*   back;
} kaapi_recv_list_t;


/**
*/
typedef struct kaapi_syncrecv_t {
  kaapi_activationlist_t    list;         /* list of tasks to activate */
  struct kaapi_syncrecv_t*  next;         /* used to push in recv list of a tasklist */
} kaapi_syncrecv_t;


/**
*/
typedef struct kaapi_move_arg_t {
  kaapi_data_t        src_data;
  kaapi_handle_t      dest;
} kaapi_move_arg_t;


/**
*/
typedef struct kaapi_finalizer_arg_t {
  kaapi_handle_t      dest;
} kaapi_finalizer_arg_t;


/** TaskDescriptor
    This data structure is add more information about task (kaapi_task_t)
    and it is pushed into the stack of each list of stack.
    To each task descriptor :
    - a task is associated (contains both the body and sp for its arguments)
    - a pointer to the next ready task (if any)
    - a FIFO list taskdescriptor to activate (push in back / pop in front).
    - a counter which correspond to the number of out dependencies.
    
    The prologue, if not null, is called after the task body execution. The
    prologue is used to encode broadcast information to other asid and remote
    threads inside each asid (one copy per asid + k signalisation --one per thread--).
    Because communication may be non blocking, the prologue is viewed as a task with
    its own activation list which are invoked to made ready tasks that may write
    data to communicate.
    
    The counter is decremented concurrently during synchronization or communication bcast/recv pair.
*/
typedef struct kaapi_taskdescr_t {
  kaapi_atomic_t                counter;   /* concurrent decr to test if task is ready */
  uint32_t                      wc;        /* value of counter before task becomes ready */
#if defined(KAAPI_TASKLIST_POINTER_TASK)
  kaapi_task_t*                 task;      /* the DFG task to execute */
#else
  kaapi_task_t                  task;      /* the DFG task to execute */
#endif
  const kaapi_format_t*         fmt;       /* format of the task */
  int                           priority;
  float				alpha;	    /* alpha value */
  kaapi_bitmap_value32_t        ocr;       /* OCR flag for the task */  
  kaapi_task_t                  tasksteal; /* used if task is pushed into remote queue */
  kaapi_taskstealready_arg_t    tasksteal_arg; /* put it together with taskdescr to avoid dynamic alloc */
  int                           mark;      /* used by some graph algorithm, initial value=0 */
  union {
    struct { /* case of tasklist use with precomputation of activation link */
      uint64_t                  date;      /* minimal logical date of production or critical path */
#if defined(KAAPI_DEBUG)
      uint64_t                  exec_date; /* execution date, debug only */
#endif
      kaapi_activationlist_t*   bcast;     /* list of bcast tasks activated to send data produced by this task */
      kaapi_activationlist_t    list;      /* list of tasks descr. activated after bcast list */
    } acl;
  } u;

} kaapi_taskdescr_t;


/** Initial capacity of each ready list work queue, used to avoid dynamic allocation for
    most of the case.
*/
#define KAAPI_TASKLIST_INITIAL_CAPACITY 32 

/** One workqueue of ready tasks 
*/
typedef struct kaapi_onereadytasklist_t {
  kaapi_workqueue_t       wq;           /* workqueue for ready tasks, used during runtime, 
                                           always: index beg or end <=0 */
  kaapi_workqueue_index_t next;         /* next rentry relative to base to pop */
  kaapi_taskdescr_t**     base;         /* data to the container base[begin]..base[0] */ 
  int                     size;         /* size of base != 0 if initial capacity was to small, which
                                           means that base if dynamically allocated */
  int                     dynallocated; /* ==1 if base was dynamically allocated */
  int                     task_pushed;  /* flag !=0 if one task has been pushed before the commit operation */
} kaapi_onereadytasklist_t;


/** The workqueue of ready tasks for several priority
*/
typedef struct kaapi_readytasklist_t {
  kaapi_onereadytasklist_t prl[KAAPI_TASKLIST_NUM_PRIORITY]; 
#if defined(KAAPI_DEBUG)
  kaapi_workqueue_index_t  max_task;
#endif
  kaapi_bitmap_value32_t   task_pushed;  /* bit ith== i-th priority workqueue has new pushed tasks */
  kaapi_taskdescr_t*       staticcontainer[KAAPI_TASKLIST_NUM_PRIORITY*KAAPI_TASKLIST_INITIAL_CAPACITY];
} kaapi_readytasklist_t;


/** TaskList
    This data structure is attached to a frame and must be considered as 
    an acceleratrice data structure in place of the standard FIFO queue of 
    tasks into a frame.
    The tasklist data structure stores the list of ready tasks as well as tasks 
    that will becomes ready on completion of previous tasks.
    At runntime, the list is managed as a LIFO queue of task descriptor: the most 
    recent pushed task descriptor is poped first. When the completion of a task 
    activate another tasks, they are
    pushed into the ready list.
    All data (task descriptors or activationlinks) are stores in allocator and 
    are deallocated in group at the end.
    
    The tasklist_t has an workqueue interface: push/pop and steal.
*/
typedef struct kaapi_tasklist_t {
  kaapi_lock_t            lock;        /* protect recvlist */
  kaapi_atomic_t          count_thief; /* count the number of thiefs for terminaison */

  /* execution state for ready task using tasklist */
  kaapi_readytasklist_t   rtl;        /* the workqueue of ready tasks */

  struct kaapi_thread_context_t* thread; /* thread that execute the task list */
  
  struct kaapi_tasklist_t*master;     /* master tasklist to signal at the end */
  kaapi_recvactlink_t*    recv;       /* next entry to receive */

  /* context to start or restart execution from suspend state */
  struct context_t {
    int                     chkpt;      /* see execframe_tasklist  */
    kaapi_taskdescr_t*      td;
    kaapi_frame_t*          fp;
    kaapi_workqueue_index_t local_beg;
  } context;

  /* constant state (after creation) */
  kaapi_activationlist_t  readylist;   /* readylist of task descriptor */
#if defined(KAAPI_DEBUG)
  kaapi_activationlist_t  allocated_td; /* list of all allocated tasks, debug only */
#endif
  uintptr_t               count_recv; /* number of extern synchronization to receive before detecting end of execution*/
  kaapi_recv_list_t       recvlist;   /* put by pushsignal into ready list to signal incomming data */
  kaapi_allocator_t       td_allocator;  /* where to push task descriptor */
  kaapi_allocator_t       allocator;  /* where to push other data structure */
  int64_t                 cnt_tasks[KAAPI_TASKLIST_NUM_PRIORITY];  /* number of tasks in the tasklist */
  uint64_t                t_infinity; /* length path in the graph of tasks */
#if defined(TASKLIST_ONEGLOBAL_MASTER) && !defined(TASKLIST_REPLY_ONETD)
  kaapi_atomic_t          pending_stealop;
#endif
  kaapi_atomic_t          cnt_exec;
  intptr_t                total_tasks;/* valid on master. Terminaison: on the master cnt_task == cnt_exec */
} kaapi_tasklist_t;


struct kaapi_version_t;

/** Link replicats of the same version
*/
typedef struct kaapi_link_version_t {
  struct kaapi_version_t*       version;        /* the version */
  struct kaapi_link_version_t*  next;           /* the next replicat version */
  kaapi_tasklist_t*             tl;             /* the container of task that acceed the version */
} kaapi_link_version_t;


/** Serves to detect: W -> R dependency or R -> W dependency but not yet cw...
*/
typedef struct kaapi_version_t {
  kaapi_access_mode_t      last_mode;       /* */
  kaapi_data_t*            handle;          /* */
  kaapi_taskdescr_t*       writer_task;     /* last writer task of the version, 0 if no indentify task (input data) */
} kaapi_version_t;


/** Find the version object associated to the addr.
    If not find, then insert into the table a new with 
      last_mode == KAAPI_ACCESS_MODE_VOID.
    The returned object should be initialized correctly 
    with a first initial access not KAAPI_ACCESS_MODE_VOID.
    See kaapi_version_add_initialaccess.
    On return islocal is set to 1 iff the access is local to the thread 'thread'.
    [Not: this is for partitioning into multiple threads; currently not used ]
*/
extern kaapi_version_t* kaapi_version_findinsert( 
    int* islocal,
    struct kaapi_thread_context_t* thread,
    kaapi_tasklist_t*       tl,
    const void*             addr 
);

/** Set the initial access of a version.
    The new version is associated with an initial task which
    dependent on the initial access mode (m=R|RM -> task move,
    m=W|CW -> task alloc).
    In case of move, the source data (host pointer) is supposed
    to be recopied into the remote address space.
*/
extern int kaapi_version_add_initialaccess( 
    kaapi_version_t*           ver, 
    kaapi_tasklist_t*          tl,
    kaapi_access_mode_t        m,
    void*                      data, 
    const kaapi_memory_view_t* view 
);


/**
*/
extern kaapi_version_t* kaapi_version_createreplicat( 
    kaapi_tasklist_t*       tl,
    kaapi_version_t*        master_version
);


/** Insert a synchronization points between the version and the master version
*/
extern int kaapi_thread_insert_synchro( 
    kaapi_tasklist_t*    tl,
    kaapi_version_t*     version, 
    kaapi_access_mode_t  m
);


/** Invalidate all replicats, except version
*/
extern int kaapi_version_invalidatereplicat( 
    kaapi_version_t*     version
);


/**/
static inline void kaapi_activationlist_clear( kaapi_activationlist_t* al )
{
  al->front = al->back = 0;
}


/**/
static inline int kaapi_activationlist_isempty( const kaapi_activationlist_t* al )
{
  return (al->front ==0);
}


/**/
static inline void kaapi_activationlist_pushback( kaapi_activationlist_t* al, kaapi_activationlink_t* l )
{
  l->next  = 0;
  if (al->back ==0)
    al->front = al->back = l;
  else {
    al->back->next = l;
    al->back = l;
  }
}


/**/
static inline kaapi_activationlink_t* kaapi_activationlist_popfront( kaapi_activationlist_t* al )
{
  kaapi_activationlink_t* retval = al->front;
  if (retval ==0) return 0;
  al->front = retval->next;
  if (retval->next == 0)
    al->back = 0;
  KAAPI_DEBUG_INST(retval->next  = 0);
  return retval;
}


/**/
static inline void kaapi_recvlist_clear( kaapi_recv_list_t* al )
{
  al->front = al->back = 0;
}


/**/
static inline int kaapi_recvlist_isempty( const kaapi_recv_list_t* al )
{
  return (al->front ==0);
}


/**/
static inline int kaapi_data_clear( kaapi_data_t* d )
{
  kaapi_pointer_setnull(&d->ptr);
  kaapi_memory_view_clear(&d->view);
  return 0;
}


/*
*/
static inline void kaapi_tasklist_newpriority_task( kaapi_tasklist_t* tasklist, int priority )
{
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );
  ++tasklist->cnt_tasks[priority];
}

/*
*/
extern int kaapi_tasklist_critical_path( kaapi_tasklist_t* tasklist );

/**/
static inline int kaapi_taskdescr_activated( kaapi_taskdescr_t* td)
{
  return (KAAPI_ATOMIC_INCR(&td->counter) % td->wc == 0);
}

/*
*/
static inline int kaapi_readytasklist_init( 
  kaapi_readytasklist_t* rtl,
  kaapi_lock_t*          lock
)
{
  int i;
  kaapi_taskdescr_t** base = rtl->staticcontainer;
#if defined(KAAPI_DEBUG)
  memset( base, ~0, sizeof(kaapi_taskdescr_t*)*KAAPI_TASKLIST_NUM_PRIORITY*KAAPI_TASKLIST_INITIAL_CAPACITY);
#endif

  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i)
  {
    kaapi_workqueue_init_with_lock(&rtl->prl[i].wq, 0, 0, lock);
    rtl->prl[i].next         = -1; /* next push to push, never use 0 == end positin outside base */
    rtl->prl[i].base         = base+KAAPI_TASKLIST_INITIAL_CAPACITY;  /* always store the "past the end entry" */
    rtl->prl[i].size         = KAAPI_TASKLIST_INITIAL_CAPACITY;
    rtl->prl[i].dynallocated = 0;
    rtl->prl[i].task_pushed  = 0;
    base += KAAPI_TASKLIST_INITIAL_CAPACITY;
  }
  KAAPI_DEBUG_INST( rtl->max_task = 0 );
  kaapi_bitmap_value_clear_32(&rtl->task_pushed);
  return 0;
}


/**/
static inline int kaapi_onereadytasklist_isempty( const kaapi_onereadytasklist_t* ortl )
{
  return (ortl->base ==0) || (ortl->next == -1) || kaapi_workqueue_isempty(&ortl->wq);
}

/* Here thread is only used to get a pointer in the stack where to store
   pointers to taskdescr during execution.
   It should be remove for partitionnig
*/
extern int kaapi_tasklist_init( kaapi_tasklist_t* tl, struct kaapi_thread_context_t* thread );

/*
*/
extern int kaapi_tasklist_destroy( kaapi_tasklist_t* tl );

/**/
static inline int kaapi_tasklist_isempty( kaapi_tasklist_t* tl )
{
  int i;
  if (tl ==0) return 1;
  if (!kaapi_recvlist_isempty(&tl->recvlist)) 
    return 0;
  for (i =0; i<KAAPI_TASKLIST_NUM_PRIORITY; ++i)
    if ( !kaapi_onereadytasklist_isempty( &tl->rtl.prl[i] ) )
      return 0;
  return 1;
}

/**/
static inline kaapi_activationlink_t* kaapi_allocator_allocate_al( kaapi_allocator_t* kal)
{
  kaapi_activationlink_t* retval =
      (kaapi_activationlink_t*)kaapi_allocator_allocate( kal, sizeof(kaapi_activationlink_t) );
  return retval;
}


/**/
static inline kaapi_activationlink_t* kaapi_tasklist_allocate_al( kaapi_tasklist_t* tl )
{
  return kaapi_allocator_allocate_al(&tl->allocator);
}


/**/
static inline kaapi_taskdescr_t* kaapi_allocator_allocate_td( 
    kaapi_allocator_t*    kal, 
    kaapi_task_t*         task, 
    const kaapi_format_t* task_fmt 
)
{
  kaapi_taskdescr_t* td = 
      (kaapi_taskdescr_t*)kaapi_allocator_allocate( kal, sizeof(kaapi_taskdescr_t) );
  KAAPI_ATOMIC_WRITE(&td->counter, 0);
  td->wc         = 0;
#if defined(KAAPI_TASKLIST_POINTER_TASK)
  td->task       = task;
#endif
  td->fmt 	     = task_fmt;
  /* TODO: here */
  td->priority   = KAAPI_TASKLIST_CPU_MIN_PRIORITY;
//  td->priority   = KAAPI_TASKLIST_MIN_PRIORITY;
#if 0 // DEPRECATED_ATTRIBUTE
  td->next       = 0;
  td->prev       = 0;
#endif
  kaapi_bitmap_value_clear_32(&td->ocr);       /* means no OCR */
  td->mark             = 0;
  td->u.acl.date       = 0;
  KAAPI_DEBUG_INST(td->u.acl.exec_date = 0);
  td->u.acl.bcast      = 0;
  td->u.acl.list.front = 0; 
  td->u.acl.list.back  = 0;
  return td;
}


/**/
static inline kaapi_taskdescr_t* kaapi_tasklist_allocate_td( 
    kaapi_tasklist_t*  tl, 
    kaapi_task_t*      task, 
    kaapi_format_t*    task_fmt
)
{
  kaapi_taskdescr_t* td;
  td = kaapi_allocator_allocate_td( &tl->td_allocator, task, task_fmt );
  ++tl->total_tasks;
  return td;
}


/**/
static inline kaapi_task_t* kaapi_tasklist_allocate_task( 
    kaapi_tasklist_t*  tl, 
    kaapi_task_bodyid_t body, 
    void* arg 
)
{
  kaapi_task_t* task = 
      (kaapi_task_t*)kaapi_allocator_allocate( &tl->allocator, sizeof(kaapi_task_t) );
  kaapi_task_init(task, body, arg);
  return task;
}

/* Push task in the front: the execution with revert it at the begining
*/
static inline void kaapi_tasklist_pushback_ready( kaapi_tasklist_t* tl, kaapi_taskdescr_t* td)
{
  kaapi_activationlink_t* al =
      (kaapi_activationlink_t*)kaapi_allocator_allocate( &tl->allocator, sizeof(kaapi_activationlink_t) );
  al->td    = td;
#if 0
  al->queue = 0;
#endif
  if (tl->readylist.front ==0)
  {
    al->next  = 0;
    tl->readylist.front = tl->readylist.back = al;
  } else {
    al->next = tl->readylist.front;
    tl->readylist.front = al;
  }
  /* call to reserved memory before execution without several memory allocation */
  kaapi_tasklist_newpriority_task( tl, td->priority );
}


/* activate and push all ready tasks in the activation list to their allocated queue
*/
extern int kaapi_tasklist_doactivationlist( kaapi_activationlist_t* al );


/**/
static inline void* kaapi_tasklist_allocate( kaapi_tasklist_t* tl, size_t size )
{
  void* retval = kaapi_allocator_allocate( &tl->allocator, size );
  return retval;
}


/** Push the task td_successor as a successor of task td.
    \param tl [IN/OUT] the task list that contains td. 
    The activation link is allocated into this tasklist structure
    \param td [IN/OUT] the predecessor of the task td_successor
    \param td_successor [IN] the successor of the task td to insert into tl_successor
*/
static inline void kaapi_tasklist_push_successor( 
    kaapi_tasklist_t*  tl, 
    kaapi_taskdescr_t* td, 
    kaapi_taskdescr_t* td_successor
)
{
  kaapi_activationlink_t* al = kaapi_tasklist_allocate_al(tl);

  al->td    = td_successor;
#if 0
  al->queue = tl;
#endif
  al->next  = 0;
  if (td->u.acl.list.back ==0)
    td->u.acl.list.front = td->u.acl.list.back = al;
  else {
    td->u.acl.list.back->next = al;
    td->u.acl.list.back = al;
  }

  /* one more synchronisation  */
  ++td_successor->wc;
}


/** Indicate that the task is waiting for a communication
*/
static inline void kaapi_tasklist_push_receivetask( 
    kaapi_tasklist_t*  tl, 
    kaapi_comtag_t     tag,
    kaapi_taskdescr_t* td
)
{
  kaapi_recvactlink_t* l 
    = (kaapi_recvactlink_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_recvactlink_t));
  l->td    = td;
  l->queue = tl;
  l->tag   = tag;
  l->next  = 0;
  if (tl->recvlist.back ==0)
    tl->recvlist.front = tl->recvlist.back = l;
  else {
    tl->recvlist.back->next = l;
    tl->recvlist.back = l;
  }

  /* one more synchronisation  */
  ++td->wc;
  ++tl->count_recv;
}


/** Push a broadcast task attached to a writer task
*/
extern void kaapi_tasklist_push_broadcasttask( 
    kaapi_tasklist_t*  tl, 
    kaapi_taskdescr_t* td_writer,
    kaapi_taskdescr_t* td_bcast
);


/** Compute the dependencies for the task with respect to access to global memory
    The task is not assumed to be pushed into the thread.
    Both computereadylist and online_computedef call this function.
    \retval 0 in case of success
*/
extern int kaapi_thread_computedep_task(
  struct kaapi_thread_context_t* thread, 
  kaapi_tasklist_t*       tasklist, 
  kaapi_task_t* task
);


/** Compute the readylist of the topframe of a thread
    \retval 0 in case of success
    \retval EBUSY if a ready list already exist for the thread
*/
extern int kaapi_thread_computereadylist( 
    struct kaapi_thread_context_t* thread, 
    kaapi_tasklist_t* tasklist 
);


/** Compute the synchronisation while pushing the task accessing with mode m to the data referenced
    by version into the tasklist tl.
*/
extern kaapi_data_t* kaapi_thread_computeready_access( 
    kaapi_tasklist_t*   tl, 
    kaapi_version_t*    version, 
    kaapi_taskdescr_t*  task,
    kaapi_access_mode_t m
);

/** Initialize task on the new declared version depending of the first access mode made by task
    For each new created version a first dummy task is pushed to create (allocated) or to allocate
    data for next tasks.
*/
extern int kaapi_thread_initialize_first_access( 
    kaapi_tasklist_t*   tl, 
    kaapi_version_t*    version, 
    kaapi_access_mode_t m,
    void*               srcdata    
);

/** Return the number of tasks inside the tasklist
*/
static inline size_t kaapi_readylist_workload( kaapi_readytasklist_t* rtl )
{
  size_t size = 0;
  kaapi_onereadytasklist_t* onertl;
  int i;

  for (i =KAAPI_TASKLIST_MAX_PRIORITY; i<(1+KAAPI_TASKLIST_MIN_PRIORITY); ++i)
  {
    onertl = &rtl->prl[i];
    if (onertl->next == -1) continue;
    size = 100*size+kaapi_workqueue_size(&onertl->wq);
  }
  return size;
}

/** To pop the next ready tasks
    Return the next task to execute if err ==0
    Return 0 if case of success
    Return EBUSY if the workqueue is empty
    Else return an error code
*/
static inline int kaapi_readylist_pop( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_onereadytasklist_t* onertl;
  int i,err;
  kaapi_workqueue_index_t local_end;
  kaapi_workqueue_index_t local_beg;

  for (i =KAAPI_TASKLIST_MAX_PRIORITY; i<(1+KAAPI_TASKLIST_MIN_PRIORITY); ++i)
  {
    onertl = &rtl->prl[i];
    if (onertl->next == -1) continue;
    err = kaapi_workqueue_pop(&onertl->wq, &local_beg, &local_end, 1);
    if (err ==0) 
    {
      kaapi_assert_debug( (local_beg< 0) && (local_beg >= -onertl->size) );
      *td = onertl->base[local_beg];
      /* next to push: */
      onertl->next = local_beg;
      return 0;
    }
    else if (err !=EBUSY) return err;
  }
  return EBUSY;
}

#if defined(KAAPI_USE_CUDA)

static inline int kaapi_readylist_pop_cpu( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_onereadytasklist_t* onertl;
  int i,err;
  kaapi_workqueue_index_t local_end;
  kaapi_workqueue_index_t local_beg;

  for( i = KAAPI_TASKLIST_CPU_MAX_PRIORITY;
	  (i >= 0) && (i >= KAAPI_TASKLIST_CPU_MIN_PRIORITY);
	  i--
    ) 
  {
    onertl = &rtl->prl[i];
    if (onertl->next == -1) continue;
    err = kaapi_workqueue_pop(&onertl->wq, &local_beg, &local_end, 1);
    if (err ==0) 
    {
      kaapi_assert_debug( (local_beg< 0) && (local_beg >= -onertl->size) );
      *td = onertl->base[local_beg];
      /* next to push: */
      onertl->next = local_beg;
#if 0
	if( (*td)->fmt == 0 ) {
	    fprintf(stdout,"%s: prio=%d\n", __FUNCTION__, i );
	    fflush(stdout);
	} else {
	    fprintf(stdout,"%s: prio=%d td=%p(wc=%d,name=%s)\n", __FUNCTION__, i,
		    (void*)*td,
		    (int)(*td)->wc,
		    (*td)->fmt->name
		);
	    fflush(stdout);
	}
#endif
      return 0;
    }
    else if (err !=EBUSY) return err;
  }
  return EBUSY;
}

static inline int kaapi_readylist_pop_gpu( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_onereadytasklist_t* onertl;
  int i,err;
  kaapi_workqueue_index_t local_end;
  kaapi_workqueue_index_t local_beg;

  for (i = KAAPI_TASKLIST_GPU_MAX_PRIORITY;
	  i <= KAAPI_TASKLIST_GPU_MIN_PRIORITY; i++ )
  {
    onertl = &rtl->prl[i];
    if (onertl->next == -1) continue;
    err = kaapi_workqueue_pop(&onertl->wq, &local_beg, &local_end, 1);
    if (err ==0) 
    {
      kaapi_assert_debug( (local_beg< 0) && (local_beg >= -onertl->size) );
      *td = onertl->base[local_beg];
      /* next to push: */
      onertl->next = local_beg;
#if 0
	if( (*td)->fmt == 0 ) {
	    fprintf(stdout,"%s: prio=%d\n", __FUNCTION__, i );
	    fflush(stdout);
	} else {
	    fprintf(stdout,"%s: prio=%d td=%p(wc=%d,name=%s)\n", __FUNCTION__, i,
		    (void*)*td,
		    (int)(*td)->wc,
		    (*td)->fmt->name
		);
	    fflush(stdout);
	}
#endif
      return 0;
    }
    else if (err !=EBUSY) return err;
  }
  return EBUSY;
}

#endif /* KAAPI_USE_CUDA */

/** Double the size of the workqueue.
    \retval 0 in case of success
    \retval ENOMEM if new allocation fail
*/
extern int _kaapi_readylist_extend_wq( kaapi_onereadytasklist_t* wq );

/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
static inline int kaapi_readylist_pushone_td( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td, int priority)
{
  kaapi_onereadytasklist_t* wq;
  kaapi_workqueue_index_t local_beg;
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );

  wq = &rtl->prl[priority];
  local_beg = wq->next; /* reuse the position of the previous executed task */
  if (local_beg < -wq->size)
    _kaapi_readylist_extend_wq( wq );

  kaapi_assert_debug( (local_beg <0) && (local_beg >= -wq->size) );
  wq->base[local_beg] = td;  /* <=> (base - wq->size)[-localbeg] */
  wq->next = --local_beg;
  wq->task_pushed = 1;
  kaapi_bitmap_value_set_32(&rtl->task_pushed, priority);
  KAAPI_DEBUG_INST( if (rtl->max_task < -local_beg) rtl->max_task = -local_beg  );
#if 0
  if( td->fmt != 0 )
      fprintf(stdout, "%s: pushed td=%p prio=%d name=%s\n", 
	      __FUNCTION__,
	      (void*)td, priority, td->fmt->name );
  else
      fprintf(stdout, "%s: pushed td=%p prio=%d\n", 
	      __FUNCTION__,
	      (void*)td, priority );
  fflush(stdout);
#endif

  return priority;
}

/** Push ready task in the correct queue
    \retval: 0 if local push into the tasklist
    \retval: 1 iff remote push into remote queue
*/
extern int kaapi_tasklist_pushready_td( 
    kaapi_tasklist_t*       tasklist, 
    kaapi_taskdescr_t*      td,
#if !defined(TASKLIST_ONEGLOBAL_MASTER)  
    kaapi_taskdescr_t**     tdref,
#endif
    int priority 
);

/** Activate and push ready tasks of an activation link.
    Return the number of ready tasks that have been activated. into local ready queue.
    Else return 0.
*/
static inline int kaapi_thread_tasklistready_pushactivated( 
    kaapi_tasklist_t*       tasklist, 
    kaapi_activationlink_t* head 
)
{
//  kaapi_readytasklist_t*  rtl = &tasklist->rtl;
  kaapi_taskdescr_t* td;
  int retval =0;
  
  while (head !=0)
  {
    td = head->td;
    if (kaapi_taskdescr_activated(td))
    {
      ++retval;
      kaapi_tasklist_pushready_td( 
              tasklist, 
              td, 
#if !defined(TASKLIST_ONEGLOBAL_MASTER)  
              &head->td,
#endif
              td->priority 
      );
    }
    head = head->next;
  }
  return retval; //0 != kaapi_bitmap_value_empty_32(&rtl->task_pushed);
}


/* Reserve capacity task per priority queue in order to avoid dynamic memory allocation.
   The initial distribution of the static container of size 
      KAAPI_TASKLIST_INITIAL_CAPACITY * KAAPI_TASKLIST_NUM_PRIORITY 
   is changed according to the new distribution in cnt_tasks.
   Thus, NO TASK can be pushed into each kaapi_onereadytasklist_t queue before calling
   this function.
   If called before filling the master tasklist, then no memory allocation will ocurrs
   except during this call.
*/
extern int kaapi_readytasklist_reserve( 
  kaapi_readytasklist_t* rtl,
  uint64_t               cnt_tasks[KAAPI_TASKLIST_NUM_PRIORITY]
);


/** Initialize the tasklist with a set of stolen task descriptors
*/
static inline int kaapi_thread_tasklistready_push_init_fromsteal( 
    kaapi_tasklist_t*       tasklist, 
    kaapi_taskdescr_t**     begin, 
    kaapi_taskdescr_t**     end
)
{
  kaapi_readytasklist_t* rtl = &tasklist->rtl;
  while (begin != end)
  {
    kaapi_tasklist_pushready_td(
        tasklist, 
        *begin, 
#if !defined(TASKLIST_ONEGLOBAL_MASTER)  
        begin,
#endif
        (*begin)->priority 
    );
    ++begin;
  }
  return 0 != kaapi_bitmap_value_empty_32(&rtl->task_pushed);
}


/** Push initial ready tasks list into the thread.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
static inline int kaapi_thread_tasklistready_push_init( kaapi_tasklist_t* tasklist, kaapi_activationlist_t* acl)
{
  kaapi_activationlink_t* head;
  kaapi_readytasklist_t* rtl = &tasklist->rtl;

  //kaapi_readytasklist_reserve( &tasklist->rtl, tasklist->cnt_tasks );

  head = acl->front;
  while (head !=0)
  {
    kaapi_tasklist_pushready_td(
        tasklist, 
        head->td, 
#if !defined(TASKLIST_ONEGLOBAL_MASTER)  
        &head->td,
#endif
        head->td->priority 
    );
    head = head->next;
  }
  return 0 != kaapi_bitmap_value_empty_32(&rtl->task_pushed);
}


/** Commit all previous pushed task descriptor to the other thieves.
    Return =0 in case of success
    Return 1 iff no task was pushed
*/
static inline int kaapi_thread_tasklist_commit_ready( kaapi_tasklist_t* tasklist )
{
  if (!kaapi_bitmap_value_empty_32(&tasklist->rtl.task_pushed))
  {
    /* ABA problem here if we suppress lock/unlock? seems to be true */
    //kaapi_atomic_lock( &tasklist->thread->stack.lock );
    while (!kaapi_bitmap_value_empty_32(&tasklist->rtl.task_pushed))
    {
      int ith = kaapi_bitmap_value_first1_and_zero_32( &tasklist->rtl.task_pushed ) -1;

kaapi_processor_incr_workload(kaapi_get_current_processor(), 
    kaapi_workqueue_range_begin( &tasklist->rtl.prl[ith].wq ) - (1+tasklist->rtl.prl[ith].next)
);
      /* do not keep tasklist->next for local exec */
      kaapi_workqueue_push(&tasklist->rtl.prl[ith].wq, 1+tasklist->rtl.prl[ith].next); 

    }
    //kaapi_atomic_unlock( &tasklist->thread->stack.lock );
    kaapi_bitmap_value_clear_32(&tasklist->rtl.task_pushed);
    return 0;
  }
  return 1;
}


/* TODO inverse order to CPU */
/** Commit all previous pushed task descriptor to the other thieves.
    Return !=0 next task descriptor to execute, else return 0.
    In case tasks have been pushed, the function reserves the last pushed task
    for local execution and returns it.
*/
#if defined(KAAPI_USE_CUDA)
static inline kaapi_taskdescr_t* kaapi_thread_tasklist_commit_ready_and_steal_gpu( 
    kaapi_tasklist_t* tasklist
)
{
    kaapi_thread_tasklist_commit_ready( tasklist );
    kaapi_taskdescr_t* td = 0;
    if( kaapi_readylist_pop_gpu( &tasklist->rtl, &td ) == 0 )
	return td;
    
    return 0;
}

#endif

static inline kaapi_taskdescr_t* kaapi_thread_tasklist_commit_ready_and_steal( 
    kaapi_tasklist_t* tasklist
)
{
  kaapi_onereadytasklist_t* onertl;
  kaapi_taskdescr_t* td_steal = 0;
  
  if (!kaapi_bitmap_value_empty_32(&tasklist->rtl.task_pushed))
  {
    /* ABA problem here if we suppress lock/unlock? seems to be true */
    while (!kaapi_bitmap_value_empty_32(&tasklist->rtl.task_pushed))
    {
      int ith = kaapi_bitmap_value_first1_and_zero_32( &tasklist->rtl.task_pushed ) -1;
      onertl = &tasklist->rtl.prl[ith];
      if (td_steal == 0)
      {
        kaapi_assert_debug( ((1+onertl->next)< 0) && ((onertl->next+1) >= -onertl->size) );
        td_steal = onertl->base[++onertl->next];
      }
	kaapi_processor_incr_workload(kaapi_get_current_processor(), 
	    kaapi_workqueue_range_begin( &onertl->wq ) - (1+onertl->next)
	);
      kaapi_workqueue_lock( &onertl->wq );
      kaapi_workqueue_push(&onertl->wq, 1+onertl->next); 
      kaapi_workqueue_unlock( &onertl->wq );
    }
  }

  return td_steal;
}

/** Used to steal a tasklist in the frame
*/
extern void kaapi_sched_stealtasklist( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_tasklist_t*             tasklist, 
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
);

#if defined(KAAPI_USE_CUDA)
extern void kaapi_sched_stealtasklist_cpu( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_tasklist_t*             tasklist, 
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
);

extern void kaapi_sched_stealtasklist_gpu( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_tasklist_t*             tasklist, 
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
);
#endif


/** How to execute task with readylist
    It is assumed that top frame is a frame with a ready list.
*/
extern int kaapi_thread_execframe_tasklist( struct kaapi_thread_context_t* thread );

/**
*/
extern int kaapi_thread_abstractexec_readylist( 
  const kaapi_tasklist_t* tasklist, 
  void (*taskdescr_executor)(kaapi_taskdescr_t*, void*),
  void* arg_executor
);

/**
*/
extern int kaapi_thread_tasklist_print( FILE* file, kaapi_tasklist_t* tl );

/*
*/
extern int kaapi_thread_tasklist_print_dot ( FILE* file, const kaapi_tasklist_t* tasklist, int clusterflags );

/** Generate the tasks of a frame in the dot format to display the data flow graph
    \param file the output file
    \param frame the input frame
    \param clusterflag is equal to 0 iff no cluster dot generation
*/
extern int kaapi_frame_print_dot  ( FILE* file, const kaapi_frame_t* frame, int clusterflag );



/** Manage synchronisation between several tid 
    Each thread may communicate between them through FIFO queue to signal incomming data
*/
extern int kaapi_tasklist_pushsignal( kaapi_pointer_t rsignal );


#if defined(KAAPI_DEBUG)
extern void kaapi_print_state_tasklist( kaapi_tasklist_t* tl );
#endif


/**
*/
extern void kaapi_thread_signalend_exec( kaapi_thread_context_t* thread );

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
