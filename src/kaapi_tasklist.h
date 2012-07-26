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
#include "kaapi_affinity.h"

/* .................................. Implementation notes ......................................*/

/* fwd decl */
struct kaapi_taskdescr_t;
struct kaapi_tasklist_t;
struct kaapi_frame_tasklist_t;


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
  kaapi_task_t*                 task;      /* the DFG task to execute */
  const kaapi_format_t*         fmt;       /* format of the task */
  int                           priority;
  kaapi_bitmap_value32_t        ocr;       /* OCR flag for the task */  
  int                           mark;      /* used by some graph algorithm, initial value=0 */

  struct kaapi_tasklist_t*  tasklist;      /* the owner, initial master tasklist */
  struct kaapi_taskdescr_t*	prev;
  struct kaapi_taskdescr_t*	next;

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

#include "tasklist/kaapi_readytasklist.h"

/** TaskList
    This data structure is used to execute a list of task descriptors computed into 
    a frame_tasklist_t data structure.

    At runntime, the list is managed as a LIFO queue of task descriptors: the most 
    recent pushed task descriptor is poped first. When the completion of a task 
    activate another tasks, they are
    pushed into the ready list.
    All data (task descriptors or activationlinks) are stores in allocator and 
    are deallocated in group at the end.
    
    The tasklist_t has an workqueue interface: push/pop and steal.
*/
typedef struct kaapi_tasklist_t {
  kaapi_readytasklist_t          rtl;            /* the workqueue of ready tasks */
  struct kaapi_tasklist_t*       master;         /* the initial tasklist or 0 */
  struct kaapi_frame_tasklist_t* frame_tasklist; /* if master ==0, then the initial frame_tasklist, or 0 */
  kaapi_atomic_t                 cnt_exec;
  intptr_t                       total_tasks;/* total number of tasks in the frame */
} kaapi_tasklist_t;


/** frame TaskList
    This data structure is attached to a frame and must be considered as 
    an acceleratrice data structure in place of the standard FIFO queue of 
    tasks into a frame.
    The frame tasklist data structure stores the list of ready tasks as well as tasks 
    that will becomes ready on completion of previous tasks.
    
    At runtime the tasks (task descriptors) are managed using the kaapi_tasklist_t which
    store data for:
      - executing tasks (list of ready tasks)
      - final synchronization.
    
    The frame_tasklist cannot be directly steal, but the list of ready tasks can.
*/
typedef struct kaapi_frame_tasklist_t {
  kaapi_tasklist_t        tasklist;   /* initial task list where ready tasks are initially pushed */  
  kaapi_recvactlink_t*    recv;       /* next entry to receive. NOT USED YET */

  /* constant data fields (after creation) */
  kaapi_activationlist_t  readylist;   /* readylist of task descriptor */
#if defined(KAAPI_DEBUG)
  kaapi_activationlist_t  allocated_td; /* list of all allocated tasks, debug only */
#endif
  uintptr_t               count_recv; /* number of extern synchronization to receive before detecting end of execution*/
  kaapi_recv_list_t       recvlist;   /* put by pushsignal into ready list to signal incomming data */
  kaapi_allocator_t       td_allocator;  /* where to push task descriptor */
  kaapi_allocator_t       allocator;  /* where to push other data structure */
  uint64_t                t_infinity; /* length path in the graph of tasks */
#if !defined(TASKLIST_REPLY_ONETD)
  kaapi_atomic_t          pending_stealop;
#endif
} kaapi_frame_tasklist_t;



#include "tasklist/kaapi_version.h"

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


/*
*/
static inline void kaapi_tasklist_newpriority_task( int priority )
{
  kaapi_assert_debug( (priority >= KAAPI_TASKLIST_MAX_PRIORITY) && (priority <= KAAPI_TASKLIST_MIN_PRIORITY) );
}

/*
*/
extern int kaapi_tasklist_critical_path( kaapi_frame_tasklist_t* tasklist );

/**/
static inline int kaapi_taskdescr_activated( kaapi_taskdescr_t* td)
{
  return (KAAPI_ATOMIC_INCR(&td->counter) % td->wc == 0);
}

/* Here thread is only used to get a pointer in the stack where to store
   pointers to taskdescr during execution.
   It should be remove for partitionnig
*/
extern int kaapi_frame_tasklist_init( kaapi_frame_tasklist_t* tl, struct kaapi_thread_context_t* thread );

/* Initialize a worker tasklist given the master task list in parameter
*/
static inline int kaapi_tasklist_init( kaapi_tasklist_t* tl, kaapi_tasklist_t* mtl )
{
  kaapi_readytasklist_init(&tl->rtl);
  
  tl->master         = mtl;
  tl->frame_tasklist = 0;
  tl->total_tasks    = 0;

  KAAPI_ATOMIC_WRITE(&tl->cnt_exec, 0);

  return 0;
}

static inline int kaapi_tasklist_destroy( kaapi_tasklist_t* tl )
{
  kaapi_readytasklist_destroy(&tl->rtl);
  
  return 0;
}

/*
*/
extern int kaapi_frame_tasklist_destroy( kaapi_frame_tasklist_t* tl );

/**/
static inline int kaapi_tasklist_isempty( kaapi_tasklist_t* tl )
{
  return kaapi_readytasklist_isempty(&tl->rtl);
}

/**/
static inline kaapi_activationlink_t* kaapi_allocator_allocate_al( kaapi_allocator_t* kal)
{
  kaapi_activationlink_t* retval =
      (kaapi_activationlink_t*)kaapi_allocator_allocate( kal, sizeof(kaapi_activationlink_t) );
  return retval;
}


/**/
static inline kaapi_activationlink_t* kaapi_tasklist_allocate_al( kaapi_frame_tasklist_t* tl )
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
  td->task       = task;
  td->fmt 	     = task_fmt;
  td->priority   = KAAPI_TASKLIST_CPU_MIN_PRIORITY;
  td->next       = 0;
  td->prev       = 0;
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
    kaapi_frame_tasklist_t*  tl, 
    kaapi_task_t*      task, 
    kaapi_format_t*    task_fmt
)
{
  kaapi_taskdescr_t* td;
  td = kaapi_allocator_allocate_td( &tl->td_allocator, task, task_fmt );
  td->tasklist = &tl->tasklist;
  ++tl->tasklist.total_tasks;
  return td;
}


/**/
static inline kaapi_task_t* kaapi_tasklist_allocate_task( 
    kaapi_frame_tasklist_t*  tl, 
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
static inline void kaapi_frame_tasklist_pushback_ready( kaapi_frame_tasklist_t* tl, kaapi_taskdescr_t* td)
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
  kaapi_tasklist_newpriority_task( td->priority );
}


/* activate and push all ready tasks in the activation list to their allocated queue
   - currently unused. To be recoded.
*/
__attribute__((deprecated))
extern int kaapi_tasklist_doactivationlist( kaapi_tasklist_t* tl, kaapi_activationlist_t* al );


/**/
static inline void* kaapi_tasklist_allocate( kaapi_frame_tasklist_t* tl, size_t size )
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
    kaapi_frame_tasklist_t*  tl, 
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
static inline void kaapi_frame_tasklist_push_receivetask( 
    kaapi_frame_tasklist_t*  tl, 
    kaapi_comtag_t     tag,
    kaapi_taskdescr_t* td
)
{
  kaapi_recvactlink_t* l 
    = (kaapi_recvactlink_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_recvactlink_t));
  l->td    = td;
  l->queue = 0;
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
extern void kaapi_frame_tasklist_push_broadcasttask( 
    kaapi_frame_tasklist_t*  tl, 
    kaapi_taskdescr_t*       td_writer,
    kaapi_taskdescr_t*       td_bcast
);


/** Compute the dependencies for the task with respect to access to global memory
    The task is not assumed to be pushed into the thread.
    Both computereadylist and online_computedef call this function.
    \retval 0 in case of success
*/
extern int kaapi_thread_computedep_task(
  struct kaapi_thread_context_t* thread, 
  kaapi_frame_tasklist_t*        tasklist, 
  kaapi_task_t*                   task
);


/** Compute the readylist of the topframe of a thread
    \retval 0 in case of success
    \retval EBUSY if a ready list already exist for the thread
*/
extern int kaapi_thread_computereadylist( 
    struct kaapi_thread_context_t* thread, 
    kaapi_frame_tasklist_t*        ftl 
);


/** Compute the synchronisation while pushing the task accessing with mode m to the data referenced
    by version into the tasklist tl.
*/
extern kaapi_data_t* kaapi_thread_computeready_access( 
    kaapi_frame_tasklist_t* ftl, 
    kaapi_version_t*        version, 
    kaapi_taskdescr_t*      task,
    kaapi_access_mode_t     m
);

/** Initialize task on the new declared version depending of the first access mode made by task
    For each new created version a first dummy task is pushed to create (allocated) or to allocate
    data for next tasks.
*/
extern int kaapi_thread_initialize_first_access( 
    kaapi_frame_tasklist_t* ftl, 
    kaapi_version_t*        version, 
    kaapi_access_mode_t     m,
    void*                   srcdata    
);

/** Push one ready task to the correct queue.
 * 1) it tests the task priority and
 * 2) affinity (if enabled)
 * 3) them pushes in the readylist rtl 
*/
static inline int kaapi_readytasklist_pushready_td( 
    kaapi_readytasklist_t*  rtl, 
    kaapi_taskdescr_t*      td,
    int priority 
)
{
    if( kaapi_processor_get_type(kaapi_get_current_processor()) ==
	    KAAPI_PROC_TYPE_CUDA ) {
	if( td->priority > KAAPI_TASKLIST_GPU_MIN_PRIORITY ) {
	    kaapi_assert_debug( td->tasklist != NULL );
	    return kaapi_readylist_push( &td->tasklist->rtl, td, priority );
	}
    } else {
	if( td->priority > KAAPI_TASKLIST_GPU_MIN_PRIORITY ) {
	  return kaapi_readylist_push( rtl, td, priority );
	}
    }

  kaapi_processor_t* kproc_remote =
    kaapi_get_current_processor()->affinity( kaapi_get_current_processor(), td );
  if( kproc_remote != kaapi_get_current_processor() ) {
      return kaapi_readylist_remote_push( kproc_remote->rtl, td, priority );
  }
  return kaapi_readylist_push( rtl, td, priority );
}

static inline int kaapi_readytasklist_push_from_activationlist( 
    kaapi_readytasklist_t*  rtl, 
    kaapi_activationlink_t*	head 
)
{
  kaapi_taskdescr_t* td;
  int retval =0;
  
  while (head !=0) {
    td = head->td;
    if (kaapi_taskdescr_activated(td)) {
      ++retval;
      kaapi_readytasklist_pushready_td( 
                                       rtl, 
                                       td, 
                                       td->priority 
      );
    }
    head = head->next;
  }
  return retval;
}

static inline uint32_t kaapi_readytasklist_pushactivated(
      kaapi_readytasklist_t*  rtl, 
      kaapi_taskdescr_t*	    td 
)
{
  uint32_t cnt_pushed= 0;
  
  /* push in the front the activated tasks */
  if (!kaapi_activationlist_isempty(&td->u.acl.list))
    cnt_pushed =
	  kaapi_readytasklist_push_from_activationlist( rtl, td->u.acl.list.front );
  else 
    cnt_pushed = 0;
  
  /* do bcast after child execution (they can produce output data) */
  if (td->u.acl.bcast !=0) 
    cnt_pushed +=
    kaapi_readytasklist_push_from_activationlist( rtl, td->u.acl.bcast->front );
  
  return cnt_pushed;
}

/* Push all activated tasks from td */
__attribute__((deprecated))
static inline uint32_t kaapi_tasklist_pushactivated(
	kaapi_tasklist_t*	tasklist,
	kaapi_taskdescr_t*	td 
    )
{
  return kaapi_readytasklist_pushactivated( &tasklist->rtl, td );
}

/** Initialize the tasklist with a set of stolen task descriptors
*/
static inline int kaapi_tasklistready_push_init_fromsteal( 
    kaapi_tasklist_t*       tasklist, 
    kaapi_taskdescr_t**     begin, 
    kaapi_taskdescr_t**     end
)
{
//  kaapi_readytasklist_t* rtl = &tasklist->rtl;
  while (begin != end)
  {
    kaapi_readytasklist_pushready_td(
        &tasklist->rtl, 
        *begin, 
        (*begin)->priority 
    );
    ++begin;
  }
  return 0;
}


/** Push initial ready tasks list into the thread.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
    DEPRECTATED: use kaapi_readytasklist_push_from_activationlist
*/
static inline int kaapi_thread_tasklistready_push_init(
	kaapi_tasklist_t* tasklist, 
  kaapi_activationlist_t* acl
)
{
  kaapi_activationlink_t* head;
  kaapi_readytasklist_t* rtl = &tasklist->rtl;

  head = acl->front;
  while (head !=0)
  {
    kaapi_readytasklist_pushready_td(
        rtl, 
        head->td, 
        head->td->priority 
    );
    head = head->next;
  }
  return 0;
}

/**
*/
extern int kaapi_thread_abstractexec_readylist( 
  const kaapi_frame_tasklist_t* tasklist, 
  void (*taskdescr_executor)(kaapi_taskdescr_t*, void*),
  void* arg_executor
);

/**
*/
extern int kaapi_frame_tasklist_print( FILE* file, kaapi_frame_tasklist_t* tl );

/*
    DEPRECTATED: use kaapi_frame_tasklist_print
*/
__attribute__((deprecated))
static inline int kaapi_thread_tasklist_print( FILE* file, kaapi_frame_tasklist_t* tl )
{
  return kaapi_frame_tasklist_print(file, tl);
}

/*
*/
extern int kaapi_frame_tasklist_print_dot ( FILE* file, const kaapi_frame_tasklist_t* tasklist, int clusterflags );

/*
    DEPRECTATED: use kaapi_frame_tasklist_print
*/
__attribute__((deprecated))
static inline int kaapi_thread_tasklist_print_dot ( FILE* file, const kaapi_frame_tasklist_t* tasklist, int clusterflags )
{ return kaapi_frame_tasklist_print_dot(file,tasklist,clusterflags); }

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
extern void kaapi_print_state_tasklist( kaapi_frame_tasklist_t* tl );
#endif


/**
*/
extern void kaapi_thread_signalend_exec( kaapi_thread_context_t* thread );


#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
