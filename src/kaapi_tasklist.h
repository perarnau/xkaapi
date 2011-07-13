/*
** kaapi_staticsched.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
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
#define _KAAPI_TASKLIST_H_

#if defined(__cplusplus)
extern "C" {
#endif

/* ........................................ Implementation notes ........................................*/

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
  struct kaapi_tasklist_t*       queue;  /* where to push the task if activated, 0 == local task list */
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


typedef enum {
  KAAPI_TASKACL_CASE,         /* task with activation link */
  KAAPI_TASKDFG_CASE          /* task stolen from DFG */
} kaapi_taskdescr_type;

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
  kaapi_task_t                  task;      /* the task to executed */
  const kaapi_format_t*         fmt;       /* format of the task */
  struct kaapi_taskdescr_t*     next;      /* next / prev to link in affinity queue */
  struct kaapi_taskdescr_t*     prev;      /* next / prev to link in affinity queue */
  struct kaapi_tasklist_t*      tl;        /* owner of the td: the tasklist to signal */
  
  kaapi_taskdescr_type          type;
  union {
    struct { /* case of tasklist use with precomputation of activation link */
      kaapi_cpuset_t            mapping;
      uint64_t                  date;      /* minimal logical date of production or critical path */
#if defined(KAAPI_DEBUG)
      uint64_t                  exec_date; /* execution date, debug only */
#endif
      kaapi_activationlist_t*   bcast;     /* list of bcast tasks activated to send data produced by this task */
      kaapi_activationlist_t    list;      /* list of tasks descr. activated after bcast list */     
    } acl;
    struct { /* case of stealing task from origin DFG stack */
      kaapi_thread_context_t*   thread;
      unsigned int              war;
    } dfg;
  } u;

} kaapi_taskdescr_t;


/** Workqueue of ready task list with init, push, pop and steal method
    as for other workqueue
*/
typedef struct kaapi_readytasklist_t {
  kaapi_workqueue_t       wq;           /* workqueue for ready tasks, used during runtime, always <=0 index */
  kaapi_workqueue_index_t next;         /* next rentry relative to base to pop */
  kaapi_taskdescr_t**     base;         /* data to the container base[begin]..base[0] */ 
  int                     task_pushed;  /*!=0 if one task has been pushed before the commit operation */
  int                     size;         /* number of pushed tasks */
} kaapi_readytasklist_t;


/** TaskList
    This data structure is attached to a frame and must be considered as an acceleratrice data structure
    in place of the standard FIFO queue of tasks into a frame.
    The tasklist data structure stores the list of ready tasks as well as tasks that will becomes
    ready on completion of previous tasks.
    At runntime, the list is managed as a LIFO queue of task descriptor: the most recent pushed
    task descriptor is poped first. When the completion of a task activate another tasks, they are
    pushed into the ready list.
    All data (task descriptors or activationlinks) are stores in allocator and are deallocated
    in group at the end.
    
    The tasklist_t has an workqueue interface: push/pop and steal.
*/
typedef struct kaapi_tasklist_t {
  kaapi_atomic_t          lock;        /* protect recvlist */
  kaapi_atomic_t          count_thief; /* count the number of thiefs for terminaison */

  /* execution state for ready task using tasklist */
  kaapi_readytasklist_t   rtl;        /* the workqueue of ready tasks */

  kaapi_thread_context_t* thread;     /* thread that execute the task list */
  
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
  uintptr_t               count_recv; /* number of extern synchronization to receive before detecting end of execution */
  kaapi_recv_list_t       recvlist;   /* put by pushsignal into ready list to signal incomming data */
  kaapi_allocator_t       allocator;  /* where to push task descriptor and other data structure */
  uint64_t                cnt_tasks;  /* number of tasks in the tasklist */
  uint64_t                t_infinity; /* length path in the graph of tasks */
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
#if 0
  kaapi_link_version_t*    master;          /* points to the master or 0 */
#endif
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
    kaapi_thread_context_t* thread,
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


/**/
static inline int kaapi_taskdescr_activated( kaapi_taskdescr_t* td)
{
  return (KAAPI_ATOMIC_INCR(&td->counter) % td->wc == 0);
}

/*
*/
static inline int kaapi_readytasklist_init( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** container )
{
  kaapi_workqueue_init(&rtl->wq, 0, 0);
  rtl->next        = 0;
  rtl->base        = container;  
  rtl->task_pushed = 0;
  return 0;
}

/* Here thread is only used to get a pointer in the stack where to store
   pointers to taskdescr during execution.
   It should be remove for partitionnig
*/
static inline int kaapi_tasklist_init( kaapi_tasklist_t* tl, kaapi_thread_context_t* thread )
{
  kaapi_sched_initlock(&tl->lock);
  KAAPI_ATOMIC_WRITE(&tl->count_thief, 0);

  kaapi_readytasklist_init( &tl->rtl, (kaapi_taskdescr_t**)thread->sfp->sp );

  tl->master          = 0;
  tl->thread          = thread;
  tl->recv            = 0;
  tl->context.chkpt   = 0;
#if defined(KAAPI_DEBUG)  
  tl->context.fp      = 0;
  tl->context.td      = 0;
#endif  
  tl->count_recv      = 0;
  kaapi_activationlist_clear( &tl->readylist );
#if defined(KAAPI_DEBUG)
  kaapi_activationlist_clear( &tl->allocated_td );
#endif
  kaapi_recvlist_clear(&tl->recvlist);
  kaapi_allocator_init( &tl->allocator );
  tl->cnt_tasks     = 0;
  tl->t_infinity    = 0;
  return 0;
}

/**/
static inline int kaapi_tasklist_destroy( kaapi_tasklist_t* tl )
{
  tl->rtl.base = 0;
  kaapi_workqueue_destroy( &tl->rtl.wq );
  kaapi_allocator_destroy( &tl->allocator );
  return 0;
}

/**/
static inline int kaapi_tasklist_isempty( kaapi_tasklist_t* tl )
{
  return (tl ==0) || 
    (
     (tl->rtl.base !=0) && 
      kaapi_workqueue_isempty(&tl->rtl.wq) && 
      kaapi_recvlist_isempty(&tl->recvlist)
    );
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
  td->type       = KAAPI_TASKACL_CASE;
  td->wc         = 0;
  td->task       = *task;
  td->fmt 	     = task_fmt;
  td->next       = 0;
  td->prev       = 0;
  KAAPI_DEBUG_INST( td->tl = 0) ;
  kaapi_cpuset_clear(&td->u.acl.mapping);
  td->u.acl.date       = 0;
  KAAPI_DEBUG_INST(td->u.acl.exec_date = 0);
  td->u.acl.bcast      = 0;
  td->u.acl.list.front = 0; 
  td->u.acl.list.back  = 0;
  /* assume here that u.dfg are also set to 0 ! */
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
  ++tl->cnt_tasks;
  td = kaapi_allocator_allocate_td( &tl->allocator, task, task_fmt );
  td->tl = tl;
  return td;
}

/**/
static inline kaapi_taskdescr_t* kaapi_allocator_allocate_td_withbody( 
    kaapi_allocator_t* kal, 
    kaapi_format_t*    task_fmt,
    kaapi_task_body_t  body,
    void*              sp
)
{
  kaapi_taskdescr_t* td = 
      (kaapi_taskdescr_t*)kaapi_allocator_allocate( kal, sizeof(kaapi_taskdescr_t) );
  KAAPI_ATOMIC_WRITE(&td->counter, 0);
  td->type       = KAAPI_TASKACL_CASE;
  td->wc         = 0;
  kaapi_task_initdfg(&td->task, body, sp );
  td->fmt 	     = task_fmt;
  td->next       = 0;
  td->prev       = 0;
  td->u.acl.date       = 0;
  KAAPI_DEBUG_INST(td->u.acl.exec_date = 0);
  td->u.acl.bcast      = 0;
  td->u.acl.list.front = 0; 
  td->u.acl.list.back  = 0;
  /* assume here that u.dfg are also set to 0 ! */
  return td;
}

/**/
static inline kaapi_taskdescr_t* kaapi_tasklist_allocate_td_withbody( 
    kaapi_tasklist_t*  tl, 
    kaapi_format_t*    task_fmt,
    kaapi_task_body_t  body,
    void*              sp
)
{
  kaapi_taskdescr_t* td;
  ++tl->cnt_tasks;
  td = kaapi_allocator_allocate_td_withbody( &tl->allocator, task_fmt, body, sp );
  td->tl = tl;
  return td;
}


/**/
static inline kaapi_task_t* kaapi_allocator_allocate_task( kaapi_allocator_t* kal, kaapi_task_bodyid_t body, void* arg )
{
  kaapi_task_t* task = 
      (kaapi_task_t*)kaapi_allocator_allocate( kal, sizeof(kaapi_task_t) );
  kaapi_task_init(task, body, arg);
  return task;
}

/**/
static inline kaapi_task_t* kaapi_tasklist_allocate_task( 
    kaapi_tasklist_t*  tl, 
    kaapi_task_bodyid_t body, 
    void* arg 
)
{
  return kaapi_allocator_allocate_task( &tl->allocator, body, arg );
}

/* Push task in the front: the execution with revert it at the begining
*/
static inline void kaapi_tasklist_pushback_ready( kaapi_tasklist_t* tl, kaapi_taskdescr_t* td)
{
  kaapi_activationlink_t* al =
      (kaapi_activationlink_t*)kaapi_allocator_allocate( &tl->allocator, sizeof(kaapi_activationlink_t) );
  al->td    = td;
  al->queue = 0;
  if (tl->readylist.front ==0)
  {
    al->next  = 0;
    tl->readylist.front = tl->readylist.back = al;
  } else {
    al->next = tl->readylist.front;
    tl->readylist.front = al;
  }
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
  al->queue = tl;
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
  kaapi_thread_context_t* thread, 
  kaapi_tasklist_t*       tasklist, 
  kaapi_task_t* task
);


/** Compute the readylist of the topframe of a thread
    \retval 0 in case of success
    \retval EBUSY if a ready list already exist for the thread
*/
extern int kaapi_thread_computereadylist( 
    kaapi_thread_context_t* thread, 
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

/** To pop the next ready tasks
    Return the next task to execute if err ==0
    Return 0 if case of success, else return non null value.
*/
static inline int kaapi_readylist_pop( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t** td )
{
  kaapi_workqueue_index_t local_end;
  kaapi_workqueue_index_t local_beg;
  int err = kaapi_workqueue_pop(&rtl->wq, &local_beg, &local_end, 1);
  if (err ==0) *td = rtl->base[local_beg];
  return err;
}


/* Return the sp stack pointer of the tasklist that can be used to push a new frame
   before executing a task.
   The current implementation is based on pushing task descriptor into the stack of the thread,
   so the method return the sp as the 0xF aligned pointer before current executed td.
*/
static inline kaapi_task_t* kaapi_thread_tasklist_getsp( kaapi_tasklist_t* tasklist )
{
#if (__SIZEOF_POINTER__ == 4)
  return (kaapi_task_t*)(((uintptr_t)(&tasklist->rtl.base[tasklist->rtl.next] -1)-sizeof(kaapi_task_t)+1) & ~0x7UL);
#else
  return (kaapi_task_t*)(((uintptr_t)(&tasklist->rtl.base[tasklist->rtl.next] -1)-sizeof(kaapi_task_t)+1) & ~0x7ULL);
#endif
}


/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
static inline int kaapi_readylist_pushone( kaapi_readytasklist_t* rtl, kaapi_taskdescr_t* td )
{
  kaapi_workqueue_index_t local_beg = rtl->next; /* reuse the position of the previous executed task */
  rtl->base[--local_beg] = td;
  rtl->next = local_beg;
  rtl->task_pushed = 1;
  return 1;
}

/** Activate and push ready tasks of an activation link.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
extern int kaapi_thread_tasklistready_pushactivated( 
    kaapi_readytasklist_t*  rtl, 
    kaapi_activationlink_t* head 
);


/** Initialize the tasklist with a set of stolen task descriptors
*/
static inline int kaapi_thread_tasklistready_push_init_fromsteal( 
    kaapi_readytasklist_t*  rtl, 
    kaapi_taskdescr_t**     begin, 
    kaapi_taskdescr_t**     end 
)
{
  kaapi_taskdescr_t** base;
  kaapi_workqueue_index_t local_beg = 0; /* */

  if (begin == end) 
    rtl->task_pushed = 0;
  else 
  {
    base = rtl->base;
    while (begin != end)
    {
      kaapi_assert_debug((char*)&base[local_beg] > (char*)(kaapi_self_thread()->sp_data));
      base[--local_beg] = *begin;
      ++begin;
    }
    rtl->task_pushed = 1;
  }

  rtl->next   = local_beg;
  return rtl->task_pushed;
}


/** Push initial ready tasks list into the thread.
    Return 1 if at least one ready task has been pushed into ready queue.
    Else return 0.
*/
static inline int kaapi_thread_tasklistready_push_init( kaapi_readytasklist_t* rtl, kaapi_activationlist_t* acl)
{
  kaapi_activationlink_t* head;
  kaapi_taskdescr_t** base;
  kaapi_workqueue_index_t local_beg = 0; /* never push in position 0 */

  base = rtl->base;
  head = acl->front;
  rtl->task_pushed = (head !=0);

  while (head !=0)
  {
    /* if non local -> push on remote queue ? */
    kaapi_assert_debug((char*)&base[local_beg] > (char*)(kaapi_self_thread()->sp_data));
    base[--local_beg] = head->td;
    head = head->next;
  }
  rtl->next   = local_beg;
  return rtl->task_pushed;
}


/** Commit all previous pushed task descriptor to the other thieves.
    Return =0 in case of success
*/
static inline int kaapi_thread_tasklist_commit_ready( kaapi_tasklist_t* tasklist )
{
  if (tasklist->rtl.task_pushed)
  {
    /* ABA problem here if we suppress lock/unlock? seems to be true */
    kaapi_sched_lock( &tasklist->thread->proc->lock );
    kaapi_workqueue_push(&tasklist->rtl.wq, tasklist->rtl.next); /* do not keep tasklist->next for local exec */
    kaapi_sched_unlock( &tasklist->thread->proc->lock );
    tasklist->rtl.task_pushed = 0;
    return 0;
  }
  return 1;
}


/** Commit all previous pushed task descriptor to the other thieves.
    Return !=0 next task descriptor to execute, else return 0.
    In case tasks have been pushed, the function reserves the last pushed task
    for local execution and returns it.
*/
static inline kaapi_taskdescr_t* kaapi_thread_tasklist_commit_ready_and_steal( kaapi_tasklist_t* tasklist )
{
  if (tasklist->rtl.task_pushed)
  {
    /* ABA problem here if we suppress lock/unlock? seems to be true */
    kaapi_sched_lock( &tasklist->thread->proc->lock );
    kaapi_workqueue_push(&tasklist->rtl.wq, 1+tasklist->rtl.next); /* keep tasklist->next_exec for local exec */
    kaapi_sched_unlock( &tasklist->thread->proc->lock );
    tasklist->rtl.task_pushed = 0;
    return tasklist->rtl.base[tasklist->rtl.next];
  }
  return 0;
}


/** Wrapper over kaapi_workqueue_steal for tasklistready_t
*/
static inline int kaapi_thread_tasklistready_steal( 
  kaapi_readytasklist_t* rtl, 
  kaapi_taskdescr_t*** begin_td_stolen, 
  kaapi_taskdescr_t*** end_td_stolen, 
  size_t size_steal
)
{
  kaapi_workqueue_index_t steal_beg;
  kaapi_workqueue_index_t steal_end;
  int err = kaapi_workqueue_steal( &rtl->wq, &steal_beg, &steal_end, size_steal);
  if (err ==0)
  {
    *begin_td_stolen = rtl->base + steal_beg;
    *end_td_stolen   = rtl->base + steal_end;
  }
  return err;
}

/** How to execute task with readylist
    It is assumed that top frame is a frame with a ready list.
*/
extern int kaapi_thread_execframe_tasklist( kaapi_thread_context_t* thread );

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
