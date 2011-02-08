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


/** Data shared between address space and task
    Such data structure is referenced through the pointer arguments of tasks using a handle.
*/
typedef struct kaapi_data_t {
  kaapi_address_space_id_t     asid;                /* address space of the access (r)   */
  void*                        addr;                /* address of data */
  kaapi_memory_view_t          view;                /* view of data */
} kaapi_data_t;


/** Handle to data.
    During generation of tasklist, each pointer parameter is replaced by a handle
    to a kaapi_data that stores the data.
    In this, way we can express dependencies between tasks independently of the memory
    allocation used for exection.
*/
typedef kaapi_data_t* kaapi_handle_t;


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


/**
*/
typedef struct kaapi_comrecv_t {
  kaapi_comtag_t            tag;          /* tag */
  struct kaapi_tasklist_t*  tasklist;     /* points to the data structure where to store activated tasks */
  kaapi_activationlist_t    list;         /* list of tasks to activate */
  struct kaapi_comrecv_t*   next;         /* used to push in recv list of a tasklist */
} kaapi_comrecv_t;


/**
*/
typedef struct kaapi_move_arg_t {
  void*               src_data;
  kaapi_memory_view_t src_view;
  kaapi_handle_t      dest;
} kaapi_move_arg_t;

/** Task with kaapi_move_arg_t as parameter
*/
extern void kaapi_taskmove_body( void*, kaapi_thread_t* );

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
  uint32_t                      date;      /* minimal logical date of production */
  kaapi_task_t*                 task;      /* the task to executed */
  kaapi_activationlist_t*       bcast;     /* list of bcast tasks activated to send data produced by this task */
  kaapi_activationlist_t        list;      /* list of tasks descr. activated after bcast list */     
  struct kaapi_taskdescr_t*     next;      /* link into the task list */
} kaapi_taskdescr_t;


/** TaskList
    This data structure is attached to a frame that has been attached to an acceleratrice data structure.
    During the partitionning of threads, the algorithm built a tasklist for each generated thread.
    The data structure stores the list of ready tasks. 
    During execution, the list is managed as a LIFO queue of task descriptor: the most recent pushed
    task descriptor is poped first.
    All other data (non ready task descriptor or activationlink) are stores
    into the stack and will be activated at runtime by beging pushed in the front of the list.
    After each task execution, the activated tasks are pushed into the list
    of ready tasks. During execution the list is managed as a LIFO list.
    
    By construction, the execution of tasks may depend on the execution located onto other threads.
    The synchronisation between two threads is based on a FIFO queue of tasks to activate.
    
    New data structures needed for execution are pushed into the stack of the attached thread.
    
*/
typedef struct kaapi_tasklist_t {
  kaapi_atomic_t          lock;       /* protect recvlist */
  kaapi_taskdescr_t*      front;      /* readylist of task descriptor */
  kaapi_taskdescr_t*      back;       /* readylist of task descriptor */
  uintptr_t               count_recv; /* number of extern synchronization to receive before detecting end of execution */
  kaapi_comrecv_t*        recvlist;   /* put by pushsignal into ready list to signal incomming data */
  kaapi_frame_t*          frame;      /* where to push task descriptor and other data structure */
} kaapi_tasklist_t;


/**
*/
typedef struct kaapi_version_t {
  kaapi_data_t        orig;              /* original data + original view */
  kaapi_handle_t      handle;            /* @data + view */
  uint32_t            date;              /* minimal logical date of production */
  kaapi_access_mode_t last_mode;         /* */
  kaapi_taskdescr_t*  last_task;         /* task attached to last access */
  int                 is_ready;          /* */
  kaapi_taskdescr_t*  writer_task;       /* last writer task of the version, 0 if no indentify task (input data) */
} kaapi_version_t;


/**
*/
static inline kaapi_version_t* kaapi_thread_newversion( void* data, kaapi_memory_view_t* view )
{
  kaapi_version_t* version = (kaapi_version_t*)malloc( sizeof(kaapi_version_t) );
  version->orig.addr    = data;
  version->orig.view    = *view;  
  version->handle       = (kaapi_data_t*)malloc(sizeof(kaapi_data_t));
  version->handle->addr = data;
  version->handle->view = *view;
  version->date         = 0;
  version->last_mode    = KAAPI_ACCESS_MODE_VOID;
  version->last_task    = 0;
  version->is_ready     = 1;
  version->writer_task  = 0;
  return version;
}

/**/
static inline void kaapi_activationlist_clear( kaapi_activationlist_t* al )
{
  al->front = al->back = 0;
}

/**/
static inline int kaapi_activationlist_isempty( kaapi_activationlist_t* al )
{
  return (al->front ==0);
}


/**/
static inline int kaapi_data_clear( kaapi_data_t* d )
{
  d->asid    = 0;
  d->addr    = 0;
  kaapi_memory_view_clear(&d->view);
  return 0;
}


/**/
static inline void kaapi_taskdescr_init( kaapi_taskdescr_t* td, kaapi_task_t* task )
{
  KAAPI_ATOMIC_WRITE(&td->counter, 0);
  td->date       = 0;
  td->task       = task;
  td->bcast      = 0;
  td->list.front = 0; 
  td->list.back  = 0;
  td->next       = 0; // debug: set when insert
}


/**/
static inline int kaapi_tasklist_init( kaapi_tasklist_t* tl, kaapi_frame_t* fp )
{
  kaapi_sched_initlock(&tl->lock);
  tl->front      = tl->back = 0;
  tl->count_recv = 0;
  tl->recvlist   = 0;
  tl->frame      = fp;
  return 0;
}


/**/
static inline int kaapi_tasklist_isempty( kaapi_tasklist_t* tl )
{
  return (tl ==0) || ((tl->front ==0) && (tl->recvlist ==0));
}


/**/
static inline kaapi_taskdescr_t* kaapi_tasklist_allocate_td( kaapi_tasklist_t* tl, kaapi_task_t* task )
{
  kaapi_taskdescr_t* retval = 
      (kaapi_taskdescr_t*)kaapi_thread_pushdata( (kaapi_thread_t*)tl->frame, sizeof(kaapi_taskdescr_t) );
  kaapi_taskdescr_init(retval, task);
  return retval;
}


/**/
static inline kaapi_task_t* kaapi_tasklist_push_task( kaapi_tasklist_t* tl, kaapi_task_bodyid_t body, void* arg )
{
  kaapi_task_t* task = kaapi_thread_toptask((kaapi_thread_t*)tl->frame);
  kaapi_task_init(task, body, arg);
  kaapi_thread_pushtask((kaapi_thread_t*)tl->frame);
  return task;
}


/**/
static inline kaapi_taskdescr_t* kaapi_tasklist_pop( kaapi_tasklist_t* tl )
{
  kaapi_taskdescr_t* retval = tl->front;
  if (retval ==0) return 0;
  tl->front = retval->next;
  if (retval->next == 0)
    tl->back = 0;
  return retval;
}


/**/
static inline void kaapi_tasklist_pushback_ready( kaapi_tasklist_t* tl, kaapi_taskdescr_t* td)
{
  td->next = 0;
  if (tl->back ==0)
    tl->front = tl->back = td;
  else {
    tl->back->next = td;
    tl->back = td;
  }
}


/**/
static inline int kaapi_tasklist_merge_activationlist( kaapi_tasklist_t* tl, kaapi_activationlist_t* al )
{
  kaapi_activationlink_t* curr = al->front;
  while (curr !=0)
  {
    if (KAAPI_ATOMIC_DECR(&curr->td->counter) == 0)
    {
      kaapi_tasklist_pushback_ready( tl, curr->td );
    }
    curr = curr->next;
  }

  return 0;
}


/**/
static inline kaapi_activationlink_t* kaapi_tasklist_allocate_al( kaapi_tasklist_t* tl)
{
  void* retval = kaapi_thread_pushdata( (kaapi_thread_t*)tl->frame, sizeof(kaapi_activationlink_t) );
  return (kaapi_activationlink_t*)retval;
}


/**/
static inline void* kaapi_tasklist_allocate( kaapi_tasklist_t* tl, size_t size )
{
  void* retval = kaapi_thread_pushdata( (kaapi_thread_t*)tl->frame, (int)size );
  return retval;
}


/**/
static inline void kaapi_tasklist_push_successor( 
    kaapi_tasklist_t* tl, 
    kaapi_taskdescr_t* td, 
    kaapi_taskdescr_t* tdsuccessor
)
{
  kaapi_activationlink_t* al = kaapi_tasklist_allocate_al(tl);

  /* one more synchronisation  */
  KAAPI_ATOMIC_INCR(&tdsuccessor->counter);

  al->td   = tdsuccessor;
  al->next = 0;
  if (td->list.back ==0)
    td->list.front = td->list.back = al;
  else {
    td->list.back->next = al;
    td->list.back = al;
  }
}


/** Compute the readylist of the topframe of a thread
    \retval 0 in case of success
    \retval EBUSY if a ready list already exist for the thread
*/
extern int kaapi_thread_computereadylist( kaapi_thread_context_t* thread );


/**
*/
extern int kaapi_thread_computeready_access( 
    kaapi_tasklist_t*   tl, 
    kaapi_version_t*    version, 
    kaapi_taskdescr_t*  task,
    kaapi_access_mode_t m 
);


/** How to execute task with readylist
    It is assumed that top frame is a frame with a ready list.
*/
extern int kaapi_thread_execframe_readylist( kaapi_thread_context_t* thread );


/**
*/
extern int kaapi_thread_readylist_print( FILE* file, kaapi_tasklist_t* tl );



/**/
static inline void kaapi_activationlist_pushback( 
    kaapi_tasklist_t* tl, 
    kaapi_activationlist_t* al, 
    kaapi_taskdescr_t* td
)
{
  kaapi_activationlink_t* l 
    = (kaapi_activationlink_t*)kaapi_tasklist_allocate(tl, sizeof(kaapi_activationlink_t));
  l->td = td;
  l->next = 0;
  if (al->back ==0)
    al->front = al->back = l;
  else {
    al->back->next = l;
    al->back = l;
  }
}


/** Manage synchronisation between several tid 
    Each thread may communicate between them through FIFO queue to signal incomming data
*/
static inline int kaapi_tasklist_pushsignal( kaapi_pointer_t rsignal )
{
  kaapi_comrecv_t* recv = (kaapi_comrecv_t*)rsignal;
  kaapi_tasklist_t* tl  = recv->tasklist;
  kaapi_sched_lock(&tl->lock);
  recv->next = tl->recvlist;
  tl->recvlist = recv;
  kaapi_sched_unlock(&tl->lock);
  return 0;
}

/** Only call by the owner
*/
static inline kaapi_comrecv_t* kaapi_tasklist_popsignal( kaapi_tasklist_t* tl )
{
  kaapi_comrecv_t* retval;
  if (tl->recvlist ==0) return 0;
  kaapi_sched_lock(&tl->lock);
  retval = tl->recvlist;
  tl->recvlist = retval->next;
  kaapi_sched_unlock(&tl->lock);
  retval->next = 0;
  return retval;
}


/**
*/
extern void kaapi_thread_signalend_exec( kaapi_thread_context_t* thread );

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
