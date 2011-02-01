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
#ifndef _KAAPI_STATICSCHED_H
#define _KAAPI_STATICSCHED_H 1

#if defined(__cplusplus)
extern "C" {
#endif

/* ........................................ Implementation notes ........................................*/
/* Static partitionning of data flow graph.
   The objective is to be able to split in different executing ressources (kaapi threads) a set of tasks 
   that are pushed online.
   Each thread able an initial mapping to an address space. An address space is a logical entity in which
   a version have only one copy before runtime. Data between address space are copied. Multi-core machine 
   may be configured to use several address spaces (for instance one per NUMA node).
   
   Several threads may be binded to the same address space: in that case, the runtime try to use as less as
   possible memory by avoid copies of a same version. The only exception is that if the runtime need to
   extract parallelism by removing WAR (Write after Read) dependencies by copying data.
   
   Before partitioning, all data may have a mapping to a given address space. If not, the data is assumed
   to be binded to the address space of the main thread (identified by tid=-1 in the static graph partitioning 
   code).
   
   The algorithm to push task into several threads is based on the knowledge of the data flow dependencies 
   computed from the access mode declared by the tasks. If the decision to put a task into a thread requires
   to satisfy data flow constraint (a task T0 in thread 0 writes a data consumed by a task T1 in thread 1),
   then the algorithm add meta information to generate synchronisation (term used if T0 and T1 shared the
   same address space) or communication (term used if T0 and T1 are mapped into 2 different address spaces).
   At the end of the algorithm, each thread is annotated by a data structure used to scheduled and manage
   synchronisation or communication of all the tasks pushed into the thread. Especially, communication or 
   synchronisation are known, as well as activation list of task (the list of tasks that may becomes activated
   at the end of execution of one task).
   
   The algorithm operates onto a set of threads. If the order to push tasks are the same for several instance
   of the algorithm, then the result are the same. Moreover, if the order to push tasks are the same, several 
   instance of the algorithm may be used to compute concurrently the contents of the set of thread as well as
   their scheduling data structure.
*/


/** A faire:
    - suppression des datas
    - save/restore des threads + list ready
        * à tester
    - utilisation de l'interface d'allocation
      * allocation GPU ? -> thread gpu
      * partie partition_barrier distribuée entre thread + data allouée à ce moment
        - en distribué: si allocation iso-address -> 
    - test itération
    - test static + poisson3D
    - supprimer les copies 
        - en distribuée si mapping de 0 no sur local gid
        - ajout d'un epilogue à une tâche de lecture ?
        - retarder les allocations
    - optimiser la synchro partition barrière & la recherche des tags pour l'assignement
    des addresses distantes
*/

/* fwd decl */
struct kaapi_taskdescr_t;
struct kaapi_tasklist_t;

/** Activationlink
*/
typedef struct kaapi_activationlink_t {
  struct kaapi_taskdescr_t*      td;
  struct kaapi_activationlink_t* next;
} kaapi_activationlink_t;


/** ActivationList
*/
typedef struct kaapi_activationlist_t {
  kaapi_activationlink_t*   front;
  kaapi_activationlink_t*   back;
} kaapi_activationlist_t;



/** Tag for communication
*/
typedef uint64_t kaapi_comtag_t;


/** TaskArg description.
    Encode for each argument of a task (at most 64),
    if i-th argument, assumed to be a pointer data into shared memory
    points to the data (bit ==0) or points to an handle which points 
    to the data (bit ==1).
    The latter mode, allows to postponed allocation of data upto the execution
    of the task and not during partitionning step.
*/
typedef uint64_t kaapi_taskarg_descr_t;


/** Data structure that points to the task descriptor that are waiting for an input communication
*/
typedef struct kaapi_comrecv_t {
  kaapi_comtag_t            tag;          /* tag */
  kaapi_globalid_t          from;         /* who is the send the data */
  int                       tid;          /* who is the recv the data */
  kaapi_reducor_t           reduce_fnc;   /* if !=0 then it is a recv with reduction */
  void*                     result;       /* result of the reduce_fnc */
  struct kaapi_tasklist_t*  tasklist;     /* points to the data structure where to store activated tasks */
  void*                     data;         /* where to store incomming data */
  kaapi_memory_view_t       view;         /* view */
  kaapi_activationlist_t    list;         /* list of the task descriptor to signal on receive */
  struct kaapi_comrecv_t*   next;         /* used to link together comrecv activated due to incomming data */
} kaapi_comrecv_t;


/** Information where to send data to a remote address space
*/
typedef struct kaapi_comsend_raddr_t {
  kaapi_comtag_t                tag;
  kaapi_address_space_id_t      asid;          /* address space id in the group for the receiver */
  kaapi_pointer_t               rsignal;       /* remote kaapi_comrecv_t data structure */
  kaapi_pointer_t               raddr;         /* remote data address */
  kaapi_memory_view_t           rview;         /* remote data view */
  struct kaapi_comsend_raddr_t* next;          /* next entry where to send */
} kaapi_comsend_raddr_t ;


/** List of remote address space where to send a data
*/
typedef struct kaapi_comsend_t {
  kaapi_comtag_t                vertag;
  void*                         data;          /* used if not task attached */
  kaapi_memory_view_t           view;          /* used if not task attached */
  int                           ith;           /* the parameter of the task that will be send */
  struct kaapi_comsend_t*       next;          /* next kaapi_comsend_t for an other tag for the same task*/
  struct kaapi_comsend_raddr_t  front;         /* list of readers */
  struct kaapi_comsend_raddr_t* back;          /* list of readers */
} kaapi_comsend_t;


/** List of arguments to send in case of multiple parameter of a task to send 
    New argument to send is added at the end of the list
*/
typedef struct kaapi_taskbcast_arg_t {
  kaapi_comsend_t         front;          /* list of task to signal FIFO order */
  kaapi_comsend_t*        back;
} kaapi_taskbcast_arg_t;


/** Link of out or in communication structure
*/
typedef struct kaapi_comlink_t {
  union {
    struct kaapi_comsend_t*  send;
    struct kaapi_comrecv_t*  recv;
  } u;
  struct kaapi_comlink_t*    next;
} kaapi_comlink_t;


/** Used for matching recv & send
    * 
*/
typedef struct kaapi_comaddrlink_t {
  kaapi_comtag_t                tag;
  struct kaapi_comsend_raddr_t* send;
  struct kaapi_comaddrlink_t*   next;
} kaapi_comaddrlink_t;


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
  kaapi_atomic_t                counter;
  kaapi_task_t*                 task;
  kaapi_taskbcast_arg_t*        bcast;
  struct kaapi_taskdescr_t*     next;
  kaapi_activationlist_t        list;
} kaapi_taskdescr_t;


/** \ingroup DFG
    Identification of a reader of a data writen in an other partition.
    This structure is only used during partitionning step, not at runtime.
*/
typedef struct kaapi_data_version_t {
  kaapi_address_space_id_t     asid;                /* address space of the access (r)   */
  kaapi_taskdescr_t*           task;                /* the last reader tasks that owns a reference to the data */
  int                          ith;                 /* index of the argument wich has read access */
  void*                        addr;                /* address of data */
  kaapi_memory_view_t          view;                /* view of data */
  kaapi_reducor_t              reducor;             /* if access is cw */
  struct kaapi_data_version_t* next;                /* next kaapi_data_version_t */
} kaapi_data_version_t;


/**/
static inline int kaapi_data_version_clear( kaapi_data_version_t* dv )
{
  dv->asid    = 0;
  dv->task    = 0;
  dv->ith     = -1;
  dv->addr    = 0;
  kaapi_memory_view_clear(&dv->view);
  dv->reducor = 0;
  dv->next    = 0;
  return 0;
}


/* list of data version */
typedef struct kaapi_data_version_list_t {
  kaapi_data_version_t*       front;               /* front + back for constant time insertion */
  kaapi_data_version_t*       back; 
} kaapi_data_version_list_t;


/**/
static inline int kaapi_data_version_list_clear( kaapi_data_version_list_t* tl )
{
  tl->front = tl->back = 0;
  return 0;
}


/**/
static inline int kaapi_data_version_list_isempty( kaapi_data_version_list_t* tl )
{
  return (tl->front ==0);
}


/**/
static inline int kaapi_data_version_list_add( kaapi_data_version_list_t* l,  kaapi_data_version_t* dv)
{
  dv->next = 0;
  if (l->back ==0)
    l->front = l->back = dv;
  else 
  {
    l->back->next = dv;
    l->back = dv;
  }
  return 0;
}


/**/
static inline int kaapi_data_version_list_append( kaapi_data_version_list_t* l1, kaapi_data_version_list_t* l2 )
{
  if (kaapi_data_version_list_isempty(l2)) return 0;
  if (l1->front ==0) 
  {
    l1->front = l2->front;
    l1->back  = l2->back;
  }
  else {
    l1->back->next = l2->front;
    l1->back = l2->back;
  }
  l2->front = l2->back = 0;
  return 0;
}


/** Allow to maintain meta information about version of data.
*/
typedef struct kaapi_version_t {
  kaapi_comtag_t              tag;             /* the tag (thread group wide) identifier of the data */
  kaapi_data_version_t        writer;          /* address space of the writer   */
  kaapi_access_mode_t         writer_mode;     /* writer access mode :W or CW */
  int                         writer_thread;   /* index of the last thread that writes the data, -1 if outside the group*/
  kaapi_data_version_list_t   copies;          /* list of copies */
  kaapi_data_version_list_t   todel;           /* list of data to delete */
} kaapi_version_t;


/**/
static inline int kaapi_version_clear( kaapi_version_t* v )
{
  v->tag  = 0;
  kaapi_data_version_clear( &v->writer );
  v->writer_thread = ~0;
  kaapi_data_version_list_clear( &v->copies );
  kaapi_data_version_list_clear( &v->todel );
  return 0;
}


/** TaskList
    This data structure is attached to each partition (thread).
    The data structure stores the list of ready task with
    the first taskdescriptor is accessible through front.
    All other data (non ready task descriptor or activationlink) are stores
    into the stack.
    After each task execution, the activated tasks are pushed into the list
    of ready tasks. During execution the list is managed as a LIFO list.
    
    During the partitioning step, the list is filled by inserting tasks at
    the end of the list.
*/
typedef struct kaapi_tasklist_t {
  kaapi_atomic_t     lock;       /* protect recvlist */
  kaapi_taskdescr_t* front;      /* readylist of task descriptor */
  kaapi_taskdescr_t* back;       /* readylist of task descriptor */
  uintptr_t          count_recv; 
  char*              stack;      /* where to push taskdecr or activationlink */
  uintptr_t          sp;         /* stack pointer */
  size_t             size;       /* size of the stack */
  kaapi_comrecv_t*   recvlist;   /* put by pushsignal into ready list to signal incomming data */
} kaapi_tasklist_t;


/**/
static inline int kaapi_tasklist_clear( kaapi_tasklist_t* tl )
{
  kaapi_sched_initlock(&tl->lock);
  tl->front      = tl->back = 0;
  tl->stack      = 0;
  tl->sp         = 0;
  tl->size       = 0;
  tl->recvlist   = 0;
  tl->count_recv = 0;
  return 0;
}


/**/
static inline int kaapi_tasklist_isempty( kaapi_tasklist_t* tl )
{
  return (tl ==0) || ((tl->front ==0) && (tl->recvlist ==0));
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
static inline void kaapi_taskdescr_init( kaapi_taskdescr_t* td, kaapi_task_t* task )
{
  KAAPI_ATOMIC_WRITE(&td->counter, 0);
  td->task  = task;
  td->bcast = 0;
  td->next  = 0;
  td->list.front = 0;
  td->list.back  = 0;
}


/**/
static inline kaapi_taskdescr_t* kaapi_tasklist_allocate_td( kaapi_tasklist_t* tl, kaapi_task_t* task )
{
  kaapi_assert_debug( tl->sp+sizeof(kaapi_taskdescr_t) < tl->size );
  kaapi_taskdescr_t* retval = (kaapi_taskdescr_t*)(tl->stack+tl->sp);
  tl->sp += sizeof(kaapi_taskdescr_t);
  kaapi_taskdescr_init(retval, task);
  return retval;
}


/**/
static inline kaapi_activationlink_t* kaapi_tasklist_allocate_al( kaapi_tasklist_t* tl)
{
  kaapi_assert_debug( tl->sp+sizeof(kaapi_activationlink_t) < tl->size );
  void* retval = (void*)(tl->stack+tl->sp);
  tl->sp += sizeof(kaapi_activationlink_t);
  return (kaapi_activationlink_t*)retval;
}


/**/
static inline void* kaapi_tasklist_allocate( kaapi_tasklist_t* tl, size_t size )
{
  kaapi_assert_debug( tl->sp+size < tl->size );
  void* retval = (void*)(tl->stack+tl->sp);
  tl->sp += size;
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
static inline void kaapi_taskdescr_push_successor( 
    kaapi_tasklist_t* tl, 
    kaapi_taskdescr_t* td, 
    kaapi_taskdescr_t* toactivate
)
{
  kaapi_activationlink_t* al = kaapi_tasklist_allocate_al(tl);

  /* one more synchronisation  */
  KAAPI_ATOMIC_INCR(&toactivate->counter);

  al->td   = toactivate;
  al->next = 0;
  if (td->list.back ==0)
    td->list.front = td->list.back = al;
  else {
    td->list.back->next = al;
    td->list.back = al;
  }
}


#if 0
/* Set the i-thbit to 1
*/
static inline void kaapi_taskargdescr_sethandle( kaapi_taskdescr_t* td, int ith )
{ 
  kaapi_assert_debug((ith>=0) && (ith < 32));
  td->flags |= (1<<ith); 
}


static inline int kaapi_taskargdescr_ishandle( kaapi_taskdescr_t* td, int ith )
{ 
  kaapi_assert_debug((ith>=0) && (ith < 32));
  return (td->flags & (1<<ith)) != 0; 
}
#endif

/**/
extern kaapi_data_version_t* kaapi_version_findasid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid );


/* find in list of copies and unlink entry (if found) before return it */
extern kaapi_data_version_t* kaapi_version_findcopiesrmv_asid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid );


/* find in list of data to dele and unlink entry (if found) before return it */
extern kaapi_data_version_t* kaapi_version_findtodelrmv_asid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid );


/** New typedef for data structure required to manage version 
*/
KAAPI_DECLARE_GENBLOCENTRIES(kaapi_dataforversion_allocator_t);


/** Basic state for the thread group.
    A thread group is used to partition a set of tasks in the following way:
    1/ the thread group is created with a specific number of thread
    2/ the tasks are pushed explicitly into one of the partition, depending
       of their access mode, extra tasks required for the synchronisation are added
    3/ the thread group begin its execution.
    
    Model (public interface= kaapi_threadgroup_t == kaapi_threadgroup_t* of the implementation
    intergace).
      kaapi_threadgroup_t* group; 
      kaapi_threadgroup_create( &group, 10 );
      
      kaapi_threadgroup_begin_partition( group );
      
      ...
      task = kaapi_threadgroup_toptask( group, 2 );
      <init of the task>
      kaapi_threadgroup_pushtask( group, 2 )
      
      kaapi_threadgroup_end_partition( group );
      
    In case of distributed execution, the threadgroup is replicated among the processors set.
    If all tasks pushed into the threadgroup are in the same order with the same arguments
    on all processors, and if the mapping is deterministic, then the different thread group
    may be built in parallel.
*/
typedef enum {
  KAAPI_THREAD_GROUP_CREATE_S,     /* created */
  KAAPI_THREAD_GROUP_PARTITION_S,  /* paritionning scheme beguns */
  KAAPI_THREAD_GROUP_MP_S,         /* multi partition ok */
  KAAPI_THREAD_GROUP_EXEC_S,       /* exec state started */
  KAAPI_THREAD_GROUP_WAIT_S        /* end of execution */
} kaapi_threadgroup_state_t;


/** This is a private view of the data structure.
    The public part of the data structure should be the same with the public definition
    of the data structure in kaapi.h
*/
typedef struct kaapi_threadgrouprep_t {
  /* public part: to be the same in kaapi.h */
  kaapi_thread_t**           threads;      /* array on top frame of each threadctxt, array[-1] = mainthread */
  int32_t                    grpid;        /* group identifier */
  int32_t                    group_size;   /* number of threads in the group */
   
  /* private part: not exported to application */
  kaapi_globalid_t           localgid;     /* local gid of this thread group rep */
  uint32_t                   nodecount;    /* number of nodes = number of copies of thread group */
  kaapi_globalid_t*          tid2gid;      /* mapping of threads onto the unix processes */
  kaapi_address_space_id_t*  tid2asid;     /* mapping of threads onto address spaces */

  /* */
  uint32_t                   localthreads;   /* number of threads local to thgrp->localgid */
  kaapi_atomic_t             endlocalthread; /* count the number of local threads that have finished */
  kaapi_atomic_t             endglobalgroup; /* count the number of remote thread group that have finished */

  kaapi_task_t*              waittask;     /* task to mark end of parallel computation */
  int volatile               startflag;    /* set to 1 when threads should starts */
  int volatile               step;         /* current iteration step */
  int                        maxstep;      /* max iteration step or -1 if not known */
  int                        signal_step;  /* current step marked as signaled of the end of iteration */
  int                        flag;         /* some flag to pass info (save / not save) */
  kaapi_frame_t              mainframe;    /* save/restore main thread */
  kaapi_thread_context_t**   threadctxts;  /* the threads (internal) */
  
  kaapi_thread_context_t*    dummy_thread; /* thread used to temporally store tasks if thread is not local */
  
  kaapi_comlink_t**          lists_send;   /* send and recv list of comsend or comrecv descriptor */
  kaapi_comlink_t**          lists_recv;
  kaapi_comaddrlink_t*       all_sendaddr;

  /* used for iterative execution: only save the tasklist data structure */
  char**                     save_readylists;
  size_t*                    size_readylists;

  
  /* state of the thread group */
  kaapi_threadgroup_state_t  state;        /* state */
  
  pthread_cond_t             cond;
  pthread_mutex_t            mutex;

  /* scheduling part / partitioning, free ad the end of the group */
  kaapi_hashmap_t            ws_khm;  
  
  /* persistant data among several execution of threadgroup */
  long                             tag_count;
  kaapi_data_version_t*            free_dataversion_list;
  kaapi_dataforversion_allocator_t allocator_version; /* used to allocate both version and version data */
  kaapi_dataforversion_allocator_t allocator; /* used to allocate comm data structure */
} kaapi_threadgrouprep_t;


/** All threadgroups are registerd into this global table.
    kaapi_threadgroup_count is the index of the next created threadgroup
*/
#define KAAPI_MAX_THREADGROUP 32
extern kaapi_threadgroup_t kaapi_all_threadgroups[KAAPI_MAX_THREADGROUP];
extern uint32_t kaapi_threadgroup_count;


/** Manage mapping from threadid in a group and address space
*/
static inline kaapi_address_space_id_t kaapi_threadgroup_tid2asid( kaapi_threadgroup_t thgrp, int tid )
{
  return thgrp->tid2asid[tid];
}

static inline int kaapi_threadgroup_asid2tid( kaapi_threadgroup_t thgrp, kaapi_address_space_id_t asid )
{
  return (int)kaapi_memory_address_space_getuser(asid);
}

static inline kaapi_globalid_t kaapi_threadgroup_tid2gid( kaapi_threadgroup_t thgrp, int tid )
{
  return thgrp->tid2gid[tid];
}

static inline kaapi_globalid_t kaapi_threadgroup_asid2gid( kaapi_threadgroup_t thgrp, kaapi_address_space_id_t asid )
{
  return kaapi_memory_address_space_getgid(asid);
}

/* Initialize the ith thread of the thread group 
   - create tasklist 
   - create the thread data specific allocator
*/
int kaapi_threadgroup_initthread( kaapi_threadgroup_t thgrp, int ith );


/** WARNING: also duplicated in kaapi.h for the API. Redefined here during compilation of the sources
*/
static inline kaapi_thread_t* kaapi_threadgroup_thread( kaapi_threadgroup_t thgrp, int partitionid ) 
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_assert_debug( (partitionid>=-1) && (partitionid < thgrp->group_size) );
  {
    kaapi_thread_t* thread = thgrp->threads[partitionid];
    return thread;
  }
}


/* ============================= internal interface to manage version ============================ */
/*
*/
kaapi_hashentries_t* kaapi_threadgroup_newversion( kaapi_threadgroup_t thgrp, kaapi_hashmap_t* hmap, int tid, kaapi_access_t* a );

/*
*/
void kaapi_threadgroup_deleteversion( kaapi_threadgroup_t thgrp, kaapi_version_t* ver );

/* Detect dependencies implies by the read access (R or RW) and update the list of tasks of
   the impacted threads.
   
   On success, the method returns 1 iff the access is ready.
   Else the method returns 0 if the access is not ready.
   Else the method returns a negative integer indicating an error.
*/
extern int kaapi_threadgroup_version_newreader( 
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_access_mode_t   mode,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access
);

/* Detect dependencies implies by the read access (RW or W) and update the list of tasks of
   the impacted threads.

   On success, the method returns 1 iff the access is ready.
   Else the method returns 0 if the access is not ready.
   Else the method returns a negative integer indicating an error.
*/
extern int kaapi_threadgroup_version_newwriter(
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_access_mode_t   mode,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access
);


/* Cumulative write is a mix of reader / writer task creation
*/
extern int kaapi_threadgroup_version_newwriter_cumulwrite( 
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver, 
    int                   tid, 
    kaapi_access_mode_t   mode,
    kaapi_taskdescr_t*    task, 
    const kaapi_format_t* fmt,
    int                   ith,
    kaapi_access_t*       access
);

/* Code to finalize reduction between gid
*/
extern int kaapi_threadgroup_version_finalize_cw(
    kaapi_threadgroup_t   thgrp, 
    kaapi_version_t*      ver
);


/* Allocate a new version
*/
static inline kaapi_version_t* kaapi_threadgroup_allocate_version( kaapi_threadgroup_t thgrp )
{
  kaapi_version_t* retval = (kaapi_version_t*)kaapi_allocator_allocate(&thgrp->allocator_version, sizeof(kaapi_version_t));
  kaapi_assert_debug( retval != 0);
  kaapi_version_clear( retval );
  return retval;
}

/* Return a new data version 
*/
static inline kaapi_data_version_t* kaapi_threadgroup_allocate_dataversion( kaapi_threadgroup_t thgrp )
{
  kaapi_data_version_t* retval;
  if (thgrp->free_dataversion_list !=0)
  {
    retval = thgrp->free_dataversion_list;
    thgrp->free_dataversion_list = retval->next;
  }
  else 
  {
    retval = (kaapi_data_version_t*)kaapi_allocator_allocate(&thgrp->allocator_version, sizeof(kaapi_data_version_t));
    kaapi_assert_debug( retval != 0);
  }
  kaapi_data_version_clear( retval );
  return retval;
}

/* Recycle an allocated data version 
*/
static inline void kaapi_threadgroup_deallocate_dataversion( kaapi_threadgroup_t thgrp, kaapi_data_version_t* dv )
{
  kaapi_assert_debug( dv->next ==0); /* should not be in list */
  dv->next = thgrp->free_dataversion_list;
  thgrp->free_dataversion_list = dv;
}

/* task to wait end of a step
*/
void kaapi_taskwaitend_body( void* sp, kaapi_thread_t* thread );

/**/
static inline int kaapi_activationlist_isempty( kaapi_activationlist_t* al )
{
  return (al->front ==0);
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
static inline void kaapi_activationlist_pushback( kaapi_threadgroup_t thgrp, kaapi_activationlist_t* al, kaapi_taskdescr_t* td)
{
  kaapi_activationlink_t* l 
    = (kaapi_activationlink_t*)kaapi_allocator_allocate(&thgrp->allocator, sizeof(kaapi_activationlink_t));
  l->td = td;
  l->next = 0;
  if (al->back ==0)
    al->front = al->back = l;
  else {
    al->back->next = l;
    al->back = l;
  }
}


/** Return 0 iff tid does not belong to the list of readers of the version.
    Else return a pointer to the entry.
*/
extern kaapi_data_version_t* kaapi_version_find_tag( const kaapi_version_t* v, int tid );

/**
*/
extern kaapi_comsend_t* kaapi_sendcomlist_find_tag( kaapi_taskbcast_arg_t* bcast, kaapi_comtag_t tag );

/**
*/
extern kaapi_comsend_raddr_t* kaapi_sendcomlist_find_asid( kaapi_comsend_t* com, kaapi_address_space_id_t asid );

/**
*/
extern kaapi_comsend_raddr_t* kaapi_sendcomlist_find_gidtid( kaapi_comsend_t* com, kaapi_globalid_t gid, int tid );

/**
*/
extern kaapi_comrecv_t* kaapi_recvcomlist_find_tag( kaapi_comlink_t* recvl, kaapi_comtag_t tag );

/**
*/
extern kaapi_comsend_raddr_t* kaapi_threadgroup_findsend_tagtid( kaapi_comaddrlink_t* list, kaapi_comtag_t tag, int tid );

/** Register the tag as an out data send by a bcast task on thread tidwriter
*/
static inline int kaapi_threadgroup_comsend_register( 
    kaapi_threadgroup_t thgrp, 
    int                 tidwriter, 
    kaapi_comtag_t      tag, 
    kaapi_comsend_t*    com 
)
{
  kaapi_comlink_t* cl = (kaapi_comlink_t*)kaapi_allocator_allocate(&thgrp->allocator, sizeof(kaapi_comlink_t));
  kaapi_assert_debug(cl !=0);
  cl->u.send = com;
  cl->next = thgrp->lists_send[tidwriter];
  thgrp->lists_send[tidwriter] = cl;
  return 0;
}


/** Register the tag as an in data that will be received by thread tidreader
*/
static inline int kaapi_threadgroup_comrecv_register( 
  kaapi_threadgroup_t thgrp, 
  int tidreader, 
  kaapi_comtag_t tag, 
  kaapi_comrecv_t* com 
)
{
  kaapi_comlink_t* cl = (kaapi_comlink_t*)kaapi_allocator_allocate(&thgrp->allocator, sizeof(kaapi_comlink_t));
  kaapi_assert_debug(cl !=0);
  cl->u.recv = com;
  cl->next = thgrp->lists_recv[tidreader];
  thgrp->lists_recv[tidreader] = cl;
  if (thgrp->tid2gid[tidreader] == thgrp->localgid)
    ++thgrp->threadctxts[tidreader]->tasklist->count_recv;
  return 0;
}


/** Call to resolved remote address for bcast data structure.
    Should be called by all participating process.
*/
extern int kaapi_threadgroup_barrier_partition( kaapi_threadgroup_t thgr );

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
extern int kaapi_threadgroup_bcast( kaapi_threadgroup_t thgrp, kaapi_address_space_id_t asid_src, kaapi_comsend_t* com);

/**
*/
extern int kaapi_threadgroup_restore_thread( kaapi_threadgroup_t thgrp, int tid );


/**
*/

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
