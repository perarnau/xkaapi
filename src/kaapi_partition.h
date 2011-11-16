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
#ifndef _KAAPI_PARTITION_H
#define _KAAPI_PARTITION_H 1

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

/** Update a tasklist data flow graph representation to compute the critical path of each tasks,
    i.e. the longest path to the output tasks
*/
extern int kaapi_staticschedtask_critical_path( kaapi_tasklist_t* tasklist );


/** fwd decl
*/
struct kaapi_partid_version_t;

/** 
*/
typedef struct kaapi_recv_arg_t {
  kaapi_comtag_t            tag;          /* tag */
  kaapi_globalid_t          from;         /* who is the send the data ? */
  void*                     data;         /* where to store incomming data */
  kaapi_handle_t            dest;         /* handle to update on receive */
  struct kaapi_comrecv_t*   next;         /* used to link together comrecv activated due to incomming data */
} kaapi_recv_arg_t;


/** 
*/
typedef struct kaapi_bcast_onedest_t {
  kaapi_pointer_t               dest;         /* where to send data */
  uintptr_t                     rsignal;      /* who to signal after data transfert (a taskdescriptor) */
  struct kaapi_bcast_onedest_t* next;         /* used to link together data to send from one task */
} kaapi_bcast_onedest_t;


/** 
*/
typedef struct kaapi_bcast_arg_t {
  kaapi_comtag_t            tag;          /* tag */
  kaapi_taskdescr_t*        td_bcast;     /* */
  kaapi_handle_t            src;          /* handle to update on receive */
  kaapi_bcast_onedest_t     front;        /* used to link together data to send from one task */
  kaapi_bcast_onedest_t*    back;
} kaapi_bcast_arg_t;



/** Data structure that points to the task descriptor that are waiting for an input communication
*/
typedef struct kaapi_comrecv_t {
  kaapi_comtag_t            tag;          /* tag */
  kaapi_globalid_t          from;         /* who is the send the data */
  kaapi_syncrecv_t          recv;         /* who is the recv the data */
  void*                     data;         /* where to store incomming data */
  kaapi_memory_view_t       view;         /* view */
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




/** Basic state for the thread group.
    A thread group is used to partition a set of tasks in the following way:
    1/ the thread group is created with a specific number of thread. Each thread
    is mapped to an address space on a given processors.
    2/ the tasks are pushed explicitly into one of the partition, depending
       of their access mode, extra tasks required for the synchronisation are added
       into the graph.
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
   
  /* state of the thread group */
  kaapi_threadgroup_state_t  state;        /* state */

  /* private part: not exported to application */
  kaapi_globalid_t           localgid;     /* local gid of this thread group rep */
  uint32_t                   nodecount;    /* number of nodes = number of copies of thread group */
  kaapi_address_space_id_t*  tid2asid;     /* mapping of threads onto address spaces */

  kaapi_thread_context_t**   threadctxts;  /* the threads (internal) */
  kaapi_thread_context_t*    dummy_thread; /* thread used to temporally store tasks if thread is not local */

  /* detection of the terminaison */
  uint32_t                   localthreads;   /* number of threads local to thgrp->localgid */
  kaapi_atomic_t             endlocalthread; /* count the number of local threads that have finished */
  kaapi_atomic_t             endglobalgroup; /* count the number of remote thread group that have finished */

  kaapi_task_t*              waittask;     /* task to mark end of parallel computation */
  int volatile               step;         /* current iteration step */
  int                        maxstep;      /* max iteration step or -1 if not known */
  int                        flag;         /* some flag to pass info (save / not save) */

  kaapi_frame_t              mainframe;    /* save/restore main thread */
  
  kaapi_comtag_t             count_tag;    /* counter of tag relatively to the thread group */
  kaapi_comsend_raddr_t**    lists_send;   /* send and recv list of comsend or comrecv descriptor */
  kaapi_comsend_raddr_t**    lists_recv;
  
  kaapi_data_t*              list_data;     /* list of data to synchronize */
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

/**
*/
static inline kaapi_globalid_t kaapi_threadgroup_tid2gid( kaapi_threadgroup_t thgrp, int tid )
{
  return kaapi_memory_address_space_getgid(thgrp->tid2asid[tid]);
}

/** Initialize the ith thread of the thread group 
   - initialize internal fields
   - create tasklist on the top frame of each threads
*/
extern int kaapi_threadgroup_initthread( kaapi_threadgroup_t thgrp, int ith );

/** Allocate and reset the tasklist into the top frame of the thread
*/
extern kaapi_tasklist_t* kaapi_threadgroup_allocatetasklist(void);

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

/**
*/
extern int kaapi_thread_readylist_print( FILE* file, kaapi_tasklist_t* tl );


/* ============================= internal interface to manage version ============================ */

/**
*/
kaapi_hashentries_t* kaapi_threadgroup_newversion
(
    kaapi_threadgroup_t  thgrp, 
    kaapi_hashmap_t*     hmap, 
    int                  tid, 
    kaapi_access_t*      access, 
    kaapi_memory_view_t* view
);

/**
*/
void kaapi_threadgroup_deleteversion( kaapi_threadgroup_t thgrp, kaapi_version_t* ver );

/** Detect dependencies implies by the read access (R or RW) and update the list of tasks of
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



/* task to wait end of a step
*/
extern void kaapi_taskwaitend_body( void* sp, kaapi_thread_t* thread );

/* task to switch from received data from other address space to local pointer
*/
extern void kaapi_taskrecv_body( void* sp, kaapi_thread_t* thread );

/* task to switch from received data from other address space to local pointer
*/
extern void kaapi_taskbcast_body( void* sp, kaapi_thread_t* thread );


#if 0

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
    ++thgrp->threadctxts[tidreader]->sfp->tasklist->count_recv;
  return 0;
}


/** Call to resolved remote address for bcast data structure.
    Should be called by all participating process.
*/
extern int kaapi_threadgroup_barrier_partition( kaapi_threadgroup_t thgr );
#endif


/**
*/
extern int kaapi_threadgroup_bcast( kaapi_threadgroup_t thgrp, kaapi_address_space_id_t asid_src, kaapi_comsend_t* com);

/**
*/
extern int kaapi_threadgroup_restore_thread( kaapi_threadgroup_t thgrp, int tid );


/**
*/
#if defined(KAAPI_USE_NETWORK)
/* network service to signal end of iteration of one tid */
void kaapi_threadgroup_signalend_service(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer );
#endif

#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_PARTITION_H */
