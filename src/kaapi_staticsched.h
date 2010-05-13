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

#include "kaapi_impl.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** \ingroup DFG
    In case of dependency W -> R with the writer and reader tasks on two different partitions,
    the task_writer->pad points on the kaapi_counters_list data structure.
*/
#define KAAPI_COUNTER_LIST_BLOCSIZE 7
typedef struct kaapi_counters_list {
  short                       size;            // max size: KAAPI_COUNTER_LIST_BLOCSIZE */
  struct kaapi_counters_list* next;            // next counters_list bloc
  struct {
    kaapi_atomic_t*           waiting_counter;
    kaapi_task_t*             waiting_task;
  } entry[KAAPI_COUNTER_LIST_BLOCSIZE];        // total size < 8*8 = 64 bytes
} kaapi_counters_list;


#define KAAPI_MAX_PARTITION 64
/** Bit field for at most 64 partitions.
*/
typedef kaapi_uint64_t kaapi_readers_t;

/** \ingroup DFG

    Allow to maintain meta information about version of data.
    A version is M producers and N readers (current M=1 because we do not support cw).
    When a new reader is added into a version, the readers information is updated:
      - the number of reader,
      - the thread index which is a reader
      - the reader task
      - the addr of the data into the thread of the reader.

    When a new writer task is added, then the previous version is not yet valid and :
      - a task to delete data is pushed into each partition that has a copy of the data 
      - the thread_readers information is set to 0.
      - these delete tasks are pointed by the task_readers field
      - a new data (renaming) is created on the thread that own the writer in order
      to avoid writing the data while an other task on an other partition is accessing it.
    The address of data referenced by the field addr_data on each partition, represents
    the address of invalidated data. If a new reader is added, then the delete_task body
    may be replaced by nop in order to reuse the data.
*/
typedef struct kaapi_deps_t {
  long             tag;                                /* the tag (thread group wide) identifier of the data */
  void*            writer_data;                        /* address of the reference data */
  int              thread_writer;                      /* index of the last thread in the group that write the data */
  kaapi_task_t*    last_writer;                        /* last writer task of the version */

  int              cnt_readers;                        /* number of bits ==1 in set_readers */
  int              thread_readers[KAAPI_MAX_PARTITION];/* bit i=1, partition i has a version of the data */
  kaapi_task_t*    task_readers[KAAPI_MAX_PARTITION];  /* the last reader tasks that owns a reference to the data */
  void*            addr_data[KAAPI_MAX_PARTITION];     /* address of data on each memory bank */
} kaapi_deps_t;

#define KAAPI_PARTITION_SET(p, i) (p[i] = 1)
#define KAAPI_PARTITION_GET(p, i) (p[i])

/* usage:
   KAAPI_FOREACH_PARTITION( index, dfginfo->thread_readers)
   {
     do some things
   }
*/
#define KAAPI_FOREACH_PARTITION(var, set) \
  for(var=0; var<KAAPI_MAX_PARTITION; ++var)\
    if (set[var] !=0)


/** Basic state for the thread group.
    A thread group is used to partition a set of tasks in the following way:
    1/ the thread group is created with a specific number of thread
    2/ the tasks are pushed explicitly into one of the partition, depending
       of their access mode, extra tasks required for the synchronisation are added
    3/ the thread group begin its execution.
    
    Model (publica interface= kaapi_threadgroup_t == kaapi_threadgroup_t* of the implementation
    intergace).
      kaapi_threadgroup_t* group; 
      kaapi_threadgroup_create( &group, 10 );
      
      kaapi_threadgroup_begin_partition( group );
      
      ...
      task = kaapi_threadgroup_toptask( group, 2 );
      <init of the task>
      kaapi_threadgroup_pushtask( group, 2 )
      
      kaapi_threadgroup_end_partition( group );
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
  /* public part */
  kaapi_thread_t**           threads;      /* array on top frame of each threadctxt */
  int                        group_size;   /* number of threads in the group */
   
  /* executive part */
  kaapi_atomic_t             countend;     /* warn: alignement ! */
  int volatile               startflag;    /* set to 1 when threads should starts */
  int volatile               step;         /* iteration step */
  kaapi_thread_context_t**   threadctxts;  /* the threads (internal) */
  
  /* state of the thread group */
  kaapi_threadgroup_state_t  state;        /* state */
  
  pthread_cond_t             cond;
  pthread_mutex_t            mutex;

  /* scheduling part / partitioning */
  kaapi_hashmap_t            ws_khm;  
  long                       tag_count;
} kaapi_threadgrouprep_t;


/* Task used to write on remote memory data and signal the waiting task
*/
void kaapi_writesignal_body( void* sp, kaapi_thread_t* stack );

/* Task used to delete memory due to allocation of temporary
*/
void kaapi_delete_body( void* sp, kaapi_thread_t* stack );


/** WARNING: also duplicated in kaapi.h for the API. Redefined here during compilation of the sources
*/
static inline kaapi_thread_t* kaapi_threadgroup_thread( kaapi_threadgroup_t thgrp, int partitionid ) 
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_assert_debug( (partitionid>=0) && (partitionid<thgrp->group_size) );
  kaapi_thread_t* thread = thgrp->threads[partitionid];
  return thread;
}

/** Equiv to kaapi_thread_toptask( thread ) 
*/
static inline kaapi_task_t* kaapi_threadgroup_toptask( kaapi_threadgroup_t thgrp, int partitionid ) 
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_assert_debug( (partitionid>=0) && (partitionid<thgrp->group_size) );

  kaapi_thread_t* thread = thgrp->threads[partitionid];
  return kaapi_thread_toptask(thread);
}

static inline int kaapi_threadgroup_pushtask( kaapi_threadgroup_t thgrp, int partitionid )
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_assert_debug( (partitionid>=0) && (partitionid<thgrp->group_size) );
  kaapi_thread_t* thread = thgrp->threads[partitionid];
  kaapi_assert_debug( thread !=0 );
  
  /* la tache a pousser est pointee par thread->sp, elle n'est pas encore pousser et l'on peut
     calculer les dépendances (appel au bon code)
  */
  kaapi_threadgroup_computedependencies( thgrp, partitionid, thread->sp );
  
  return kaapi_thread_pushtask(thread);
}


#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
