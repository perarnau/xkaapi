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

/** Basic state for the thread group.
    A thread group is used to partition a set of tasks in the following way:
    1/ the thread group is created with a specific number of thread
    2/ the tasks are pushed explicitly into one of the partition, depending
       of their access mode, extra tasks required for the synchronisation are added
    3/ the thread group begin its execution.
*/
typedef enum {
  KAAPI_THREAD_GROUP_CREATE_S,     /* created */
  KAAPI_THREAD_GROUP_PARTITION_S,  /* paritionning scheme beguns */
  KAAPI_THREAD_GROUP_MP_S,         /* multi partition ok */
  KAAPI_THREAD_GROUP_EXEC_S,       /* exec state started */
  KAAPI_THREAD_GROUP_WAIT_S        /* end of execution */
} kaapi_threadgroup_state_t;


/** This is a private view of the data structure, may be the public
    view should only export as well as the functions in kaapi.h 
*/
typedef struct kaapi_threadgroup_t {
  /* public part */
  kaapi_thread_t**           threads;      /* array on top frame of each threadctxt */
  int                        group_size;   /* number of threads in the group */
   
  /* executive part */
  kaapi_atomic_t             countend;     /* warn: alignement ! */
  int volatile               startflag;    /* set to 1 when threads should starts */
  kaapi_thread_context_t**   threadctxts;  /* the threads (internal) */
  
  /* state of the thread group */
  kaapi_threadgroup_state_t state;        /* state */

  /* scheduling part / partitioning */
  //kaapi_hashmap_for_theo hm + extra data for the partitioning step
  
} kaapi_threadgroup_t;


/** Create a thread group with size threads. 
    Return 0 in case of success or the error code.
*/
extern int kaapi_threadgroup_create(kaapi_threadgroup_t** thgrp, int size );


/**
*/
extern int kaapi_threadgroup_begin_parition(kaapi_threadgroup_t* thgrp );

/**
*/
static inline kaapi_thread_t* kaapi_threadgroup_thread( kaapi_threadgroup_t* thgrp, int i ) 
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_thread_t* thread = thgrp->threads[i];
  return thread;
}

/** Equiv to kaapi_thread_toptask( thread ) 
*/
static inline kaapi_task_t* kaapi_threadgroup_toptask( kaapi_threadgroup_t* thgrp, int i ) 
{
  kaapi_assert_debug( thgrp !=0 );

  kaapi_thread_t* thread = thgrp->threads[i];
  return kaapi_thread_pushtask(thread);
}


/** Equiv to kaapi_thread_pushtask( thread ) 
*/
static inline int kaapi_threadgroup_pushtask( kaapi_threadgroup_t* thgrp, int i )
{
  kaapi_assert_debug( thgrp !=0 );
  kaapi_thread_t* thread = thgrp->threads[i];
  
  /* la tache a pousser est pointee par thread->sp, elle n'est pas encore pousser et l'on peut
     calculer les dépendances (appel au bon code)
  */
  kaapi_threadgroup_computedependencies( thgrp, i, thread->sp ); /* à changer */
  
  return kaapi_thread_pushtask(thread);
}


/**
*/
extern int kaapi_threadgroup_end_parition(kaapi_threadgroup_t* thgrp );



/**
*/
extern int kaapi_threadgroup_begin_execute(kaapi_threadgroup_t* thgrp );


/**
*/
extern int kaapi_threadgroup_end_execute(kaapi_threadgroup_t* thgrp );


/**
*/
extern int kaapi_threadgroup_destroy(kaapi_threadgroup_t* thgrp );


/** 
*/
static inline int kaapi_thread_allocatefifodata( kaapi_fifo_t* fdata, kaapi_thread_t* thread, size_t sizedata )
{
  fdata->flag = 0;
  fdata->size = sizedata;
#if defined(KAAPI_DEBUG)
  fdata->data = calloc(1, sizedata);
#else
  fdata->data = malloc(sizedata);
#endif
  fdata->task_writer   = 0;
  fdata->task_reader   = 0;
  fdata->thread_reader = 0;
  return 0;
}


/**
*/
static inline int kaapi_fifo_write( kaapi_fifo_t* fifo )
{
  kaapi_task_t* rtask = fifo->task_reader;
  if (rtask !=0) 
  {
    /* restore original body */
    rtask->body = rtask->ebody;
    /* optimization: put the thread into ready list of thread */
  }
  kaapi_writemem_barrier();
  fifo->flag = 1;
  return 0;
}


/** return 0 if not ready
*/
static inline void* kaapi_fifo_read( kaapi_fifo_t* fifo )
{
  if (fifo->flag ==0) return 0;
  kaapi_readmem_barrier();
  return fifo->data;
}


/** Push task with at least one fifo access
    - update body if it is a writer
    - update body if it is a reader
*/
extern int kaapi_thread_pushfifotask(kaapi_thread_t* thread);


/** Update information of Fifo parameter of a task during creation
*/
extern int kaapi_task_bindfifo(kaapi_task_t* task, int ith, kaapi_access_mode_t m, kaapi_fifo_t* param);


#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_STATICSCHED_H */
