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

/** Single assignment variable.
*/
typedef struct kaapi_fifo_t {
  void*                   data;
  size_t                  size;
  int volatile            flag;            /* 1 iff data ready */
  kaapi_task_t*           task_writer;     /* the task that write the data */
  int                     param_writer;    /* index of the parameter that write the data */
  kaapi_task_t*           task_reader;     /* the task that read the data */
  int                     param_reader;    /* index of the parameter that read the data */
  kaapi_thread_context_t* thread_reader;   /* the thread context that owns task_reader */
} kaapi_fifo_t;


/**
*/
typedef struct kaapi_thread_group_t {
  int volatile             startflag;
  kaapi_atomic_t           countend;
  int                      group_size;
  kaapi_thread_context_t** threads;
  pthread_mutex_t          mutex;
  pthread_cond_t           cond;
} kaapi_thread_group_t;


/**
*/
extern int kaapi_thread_group_create(kaapi_thread_group_t* thgrp, int size );


/**
*/
extern int kaapi_thread_group_begin_execute(kaapi_thread_group_t* thgrp );


/**
*/
extern int kaapi_thread_group_end_execute(kaapi_thread_group_t* thgrp );


/**
*/
extern int kaapi_thread_group_destroy(kaapi_thread_group_t* thgrp );


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
