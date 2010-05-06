/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** theo.trouillon@imag.fr
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
#include "kaapi_impl.h"
#include "kaapi_staticsched.h"


/** 2 gros bug dans le code de Theo:
    hashmap avec access au lieu de access->data
    ATOMIC_DECR
*/

/**
 */
int kaapi_threadgroup_computedependencies(kaapi_threadgroup_t thgrp, kaapi_thread_t* thread, kaapi_task_t* task)
{
  kaapi_task_t*        task_writer;
  kaapi_counters_list* wc_list;
  kaapi_format_t* task_fmt;

  if(task->body==kaapi_suspend_body || task->body==kaapi_exec_body)
    task_fmt= kaapi_format_resolvebybody(task->ebody);
  else 
    task_fmt= kaapi_format_resolvebybody(task->body);
  
  if (task_fmt ==0) return EINVAL;
  
  /* initialize the pad for every pushed task */
  task->pad = 0;

  /* find the last writer for each args and in which partition it is 
     -> if all writers are in the same partition do nothing, push the task in the i-th partition
     -> if one of the writer is in a different partition, then change the body of the writer
     in order to add information to signal the task that are waiting for parameter
     
    ASSUMPTIONS:
    1- all the threads in the group are inactive and not subject to steal operation
    2- we only consider R,W and RW dependencies, no CW that implies multiple writers
  */
  kaapi_hashentries_t* entry; //Current argument's entry in the Hashmap
  kaapi_atomic_t* counter=0;  //Waiting counter for the task
  
  for (int i=0;i<task_fmt->count_params;i++) 
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(task_fmt->mode_params[i]);
    if (m == KAAPI_ACCESS_MODE_V) 
      continue;
    
    /* its an access */
    kaapi_access_t* access = (kaapi_access_t*)(task_fmt->off_params[i] + (char*)task->sp);
    entry = 0;

    /* find the last writer (task & thread) using the hash map */
    entry = kaapi_hashmap_find(&thgrp->ws_khm, access->data);
    
    if (KAAPI_ACCESS_IS_READ(m))
    {
      /* if entry ==0, access is the first access */
      if ((entry !=0) && (entry->u.datas.last_writer_thread != thread))
      {
        /* create the waiting counter for this task and change its state */
        if (counter ==0) 
        {
          counter =(kaapi_atomic_t*)kaapi_thread_pushdata_align(thread, sizeof(kaapi_atomic_t),8);
          KAAPI_ATOMIC_WRITE(counter, 1);
        }
        else
          KAAPI_ATOMIC_INCR(counter);

        kaapi_task_setbody( task, kaapi_suspend_body);

        /* add task into the reader list */
        task_writer = entry->u.datas.last_writer;
        if (kaapi_task_getbody(task_writer) != kaapi_dependenciessignal_body)
        {
          /* create its waiting readers list */
          wc_list = kaapi_thread_pushdata(
                entry->u.datas.last_writer_thread, 
                sizeof(kaapi_counters_list)
          );
          memset(wc_list, 0, sizeof(kaapi_counters_list) );
          task_writer->pad = wc_list;
          kaapi_task_setbody(task_writer, kaapi_dependenciessignal_body);
        }
        else 
          wc_list = (kaapi_counters_list*)task_writer->pad;
        
        /* allocate a new entry in the list wc_list */
        kaapi_assert(wc_list->size <= KAAPI_COUNTER_LIST_BLOCSIZE);
        wc_list->entry[wc_list->size].waiting_counter = counter;
        wc_list->entry[wc_list->size].waiting_task    = task;
        ++wc_list->size;
      }
    }
    if (KAAPI_ACCESS_IS_WRITE(m))
    {
      /* find the last writer (task & thread) using the hash map */
      if (entry ==0) entry = kaapi_hashmap_insert(&thgrp->ws_khm, access->data);

      entry->u.datas.last_writer=task;
      entry->u.datas.last_writer_thread=thread;      
    }
    
  } /* end for all arguments of the task */

  return 0;
}


/** This is the body of a writer task that has been changed to execute
    1/ the original body
    2/ signal other readers tasks/threads
 */
void kaapi_dependenciessignal_body( void* sp, kaapi_thread_t* thread )
{
  /* thread->pc is the executing task */
  kaapi_task_t*        self    = thread->pc;
  kaapi_counters_list* wc_list = (kaapi_counters_list*)self->pad;
  kaapi_atomic_t* counter;
  kaapi_task_t* task;
  short i;

  thread->pc->pad=0;

  (*self->ebody)(sp,thread); // Execution of the real body
  
  /* write memory barrier to ensure that other threads will view the data produced */
  kaapi_mem_barrier();

  /* signal all readers */  
  while(wc_list != 0) 
  {
    for (i=0; i<wc_list->size; ++i)
    {
      counter = wc_list->entry[i].waiting_counter;
      task = wc_list->entry[i].waiting_task;
      
      if (KAAPI_ATOMIC_DECR(counter) ==0)
        kaapi_task_setbody(task, task->ebody);
    }
    wc_list = wc_list->next;
  }
}
