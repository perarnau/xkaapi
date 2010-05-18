/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
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
#include "kaapi_impl.h"
#include "kaapi_staticsched.h"


/*
*/
kaapi_hashentries_t* kaapi_threadgroup_newversion( kaapi_threadgroup_t thgrp, kaapi_hashmap_t* hmap, int tid, kaapi_access_t* access )
{
  kaapi_hashentries_t* entry;
  kaapi_version_t* ver;
  entry = kaapi_hashmap_insert(&thgrp->ws_khm, access->data);
   
  /* here a stack allocation attached with the thread group */
  ver = entry->u.dfginfo = calloc( 1, sizeof(kaapi_version_t) );
  ver->tag = ++thgrp->tag_count;
  ver->writer_task = 0;
  ver->writer_thread = -1;
  ver->original_data = ver->writer_data = access->data;
  ver->com = 0;
  memset( &ver->readers, 0, sizeof(ver->readers));
  ver->cnt_readers = 0;
  return entry;
}


/*
*/
void kaapi_threadgroup_deleteversion( kaapi_threadgroup_t thgrp, kaapi_version_t* ver )
{
}


/* add a new reader
*/
kaapi_task_t* kaapi_threadgroup_version_newreader( 
    kaapi_threadgroup_t thgrp, 
    kaapi_version_t* ver, 
    int tid, 
    kaapi_task_t* task, 
    kaapi_access_t* access
)
{
  kaapi_taskbcast_arg_t* argbcast =0;
  kaapi_task_t* taskbcast;
  
  if (ver->writer_task == 0)
  { /* this is a first read, without defined writer -> the implicit writer is on the mainthread */
    /* push task bcast */
    taskbcast = kaapi_thread_toptask(thgrp->mainthread);
    kaapi_task_init(taskbcast, kaapi_taskbcast_body, kaapi_thread_pushdata(thgrp->mainthread, sizeof(kaapi_taskbcast_arg_t) ) );
    argbcast = kaapi_task_getargst( taskbcast, kaapi_taskbcast_arg_t );
    memset(argbcast, 0, sizeof(kaapi_taskbcast_arg_t) );
    argbcast->last     = &argbcast->head;
    taskbcast->pad     = argbcast;
    ver->writer_task   = taskbcast;
    ver->writer_data   = access->data;
    ver->writer_thread = -1;
    ver->com           = &argbcast->head;
    ver->com->tag      = ver->tag;
    ver->com->a        = *access;
    /* push the task */
    kaapi_thread_pushtask(thgrp->mainthread);
  }
  else {
    taskbcast= 0;
    argbcast = (kaapi_taskbcast_arg_t*)ver->writer_task->pad;
    /* if 0 -> mute the task + allocate the bcast arguments */
    if ((argbcast ==0) && (ver->writer_thread != tid ))
    {
      kaapi_assert(kaapi_task_getbody(ver->writer_task) != kaapi_taskbcast_body);
      kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, tid );
      argbcast = (kaapi_taskbcast_arg_t*)kaapi_thread_pushdata(thread, sizeof(kaapi_taskbcast_arg_t) );
      memset(argbcast, 0, sizeof(kaapi_taskbcast_arg_t) );
      argbcast->last     = &argbcast->head;
      ver->writer_task->pad = argbcast;
      ver->com           = &argbcast->head; 
      ver->com->tag      = ver->tag;
      ver->com->a        = *access;
      kaapi_task_setbody(ver->writer_task, kaapi_taskbcast_body );
    } else {
      ver->com = kaapi_thread_pushdata( kaapi_threadgroup_thread( thgrp, ver->writer_thread), sizeof(kaapi_com_t) );
      memset( ver->com, 0, sizeof(kaapi_com_t) );
      argbcast->last->next = ver->com;
      argbcast->last       = ver->com;
      ver->com->tag        = ver->tag;
      ver->com->a          = *access;
    }
  }

  /* writer not on the same partition 
     - mute the writer task to be a bcast task
     - add a new reader in the list of the bcast task 
  */

  /* if already has a reader do nothing */    
  if (!ver->readers[tid].used)
  { /* add a new reader:
       - push a recv task with one access to the data in suspend mode
    */
    kaapi_task_t savedtask;
    kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, tid );

    if (ver->writer_thread != tid )
    {
      /* save the top task */
      savedtask = *task;
      /* push task recv */
      kaapi_task_t* taskrecv = kaapi_thread_toptask(thread);
      kaapi_taskrecv_arg_t* argrecv;
      kaapi_task_init(taskrecv, kaapi_taskrecv_body, kaapi_thread_pushdata(thread, sizeof(kaapi_taskrecv_arg_t) ) );
      argrecv = kaapi_task_getargst( taskrecv, kaapi_taskrecv_arg_t );
      argrecv->tag = ver->tag;
      argrecv->a   = *access;
      kaapi_task_setbody(taskrecv, kaapi_suspend_body );
      ver->readers[tid].task = taskrecv;

      /* push the task */
      kaapi_thread_pushtask(thread);
      
      /* restore savedtask on the top task position */
      task = kaapi_thread_toptask(thread);
      *task = savedtask;

      /* allocate a new entry in the list wc_list if not on the same partition */
      kaapi_assert(ver->com->size <= KAAPI_BCASTENTRY_SIZE);
      ver->com->entry[ver->com->size].recv_task = taskrecv;
      ver->com->entry[ver->com->size].addr      = access->data;
      ++ver->com->size;
    }
    else /* on the same partition, only store the task info about reading the data */
      ver->readers[tid].task = task;
    
    /* WARNING here the data should be allocated in the THREAD if we want to avoid WAR */
    access->data = ver->readers[tid].addr = ver->writer_data;
    ver->readers[tid].used = true;
    

    ++ver->cnt_readers;
  }
  else {
    /* mute the data */
  }

  return task;
}


/* New writer
*/
kaapi_task_t* kaapi_threadgroup_version_newwriter( 
    kaapi_threadgroup_t thgrp, 
    kaapi_version_t* ver, 
    int tid, 
    kaapi_task_t* task, 
    kaapi_access_t* access 
)
{
  /* an entry alread exist: 
     - W mode => mute the data => invalidate copies
     - if other copies exist in the thread group then be carefull in order to not modify other read copies
     through the write.
     - so if #task_readers >1 or task_readers[i] are not in the same partition, then make a copy
     by forking task_copy just before:
        -> recopy the task to temporary
        -> mute the current task to be a task_copy
        -> push the original task after
  */
  if (ver->writer_task == 0)
  { /* this is the first writer */
    kaapi_assert( ver->cnt_readers == 0);
    ver->writer_data   = access->data;
    ver->writer_task   = task;
    ver->writer_thread = tid;
  }
  else 
  { /* a writer already exist:
       - invalid all copies in all the threads.
         * add task delete on all the threads or marks threads with data to delete at the end
         * update the writer information, reset the list of readers
    */
    /* print a warning about WAR dependencies */
    if ((ver->cnt_readers>1) || !(ver->readers[tid].used))
    {
      printf("***Warning, WAR dependency writer not correctly handle on task: %p, data:%p\n", (void*)task, (void*)ver->original_data);
    }

    /* mark data on each partition deleted */
    int i, r;
    for (i=0, r=0; r < ver->cnt_readers; ++i)
    {
      if (ver->readers[i].used)
      {
        if (ver->readers[i].addr != ver->original_data)
          ver->delete_data[i] = ver->readers[i].addr;
        ver->readers[i].addr = 0;
        ++r;
        ver->readers[i].used = false;
      }
      else ver->delete_data[i] = 0;
    }
    ver->cnt_readers =0;
    
    if (ver->delete_data[tid] !=0)
    {
      ver->writer_data = access->data = ver->delete_data[tid];
    }
    else {
      kaapi_assert( ver->original_data == access->data );
      ver->writer_data = ver->original_data;
    }
    ver->writer_thread = tid;
    ver->writer_task = task;
    ver->tag = ++thgrp->tag_count;
  }
  
  return task;
}

