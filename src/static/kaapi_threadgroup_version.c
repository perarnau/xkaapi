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

/* ........................................ Implementation notes ........................................*/
/* Each time a task is forked on a partition, the kaapi_threadgroup_computedependencies method is called
   to compute and update dependencies. 
   At the difference with the pure workstealing execution, here the read after write dependencies (RAW)
   are handle in order to:
      1- envelop the writer task to first execute the original task and then forward dependencies by
      moving state of the reader to ready (in fact: decrement a counter + moving kaapi_suspend_body body
      to the original read task body). The envelopp is build by storing into the free pad field the
      a pointer to a kaapi_taskbcast_arg_t data structure and replacing the body by the
      new bcast body.
      2- add a counter in the pad field of each reader that has a dependency from a writer not on the same 
      partition. The counter is managed in 2 part: high level bits are bit fields to indicate wich data
      are waiting. It is store in the stack of the reader thread and it is referenced by the pad data structure.
      The purpose of this bit field is to add attach fifo channel to each input data from a remotely distant 
      node. The lower part (at most 8 bits = 255 values) are used to count the number of non waiting effective 
      parameters.
    
  In case of RW access (both read and write accesses), the pad points to a structure with is 
    - kaapi_taskbcast_arg_t if we consider the W access
    - kaapi_taskrecv_arg_t if we consider the R access
  Both this data structures shared the same fields: we can view this as if kaapi_taskbcast_arg_t inerit 
  from kaapi_taskrecv_arg_t.
  
*/


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
    kaapi_access_t* access,
    int ith
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
  else 
  { /* already has a non null pad  */
    taskbcast= 0;
    argbcast = (kaapi_taskbcast_arg_t*)ver->writer_task->pad;
    if (ver->writer_task->body == kaapi_taskrecv_body) /* case of read-write access */
      argbcast = 0;
    
    if ((argbcast ==0) && (ver->writer_thread != tid ))
    {
      kaapi_assert(kaapi_task_getbody(ver->writer_task) != kaapi_taskbcast_body);
      if (ver->writer_task->body == kaapi_taskrecv_body) /* case of read-write access */
      {
        kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, ver->writer_thread );
        argbcast = (kaapi_taskbcast_arg_t*)kaapi_thread_pushdata(thread, sizeof(kaapi_taskbcast_arg_t) );
        memset( argbcast, 0, sizeof(kaapi_taskbcast_arg_t));
        argbcast->common = *(kaapi_taskrecv_arg_t*)ver->writer_task->pad;
        argbcast->last = &argbcast->head;
        kaapi_task_setbody(ver->writer_task, kaapi_taskrecvbcast_body);
      }
      else {
        kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, tid );
        argbcast = (kaapi_taskbcast_arg_t*)kaapi_thread_pushdata(thread, sizeof(kaapi_taskbcast_arg_t) );
        memset(argbcast, 0, sizeof(kaapi_taskbcast_arg_t) );
        argbcast->last     = &argbcast->head;
        kaapi_task_setbody(ver->writer_task, kaapi_taskbcast_body );
      }
      ver->writer_task->pad = argbcast;
      ver->com           = &argbcast->head; 
      ver->com->tag      = ver->tag;
      ver->com->a        = *access;
    } 
    else {
      /* on the same site, do nothing */
      if (ver->com == 0) 
      {
        /* find it into the list of version in bcast*/
        kaapi_com_t* c = &argbcast->head;
        while (c !=0)
        {
          if (c->tag == ver->tag) break;
          c = c ->next;
        }
        ver->com = c;
        if (ver->com ==0)
        {
          if (ver->writer_thread == -1)
            ver->com = kaapi_thread_pushdata( thgrp->mainthread, sizeof(kaapi_com_t) );
          else
            ver->com = kaapi_thread_pushdata( kaapi_threadgroup_thread( thgrp, ver->writer_thread), sizeof(kaapi_com_t) );
          memset( ver->com, 0, sizeof(kaapi_com_t) );
          argbcast->last->next = ver->com;
          argbcast->last       = ver->com;
          ver->com->tag        = ver->tag;
          ver->com->a          = *access;
        }
      }
    }
  }


  /* if already has a reader do nothing */    
  if (!ver->readers[tid].used)
  {
    kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, tid );

    if (ver->writer_thread != tid )
    {
      kaapi_taskrecv_arg_t* argrecv = (kaapi_taskrecv_arg_t*)task->pad;
      if (argrecv ==0) {
        task->pad = argrecv = kaapi_thread_pushdata( thread, sizeof(kaapi_taskrecv_arg_t) );
        memset( argrecv, 0, sizeof(kaapi_taskrecv_arg_t) );
      }

      KAAPI_THREADGROUP_SETRECVPARAM( argrecv, ith );
        
      if (ver->readers[tid].addr ==0)
        ver->readers[tid].addr = ver->delete_data[tid];
      if (ver->readers[tid].addr == 0)
        ver->readers[tid].addr = ver->writer_data;
      kaapi_assert (ver->readers[tid].addr != 0);
      
      if (task->body == kaapi_taskbcast_body) 
        kaapi_task_setbody(task, kaapi_taskrecvbcast_body );
      else
        kaapi_task_setbody(task, kaapi_taskrecv_body );

      ver->readers[tid].task = task;

      /* allocate a new entry in the list wc_list if not on the same partition */
      kaapi_assert(ver->com->size <= KAAPI_BCASTENTRY_SIZE);
      ver->com->entry[ver->com->size].task = task;
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
      ver->com->entry[ver->com->size].addr = ver->readers[tid].addr;
#else
      ver->com->entry[ver->com->size].addr = access->data;
#endif      
      ++ver->com->size;
    }
    else { /* on the same partition, only store the task info about reading the data */
      ver->readers[tid].task = task;
      ver->readers[tid].addr = ver->writer_data;
    }
      
    
    /* WARNING here the data should be allocated in the THREAD if we want to avoid WAR */
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
    access->data = ver->readers[tid].addr;
#else
    access->data = ver->readers[tid].addr = ver->writer_data;
#endif
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
    kaapi_access_t* access,
    int ith
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
    int war = (ver->cnt_readers>1) || !(ver->readers[tid].used);
    if (war)
    {
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
      printf("***Not yet implemented WAR dependency on task: %p, data:%p\n", (void*)task, (void*)ver->original_data);
#else
      printf("***Warning, WAR dependency writer not correctly handle on task: %p, data:%p\n", (void*)task, (void*)ver->original_data);
#endif
      fflush( stdout );
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
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
      /* two possibilities here:
         - allocate (rename) before execution the data to access.
         - flag the task in order it reallocates at runtime the data 
         and let it to forward the new pointer following the access chain.
      */
      if (war)
      {
        access->data = ver->writer_data = malloc( sizeof(void*) );
        exit(1);
      }
      else 
        access->data = ver->writer_data = ver->original_data;
#else
      kaapi_assert( ver->original_data == access->data );
      ver->writer_data = ver->original_data;
#endif
    }
    ver->writer_thread = tid;
    ver->writer_task   = task;
    ver->tag           = ++thgrp->tag_count;
    ver->com           = 0;
  }
  
  return task;
}

