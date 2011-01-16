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
   The read after write dependencies (RAW) are handle in order to:
      1- envelop the writer task to first execute the original task and then forward dependencies by
      moving state of the reader to ready (in fact: decrement a counter + moving kaapi_suspend_body body
      to the original read task body). The envelopp is build by storing original body and sp fields the
      of the original task into a kaapi_taskbcast_arg_t data structure and replacing the body by the
      new bcast body.
      2- add a counter in the ... field of each reader that has a dependency from a writer not on the same 
      partition. The counter is managed in 2 part: high level bits are bit fields to indicate wich data
      are waiting. It is store in the stack of the reader thread and it is referenced by the XX data structure.
      The purpose of this bit field is to add attach fifo channel to each input data from a remotely distant 
      node. The lower part (at most 8 bits = 255 values) are used to count the number of non waiting effective 
      parameters.
    
  In case of RW access (both read and write accesses), the XX points to a structure with is 
    - kaapi_taskbcast_arg_t if we consider the W access
    - kaapi_taskrecv_arg_t if we consider the R access
  Both these data structures shared the same fields: we can view this as if kaapi_taskbcast_arg_t inerit 
  from kaapi_taskrecv_arg_t.
  
*/

/*
#error
NBUG:
- task_RWW_R_R: encapsulation d'une tâche kaapi_task_recvbcast par une tâche kaapi_bcast.
recvbcast <=> bcast + state à steal. Donc ne devrait rien changer.... ? Si ce n'est des asserts
*/

/*
*/
kaapi_hashentries_t* kaapi_threadgroup_newversion( 
    kaapi_threadgroup_t thgrp, 
    kaapi_hashmap_t* hmap, 
    int tid, 
    kaapi_access_t* access 
)
{
  kaapi_hashentries_t* entry;
  kaapi_version_t* ver;
  entry = kaapi_hashmap_insert( &thgrp->ws_khm, access->data );
   
  /* here a stack allocation attached with the thread group */
  ver = entry->u.dfginfo = kaapi_versionallocator_allocate( &thgrp->ver_allocator );
  kaapi_assert( ver != 0 );
  ver->tag = 0; //++thgrp->tag_count;
  ver->writer_task = 0;
  ver->writer_thread = -1; /* main thread */
  ver->original_data = ver->writer_data = access->data;
  ver->com = 0;
  memset( &ver->readerslist, 0, sizeof(ver->readerslist));
  ver->cnt_readers = 0;
  return entry;
}



/*
*/
void kaapi_threadgroup_version_addfirstreader( 
    kaapi_threadgroup_t thgrp, 
    kaapi_vector_t* v, 
    int tid, 
    kaapi_task_t* task, 
    kaapi_access_t* access, 
    int ith 
)
{
  kaapi_pidreader_t* entry;
  entry = kaapi_vector_pushback(v);
  entry->addr = access->data;
  entry->tid  = tid;
  entry->used = (char)ith;
  entry->task = task;
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
    kaapi_version_t*    ver, 
    int                 tid, 
    kaapi_task_t*       task, 
    kaapi_access_t*     access,
    size_t data_size,
    int ith
)
{
  kaapi_taskbcast_arg_t* argbcast =0;
  kaapi_task_t* taskbcast;
  kaapi_thread_t* writer_thread = 0;
  kaapi_part_datainfo_t* kpdi;

  kaapi_assert( -1 <= tid );

  /* initialize the writer data structure if it not on the same partition (else only
     update reader data structure without changing the writer code
  */
  if (ver->writer_thread != tid)
  {
    writer_thread = kaapi_threadgroup_thread( thgrp, ver->writer_thread );

    /* if not exist -> first reader, create a pure send task */
    if (ver->writer_task == 0)
    { 
      taskbcast = kaapi_thread_toptask(writer_thread);
      kaapi_task_init(
            taskbcast, 
            kaapi_taskbcast_body, 
            kaapi_thread_pushdata(writer_thread, sizeof(kaapi_taskbcast_arg_t) ) );
      argbcast = kaapi_task_getargst( taskbcast, kaapi_taskbcast_arg_t );
      memset(argbcast, 0, sizeof(kaapi_taskbcast_arg_t) );
      argbcast->last     = &argbcast->head;
      taskbcast->sp      = argbcast;
      ver->writer_task   = taskbcast;
      ver->writer_data   = access->data;
//      ver->tag           = ++thgrp->tag_count;
      /* push the task */
      kaapi_thread_pushtask(writer_thread);
    }
    else {
      kaapi_task_body_t body = kaapi_task_getbody(ver->writer_task);
      /* if the writer is in fact a recv (case of task with both w and r access), mute the recv data structure */
      if (body == kaapi_taskrecv_body)
      {
        kaapi_assert( kaapi_task_state_issteal( kaapi_task_getstate( ver->writer_task) ) );
        argbcast = (kaapi_taskbcast_arg_t*)kaapi_thread_pushdata(writer_thread, sizeof(kaapi_taskbcast_arg_t) );
        memset( argbcast, 0, sizeof(kaapi_taskbcast_arg_t));
        /* recopy old field */
        argbcast->common = *(kaapi_taskrecv_arg_t*)ver->writer_task->sp;
        ver->writer_task->sp   = argbcast;

#if (SIZEOF_VOIDP == 4)
        kaapi_task_setbody(ver->writer_task, kaapi_taskbcast_body);
#else
        kaapi_task_setstate(ver->writer_task, kaapi_task_state_setsteal( kaapi_task_body2state(kaapi_taskbcast_body) ) );
#endif

//        ver->tag           = ++thgrp->tag_count;
      }

      /* writer already exist: if it is not a bcast, encapsulate the task by a bcast task
      */
      else if (body != kaapi_taskbcast_body)
      {
//        kaapi_assert(kaapi_task_getbody(ver->writer_task) != kaapi_taskbcast_body);
        argbcast = (kaapi_taskbcast_arg_t*)kaapi_thread_pushdata(writer_thread, sizeof(kaapi_taskbcast_arg_t) );
        memset(argbcast, 0, sizeof(kaapi_taskbcast_arg_t) );
        argbcast->common.original_sp   = ver->writer_task->sp;
        argbcast->common.original_body = body;
        ver->writer_task->sp = argbcast;
//        ver->tag           = ++thgrp->tag_count;
        kaapi_task_setbody(ver->writer_task, kaapi_taskbcast_body );
//        kaapi_task_setextrabody(ver->writer_task, kaapi_taskbcast_body );
      } 
      else {
        /* else its already a bcast kind of task */
        argbcast = (kaapi_taskbcast_arg_t*)ver->writer_task->sp;
      }
      argbcast->last          = &argbcast->head;
    }

    if (ver->com == 0) 
    {
      /* find com tag into the list of version in bcast list */
      kaapi_com_t* c = &argbcast->head;
      if (c->size !=0) 
      {
        while (c !=0)
        {
          if (c->tag == ver->tag) break;
          c = c ->next;
        }
      }
      ver->com = c;
      if (ver->com ==0)
      {
        ver->com = kaapi_thread_pushdata( writer_thread, sizeof(kaapi_com_t) );
        memset( ver->com, 0, sizeof(kaapi_com_t) );
        argbcast->last->next = ver->com;
        argbcast->last       = ver->com;
        ver->com->a          = *access;
      }
    }
  }

  /* if already has a reader do nothing, else update the reader information */
  kpdi = kaapi_version_reader_find_insert_tid( thgrp, ver, tid );
  if (!kpdi->reader.used)
  {
    kaapi_thread_t* thread = kaapi_threadgroup_thread( thgrp, tid );

    /* if it is a recv, mark the task as suspended */
    if (ver->writer_thread != tid )
    {
      kaapi_taskrecv_arg_t* argrecv = 0;
      kaapi_task_body_t body = kaapi_task_getbody(task);
      if ((body != kaapi_taskbcast_body) && (body != kaapi_taskrecv_body))
      {
        argrecv = kaapi_thread_pushdata( thread, sizeof(kaapi_taskrecv_arg_t) );
        memset( argrecv, 0, sizeof(kaapi_taskrecv_arg_t) );
        argrecv->original_body = body;
        argrecv->original_sp   = task->sp;
        task->sp = argrecv;
      } else {
        argrecv = (kaapi_taskrecv_arg_t*)task->sp;
      }
//      KAAPI_ATOMIC_INCR( &argrecv->counter );
      ++argrecv->original_counter;
      KAAPI_THREADGROUP_SETRECVPARAM( argrecv, ith );
        
      if (kpdi->reader.addr ==0)
        kpdi->reader.addr = kpdi->delete_data;
      if (kpdi->reader.addr == 0)
        kpdi->reader.addr = ver->writer_data;
      kaapi_assert (kpdi->reader.addr != 0);
      
      /* The task is waiting for a parameter -> suspend state */

#if (SIZEOF_VOIDP == 4)
      kaapi_task_setstate(task, KAAPI_MASK_BODY_STEAL);
      kaapi_task_setbody(task, kaapi_taskrecv_body);
#else
      kaapi_task_setstate(task, kaapi_task_state_setsteal( kaapi_task_body2state(kaapi_taskrecv_body) ) );
#endif

      if (ver->com->tag ==0) ver->com->tag = ++thgrp->tag_count;
      kaapi_comentry_t* readerentry = kaapi_thread_pushdata(writer_thread, sizeof(kaapi_comentry_t) );
      kaapi_assert_m(readerentry !=0,"cannot allocated new reader");
      readerentry->tid  = tid;
      readerentry->task = task;
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
      readerentry->addr = kpdi->reader.addr;
#else
      readerentry->addr = access->data;
#endif
      readerentry->size = data_size;
      //printf("New comentry: %p\n", (void*)readerentry ); fflush(stdout);

      /* link it */
      readerentry->next = ver->com->entry;
      ver->com->entry   = readerentry;

      ++ver->com->size;
    }
    else 
    { /* on the same partition, only store the task info about who is reading the data */
      kpdi->reader.addr = ver->writer_data;
    }

    /* WARNING here the data should be allocated in the THREAD if we want to avoid WAR */
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
    access->data = kpdi->reader.addr;
#else
    access->data = kpdi->reader.addr = ver->writer_data;
#endif
    kpdi->reader.task = task;
    kpdi->reader.used = 1;
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
  kaapi_assert( -1 <= tid );

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
  { /* this is the first writer or a reader exist on -1 */
    kaapi_assert( (ver->cnt_readers == 0) || kaapi_version_hasreader_tid(ver, -1)); 
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
    kaapi_part_datainfo_t* kpdi = kaapi_version_reader_find_tid(ver, tid);
    int war = (ver->cnt_readers>1) || !kpdi->reader.used;
    if (war)
    {
#if defined(KAAPI_STATIC_HANDLE_WARWAW)
//      printf("***Not yet implemented WAR dependency on task: %p, data:%p\n", (void*)task, (void*)ver->original_data);
//      fflush( stdout );
#else
      printf("***Warning, WAR dependency writer not correctly handle on task: %p, data:%p\n", (void*)task, (void*)ver->original_data);
      fflush( stdout );
#endif
    }

    /* for all readers, mark data on each partition has deleted and store its address */
    kaapi_part_datainfo_t* curr = ver->readerslist;
    while (curr != 0)
    {
      if (curr->reader.used)
      {
        if (curr->reader.addr != ver->original_data)
          curr->delete_data = curr->reader.addr;
        curr->reader.addr = 0;
        curr->reader.used = 0;
      }
      else curr->delete_data = 0;
      
      curr = curr->next;
    }
    ver->cnt_readers =0;
        
    /* re-use data if it was delete in partition tid */
    if (kpdi->delete_data !=0)
    {
      ver->writer_data = access->data = kpdi->delete_data;
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
        access->data = ver->writer_data = ver->original_data;/* alias ? malloc( sizeof(void*) ); */
//WARNING        exit(1);
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
    ver->com           = 0;
  }
  
  return task;
}


/*
*/
kaapi_part_datainfo_t* kaapi_version_reader_find_tid( const kaapi_version_t* v, int tid )
{
  kaapi_part_datainfo_t* curr = v->readerslist;
  while (curr !=0)
  {
    if (curr->tid == tid) return curr;
    curr = curr->next;
  }
  return curr;
}


/*
*/
kaapi_part_datainfo_t* kaapi_version_reader_find_insert_tid( kaapi_threadgroup_t thgrp, kaapi_version_t* v, int tid )
{
  kaapi_part_datainfo_t* curr = v->readerslist;
  while (curr !=0)
  {
    if (curr->tid == tid) return curr;
    curr = curr->next;
  }
  curr = kaapi_part_datainfo_allocate(thgrp); /* allocate into the group */
  curr->tid = tid;
  
  /* link */
  curr->next = v->readerslist;
  v->readerslist = curr;
  return curr;
}

