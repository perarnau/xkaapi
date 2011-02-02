/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#include "misc/kaapi_hashmap.h"

static int kaapi_threadgroup_clear( kaapi_threadgroup_t thgrp )
{
  for (int i=-1; i<thgrp->group_size; ++i)
  {
    /* delete comlink */
    thgrp->lists_send[i] = 0;
    thgrp->lists_recv[i] = 0;


    if (thgrp->localgid == thgrp->tid2gid[i]) 
    {
      /* init the thread from the thread context */
      kaapi_tasklist_t* tasklist = thgrp->threadctxts[i]->tasklist;
      kaapi_thread_clear( thgrp->threadctxts[i] );
      
      /* reset the task list */
      tasklist->sp    = 0;
      tasklist->front = 0;
      tasklist->back  = 0;
      thgrp->threadctxts[i]->tasklist = tasklist;
      thgrp->threadctxts[i]->partid   = i;
      thgrp->threadctxts[i]->the_thgrp = thgrp;
    }
  }
  /* delete allocator: free all temporary memory used by managing activation list and communication */
  kaapi_allocator_destroy(&thgrp->allocator);
  
  /* update the list of version in the hashmap to suppress reference to previously executed task */
  for (int i=0; i<KAAPI_HASHMAP_SIZE; ++i)
  {
    kaapi_hashentries_t* entry = _get_hashmap_entry(&thgrp->ws_khm, i);
    while (entry != 0)
    {
      kaapi_version_t* ver = entry->u.version;
      ver->writer_mode = KAAPI_ACCESS_MODE_VOID;
      ver->writer.task = 0;
      ver->writer.ith  = -1;
      kaapi_data_version_list_append( &ver->todel, &ver->copies );
      entry = entry->next;
    }
  }

  thgrp->state              = KAAPI_THREAD_GROUP_CREATE_S;
  thgrp->flag               = 0;
  thgrp->tag_count          = 0;
  thgrp->maxstep            = -1;
  thgrp->step               = -1;
  thgrp->signal_step        = -1;
 
  return 0;
}


/**
*/
int kaapi_threadgroup_begin_partition(kaapi_threadgroup_t thgrp, int flag)
{
  kaapi_processor_t* kproc;

  if (thgrp->state == KAAPI_THREAD_GROUP_MP_S)
  {
    /* the previous execution is finish: clear all entries as if the thread group was created */
    kaapi_threadgroup_clear( thgrp );
  }
  else {
    /* do not init the hash map if previous execution of the thread group */
    kaapi_hashmap_init( &thgrp->ws_khm, 0 );
  }

  if (thgrp->state != KAAPI_THREAD_GROUP_CREATE_S) return EINVAL;
  thgrp->state = KAAPI_THREAD_GROUP_PARTITION_S;
  
  /* same the main thread frame to restore it at the end of parallel computation */
  if (thgrp->localgid == thgrp->tid2gid[-1])
  {
    kaapi_thread_save_frame(thgrp->threads[-1], &thgrp->mainframe);

    /* avoid thief to steal the main thread will tasks are added */
    kproc = thgrp->threadctxts[-1]->proc;
    kaapi_sched_lock(&kproc->lock);
    thgrp->threadctxts[-1]->unstealable = 1;
    kaapi_sched_unlock(&kproc->lock);
  }
  
  kaapi_assert_debug( (flag == 0) || (flag == KAAPI_THGRP_SAVE_FLAG) );
  thgrp->flag = flag;
  
  return 0;
}


/**
*/
int kaapi_threadgroup_end_partition(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_PARTITION_S) 
    return EINVAL;

  /* */
//  kaapi_threadgroup_print( stdout, thgrp );
  
  /* save if required and update remote reference */
  kaapi_threadgroup_barrier_partition( thgrp );
  
  /* */
  /* kaapi_threadgroup_print( stdout, thgrp ); */
  
  kaapi_mem_barrier();
  
  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
