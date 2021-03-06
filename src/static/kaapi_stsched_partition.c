/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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

static int kaapi_threadgroup_clear( kaapi_threadgroup_t thgrp )
{
#if 0
  for (int i=-1; i<thgrp->group_size; ++i)
  {
    /* delete comlink */
    thgrp->lists_send[i] = 0;
    thgrp->lists_recv[i] = 0;

    if (thgrp->localgid == thgrp->tid2gid[i]) 
    {
      /* init the thread from the thread context */
      kaapi_tasklist_t* tasklist;
      if (i == -1) 
        tasklist = thgrp->tasklist_main;
      else 
        tasklist = thgrp->threadctxts[i]->sfp->tasklist;
      kaapi_assert_debug(tasklist != 0);
      if (i != -1)
        kaapi_thread_clear( thgrp->threadctxts[i] );
      
      /* reset the task list */
      tasklist->sp    = 0;
      tasklist->front = 0;
      tasklist->back  = 0;
      if (i != -1)
        thgrp->threadctxts[i]->sfp->tasklist  = tasklist;
      thgrp->threadctxts[i]->partid         = i;
      thgrp->threadctxts[i]->the_thgrp      = thgrp;
      
    }
  }
  /* delete allocator: free all temporary memory used by managing activation list and communication */
  kaapi_allocator_destroy(&thgrp->allocator);
  
  /* update the list of version in the hashmap to suppress reference to previously executed tasks */
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
#endif
  return 0;
}

/**
*/
int kaapi_threadgroup_begin_partition(kaapi_threadgroup_t thgrp, int flag)
{
  kaapi_processor_t* kproc;
  kaapi_frame_t* fp;
  kaapi_thread_context_t* threadctxtmain;

#if 0
  printf("\n\n>>>>>>>>>>>>> Begin partition \n");
  kaapi_thread_print(stdout, thgrp->threadctxts[-1]);
#endif

  if (thgrp->state == KAAPI_THREAD_GROUP_MP_S)
  {
    /* the previous execution is finished: clear all entries as if the thread group was created */
    kaapi_threadgroup_clear( thgrp );
  }

  if (thgrp->state != KAAPI_THREAD_GROUP_CREATE_S) 
    return EINVAL;

  thgrp->state = KAAPI_THREAD_GROUP_PARTITION_S;
  
  if (thgrp->localgid == kaapi_threadgroup_tid2gid(thgrp, -1))
  {
    /* save the main thread frame to restore at the end of parallel computation */
    kaapi_thread_save_frame(thgrp->threads[-1], &thgrp->mainframe);

    /* avoid thief to steal the main thread while tasks are added */
    threadctxtmain = thgrp->threadctxts[-1];
    kproc = threadctxtmain->proc;
    kaapi_sched_lock(&kproc->lock);
    threadctxtmain->unstealable = 1;
    kaapi_sched_unlock(&kproc->lock);
    threadctxtmain->partid = -1;

    fp = (kaapi_frame_t*)threadctxtmain->sfp;

    /* Push the task that will be active when parallel computation will finish */
    threadctxtmain->sfp[1].sp_data   = fp->sp_data;
    threadctxtmain->sfp[1].pc        = fp->sp;
    threadctxtmain->sfp[1].sp        = fp->sp;
    threadctxtmain->sfp[1].tasklist  = 0;

    fp = (kaapi_frame_t*)++threadctxtmain->sfp;
    thgrp->waittask = kaapi_thread_toptask( kaapi_threadcontext2thread(threadctxtmain) );
    kaapi_task_init_with_state( thgrp->waittask, kaapi_taskwaitend_body, KAAPI_MASK_BODY_STEAL, thgrp );
    kaapi_thread_pushtask( kaapi_threadcontext2thread(threadctxtmain) );    

    /* push the frame to store the ready list of spawned task */
    threadctxtmain->sfp[1].sp_data   = fp->sp_data;
    threadctxtmain->sfp[1].pc        = fp->sp;
    threadctxtmain->sfp[1].sp        = fp->sp;
    ++threadctxtmain->sfp;
    kaapi_threadgroup_initthread(thgrp, -1);
  }
  
  kaapi_assert_debug( (flag == 0) || (flag == KAAPI_THGRP_SAVE_FLAG) );
  thgrp->flag = flag;

#if 0
  /* */
  printf("\n\n----------- Begin partition \n");
  kaapi_thread_print(stdout, thgrp->threadctxts[-1]);
#endif
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
  
#if 0
  /* save if required and update remote reference */
  kaapi_threadgroup_barrier_partition( thgrp );
#endif
  
  /* */
  
#if 0 /* ... */
  printf("\n\n<<<<<<<<<<<<< End partition \n");
  kaapi_thread_print(stdout, thgrp->threadctxts[-1]);
  printf("\n\n\n");

char filename[128];
sprintf(filename, "/tmp/graph.dot");
FILE* filedot = fopen(filename, "w");
//  kaapi_thread_print( stdout, thread );
fprintf( filedot, "digraph G {\n" );
for (int i=-1; i<thgrp->group_size; ++i)
{
  kaapi_frame_print_dot( filedot, thgrp->threadctxts[i]->sfp, 1 );
}
fprintf( filedot, "\n\n}\n" );
fclose(filedot);
#endif /* ... */

  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
