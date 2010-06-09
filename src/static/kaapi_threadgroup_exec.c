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


/**
*/
int kaapi_threadgroup_begin_execute(kaapi_threadgroup_t thgrp )
{
  int nproc;
  int blocsize;
  int i;
  
  if (thgrp->state != KAAPI_THREAD_GROUP_MP_S) return EINVAL;
  thgrp->state = KAAPI_THREAD_GROUP_EXEC_S;

  /* Push the task that will mark synchronisation on the main thread */
  thgrp->waittask = kaapi_thread_toptask( thgrp->mainthread);
  kaapi_task_init(thgrp->waittask, kaapi_taskwaitend_body, thgrp );
  kaapi_task_setbody(thgrp->waittask, kaapi_suspend_body );
  kaapi_thread_pushtask(thgrp->mainthread);    
  
  ++thgrp->step;
  kaapi_mem_barrier();
  
  thgrp->startflag = 1;
  
  /* dispatch thread context to processor ? */
  nproc = kaapi_count_kprocessors;
  /* dispatch them using bloc destribution of size floor(kaapi_count_kprocessors/thgrp->group_size) */
  blocsize = (thgrp->group_size+nproc-1)/ nproc;
  
  for (i=0; i<thgrp->group_size; ++i)
  {
    kaapi_processor_id_t victim_procid = i/blocsize;
#if 1 // Method should be implemented. Currently push locally and wait stealer    
    kaapi_processor_t* victim_kproc = kaapi_all_kprocessors[victim_procid];
    kaapi_thread_clearaffinity( thgrp->threadctxts[i] );
    kaapi_thread_setaffinity( thgrp->threadctxts[i], victim_procid );
    thgrp->threadctxts[i]->proc = victim_kproc;

    if (kaapi_thread_isready(thgrp->threadctxts[i]))
    {
      kaapi_sched_lock( victim_kproc ); 
      kaapi_sched_pushready( victim_kproc, thgrp->threadctxts[i] );
      kaapi_sched_unlock( victim_kproc ); 
    }
    else {
      /* put pad of the first non ready task as if the thread was suspended (but not into a queue) */
      kaapi_wc_structure_t* wcs = &thgrp->threadctxts[i]->wcs;
      wcs->wclist   = 0;
      wcs->wccell   = 0;
      wcs->affinity = thgrp->threadctxts[i]->affinity;
      kaapi_task_t* task = thgrp->threadctxts[i]->sfp->pc;
      task->pad = wcs;
//      kaapi_wsqueuectxt_lockpush( &victim_kproc->lsuspend, thgrp->threadctxts[i] );
    }
#else
    kaapi_processor_t* current_proc = kaapi_get_current_processor();
    kaapi_thread_clearaffinity( thgrp->threadctxts[i] );
    kaapi_thread_setaffinity( thgrp->threadctxts[i], victim_procid );
    thgrp->threadctxts[i]->proc = current_proc;
    kaapi_wsqueuectxt_push( &current_proc->lsuspend, thgrp->threadctxts[i] );
#endif    
  }
  
  return 0;
}


/**
*/
int kaapi_threadgroup_begin_step(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_MP_S) return EINVAL;
  return kaapi_threadgroup_begin_execute( thgrp );
}


/**
*/
int kaapi_threadgroup_end_step(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_EXEC_S) return EINVAL;
  if (thgrp->state == KAAPI_THREAD_GROUP_WAIT_S) return 0;

  kaapi_sched_sync();
  /* counter reset by THE waittask */
  kaapi_assert( KAAPI_ATOMIC_READ(&thgrp->countend) == 0 );
  
  thgrp->startflag = 0;
  thgrp->state = KAAPI_THREAD_GROUP_WAIT_S;
  return 0;
}


/**
*/
int kaapi_threadgroup_end_execute(kaapi_threadgroup_t thgrp )
{
  kaapi_threadgroup_end_step(thgrp);
  
  for (int i=0; i<thgrp->group_size; ++i)
    kaapi_thread_clear(thgrp->threadctxts[i]);

  kaapi_thread_restore_frame(thgrp->mainthread, &thgrp->mainframe);
#if 0
  kaapi_thread_save_frame(thgrp->mainthread, &thgrp->mainframe);
  fprintf(stdout, "Restore frame:: pc:%p, sp:%p, spd:%p\n", 
    (void*)thgrp->mainframe.pc, 
    (void*)thgrp->mainframe.sp, 
    (void*)thgrp->mainframe.sp_data 
  );
#endif
  
  thgrp->state = KAAPI_THREAD_GROUP_CREATE_S;
  return 0;
}
