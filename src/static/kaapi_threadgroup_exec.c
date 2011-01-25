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
  int i;
  kaapi_frame_t* fp;
  kaapi_thread_context_t* threadctxtmain;
  
  if (thgrp->state != KAAPI_THREAD_GROUP_MP_S) return EINVAL;
  thgrp->state = KAAPI_THREAD_GROUP_EXEC_S;

  if (thgrp->localgid == 0)
  {
    /* Push the task that will mark synchronisation on the main thread */
    threadctxtmain = thgrp->threadctxts[-1];

    /* push the frame for the next task to execute */
    fp = (kaapi_frame_t*)threadctxtmain->sfp;
    threadctxtmain->sfp[1].sp_data = fp->sp_data;
    threadctxtmain->sfp[1].pc = fp->sp;
    threadctxtmain->sfp[1].sp = fp->sp;
    ++threadctxtmain->sfp;

    thgrp->waittask = kaapi_thread_toptask( kaapi_threadcontext2thread(threadctxtmain) );
    kaapi_task_init_with_state( thgrp->waittask, kaapi_taskwaitend_body, KAAPI_MASK_BODY_STEAL, thgrp );
    kaapi_thread_pushtask( kaapi_threadcontext2thread(threadctxtmain) );    
  }
    
  ++thgrp->step;
  kaapi_mem_barrier();
  
  thgrp->startflag = 1;
  
  /* dispatch each thread context on the local gid to processor (i/nodecount)%nproc */
  nproc = kaapi_count_kprocessors;
  
  for (i=0; i<thgrp->group_size; ++i)
  {
    if (thgrp->localgid == thgrp->tid2gid[i])
    {
      kaapi_processor_id_t victim_procid = (kaapi_processor_id_t)(i/thgrp->nodecount) % nproc;
      kaapi_processor_t* victim_kproc = kaapi_all_kprocessors[victim_procid];

      kaapi_cpuset_clear( &thgrp->threadctxts[i]->affinity);
      kaapi_cpuset_set( &thgrp->threadctxts[i]->affinity, victim_procid );
      thgrp->threadctxts[i]->proc = victim_kproc;
      thgrp->threadctxts[i]->partid = i;
      thgrp->threadctxts[i]->unstealable = 1;/* do not allow threads to steal tasks inside ??? */

//printf("Push thread: %i, %p, on processor kid:%i\n", i, (void*)thgrp->threadctxts[i], victim_kproc->kid);
      if (!kaapi_tasklist_ready_isempty(thgrp->threadctxts[i]->readytasklist))
      {
        kaapi_sched_lock( &victim_kproc->lock ); 
        kaapi_sched_pushready( victim_kproc, thgrp->threadctxts[i] );
        kaapi_sched_unlock( &victim_kproc->lock ); 
      }
      else {
        /* put thread into waiting queue of the kproc and initialize the wcs field */
  //printf("First task not ready, push in suspended list\n");
        kaapi_wsqueuectxt_push( victim_kproc, thgrp->threadctxts[i] );
      }
    }
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

  if (thgrp->localgid == thgrp->tid2gid[-1])
  {
    kaapi_threadgroup_execframe( thgrp->threadctxts[-1] );

    kaapi_sched_sync();
    /* pop pushed wait task frame */
    --thgrp->threadctxts[-1]->sfp;

    if (((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0))
    {
      if (thgrp->maxstep != -1) 
      {
        /* avoir restore for the last step */
        if (thgrp->step + 1 <thgrp->maxstep)
          kaapi_threadgroup_restore_thread(thgrp, -1);
      }
      else 
        kaapi_threadgroup_restore_thread(thgrp, -1);
    }
  }
  else {
    kaapi_assert(0); // global barrier 
  }

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
  
  thgrp->state = KAAPI_THREAD_GROUP_CREATE_S;
  return 0;
}
