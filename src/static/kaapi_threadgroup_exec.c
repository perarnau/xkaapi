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


/**
*/
int kaapi_threadgroup_begin_execute(kaapi_threadgroup_t thgrp )
{
  int nproc;
  int i;
  
  if (thgrp->state != KAAPI_THREAD_GROUP_MP_S) return EINVAL;
  thgrp->state = KAAPI_THREAD_GROUP_EXEC_S;

#if 0
  if (thgrp->localgid ==0)
    printf("%i::[kaapi_threadgroup] begin step : %i\n", thgrp->localgid, 1+thgrp->step);
#endif

  /* reset counter for the next iteration */
  KAAPI_ATOMIC_WRITE_BARRIER( &thgrp->endlocalthread, 0 );
    
#if 0
  kaapi_threadgroup_print( stdout, thgrp );
#endif

  ++thgrp->step;
  
  /* dispatch each thread context on the local gid to processor (i/nodecount)%nproc 
     Here a better and finer mapping should be given by the user
  */
  nproc = kaapi_count_kprocessors;
  
  if (thgrp->localgid != thgrp->tid2gid[-1])
  {
    /* if I'm a slave, push a waittask signaled by the master process */
    kaapi_thread_context_t* thread_slavecurrent = kaapi_get_current_processor()->thread;
    thgrp->waittask = kaapi_thread_toptask( kaapi_threadcontext2thread(thread_slavecurrent) );
    kaapi_task_init_with_state( thgrp->waittask, kaapi_taskwaitend_body, KAAPI_MASK_BODY_STEAL, thgrp );
    kaapi_thread_pushtask( kaapi_threadcontext2thread(thread_slavecurrent) );    
  }
  kaapi_mem_barrier();
    
  for (i=0; i<thgrp->group_size; ++i)
  {
    if (thgrp->localgid == thgrp->tid2gid[i])
    {
      kaapi_processor_id_t victim_procid = (kaapi_processor_id_t)(i/thgrp->nodecount) % nproc;
      kaapi_processor_t* victim_kproc = kaapi_all_kprocessors[victim_procid];

      kaapi_cpuset_clear( &thgrp->threadctxts[i]->affinity);
      kaapi_cpuset_set( &thgrp->threadctxts[i]->affinity, victim_procid );
      thgrp->threadctxts[i]->proc        = victim_kproc;
      thgrp->threadctxts[i]->partid      = i;
      thgrp->threadctxts[i]->unstealable = 1;/* do not allow threads to steal tasks inside ??? */

      if (kaapi_thread_isready(thgrp->threadctxts[i]))
      {
        kaapi_sched_lock( &victim_kproc->lock ); 
        kaapi_sched_pushready( victim_kproc, thgrp->threadctxts[i] );
        kaapi_sched_unlock( &victim_kproc->lock ); 
      }
      else {
        /* put thread into waiting queue of the kproc and initialize the wcs field */
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
  kaapi_thread_context_t* threadctxtmain;
  if (thgrp->state != KAAPI_THREAD_GROUP_EXEC_S) return EINVAL;
  if (thgrp->state == KAAPI_THREAD_GROUP_WAIT_S) return 0;

#if 0
  printf("%i::[threadgroup exec] begin execution on #local threads: %i, kid=%i\n", thgrp->localgid, thgrp->localthreads, kaapi_get_current_processor()->kid);
#if 0
  /* */
  kaapi_thread_print(stdout, thgrp->threadctxts[-1]);
#endif
#endif
  if (thgrp->localgid == thgrp->tid2gid[-1])
  {
    threadctxtmain = thgrp->threadctxts[-1];
    
    /* execute task into the readylist */
    kaapi_sched_sync();

#if 0
  printf("%i::[threadgroup exec] master thread finished execute local ready list\n", thgrp->localgid);
#endif

    /* pop frame for task in the ready list */
    --threadctxtmain->sfp;

    /* wait global terminaison and execution of waitend */
    kaapi_sched_sync();

#if 0
  printf("%i::[threadgroup exec] master thread finished execute local wait term task\n", thgrp->localgid);
#endif

    /* pop frame for task in the ready list */
    --threadctxtmain->sfp;

    /* counter reset by THE waittask */
    kaapi_assert_debug(KAAPI_ATOMIC_READ(&thgrp->endglobalgroup) ==0);

    kaapi_mem_barrier();
    
    /* reset main thread */
    if ((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0)
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
#if defined(KAAPI_USE_NETWORK)
    for (kaapi_globalid_t gid=0; gid < thgrp->nodecount; ++gid)
    {
      if (gid != thgrp->localgid)
      {
#if 0
        printf("%i::[kaapi_threadgroup_signalend_service] master send signal end to:%i\n", thgrp->localgid, gid);
        fflush(stdout);
#endif
        /* remote address space -> communication, return end */
        kaapi_network_am(
            gid,
            kaapi_threadgroup_signalend_service, 
            &thgrp->grpid, sizeof(thgrp->grpid)
        );
#if 0
        printf("%i::[kaapi_threadgroup_signalend_service] master end send signal end to:%i\n", thgrp->localgid, gid);
        fflush(stdout);
#endif
      }
    }
#endif // KAAPI_USE_NETWORK
  }
  else {
#if 0
  printf("%i::[threadgroup exec] slave thread begin execute threads\n", thgrp->localgid);
#endif
    /* wait terminaison of the waiting task of the local main thread */
    kaapi_sched_sync();
#if 0
  printf("%i::[threadgroup exec] end thread begin execute threads\n", thgrp->localgid);
#endif
  }
#if defined(KAAPI_USE_NETWORK)
#if 0
  printf("%i::[threadgroup exec] begin barrier\n", thgrp->localgid);
  fflush(stdout);
#endif
  kaapi_memory_global_barrier();
#if 0
  printf("%i::[threadgroup exec] end barrier\n", thgrp->localgid);
  fflush(stdout);
#endif
#endif
#if 0
  if (thgrp->localgid ==0)
    printf("%i::[kaapi_threadgroup_exec] end step :%i, countend:%i\n", 
        thgrp->localgid, thgrp->step, KAAPI_ATOMIC_READ(&thgrp->endglobalgroup) );
#if 0
    kaapi_threadgroup_print( stdout, thgrp );
  /* */
  kaapi_thread_print(stdout, thgrp->threadctxts[-1]);
#endif
#endif

  thgrp->state = KAAPI_THREAD_GROUP_WAIT_S;
  return 0;
}


/**
*/
int kaapi_threadgroup_end_execute(kaapi_threadgroup_t thgrp )
{
  if (thgrp->state != KAAPI_THREAD_GROUP_EXEC_S) return EINVAL;
  kaapi_threadgroup_end_step(thgrp);
  
  thgrp->state = KAAPI_THREAD_GROUP_MP_S;
  return 0;
}
