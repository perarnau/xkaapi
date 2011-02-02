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

/* fwd decl 
*/

#if defined(KAAPI_USE_NETWORK)
/* network service to signal end of iteration of one tid */
static void kaapi_threadgroup_signalend_service(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
#if 0
  printf("[kaapi_threadgroup_signalend_service] begin receive signal tag\n"); fflush(stdout);
#endif
  int32_t grpid;
  kaapi_threadgroup_t thgrp;
  memcpy(&grpid, buffer, sizeof(int32_t));
  thgrp = kaapi_all_threadgroups[grpid];

  if (KAAPI_ATOMIC_INCR( &thgrp->endglobalgroup ) == (int)thgrp->nodecount)
  {
    KAAPI_ATOMIC_WRITE_BARRIER( &thgrp->endglobalgroup, 0 );
    kaapi_task_orstate( thgrp->waittask, KAAPI_MASK_BODY_TERM );
#if 1
    printf("%i::[kaapi_threadgroup_signalend_service] set waitting task to term\n", thgrp->localgid);
#endif
  }
}
#endif

/**
*/
static void kaapi_threadgroup_signalend_tid( int tid, kaapi_threadgroup_t thgrp )
{
  /* master group is where is mapped the main thread */
  if (thgrp->tid2gid[-1] == thgrp->localgid)
  {
    if (KAAPI_ATOMIC_INCR(&thgrp->endlocalthread) == (int)thgrp->localthreads)
    {
      KAAPI_ATOMIC_WRITE_BARRIER( &thgrp->endlocalthread, 0 );
      if (KAAPI_ATOMIC_INCR( &thgrp->endglobalgroup ) == (int)thgrp->nodecount)
      {
        KAAPI_ATOMIC_WRITE_BARRIER( &thgrp->endglobalgroup, 0 );
        kaapi_task_orstate( thgrp->waittask, KAAPI_MASK_BODY_TERM );
#if 0
        printf("Put waitting task to term\n");
#endif
      }
    }
  }
  else 
  {
#if defined(KAAPI_USE_NETWORK)
    /* remote signal */
    if (KAAPI_ATOMIC_INCR(&thgrp->endlocalthread) == (int)thgrp->localthreads)
    {
#if 0
      printf("%i::[kaapi_threadgroup_signalend_tid] signalend to master thread group: counter:%i should be:%i\n", thgrp->localgid, KAAPI_ATOMIC_READ(&thgrp->endlocalthread), (int)thgrp->localthreads);
#endif
      /* remote address space -> communication */
      kaapi_network_am(
          thgrp->tid2gid[-1],
          kaapi_threadgroup_signalend_service, 
          &thgrp->grpid, sizeof(thgrp->grpid)
      );
    }
#else
    /* where is the maste threadgroup ? */
    kaapi_assert_debug( 0 );
#endif
  }
}

/** kaapi_threadgroup_execframe
    Use the list of ready task to execute program.
    Once a task has been executed from the ready list, then call execframe
    of the possible created tasks.
    If a task activates new ready task, then beguns by executes these tasks
    prior to other tasks.
*/


/*
*/
int kaapi_threadgroup_execframe( kaapi_thread_context_t* thread )
{
  kaapi_task_t*              pc;      /* cache */
  kaapi_taskdescr_t*         td;      /* cache */
  kaapi_task_body_t          body;
  kaapi_frame_t*             fp;
  kaapi_tasklist_t*          tasklist;
  kaapi_comrecv_t*           recv;
#if defined(KAAPI_USE_PERFCOUNTER)
  uint32_t                   cnt_tasks = 0;
#endif

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);
  tasklist = thread->tasklist;
  
  fp = (kaapi_frame_t*)thread->sfp;

  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc      = fp->sp;
  thread->sfp[1].sp      = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
//  KAAPI_DEBUG_INST(save_fp = (kaapi_frame_t*)thread->sfp);
  
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);

  while (!kaapi_tasklist_isempty( tasklist ))
  {
//    kaapi_assert_debug( thread->sfp == save_fp);
    td = kaapi_tasklist_pop( tasklist );

    /* execute td->task */
    if (td !=0)
    {
      pc = td->task;
      if (pc !=0)
      {
#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
        body = kaapi_task_getbody(pc);
#endif /* KAAPI_USE_EXECTASK_METHOD */

        /* here... call the task*/
        body( pc->sp, (kaapi_thread_t*)thread->sfp );
    //      printf("e:%p\n", (void*)pc); fflush(stdout);
      }    
#if defined(KAAPI_USE_PERFCOUNTER)
      ++cnt_tasks;
#endif
    }
    
    /* recv ? */
    recv = kaapi_tasklist_popsignal( tasklist );
    if ( recv != 0 ) /* here may be a while loop */
    {
      /* is cw ? (associative + cumulative) */
      if (recv->reduce_fnc !=0)
      {
#if 0
        printf("%i::[do recv CW]: oldvalue: %i, incr: %i\n", 
            thread->the_thgrp->localgid,
            *(int*)recv->result,
            *(int*)recv->data
        );
#endif
        recv->reduce_fnc( recv->result, recv->data );
#if 0
        printf("%i::[do recv CW]: value: %i\n", 
            thread->the_thgrp->localgid,
            *(int*)recv->result
        );
#endif
      }
      kaapi_tasklist_merge_activationlist( tasklist, &recv->list );
      --tasklist->count_recv;
#if 0
      printf("%i::[kaapi_threadgroup_execframe] tid:%i, recv input data, #wc_recv:%i\n", 
          thread->the_thgrp->localgid, 
          thread->partid, 
          tasklist->count_recv
      );
#endif
    }
    
    /* bcast? management of the output communication after puting input comm */
    if (td !=0) 
    {
      if (td->bcast !=0) 
        kaapi_threadgroup_bcast( thread->the_thgrp, 
                                 kaapi_threadgroup_tid2asid(thread->the_thgrp, thread->partid), 
                                 &td->bcast->front );      

      /* post execution: new tasks created ??? */
      if (unlikely(fp->sp > thread->sfp->sp))
      {
        int err = kaapi_thread_execframe( thread );
        if ((err == EWOULDBLOCK) || (err == EINTR)) return err;
        kaapi_assert_debug( err == 0 );
      }
      
      if (!kaapi_activationlist_isempty(&td->list))
      { /* push in the front the activated tasks */
        kaapi_tasklist_merge_activationlist( tasklist, &td->list );
      }
    }
    
  } /* while */

  /* pop frame */
  --thread->sfp;

#if 0
  printf("%i::[kaapi_threadgroup_execframe] end tid:%i, #task:%p, #recvlist:%p, #wc_recv:%i\n", 
    thread->the_thgrp->localgid, 
    thread->partid, 
    (void*)tasklist->front,
    (void*)tasklist->recvlist,
    tasklist->count_recv);
#endif
  
  kaapi_threadgroup_t thgrp = thread->the_thgrp;
  kaapi_assert_debug (thgrp !=0);

  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
  if (tasklist->count_recv == 0) 
  {
    /* restore before signaling end of execution */
    if (((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0))
    {
      if (thgrp->maxstep != -1) 
      {
        /* avoir restore for the last step */
        if (thgrp->step + 1 <thgrp->maxstep)
          kaapi_threadgroup_restore_thread(thgrp, thread->partid);
      }
      else 
        kaapi_threadgroup_restore_thread(thgrp, thread->partid);
    }
    
    kaapi_threadgroup_signalend_tid( thread->partid, thgrp );

#if 0
    printf("%i::[Detach thread] tid:%i\n", thgrp->localgid, thread->partid);
#endif

    if (thread->partid != -1)
    { 
      /* detach the thread: may it should be put into the execframe function */
      kaapi_sched_lock(&thread->proc->lock);
      thread->proc->thread = 0;
      kaapi_sched_unlock(&thread->proc->lock);
    }
    /* reset the exec frame for the frame to be 0 (default) value, never reset */
//    thread->sfp->execframe = 0;
    return ECHILD;
  }
#if defined(KAAPI_USE_NETWORK)
  kaapi_network_poll();
#endif

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
}
