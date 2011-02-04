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
#include "kaapi_cuda_execframe.h"

#if defined(KAAPI_USE_NETWORK)
/* network service to signal end of iteration of one tid */
void kaapi_threadgroup_signalend_service(int err, kaapi_globalid_t source, void* buffer, size_t sz_buffer )
{
  int32_t grpid;
  kaapi_threadgroup_t thgrp;
  memcpy(&grpid, buffer, sizeof(int32_t));
  kaapi_assert_debug( (grpid>=0) && (grpid <KAAPI_MAX_THREADGROUP) );

  thgrp = kaapi_all_threadgroups[grpid];
#if 0
  printf("%i::[kaapi_threadgroup_signalend_service] begin receive signal tag\n",
      thgrp->localgid
  ); 
  fflush(stdout);
#endif

  if (thgrp->localgid == thgrp->tid2gid[-1])
  {
    if (KAAPI_ATOMIC_INCR( &thgrp->endglobalgroup ) == (int)thgrp->nodecount)
    {
#if 0
      printf("%i::[kaapi_threadgroup_signalend_service] master , #nodes=%i\n", thgrp->localgid, thgrp->nodecount);
      fflush(stdout);
#endif
      KAAPI_ATOMIC_WRITE_BARRIER( &thgrp->endglobalgroup, 0 );
      kaapi_task_orstate( thgrp->waittask, KAAPI_MASK_BODY_TERM );      
    }
#if 0
    else {
      printf("%i::[kaapi_threadgroup_signalend_service] master not terminated:counter %i\n", 
          thgrp->localgid, KAAPI_ATOMIC_READ( &thgrp->endglobalgroup ));
      fflush(stdout);
    }
#endif
  }
  else {
    kaapi_task_orstate( thgrp->waittask, KAAPI_MASK_BODY_TERM );
#if 0
    printf("%i::[kaapi_threadgroup_signalend_service] slave terminate\n", thgrp->localgid);
#endif
  }
}
#endif

static void kaapi_threadgroup_signalend_tid( unsigned int dummy , kaapi_threadgroup_t thgrp )
{
#if 0
  printf("%i::[kaapi_threadgroup_signalend_tid] signalend to master thread group at:%i\n", 
      thgrp->localgid, thgrp->tid2gid[-1] );
  fflush(stdout);
#endif
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
      fflush(stdout);
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
int kaapi_cuda_threadgroup_execframe( kaapi_thread_context_t* thread )
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
  tasklist = thread->sfp->tasklist;
  
  fp = (kaapi_frame_t*)thread->sfp;

  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc = fp->sp;
  thread->sfp[1].sp = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  KAAPI_DEBUG_INST(save_fp = (kaapi_frame_t*)thread->sfp);
  
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);

  while (!kaapi_tasklist_isempty( tasklist ))
  {
    /* kaapi_assert_debug( thread->sfp == save_fp); */
    td = kaapi_tasklist_pop( tasklist );

    /* execute td->task */
    if (td !=0)
    {
      /* is cw ? (associative + cumulative) */
      if (td->reduce_fnc !=0)
      {
        td->reduce_fnc( td->context, td->value );
      }

      pc = td->task;
      if (pc !=0)
      {
	kaapi_format_t* format;

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
        body = kaapi_task_getbody(pc);
#endif /* KAAPI_USE_EXECTASK_METHOD */

	format = kaapi_format_resolvebybody(body);
	kaapi_assert_debug(format != NULL);
	kaapi_cuda_exectask(thread, pc->sp, format);
      }
#if defined(KAAPI_USE_PERFCOUNTER)
      ++cnt_tasks;
#endif
    }
    
    /* recv ? */
    recv = kaapi_tasklist_popsignal( tasklist );
    if ( recv != 0 ) /* here may be a while loop */
    {
      kaapi_tasklist_merge_activationlist( tasklist, &recv->list );
      --tasklist->count_recv;
    }
    
    /* bcast? management of the communication */
    if (td !=0) 
    {
      /* post execution: new tasks created */
      if (unlikely(fp->sp > thread->sfp->sp))
      {
        int err = kaapi_thread_execframe( thread );
        if ((err == EWOULDBLOCK) || (err == EINTR)) return err;
        kaapi_assert_debug( err == 0 );
      }

      /* do bcast after child execution (they can produce output data) */
      if (td->bcast !=0) 
        kaapi_threadgroup_bcast( thread->the_thgrp, 
                                 kaapi_threadgroup_tid2asid(thread->the_thgrp, thread->partid), 
                                 &td->bcast->front );      
      
      if (!kaapi_activationlist_isempty(&td->list))
      { /* push in the front the activated tasks */
        kaapi_tasklist_merge_activationlist( tasklist, &td->list );
      }
    }
    
  } /* while */

  /* pop frame */
  --thread->sfp;

  kaapi_threadgroup_t thgrp = thread->the_thgrp;
  kaapi_assert_debug (thgrp !=0);
  
  /* signal end of step if no more recv (and then no ready task activated) */
  if (tasklist->count_recv == 0) 
  {
    /* restore before signaling end of execution */
    if ((thread->partid != -1) && ((thgrp->flag & KAAPI_THGRP_SAVE_FLAG) !=0))
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

    if (thread->partid != -1)
    { 
      /* detach the thread: may it should be put into the execframe function */
      kaapi_sched_lock(&thread->proc->lock);
      thread->proc->thread = 0;
      kaapi_sched_unlock(&thread->proc->lock);

#if 0
      printf("%i::[Detach thread] tid:%i\n", thgrp->localgid, thread->partid);
#endif

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
