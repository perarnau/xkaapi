/*
** kaapi_stack_execframe.c
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

/** kaapi_stack_execframe
    Here the stack of task is organised like this, task1 is pointed by pc and
    it will be the first running task.

               | task0  | 
    thread->pc | task1  |< thread->sfp->pc
               | task2  |
               | task3  |
              -----------
               | task4  |< thread->sfp->sp
               | task5  |
               | task6  |
              -----------
               | task7  |
      stack->sp| ....   |
     
   Where task0 is only here to say that a possible task may exists before the frame.
   In case of whole stack execution, the caller should only call execframe with the
   first frame correctly set.
   Initialy thread->sfp store information about the frame the first frame [pc:sfp->sp). 
   
   On return, we leave the stack in the following state after execution of all tasks into the
   frame pointed by sfp, including all the child tasks.
   All task body are set to executed.
   
                     | task0  |
thread->pc=stack->sp | xxxxx  |< thread->sfp->pc = thread->sfp->sp
                     | xxxxx  |
                     | xxxxx  |
                    -----------
                     | xxxxx  | 
                     | xxxxx  |
                     | xxxxx  |
                    -----------
                     | xxxxx  |
                     | ....   |

  The method could returns EWOULDBLOCK in order to indicate that a task cannot be executed.
  In that case, thread->pc points to the tasks (body== kaapi_suspend_body) and thread->sfp
  points to the current frame where the suspend task has been tried to be executed.

  Where the task becomes ready, one may continue the execution simply by calling
  execframe with the state that has been set on return with EWOULDBLOCK.
*/

//#ifdef KAAPI_USE_CASSTEAL
//#undef KAAPI_USE_CASSTEAL
//#ndif
//#ifndef KAAPI_CONCURRENT_WS
//#undef KAAPI_CONCURRENT_WS 1
//#endif

//#define KAAPI_USE_CASSTEAL 1

/*
*/
int kaapi_stack_execframe( kaapi_thread_context_t* thread )
{
  kaapi_frame_t*             fp;
  kaapi_task_body_t          body;
  kaapi_frame_t*             eframe = thread->sfp;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint32_t             cnt_tasks = 0;
#endif 

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);
  
push_frame:
  fp = thread->sfp;
  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc = fp->sp;
  thread->sfp[1].sp = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);      
  
#if defined(KAAPI_USE_CASSTEAL)
begin_loop:
#endif
  /* stack growth down ! */
  while (fp->pc != fp->sp)
  {
    kaapi_assert_debug( fp->pc > fp->sp );

    body = fp->pc->body;
    kaapi_assert_debug( body != kaapi_exec_body);

#if defined(KAAPI_USE_CASSTEAL)
    if (!KAAPI_ATOMIC_CASPTR( (kaapi_atomic_t*)&fp->pc->body, body, kaapi_exec_body)) 
    { 
      kaapi_assert_debug((fp->pc->body == kaapi_suspend_body) || (fp->pc->body == kaapi_aftersteal_body) );
      body = fp->pc->body;
      if (body == kaapi_suspend_body)
        goto error_swap_body;
      /* else ok its aftersteal */
      body = kaapi_aftersteal_body;
    }
#else
    fp->pc->body = kaapi_exec_body;
#endif

    /* task execution */
    body( fp->pc->sp, (kaapi_thread_t*)thread->sfp );
#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif
    if (unlikely(thread->errcode)) goto backtrack_stack;

restart_after_steal:    
    if (unlikely(fp->sp != thread->sfp->sp))
    {
      goto push_frame;
    }
    --fp->pc;
  } /* end of the loop */

  kaapi_assert_debug( fp >= eframe);
  kaapi_assert_debug( fp->pc == fp->sp );

  if (fp != eframe)
  {
#if defined(KAAPI_USE_CASSTEAL)
    /* here it's a pop of frame: we lock the thread */
    while (!KAAPI_ATOMIC_CAS(&thread->lock, 0, 1));
#endif
    /* pop dummy frame */
    --fp;
    while (fp != eframe)
    {
      --fp->pc;
      if (fp->pc > fp->sp)
      {
#if defined(KAAPI_USE_CASSTEAL)
        KAAPI_ATOMIC_WRITE(&thread->lock, 0);
#endif
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
      /* else pop */  
      --fp;
    };
    --fp->pc;
#if defined(KAAPI_USE_CASSTEAL)
    kaapi_writemem_barrier();
    KAAPI_ATOMIC_WRITE(&thread->lock, 0);
#endif
  }
  thread->sfp = fp;
  
  /* end of the pop: we have finish to execute all the task */
  kaapi_assert_debug( fp->pc == fp->sp );
  kaapi_assert_debug( thread->sfp == eframe );

  /* note: the stack data pointer is the same as saved on enter */

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return 0;

#if defined(KAAPI_USE_CASSTEAL)
error_swap_body:
  if (fp->pc->body == kaapi_aftersteal_body) goto begin_loop;
  kaapi_assert_debug(pc->body == kaapi_suspend_body);
  return EWOULDBLOCK;
#endif

backtrack_stack:
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
#if !defined(KAAPI_CONCURRENT_WS)
  if ((thread->errcode & 0x1) !=0) 
  {
    kaapi_sched_advance(thread->proc);
    thread->errcode = thread->errcode & ~0x1;
    if (thread->errcode ==0) goto restart_after_steal;
  }
#endif
#if 0
  if ((thread->errcode >> 8) == EWOULDBLOCK) 
  {
    thread->errcode = thread->errcode & ~( 0xFF << 8);
    if (!KAAPI_ATOMIC_CASPTR( &thread->pc->body, kaapi_exec_body, kaapi_suspend_body ))
    { /* the only way the cas fails is during the thief that update to body to aftersteal body */
      kaapi_assert_debug( thread->pc->body == kaapi_aftersteal_body);
      goto begin_loop;
    }
    return EWOULDBLOCK;
  }
#endif

  /* here back track the kaapi_stack_execframe until go out */
  return thread->errcode;
}
























#if 0 /* SAVE BEST CODE */


/*
*/
int kaapi_stack_execframe( kaapi_thread_context_t* thread )
{
  kaapi_task_body_t          body;
  kaapi_task_t*              sp;
  kaapi_stack_t*             stack = kaapi_threadcontext2stack(thread);
  kaapi_frame_t*             eframe = thread->sfp;
  kaapi_task_t*              pc0;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint32_t             cnt_tasks = 0;
#endif 

  kaapi_assert_debug( thread->pc == thread->sfp->pc );
  pc0 = thread->pc;
  thread->sfp->sp_data = stack->sp_data;

  thread->pc = thread->sfp->pc;
  sp = thread->sfp->sp;

begin_loop:
  /* stack growth down ! */
  while (thread->pc != sp)
  {
    kaapi_assert_debug( thread->pc != sp );
    kaapi_assert_debug( (char*)thread->pc > (char*)stack->sp_data );

    body = thread->pc->ebody;
    kaapi_assert_debug( body != kaapi_exec_body);

#if defined(KAAPI_USE_CASSTEAL)
    if (!KAAPI_ATOMIC_CASPTR( (kaapi_atomic_t*)&thread->pc->body, body, kaapi_exec_body)) 
    { 
      kaapi_assert_debug((thread->pc->body == kaapi_suspend_body) || (thread->pc->body == kaapi_aftersteal_body) );
      body = thread->pc->body;
      if (body == kaapi_suspend_body)
        goto eror_swap_body;
      /* else ok its aftersteal */
    }
#else
    thread->pc->body = kaapi_exec_body;
#endif

    /* task execution */
    body( thread->pc, stack );
#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif
    if (unlikely(thread->errcode)) goto backtrack_stack;

restart_after_steal:    
    if (unlikely(sp != stack->sp))
    {
      /* here it's a push of a frame : no lock just memory barrier before update of the frame */
      thread->sfp[0].pc = thread->pc;
      thread->pc = sp;
      sp = stack->sp;
      thread->sfp[1].pc = thread->pc;
      thread->sfp[1].sp = sp;
      thread->sfp[1].sp_data = stack->sp_data;
      
      /* force previous write before next write */
      kaapi_writemem_barrier();

      ++thread->sfp;
      kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);      
      continue;
    }
    --thread->pc;
  } /* end of the loop */

  kaapi_assert_debug( thread->sfp >= eframe);

  if (thread->sfp != eframe)
  {
    /* here it's a pop of frame: we lock the thread */
    while (!KAAPI_ATOMIC_CAS(&thread->lock, 0, 1));
    while (thread->sfp != eframe)
    {
      --thread->sfp;
      --thread->sfp->pc;
      if (thread->sfp->pc > thread->sfp->sp)
      {
        thread->pc = thread->sfp->pc;
        sp = stack->sp = thread->sfp->sp;
        stack->sp_data = thread->sfp->sp_data;
        kaapi_writemem_barrier();
        KAAPI_ATOMIC_WRITE(&thread->lock, 0);
        goto begin_loop; /* remains work do do */
      }
      /* else pop */  
    }
    kaapi_writemem_barrier();
    KAAPI_ATOMIC_WRITE(&thread->lock, 0);
  }
  
  /* end of the pop: we have finish to execute all the task */
  kaapi_assert_debug( thread->pc == sp );
  kaapi_assert_debug( thread->sfp == eframe );
  thread->pc = stack->sp = thread->sfp->pc = thread->sfp->sp = pc0;
  stack->sp_data = thread->sfp->sp_data;

  /* note: the stack data pointer is the same as saved on enter */

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return 0;

eror_swap_body:
  if (thread->pc->body == kaapi_aftersteal_body) goto begin_loop;
  kaapi_assert_debug(thread->pc->body == kaapi_suspend_body);
  return EWOULDBLOCK;

backtrack_stack:
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
#if !defined(KAAPI_CONCURRENT_WS)
  if ((thread->errcode & 0x1) !=0) 
  {
    kaapi_sched_advance(thread->proc);
    thread->errcode = thread->errcode & ~0x1;
    if (thread->errcode ==0) goto restart_after_steal;
  }
#endif
#if 0
  if ((thread->errcode >> 8) == EWOULDBLOCK) 
  {
    thread->errcode = thread->errcode & ~( 0xFF << 8);
    if (!KAAPI_ATOMIC_CASPTR( &thread->pc->body, kaapi_exec_body, kaapi_suspend_body ))
    { /* the only way the cas fails is during the thief that update to body to aftersteal body */
      kaapi_assert_debug( thread->pc->body == kaapi_aftersteal_body);
      goto begin_loop;
    }
    return EWOULDBLOCK;
  }
#endif

  /* here back track the kaapi_stack_execframe until go out */
  return thread->errcode;
}

#endif
