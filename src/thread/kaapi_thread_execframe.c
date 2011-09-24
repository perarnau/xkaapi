/*
** kaapi_thread_execframe.c
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


/** kaapi_thread_execframe
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
   All task bodies are set to executed.
   
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
  In that case, thread->pc points to the tasks (body is marked as suspended) and thread->sfp
  points to the current frame where the suspend task has been tried to be executed.

  Where the task becomes ready, one may continue the execution simply by calling
  execframe with the state that has been set on return with EWOULDBLOCK.
*/

/*
*/
int kaapi_thread_execframe( kaapi_thread_context_t* thread )
{
  kaapi_task_t*              sp; /* cache */
  kaapi_task_t*              pc; /* cache */
  kaapi_frame_t*             fp; /* cache for thread->sfp */

  uintptr_t	                 state;
  kaapi_task_body_t          body;
  kaapi_frame_t*             eframe = thread->esfp;
#if defined(KAAPI_USE_PERFCOUNTER)
  uint32_t                   cnt_tasks = 0;
#endif

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);

  fp = (kaapi_frame_t*)thread->sfp;

push_frame:

  sp = fp->sp;
  pc = fp->pc;

  /* init new frame for the next task to execute */
  thread->sfp[1].pc        = sp;
  thread->sfp[1].sp        = sp;
  thread->sfp[1].sp_data   = fp->sp_data;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* push and update the current frame */
  thread->sfp = ++fp;
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);
  
  /* stack of task growth down ! */
  while (pc != sp)
  {
    kaapi_assert_debug( pc > sp );

    state = kaapi_task_orstate( pc, KAAPI_MASK_BODY_EXEC );
#if (__SIZEOF_POINTER__ == 4)
    body = pc->u.body;
#else
    body = kaapi_task_state2body( state );
#endif /* __SIZEOF_POINTER__ */


    if (likely( kaapi_task_state_isnormal(state) ))
    {
      /* task execution */
      kaapi_assert_debug(pc == thread->sfp[-1].pc);

      /* here... */
      body( pc->sp, (kaapi_thread_t*)fp );
    }
    else
    { 
      /* It is a special task: it means that before atomic or update, the body
         has already one of the special flag set (either exec, either suspend).
         Test the following case with THIS (!) order :
         - kaapi_task_body_isaftersteal(body) -> call aftersteal body
         - kaapi_task_body_issteal(body) -> error
         - else it means that the task has been executed by a thief, but it 
         does not require aftersteal body to merge results.
      */
      if ( kaapi_task_state_isaftersteal( state ) )
      {
        /* means that task has been steal & not yet terminated due
           to some merge to do
        */
        kaapi_assert_debug( kaapi_task_state_issteal( state ) );
        kaapi_aftersteal_body( pc->sp, (kaapi_thread_t*)fp );      
      }
      else if ( kaapi_task_state_isterm( state ) )
      {
        /* means that task has been steal or set to term if no debug */
        kaapi_assert_debug( kaapi_task_state_issteal( state ) );
      }
      else if ( kaapi_task_state_issteal( state ) ) /* but not terminate ! so swap */
      {
        goto error_swap_body;
      }
      else {
        kaapi_assert_debug(0);
      }
    }
#if defined(KAAPI_DEBUG)
    const uintptr_t debug_state = kaapi_task_orstate(pc, KAAPI_MASK_BODY_TERM );
    kaapi_assert_debug( !kaapi_task_state_isterm(debug_state) || (kaapi_task_state_isterm(debug_state) && kaapi_task_state_issteal(debug_state))  );
    kaapi_assert_debug( kaapi_task_state_isexec(debug_state) );
#endif    

#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif

    /* post execution: new tasks created ??? */
    if (unlikely(sp > fp->sp))
    {
      goto push_frame;
    }
#if defined(KAAPI_DEBUG)
    else if (unlikely(sp < fp->sp))
    {
      kaapi_assert_debug_m( 0, "Should not appear: a task was popping stack ????" );
    }
#endif

    /* next task to execute, store pc in memory */
    --pc;    
  } /* end of the loop */
  kaapi_assert_debug( pc == sp );

  --fp;
  fp->pc = pc;

  kaapi_assert_debug( fp >= eframe);

//  kaapi_sched_lock(&thread->proc->lock);
//  kaapi_sched_lock(&thread->lock);
  if (fp >= eframe)
  {
    /* here it's a pop of frame: we lock the thread */
    while (fp > eframe) 
    {
      --fp;

      /* finish to execute child tasks, pop current task of the frame */
      if (--fp->pc > fp->sp)
      {
        thread->sfp = fp;
        kaapi_writemem_barrier();        
        if (thread->thieffp > fp)
        {
          kaapi_mem_barrier();
          while (thread->thieffp > fp)
            kaapi_slowdown_cpu();
        }
        goto push_frame; /* remains work do do */
      }
    } 
    fp->sp = fp->pc;
  }
  thread->sfp = fp;
  kaapi_writemem_barrier();        
  if (thread->thieffp > fp)
  {
    kaapi_mem_barrier();
    while (thread->thieffp > fp)
      kaapi_slowdown_cpu();
  }
  
  /* end of the pop: we have finish to execute all the tasks */
  kaapi_assert_debug( fp->pc == fp->sp );
  kaapi_assert_debug( thread->sfp == eframe );

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return 0;


error_swap_body:
  /* write back to memory some data */
  fp[-1].pc = pc;    
  kaapi_assert_debug(thread->sfp- fp == 1);
  /* implicityly pop the dummy frame */
  thread->sfp = fp-1;
  
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
}
