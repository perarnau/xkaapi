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
#if ((KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD) || (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD))
int kaapi_thread_execframe( kaapi_thread_context_t* thread )
{
  kaapi_task_t*              pc; /* cache */
  kaapi_frame_t*             fp;
  kaapi_task_body_t          body;
  uintptr_t	                 state;
  kaapi_frame_t*             eframe = thread->esfp;
#if defined(KAAPI_USE_PERFCOUNTER)
  uint32_t                   cnt_tasks = 0;
#endif

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);
  
push_frame:
  fp = (kaapi_frame_t*)thread->sfp;
  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc = fp->sp;
  thread->sfp[1].sp = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);

  pc = fp->pc;
  
  /* stack of task growth down ! */
  while (pc != fp->sp)
  {
    kaapi_assert_debug( pc > fp->sp );

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD)
    body = pc->body;

#if (SIZEOF_VOIDP == 4)
    state = pc->state;
#else
    state = kaapi_task_body2state(body);
#endif

    kaapi_assert_debug( body != kaapi_exec_body);
    pc->body = kaapi_exec_body;
    /* task execution */
    kaapi_assert_debug(pc == thread->sfp[-1].pc);
    kaapi_assert_debug( kaapi_isvalid_body( body ) );

    /* here... */
    body( pc->sp, (kaapi_thread_t*)thread->sfp );      

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)

    state = kaapi_task_orstate( pc, KAAPI_MASK_BODY_EXEC );

#if (SIZEOF_VOIDP == 4)
    body = pc->body;
#else
    body = kaapi_task_state2body( state );
#endif /* SIZEOF_VOIDP */

#endif /* KAAPI_USE_EXECTASK_METHOD */

    if (likely( kaapi_task_state_isnormal(state) ))
    {
      /* task execution */
      kaapi_assert_debug(pc == thread->sfp[-1].pc);

      /* here... */
      body( pc->sp, (kaapi_thread_t*)thread->sfp );
//      printf("e:%p\n", (void*)pc); fflush(stdout);
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
        kaapi_aftersteal_body( pc->sp, (kaapi_thread_t*)thread->sfp );      
      }
      else if ( kaapi_task_state_isterm( state ) )
      {
        /* means that task has been steal */
        kaapi_assert_debug( kaapi_task_state_issteal( state ) );
      }
      else if ( kaapi_task_state_issteal( state ) ) /* but not terminate ! so swap */
      {
//        printf("Suspend thread: %p on pc:%p\n", thread, pc );
//        fflush(stdout);
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
    if (unlikely(fp->sp > thread->sfp->sp))
    {
      goto push_frame;
    }
#if defined(KAAPI_DEBUG)
    else if (unlikely(fp->sp < thread->sfp->sp))
    {
      kaapi_assert_debug_m( 0, "Should not appear: a task was popping stack ????" );
    }
#endif

    /* next task to execute, store pc in memory */
    fp->pc = --pc;
    
    kaapi_writemem_barrier();
  } /* end of the loop */

  kaapi_assert_debug( fp >= eframe);
  kaapi_assert_debug( fp->pc == fp->sp );

  if (fp >= eframe)
  {
#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD)
    while (fp > eframe) 
    {
      --fp;
      /* pop dummy frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
    /* here it's a pop of frame: we lock the thread */
    kaapi_sched_lock(&thread->proc->lock);
    while (fp > eframe) 
    {
      --fp;

      /* pop dummy frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        kaapi_sched_unlock(&thread->proc->lock);
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

    kaapi_sched_unlock(&thread->proc->lock);
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


#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD) 
error_swap_body:
  kaapi_assert_debug(thread->sfp- fp == 1);
  /* implicityly pop the dummy frame */
  thread->sfp = fp;
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
#endif

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif

  /* here back track the kaapi_thread_execframe until go out */
  return 0;
}

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
int kaapi_thread_execframe( kaapi_thread_context_t* thread )
{
  return 0;
}
#endif
