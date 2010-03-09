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
    Here the stack frame is organised like this, task1 is the running task.

               | task1  |
              ----------
    frame_sp ->| task2  |
               | task3  |
               | task4  |
               | task5  |
          sp ->| ....   |
     
  
   We first push a retn task and execute all the task in the frame in [pc+1, ...,  sp). 
   The task retn serves to mark the stack structure in case of stealing (...) ? Sure ?
   
   On return, we leave the stack in this state after execution of all tasks into the
   frame including the child tasks.

               | task1  |
               ----------
    frame_sp ->| x x x  |
               | x x x  |
               | x x x  |
               | x x x  |
          sp ->| ....   |
*/

/* iterative version */
int kaapi_stack_execframe( kaapi_stack_t* stack )
{
  register kaapi_task_t*     pc;
  kaapi_frame_t*             eframe = stack->epfsp;
  kaapi_task_body_t          body;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint32_t             cnt_tasks = 0;
#endif  

  if (stack->pfsp ==0)
  {
    eframe = stack->pfsp = stack->stackframe;
    kaapi_stack_save_frame(stack, stack->pfsp);
  }

enter_loop:
  stack->frame_sp = stack->pfsp->sp;

begin_loop:
  /* stack growth down ! */
  for (; stack->pfsp->pc != stack->pfsp->sp; --stack->pfsp->pc)
  {
    pc = stack->pfsp->pc;
    body = pc->body;
    kaapi_assert_debug( body != kaapi_exec_body);
    kaapi_assert_debug( (body == pc->ebody) || (body == kaapi_suspend_body) || (body == kaapi_aftersteal_body) );

#if defined(KAAPI_CONCURRENT_WS)
    KAAPI_ATOMIC_CASPTR( &pc->body, body, kaapi_exec_body );
#else
    pc->body = kaapi_exec_body;
#endif
    body( pc, stack );
#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif
    if (__builtin_expect(stack->errcode,0)) goto backtrack_stack;

restart_after_steal:    
    if (__builtin_expect(stack->pfsp->sp != stack->sp,0))
    {
      stack->frame_sp = stack->pfsp->sp;
      /* here it's a push of frame */
      ++stack->pfsp;
      kaapi_stack_save_frame(stack, stack->pfsp);
      kaapi_assert_debug( stack->pfsp - eframe <KAAPI_MAX_RECCALL);
      goto enter_loop;
    }
  }
  if (stack->pfsp != eframe)
  {
    /* here it's a pop of frame */
    --stack->pfsp;
    kaapi_assert_debug( stack->pfsp >= eframe);
    --stack->pfsp->pc;
    stack->sp = stack->pfsp->sp;
    stack->sp_data = stack->pfsp->sp_data;
    stack->frame_sp = stack->pfsp->sp;
    goto begin_loop;
  }
  stack->frame_sp = stack->pfsp->sp;

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(stack->_proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return 0;

backtrack_stack:
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(stack->_proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
#if !defined(KAAPI_CONCURRENT_WS)
  if ((stack->errcode & 0x1) !=0) 
  {
    kaapi_sched_advance(stack->_proc);
    stack->errcode = stack->errcode & ~0x1;
    if (stack->errcode ==0) goto restart_after_steal;
  }
#endif
  if ((stack->errcode >> 8) == EWOULDBLOCK) 
  {
    stack->errcode = stack->errcode & ~( 0xFF << 8);
    if (!KAAPI_ATOMIC_CASPTR( &pc->body, kaapi_exec_body, kaapi_suspend_body ))
    { /* the only way the cas fails is during the thief that update to body to aftersteal body */
      kaapi_assert_debug( pc->body == kaapi_aftersteal_body);
      goto begin_loop;
    }
    return EWOULDBLOCK;
  }

  /* here back track the kaapi_stack_execframe until go out */
  return stack->errcode;
}
