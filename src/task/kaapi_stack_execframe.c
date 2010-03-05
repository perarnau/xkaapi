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
#if 0/* */
int kaapi_stack_execframe( kaapi_stack_t* stack )
{
  kaapi_frame_t     frame;

  kaapi_stack_save_frame(stack, &frame);
  stack->frame_sp = frame.sp;

  /* stack growth down ! */
  for (; frame.pc != frame.sp; --frame.pc)
  {
    kaapi_task_body_t body = frame.pc->body;
#if defined(KAAPI_CONCURRENT_WS)
    OSAtomicCompareAndSwap32( (int32_t)body, (int32_t)kaapi_exec_body, (volatile int32_t*)&frame.pc->body);
#else
    frame.pc->body = kaapi_exec_body;
#endif
    body( frame.pc, stack );
    if (__builtin_expect(stack->errcode,0)) goto backtrack_stack;
    
    if (frame.sp != stack->sp)
    {
      stack->frame_sp = frame.sp;
      kaapi_stack_execframe(stack);
      if (__builtin_expect(stack->errcode,0)) goto backtrack_stack_return;

      stack->sp = frame.sp;
      stack->sp_data = frame.sp_data;
    }
  }
  stack->frame_sp = frame.sp;
  return 0;

backtrack_stack:
  /* here back track the kaapi_stack_execframe until go out */
  return stack->errcode;

backtrack_stack_return:
  return 1+stack->errcode;
}
#else
/* iterative version */
#define KAAPI_MAX_RECCALL 1024
int kaapi_stack_execframe( kaapi_stack_t* stack )
{
  register kaapi_task_t*     pc;
  kaapi_frame_t     frame[KAAPI_MAX_RECCALL];
  kaapi_frame_t*    pframe = frame;
  kaapi_task_body_t body;

enter_loop:
  kaapi_stack_save_frame(stack, pframe);
  stack->frame_sp = pframe->sp;

begin_loop:
  /* stack growth down ! */
  for (; pframe->pc != pframe->sp; --pframe->pc)
  {
    pc = pframe->pc;
    body = pc->body;
    kaapi_assert_debug( body != kaapi_exec_body);
    kaapi_assert_debug( body == pc->ebody);
#if defined(KAAPI_CONCURRENT_WS)
    OSAtomicCompareAndSwap32( (int32_t)body, (int32_t)kaapi_exec_body, (volatile int32_t*)&pc->body);
#else
    pc->body = kaapi_exec_body;
#endif
    body( pc, stack );
    if (__builtin_expect(stack->errcode,0)) goto backtrack_stack;
    
    if (__builtin_expect(pframe->sp != stack->sp,0))
    {
      stack->frame_sp = pframe->sp;
      ++pframe;
      goto enter_loop;
    }
  }
  if (pframe != frame)
  {
    --pframe;
    --pframe->pc;
    stack->sp = pframe->sp;
    stack->sp_data = pframe->sp_data;
    stack->frame_sp = pframe->sp;
    goto begin_loop;
  }
  stack->frame_sp = pframe->sp;
  return 0;

backtrack_stack:
  /* here back track the kaapi_stack_execframe until go out */
  return stack->errcode;
}

#endif
