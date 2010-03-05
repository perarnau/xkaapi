/*
** kaapi_sched_sync.c
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


#if 0
static int kaapi_stack_execframe( kaapi_stack_t* stack )
{
  kaapi_frame_t     frame;
  kaapi_task_body_t body;

  kaapi_stack_save_frame(stack, &frame);
  stack->frame_sp = frame.sp;

  /* stack growth down ! */
  for (; frame.pc != frame.sp; --frame.pc)
  {
    body = frame.pc->body;
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
//      stack->pc = frame.pc;
      stack->sp = frame.sp;
      stack->sp_data = frame.sp_data;
      stack->frame_sp = frame.sp;
//      kaapi_stack_restore_frame(stack, &frame);
//      stack->frame_sp = frame.sp;
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
#endif


/** kaapi_sched_sync
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
   
   On return, we leave the stack such that (the pushed tasks between frame_sp and sp)
   are poped :

               | task1  |
               ----------
frame_sp, sp ->| x x x  |
               | x x x  |
               | x x x  |
               | x x x  |
               | ....   |
*/
int kaapi_sched_sync(kaapi_stack_t* stack)
{
  int           err;
  kaapi_frame_t frame;
  kaapi_task_t* frame_sp;

  if (kaapi_stack_isempty( stack ) ) return 0;

  /* save here, do not restore pushed retn */
  kaapi_stack_save_frame(stack, &frame);

  frame_sp = stack->frame_sp; /* should correspond to the pc counter */
  
  /* increment new frame ~ push */
  ++stack->pfsp;
  
redo:
  err = kaapi_stack_execframe(stack);
  if (err == EWOULDBLOCK)
  {
    kaapi_sched_suspend( kaapi_get_current_processor() );
    goto redo;
  }
  if (err) /* but do not restore stack */
    return err;
  kaapi_stack_restore_frame(stack, &frame);
  stack->sp = stack->frame_sp = frame_sp;

  /* decrement frame pointer ~ pop */
  --stack->pfsp;

  /* mark the next task of current running task as nop to avod rexec */
  kaapi_task_setbody(frame_sp, kaapi_nop_body);

  return err;
}
