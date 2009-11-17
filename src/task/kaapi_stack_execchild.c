/*
** kaapi_task_exec.c
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


/**kaapi_stack_execchild
*/
int kaapi_stack_execchild(kaapi_stack_t* stack, kaapi_task_t* task)
{
  kaapi_frame_t frame;
  kaapi_frame_t* pframe;
  kaapi_task_t* stop_sp;
  kaapi_task_t* retn;
  if (stack ==0) return EINVAL;
  if (task ==0) return 0;

  if (task->body ==0) 
    return 0;
  if (task->body == &kaapi_suspend_body) 
      return EWOULDBLOCK;
  kaapi_stack_save_frame(stack, &frame);
  (*task->body)(task, stack);
  task->body = 0;
    
  /* if no pushed tasks return */
  if (frame.sp == stack->sp)
  {
    kaapi_stack_restore_frame(stack, &frame);
    return 0;
  }
  
  /* stop execution until sp reach stop_sp, the saved stack pointer */
  stop_sp = frame.sp;
  goto push_retn;

redo_work:
  if (stack->pc == stop_sp) return 0;
  if (task->body ==0) goto next_task;
  if (task->body == &kaapi_suspend_body) 
    return EWOULDBLOCK;

  frame.pc      = task;
  frame.sp      = stack->sp;
  frame.sp_data = stack->sp_data;
  (*task->body)(task, stack);
  task->body = 0;

push_retn:    
  /* push restore_frame task if pushed tasks */
  if (frame.sp < stack->sp)
  {
    retn = kaapi_stack_toptask(stack);
    kaapi_task_init(stack, retn, KAAPI_TASK_STICKY);
    retn->body  = &kaapi_retn_body;
    pframe = (kaapi_frame_t*)kaapi_stack_pushdata(stack, sizeof(kaapi_frame_t));
    retn->sp = (void*)pframe;
    *pframe = frame;
    kaapi_stack_pushtask(stack);

    /* update pc to the first forked task */
    task = stack->pc = frame.sp;
    goto redo_work;
  }
     
  task = ++stack->pc;
  goto redo_work;

next_task:
  task = ++stack->pc;
  if (stack->pc >= stack->sp) 
  {
    /* empty stack: reset pointer to begin of the stack */
    stack->pc = stack->sp = stack->task;
    return 0;
  }
  goto redo_work;
}
