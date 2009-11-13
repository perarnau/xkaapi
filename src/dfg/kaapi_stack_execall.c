/*
** kaapi_stack_exec.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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

/**kaapi_stack_taskexecall
*/
int kaapi_stack_execall(kaapi_stack_t* stack) 
{
  register kaapi_task_t* task;
  kaapi_task_t*          saved_sp;
  kaapi_frame_t*         frame;
  char*                  saved_sp_data;
  kaapi_task_t*          retn;

  if (stack ==0) return EINVAL;
  if (kaapi_stack_isempty( stack ) ) return 0;
  task = stack->pc;

redo_work: 
  {
    if (task->body[0] ==0) goto next_task;
#if 0 /* optimization */
    if (task->body[0] == &kaapi_suspend_body) 
    {
      return EWOULDBLOCK;
    }
    else if (task->body[0] == &kaapi_retn_body) 
      /* do not save stack frame before execution */
      kaapi_retn_body(task, stack);
    else
#endif
    {
      saved_sp      = stack->sp;
      saved_sp_data = stack->sp_data;
      (*task->body[0])(); //task->sp_data, stack;
      task->body[0] = 0;
    
      /* push restore_frame task if pushed tasks */
      if (saved_sp < stack->sp)
      {
        retn = kaapi_stack_top(stack);
/*        retn->state = KAAPI_TASK_INIT;*/
        retn->body[0]  = &kaapi_retn_body;
        void** arg_retn = kaapi_stack_pushdata(stack, 3*sizeof(void*));
//        retn->sp_data = (void*)arg_retn;
        arg_retn[0] = task; /* <=> save pc */
        arg_retn[1] = saved_sp;
        arg_retn[2] = saved_sp_data;
        kaapi_stack_push(stack);

        /* update pc to the first forked task */
        task = stack->pc = saved_sp;
        goto redo_work;
      }
       
      ++stack->pc;
      task = stack->pc;
      goto redo_work;
    }
  }
next_task:
  task = ++stack->pc;
  if (stack->pc >= stack->sp) return 0;
  goto redo_work;
}
