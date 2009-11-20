/*
** kaapi_stack_exec.c
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
#include <unistd.h>

/*#define KAAPI_TRACE_DEBUG*/

/**kaapi_stack_taskexecall
*/
int kaapi_stack_execall(kaapi_stack_t* stack) 
{
  register kaapi_task_t* task;
  kaapi_task_t*          saved_sp;
  char*                  saved_sp_data;
  kaapi_task_t*          retn;
  void** arg_retn;
  kaapi_task_body_t      body;

#if defined(KAAPI_TRACE_DEBUG)  
  int level =0;
#endif  

  if (stack ==0) return EINVAL;
  if (kaapi_stack_isempty( stack ) ) return 0;
  task = stack->pc;

redo_work: 
  {
    if (task->body ==0) return 0;
    if (task->body == &kaapi_suspend_body)
    {
      if (kaapi_task_issync(task))
      {
        KAAPI_LOG(50,"Would block task: 0x%x\n", task );
        return EWOULDBLOCK;
      }

      /* ignore the task */
      ++stack->pc;
      task = stack->pc;
      goto redo_work;
    }
    else if (task->body == &kaapi_retn_body) 
    {
      /* do not save stack frame before execution */
      kaapi_retn_body(task, stack);
      KAAPI_LOG(100, "stackexec: exec retn 0x%x, pc: 0x%x\n",task, stack->pc );
      ++stack->pc;
      task = stack->pc;
#if defined(KAAPI_TRACE_DEBUG)  
      --level;
#endif  
      goto redo_work;
    }
    else
    {
      saved_sp      = stack->sp;
      saved_sp_data = stack->sp_data;
#if defined(KAAPI_TRACE_DEBUG)  
      { int k; for (k=0; k<level; ++k) printf("--------"); }
      printf("level:%i  ", level);
#endif  
      body = task->body;
//      task->format = body;
      KAAPI_LOG(100, "stackexec: task 0x%x, pc: 0x%x\n", task, stack->pc );
      (*body)(task, stack);
      task->body = 0;

      /* process steal request */
      kaapi_stealpoint_isactive( stack, task );
        
      /* push restore_frame task if pushed tasks */
      if (saved_sp < stack->sp)
      {
        retn = kaapi_stack_toptask(stack);
        kaapi_task_init(stack, retn, KAAPI_TASK_STICKY);
        retn->body  = &kaapi_retn_body;
        arg_retn = kaapi_stack_pushdata(stack, 3*sizeof(void*));
        retn->sp = (void*)arg_retn;
        arg_retn[0] = task; /* <=> save pc */
        arg_retn[1] = saved_sp;
        arg_retn[2] = saved_sp_data;
        kaapi_stack_pushtask(stack);

  KAAPI_LOG(100, "stackexec: push retn: 0x%x, pc: 0x%x\n", 
      retn, 
      stack->pc );

        /* update pc to the first forked task */
        task = stack->pc = saved_sp;
#if defined(KAAPI_TRACE_DEBUG)  
        ++level;
#endif  
        goto redo_work;
      }

      ++stack->pc;
      task = stack->pc;
      goto redo_work;
    }
  }
/*next_task: */
  task = ++stack->pc;
  if (stack->pc >= stack->sp) return 0;
  goto redo_work;
}
