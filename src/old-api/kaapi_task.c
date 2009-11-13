/*
** kaapi_task.c
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

/** Nop task 
*/
void kaapi_nop_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
}

/** Restore stack 
*/
void kaapi_retn_body( kaapi_task_t* task, kaapi_stack_t* stack)
{
  kaapi_frame_t* frame = (kaapi_frame_t*)task->param.pdata;
  kaapi_stack_restore_frame( stack, frame );
}

/** kaapi_task_condeval
*/
int kaapi_task_condeval(kaapi_stack_t* stack, kaapi_task_t* task)
{
  if (task ==0) return EINVAL;
  if (task->sf.sf.xstate != KAAPI_TASK_WAITING) return EINVAL;
  /* here task should not change its state.... body restaure value ?*/
  if (task->body !=0) (*task->body)( task, stack );
  if (task->sf.sf.xstate == KAAPI_TASK_WAITING) return EAGAIN;
  return 0;
}

/** kaapi_stack_alloc
*/
int kaapi_stack_alloc( kaapi_stack_t* stack, size_t count_task, size_t size_data )
{
  if ((stack == 0) || (count_task ==0)) return EINVAL;
  stack->task = (kaapi_task_t*)malloc(sizeof(kaapi_task_t)*count_task);
  if (stack->task ==0) return ENOMEM;
  stack->data = (char*)malloc(size_data);
  if (stack->data ==0) return ENOMEM;

  stack->pc = stack->sp = stack->task;
  stack->sp_data = stack->data;
#if defined(KAAPI_DEBUG)
  stack->end_sp      = stack->task + count_task;
  stack->end_sp_data = stack->data + size_data;
#endif
  return 0;
}

/** kaapi_stack_free
*/
int kaapi_stack_free( kaapi_stack_t* stack )
{
  if (stack ==0) return EINVAL;
  if (stack->task !=0) free( stack->task );
  stack->pc = stack->sp = stack->task = 0;
#if defined(KAAPI_DEBUG)
  stack->end_sp = 0;
#endif
  if (stack->data !=0) free( stack->data );
  stack->sp_data = stack->data = 0;
  return 0;
}

/** kaapi_stack_init
*/
int kaapi_stack_init( kaapi_stack_t* stack, 
                      size_t size_task_buffer, void* task_buffer,
                      size_t size_data_buffer, void* data_buffer 
)
{
  if (stack == 0) return EINVAL;
  if (size_task_buffer ==0) 
  { 
    stack->pc = stack->sp = stack->task = 0; 
#if defined(KAAPI_DEBUG)
    stack->end_sp = 0;
#endif
  }
  else {
    if (size_task_buffer / sizeof(kaapi_task_t) ==0) return EINVAL;    
    stack->task = (kaapi_task_t*)task_buffer;
    stack->pc = stack->sp = stack->task;
#if defined(KAAPI_DEBUG)
    stack->end_sp      = stack->task + size_task_buffer/sizeof(kaapi_task_t);
#endif
  }
  if (size_data_buffer ==0) 
  {
    stack->sp_data = stack->data = 0;
  }
  else 
  {
    stack->sp_data = stack->data = data_buffer;
#if defined(KAAPI_DEBUG)
    stack->end_sp_data = stack->data + size_data_buffer;
#endif
  }
  return 0;
}

/** kaapi_stack_clear
*/
int kaapi_stack_clear( kaapi_stack_t* stack )
{
  if (stack == 0) return EINVAL;
  stack->pc = 0;
  stack->sp = stack->task;
  stack->sp_data = stack->data;
  return 0;
}

#if defined(KAAPI_DEBUG)
/** kaapi_stack_taskexec
*/
int kaapi_stack_taskexec(kaapi_stack_t* stack)
{
  if (stack ==0) return EINVAL;
  if (stack->pc >= stack->sp) return ENOEXEC; /* debug only */
  kaapi_task_t* task     = stack->pc;
  if (task ==0) return ENOEXEC;
  if (task->body ==0) return ENOEXEC;
  if (task->xstate != KAAPI_TASK_INIT) return EWOULDBLOCK;

  kaapi_task_t* saved_sp      = stack->sp;
  char*         saved_sp_data = stack->sp_data;
  (*task->body)(task, stack);
  task->body = 0;
  
  /* push restore_frame task if pushed tasks */
  if (saved_sp < stack->sp)
  {
    kaapi_task_t* retn = kaapi_stack_top(stack);
    retn->body  = &kaapi_retn_body;
    retn->state = KAAPI_TASK_INIT | KAAPI_TASK_F_STICKY;
    /* next line is equiv to saving a frame. retn->pdata should be viewed as a kaapi_frame_t */
    retn->pdata[0] = task; /* <=> saved_pc */
    retn->pdata[1] = saved_sp;
    retn->pdata[2] = saved_sp_data;
    kaapi_stack_push(stack);
    /* update pc to the first forked task */
    stack->pc = saved_sp;
    return 0;
  }
  ++stack->pc;
  return 0;
}
#endif


/**kaapi_stack_taskexecall
*/
int kaapi_stack_taskexecall(kaapi_stack_t* stack) 
{
  register kaapi_task_t* task;
  kaapi_task_t* saved_sp;
  char*         saved_sp_data;
  kaapi_task_t* retn;

  if (stack ==0) return EINVAL;
  if (kaapi_stack_isempty( stack ) ) return 0;
  task = stack->pc;

redo_work: 
  {
    if (task->body ==0) return 0;
    if (task->body == &kaapi_suspend_body) 
    {
      return EWOULDBLOCK;
    }
#if 0 /* optimization */
    else if (task->body == &kaapi_retn_body) 
      /* do not save stack frame before execution */
      kaapi_retn_body(task, stack);
#endif
    else
    {
      saved_sp      = stack->sp;
      saved_sp_data = stack->sp_data;
      (*task->body)(task, stack);
      task->body = 0;
    
      /* push restore_frame task if pushed tasks */
      if (saved_sp < stack->sp)
      {
        retn = kaapi_stack_top(stack);
        retn->body  = &kaapi_retn_body;
        /* next line is equiv to saving a frame. retn->pdata should be viewed as a kaapi_frame_t */
        retn->param.pdata[0] = task; /* <=> save pc */
        retn->param.pdata[1] = saved_sp;
        retn->param.pdata[2] = saved_sp_data;
        kaapi_stack_push(stack);

        /* update pc to the first forked task */
        task = stack->pc = saved_sp;
        goto redo_work;
      }
       
      task = ++stack->pc;
      goto redo_work;
    }
  }
  task = ++stack->pc;
  if (stack->pc >= stack->sp) return 0;
  goto redo_work;
}
