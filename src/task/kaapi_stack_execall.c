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
  /*register kaapi_task_t* pc;*/
  kaapi_task_t*          saved_sp;
  char*                  saved_sp_data;
  kaapi_task_t*          retn;
  void** arg_retn;

#if defined(KAAPI_TRACE_DEBUG)  
  int level =0;
#endif  

  if (stack ==0) return EINVAL;
  if (kaapi_stack_isempty( stack ) ) return 0;
  /*pc = stack->pc;*/

redo_work: 
  if (stack->pc->body == &kaapi_suspend_body)
  {
    /* rewrite pc into memory */
    /* stack->pc = pc; */
    return EWOULDBLOCK;
#if 0
    if (kaapi_task_issync(stack->pc))
    {
      KAAPI_LOG(50,"Would block task: 0x%p\n", (void*)pc );
      /* rewrite pc into memory */
      stack->pc = pc;
      return EWOULDBLOCK;
    }

    /* ignore the task */
    ++pc;
    goto redo_work;
#endif
  }
  else if (stack->pc->body == &kaapi_retn_body) 
  {
    /* do not save stack frame before execution */
//    kaapi_retn_body(pc, stack);
    kaapi_frame_t* frame = kaapi_task_getargst( stack->pc, kaapi_frame_t);
    kaapi_task_setstate( frame->pc, KAAPI_TASK_S_TERM );
    kaapi_stack_restore_frame( stack, frame );
    /* read from memory */
    /*pc = stack->pc; */
#if defined(KAAPI_TRACE_DEBUG)  
    KAAPI_LOG(100, "stackexec: exec retn 0x%p, pc: 0x%p\n",(void*)stack->pc, (void*)stack->pc );
    --level;
#endif
    ++stack->pc;
    if (stack->pc >= stack->sp) return 0;
    goto redo_work;
  }
  else
  {
    saved_sp      = stack->sp;
    saved_sp_data = stack->sp_data;
#if defined(KAAPI_TRACE_DEBUG)  
    { int k; for (k=0; k<level; ++k) printf("--------"); }
    printf("level:%i  ", level);
    KAAPI_LOG(100, "stackexec: task 0x%p, pc: 0x%p\n", (void*)stack->pc, (void*)stack->pc );
#endif  
    kaapi_task_setstate( stack->pc, KAAPI_TASK_S_EXEC );
#if 0
    extern void fibo_body( kaapi_task_t* task, kaapi_stack_t* stack );
    extern void sum_body( kaapi_task_t* task, kaapi_stack_t* stack );
    
    if (stack->pc->body == &fibo_body)
    {
      fibo_body(stack->pc, stack);
    }
    else if (stack->pc->body == &sum_body)
    {
      sum_body(stack->pc, stack);
    }
    else
#endif
    {
      (*stack->pc->body)(stack->pc, stack);
    }

    /* push restore_frame task if pushed tasks */
    if (saved_sp < stack->sp)
    {
      retn = kaapi_stack_toptask(stack);
      kaapi_task_init(stack, retn, KAAPI_TASK_STICKY);
      retn->body  = &kaapi_retn_body;
      arg_retn = kaapi_stack_pushdata(stack, 3*sizeof(void*));
      retn->sp = (void*)arg_retn;
      arg_retn[0] = stack->pc; /* <=> save pc, will mark this task as term after pop !! */
      arg_retn[1] = saved_sp;
      arg_retn[2] = saved_sp_data;
      kaapi_stack_pushtask(stack);

#if defined(KAAPI_TRACE_DEBUG)  
      KAAPI_LOG(100, "stackexec: push retn: 0x%p, pc: 0x%p\n", 
        (void*)retn, 
        (void*)stack->pc );
#endif

      /* update pc to the first forked task */
      stack->pc = saved_sp;
#if defined(KAAPI_TRACE_DEBUG)  
      ++level;
#endif  
      /* process steal request 
         - here we always see the retn to split stack into frame.
      */
      if (*stack->hasrequest !=0) {
        /*stack->pc = pc; */
        kaapi_sched_advance( stack->_proc );
      }
        
      goto redo_work;
    }
    kaapi_task_setstate( stack->pc, KAAPI_TASK_S_TERM );
    
    /* process steal request 
       - here we always see the retn to split stack into frame.
    */
    if (*stack->hasrequest !=0) {
      /*stack->pc = pc;*/
      kaapi_sched_advance( stack->_proc );
    }
  }

  /*next_task: */
  ++stack->pc;
  if (stack->pc >= stack->sp) return 0;
  goto redo_work;
}
