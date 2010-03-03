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
  register kaapi_task_t* pc;

#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint32_t         cnt_tasks = 0;
#endif  
  char*                  saved_sp_data;
  int                    err =0;
  kaapi_frame_t*         frame;
  kaapi_task_t*          retn;
  kaapi_task_body_t      body;

  if (stack ==0) return EINVAL;
  if (kaapi_stack_isempty( stack ) ) return 0;

  /* main loop */
  pc = stack->pc;
  while (pc != stack->sp)
  {
    body = pc->body;

    if (!kaapi_task_casstate(pc, body, kaapi_exec_body)) 
    {
      err= EWOULDBLOCK;
      goto label_return;
    }

    /* save the state of the stack */
    stack->saved_sp      = stack->sp;
    saved_sp_data = stack->sp_data;
    
    /* exec la tache */
    body( pc, stack );
#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif
    if ((err = stack->errcode)) 
    {
      if (err != -EAGAIN) /* set by retn */
        goto label_return;
      pc = stack->pc; 
      --pc;
      err = stack->errcode = 0;
      continue;
    }

    /* push restore_frame task if pushed tasks (growth down) */
    if (stack->saved_sp > stack->sp)
    {
      /* inline version of kaapi_stack_pushretn in order to avoid to save all frame structure */
      retn = kaapi_stack_toptask(stack);
      retn->body  = kaapi_retn_body;

      frame = kaapi_stack_pushdata(stack, sizeof(kaapi_frame_t));
      retn->sp = (void*)frame;
      frame->pc = pc; /* <=> save pc, will mark this task as term after pop !! */
      frame->sp = stack->saved_sp;
      frame->sp_data = saved_sp_data;
      kaapi_stack_pushtask(stack);

      /* update pc to the first forked task */
      pc = stack->saved_sp;
    }
    else {
      kaapi_task_setbody(pc, kaapi_nop_body);
      --pc;
    }

    /* process steal request 
       - here we always see the retn to split stack into frame.
    */
#if !defined(KAAPI_CONCURRENT_WS)
    if (stack->hasrequest !=0) 
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      KAAPI_PERF_REG(stack->_proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
      cnt_tasks = 0;
#endif
      kaapi_sched_advance(stack->_proc);
    }
#endif
  }

label_return:
  /* rewrite pc into memory */
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(stack->_proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
#endif
  if (err == EINTR)
  {
    /* mark current task executed */
    kaapi_task_setbody(pc, kaapi_nop_body);
    --pc;
  }
  stack->pc = pc;
  stack->errcode = 0;
  return err;
}
