/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** Joao.Lima@imag.fr / joao.lima@inf.ufrgs.br
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

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_tasklist.h"

#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_event.h"
#include "kaapi_cuda_stream.h"

/* cuda task body */
typedef void (*kaapi_cuda_task_body_t) (void *, cudaStream_t);

static inline void
kaapi_cuda_thread_tasklist_activate_deps(kaapi_taskdescr_t * td)
{
  kaapi_readytasklist_pushactivated(kaapi_get_current_processor()->rtl,
                                    td);
  KAAPI_ATOMIC_ADD(&td->tasklist->cnt_exec, 1);
}

/* call back to push ready task into the tasklist after terminaison of a task */
static int
kaapi_cuda_gpu_task_callback2_after_kernel(kaapi_cuda_stream_t * kstream,
                                           void *arg)
{
  kaapi_taskdescr_t *const td = (kaapi_taskdescr_t *) arg;
  kaapi_memory_taskdescr_epilogue(td);
  /* default write-back policy (lazy) */
  kaapi_cuda_thread_tasklist_activate_deps(td);
  return 0;
}

static int
kaapi_cuda_gpu_task_callback1_exec_task(kaapi_cuda_stream_t * kstream,
                                        void *arg)
{
  kaapi_taskdescr_t *const td = (kaapi_taskdescr_t *) arg;
  kaapi_cuda_task_body_t body = (kaapi_cuda_task_body_t)
  kaapi_format_get_task_bodywh_by_arch(td->fmt,
                                       KAAPI_PROC_TYPE_CUDA);
  kaapi_assert_debug(body != 0);
  kaapi_assert_debug(td->task != 0);
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_STATIC_TASK_BEG);  
  kaapi_cuda_ctx_push();
  body(kaapi_task_getargs(td->task), kaapi_cuda_kernel_stream());
  kaapi_cuda_ctx_pop();
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_STATIC_TASK_END);
  kaapi_cuda_stream_push(kstream, KAAPI_CUDA_OP_KER,
                         kaapi_cuda_gpu_task_callback2_after_kernel, arg);
  return 0;
}

static int
kaapi_cuda_gpu_task_callback0_sync_gpu(kaapi_cuda_stream_t * kstream,
                                       void *arg)
{
  kaapi_taskdescr_t *const td = (kaapi_taskdescr_t *) arg;
  kaapi_cuda_ctx_push();
  kaapi_memory_taskdescr_prologue(td);
  kaapi_cuda_ctx_pop();
  kaapi_cuda_stream_push(kstream, KAAPI_CUDA_OP_H2D,
                         kaapi_cuda_gpu_task_callback1_exec_task, arg);
  return 0;
}

static int
kaapi_cuda_host_task_callback1_exec_task(kaapi_cuda_stream_t * kstream,
                                         void *arg)
{
  kaapi_taskdescr_t *const td = (kaapi_taskdescr_t *) arg;
  kaapi_task_body_t body = kaapi_format_get_task_bodywh_by_arch(td->fmt, KAAPI_PROC_TYPE_HOST);
  kaapi_assert_debug(body != 0);
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_STATIC_TASK_BEG);
  body(kaapi_task_getargs(td->task), 0);
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_STATIC_TASK_END);  
  kaapi_cuda_thread_tasklist_activate_deps(td);
  return 0;
}

static int
kaapi_cuda_host_task_callback0_sync_host(kaapi_cuda_stream_t * kstream, void *arg)
{
//  kaapi_taskdescr_t *const td = (kaapi_taskdescr_t *) arg;
  kaapi_assert(0); /* TODO */
  kaapi_cuda_stream_push( kstream, KAAPI_CUDA_OP_D2H,
                          kaapi_cuda_host_task_callback1_exec_task, arg);
  return 0;
}

static inline int
kaapi_cuda_thread_exec_task(kaapi_cuda_stream_t * const kstream,
                            kaapi_stack_t * const stack,
                            kaapi_taskdescr_t * const td)
{
  /* get the correct body for the proc type */
  if (td->fmt == 0) 
  {
    /* currently some internal tasks do not have format */
    kaapi_task_body_t body = kaapi_task_getbody(td->task);
    kaapi_assert_debug(body != 0);
    body(kaapi_task_getargs(td->task), (kaapi_thread_t *) stack->sfp);
    kaapi_cuda_thread_tasklist_activate_deps(td);
  } else
    if (kaapi_format_get_task_bodywh_by_arch(td->fmt, KAAPI_PROC_TYPE_CUDA)== 0) 
    {
      /* execute a CPU task in the GPU thread */
      kaapi_cuda_stream_push(kstream, KAAPI_CUDA_OP_D2H,
                             kaapi_cuda_host_task_callback0_sync_host,
                             (void *) td);
    } else {
      kaapi_cuda_stream_push(kstream, KAAPI_CUDA_OP_H2D,
                             kaapi_cuda_gpu_task_callback0_sync_gpu,
                             (void *) td);
      kaapi_cuda_stream_window_test(kstream);
    }
  
  return 0;
}

int kaapi_cuda_thread_execframe_tasklist(kaapi_thread_context_t * thread)
{
  kaapi_stack_t *const stack = &thread->stack;
  kaapi_task_t *pc;		/* cache */
  kaapi_tasklist_t *tasklist;
  kaapi_taskdescr_t *td;
  kaapi_frame_t *fp;
  unsigned int proc_type;
  int err = 0;
  kaapi_cuda_stream_t *kstream;
  
  kaapi_assert_debug(stack->sfp >= stack->stackframe);
  kaapi_assert_debug(stack->sfp < stack->stackframe + KAAPI_MAX_RECCALL);
  tasklist = stack->sfp->tasklist;
  
  /* here... begin execute frame tasklist */
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_BEG);
  
  /* get the processor type to select correct entry point */
  proc_type = stack->proc->proc_type;
  
  kstream = stack->proc->cuda_proc.kstream;
  
  kaapi_assert_debug(tasklist != 0);
  
  /* force previous write before next write */
  KAAPI_DEBUG_INST(kaapi_tasklist_t save_tasklist
                   __attribute__ ((unused)) = *tasklist);
  
  while (!kaapi_tasklist_isempty(tasklist)) {
    err = kaapi_readylist_pop(&tasklist->rtl, &td);
    
    if (err == 0) {
      kaapi_processor_decr_workload(stack->proc, 1);
    execute_first:
      pc = td->task;
      if (pc != 0) {
        /* push the frame for the running task: pc/sp = one before td (which is in the stack)à */
        fp = (kaapi_frame_t *) stack->sfp;
        stack->sfp[1] = *fp;
        
        stack->sfp = ++fp;
        kaapi_assert_debug((char *) fp->sp > (char *) fp->sp_data);
        kaapi_assert_debug(stack->sfp - stack->stackframe <
                           KAAPI_MAX_RECCALL);

        /* start execution of the user body of the task */
        KAAPI_DEBUG_INST(kaapi_assert(td->u.acl.exec_date == 0));
        kaapi_cuda_thread_exec_task(kstream, stack, td);
        KAAPI_DEBUG_INST(td->u.acl.exec_date = kaapi_get_elapsedns());
        
        /* new tasks created ? */
        if (unlikely(fp->sp > stack->sfp->sp)) {
          err = kaapi_stack_execframe(&thread->stack);
          if (err == EWOULDBLOCK) {
            return EWOULDBLOCK;
          }
          kaapi_assert_debug(err == 0);
        }
        
        /* pop the frame, even if not used */
        stack->sfp = --fp;
      }
    }
    
    KAAPI_DEBUG_INST(save_tasklist = *tasklist);
    
  }				/* while */
  
  if (!kaapi_readytasklist_isempty(stack->proc->rtl)) {
    if (kaapi_readylist_pop(stack->proc->rtl, &td) == 0)
      goto execute_first;
  }
  
  if (!kaapi_readytasklist_isempty(stack->proc->rtl_remote)) {
    if (kaapi_readylist_pop(stack->proc->rtl_remote, &td) == 0)
      goto execute_first;
  }
  
  /* here... end execute frame tasklist */
  KAAPI_EVENT_PUSH0(stack->proc, thread, KAAPI_EVT_FRAME_TL_END);
  
  
  kaapi_assert(kaapi_tasklist_isempty(tasklist));
  
  if ((tasklist->master == 0)
      && (KAAPI_ATOMIC_READ(&tasklist->cnt_exec) != tasklist->total_tasks))
    return EWOULDBLOCK;
  
  return 0;
  
}
