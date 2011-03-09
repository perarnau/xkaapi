/*
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

#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/types.h>
#include "kaapi_impl.h"
#include "kaapi_tasklist.h"
#include "kaapi_cuda_error.h"


/* cuda task body */
typedef void (*cuda_task_body_t)(void*, CUstream);


/* wait fifo.
   cuda stream notifications dont pass
   data back to the user, so we need a
   way to associate a stream event with
   some app specific data.
 */

typedef struct wait_node
{
  void* data;
  struct wait_node* prev;
  struct wait_node* next;
} wait_node_t;

typedef struct wait_fifo
{
  CUstream stream;
  wait_node_t* head;
  wait_node_t* tail;
} wait_fifo_t;

static inline int wait_fifo_create(wait_fifo_t* fifo)
{
  const CUresult res = cuStreamCreate(&fifo->stream, 0);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamCreate", res);
    return -1;
  }

  fifo->head = NULL;
  fifo->tail = NULL;

  return 0;
}

static inline int wait_fifo_destroy(wait_fifo_t* fifo)
{
  wait_node_t* pos = fifo->head;

  while (pos != NULL)
  {
    wait_node_t* const tmp = pos;
    pos = pos->next;
    free(tmp);
  }

  fifo->head = NULL;
  fifo->tail = NULL;

  cuStreamDestroy(fifo->stream);

  return 0;
}

static int wait_fifo_push(wait_fifo_t* fifo, void* data)
{
  /* todo: fifo specific allocator */
  wait_node_t* const node = malloc(sizeof(wait_node_t));
  if (node == NULL) return -1;

  if (fifo->tail == NULL)
    fifo->tail = node;
  else
    fifo->head->prev = node;

  node->prev = NULL;
  node->next = fifo->head;
  fifo->head = node;

  node->data = data;

  return 0;
}

static void* wait_fifo_pop(wait_fifo_t* fifo)
{
  /* assume fifo not empty */
  wait_node_t* const node = fifo->tail;
  void* const data = node->data;

  fifo->tail = node->prev;

  if (node->prev == NULL)
    fifo->head = NULL;
  else
    fifo->tail->next = NULL;

  free(node);

  return data;
}

static inline unsigned int wait_fifo_is_empty(wait_fifo_t* fifo)
{
  return fifo->head == NULL;
}

static inline unsigned int wait_fifo_is_ready(wait_fifo_t* fifo)
{
  /* return 1 if the top node is ready */
  const CUresult res = cuStreamQuery(fifo->stream);

  /* query and mark as empty is signaled */
  if (res == CUDA_SUCCESS) return 1;

#if defined(KAAPI_DEBUG)
  if (res != CUDA_ERROR_NOT_READY)
  {
    kaapi_cuda_error("cuStreamQuery", res);
    exit(-1);
  }
#endif

  return 0;
}

static inline void* wait_fifo_next(wait_fifo_t* fifo)
{
  /* return NULL if empty or not ready */
  if (wait_fifo_is_empty(fifo)) return NULL;
  else if (wait_fifo_is_ready(fifo) == 0) return NULL;
  return wait_fifo_pop(fifo);
}


/* wait port.
   a wait port consists of 3 wait fifos
   for input, kernel and output streams.
 */

typedef struct wait_port
{
  wait_fifo_t input_fifo;
  wait_fifo_t output_fifo;
  wait_fifo_t kernel_fifo;
} wait_port_t;

static void wait_port_destroy(wait_port_t* wp)
{
  /* assume wait_port_create() returned 0 */
  wait_fifo_destroy(&wp->input_fifo);
  wait_fifo_destroy(&wp->output_fifo);
  wait_fifo_destroy(&wp->kernel_fifo);
}

static int wait_port_create(wait_port_t* wp)
{
  if (wait_fifo_create(&wp->input_fifo))
  {
    return -1;
  }
  if (wait_fifo_create(&wp->output_fifo))
  {
    wait_fifo_destroy(&wp->input_fifo);
    return -1;
  }
  else if (wait_fifo_create(&wp->kernel_fifo))
  {
    wait_fifo_destroy(&wp->input_fifo);
    wait_fifo_destroy(&wp->output_fifo);
    return -1;
  }

  return 0;
}

static int wait_port_next(wait_port_t* wp, kaapi_taskdescr_t** td)
{
  /* return 0 if something ready, -1 otherwise */

  if ((*td = wait_fifo_next(&wp->input_fifo))) return 0;
  else if ((*td = wait_fifo_next(&wp->output_fifo))) return 0;
  else if ((*td = wait_fifo_next(&wp->kernel_fifo))) return 0;
  return -1;
}

static unsigned int wait_port_is_empty(wait_port_t* wp)
{
  /* return 1 if all fifos empty */

  if (wait_fifo_is_empty(&wp->input_fifo))
    if (wait_fifo_is_empty(&wp->kernel_fifo))
      if (wait_fifo_is_empty(&wp->output_fifo))
	return 1;

  return 0;
}


/* inlined internal bodies
 */

static int taskmove_body
(wait_port_t* wp, kaapi_taskdescr_t* td, void* sp, kaapi_thread_t* thread)
{
  kaapi_move_arg_t* const arg = (kaapi_move_arg_t*)sp;

  kaapi_processor_t* const proc = kaapi_get_current_processor();

  printf("%s: [%u:%u] (%lx:%lx -> %lx:%lx) %lx\n",
	 __FUNCTION__,
	 proc->kid, proc->proc_type,
	 arg->src_data->ptr.asid, (uintptr_t)arg->src_data->ptr.ptr,
	 arg->dest->ptr.asid, (uintptr_t)arg->dest->ptr.ptr,
	 (uintptr_t)arg->dest->mdi);

  return 0;
}

static int taskalloc_body
(wait_port_t* wp, kaapi_taskdescr_t* td, void* sp, kaapi_thread_t* thread)
{
  printf("%s\n", __FUNCTION__);
  return 0;
}

static int taskfinalizer_body
(wait_port_t* wp, kaapi_taskdescr_t* td, void* sp, kaapi_thread_t* thread)
{
  printf("%s\n", __FUNCTION__);
  return 0;
}


/* refer to kaapi_thread_execframe_tasklist.c
   for general comments.
   todos:
   . newly created tasks not implemented
 */
int kaapi_cuda_thread_execframe_tasklist
(kaapi_thread_context_t* thread)
{
  /* thread->sfp->tasklist */
  kaapi_tasklist_t* tl;

  /* tl->td_top */
  kaapi_taskdescr_t** td_top;

  /* the current task descriptor */
  kaapi_taskdescr_t* td;

  /* the current activation link */
  kaapi_activationlink_t* al;

  /* the task to process */
  kaapi_task_t* pc;

  /* tasks are in the ready queue */
  unsigned int has_ready;

  /* tasks have been pushed */
  unsigned int has_pushed;

  /* local workqueue bounds */
  kaapi_workqueue_index_t local_beg;
  kaapi_workqueue_index_t local_end;

  /* current frame */
  kaapi_frame_t* fp;

  /* executed task counter */
  uint32_t cnt_exec = 0;

  /* temporary error */
  int err;

  /* proc->proc_type */
  const unsigned int proc_type = thread->proc->proc_type;

  /* completion port */
  wait_port_t wp;

  /* toremove */
  printf("%s\n", __FUNCTION__);
  /* toremove */

  /* todo_remove, move in kproc */
  if (cuCtxPushCurrent(thread->proc->cuda_proc.ctx) != CUDA_SUCCESS)
    return -1;

  /* todo: should be done once per proc */
  if (wait_port_create(&wp) == -1) return -1;

  /* bootstrap the readylist */
  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe + KAAPI_MAX_RECCALL);
  kaapi_assert_debug(thread->sfp->tasklist != 0);
  tl = thread->sfp->tasklist;

  if (tl->td_ready == 0)
  {
    kaapi_workqueue_index_t ntasks = 0;

    tl->td_ready = malloc(sizeof(kaapi_taskdescr_t*) * tl->cnt_tasks);
    kaapi_assert_debug(tl->td_ready != NULL);

    tl->recv = tl->recvlist.front;
    
    td_top = tl->td_ready + tl->cnt_tasks;

    for (al = tl->readylist.front; al != NULL; al = al->next)
    {
      *--td_top = al->td;
      ++ntasks;
    }

    /* the initial workqueue is td_top[-ntasks, 0) */
    kaapi_writemem_barrier();
    tl->td_top = tl->td_ready + tl->cnt_tasks;
    kaapi_writemem_barrier();
    kaapi_workqueue_init(&tl->wq_ready, -ntasks, 0);
  }

  /* assume execframe was already called, only reset td_top */
  td_top = tl->td_top;

  /* jump to previous state if return from suspend (EWOULDBLOCK) */
  if (tl->context.chkpt == 1)
  {
    td = tl->context.td;
    fp = tl->context.fp;

    err = kaapi_thread_execframe(thread);
    if ((err == EWOULDBLOCK) || (err == EINTR)) 
    {
      tl->context.chkpt = 1;
      tl->context.td = td;
      tl->context.fp = fp;
      tl->cnt_exectasks += cnt_exec;
      return err;
    }

    has_ready = 0;

    goto do_activation;
  }
  
  /* push the frame for the next task to execute */
  fp = (kaapi_frame_t*)thread->sfp;
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc = fp->sp;
  thread->sfp[1].sp = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  kaapi_assert_debug(thread->sfp - thread->stackframe < KAAPI_MAX_RECCALL);

  /* execute all the ready tasks */
 do_ready:
  while (kaapi_workqueue_isempty(&tl->wq_ready) == 0)
  {
    err = kaapi_workqueue_pop(&tl->wq_ready, &local_beg, &local_end, 1);
    if (err) continue ;

    td = td_top[local_beg];

    kaapi_assert_debug(td != NULL);
    kaapi_assert_debug(td->task != NULL);
    KAAPI_DEBUG_INST(td_top[local_beg] = NULL);

    /* next ready task */
    pc = td->task;

    if (td->fmt != NULL) /* cuda user task */
    {
      cuda_task_body_t body = (cuda_task_body_t)
	td->fmt->entrypoint_wh[proc_type];
      kaapi_assert_debug(body);

      err = wait_fifo_push(&wp.kernel_fifo, (void*)td);
      kaapi_assert_debug(err != -1);

      body(pc->sp, wp.kernel_fifo.stream);
    }
    else /* internal task */
    {
      kaapi_task_body_t body = kaapi_task_getuserbody(pc);
      kaapi_assert_debug(body);

      /* currently, inline those tasks. minor modifs
	 are needed to execute the body directly, ie.
	 passing the taskdescr and wait port.
       */
      if (body == kaapi_taskmove_body)
	taskmove_body(&wp, td, pc->sp, (kaapi_thread_t*)thread->sfp);
      else if (body == kaapi_taskalloc_body)
	taskalloc_body(&wp, td, pc->sp, (kaapi_thread_t*)thread->sfp);
      else if (body == kaapi_taskfinalizer_body)
	taskfinalizer_body(&wp, td, pc->sp, (kaapi_thread_t*)thread->sfp);
      else
	body(pc->sp, (kaapi_thread_t*)thread->sfp);
    }

    ++cnt_exec;
  } /* while_ready */

  has_ready = 0;

  /* receive incoming sync */
/*  do_recv: */
  if (tl->recv != NULL)
  {
    td_top[local_beg--] = tl->recv->td;
    tl->recv = tl->recv->next;
    kaapi_workqueue_push(&tl->wq_ready, 1 + local_beg);
    has_ready = 1;
  }

  /* wait for any completion */
/*  do_wait: */
  while (wait_port_is_empty(&wp) == 0)
  {
    /* event pump, wait for next to complete */
    while (wait_port_next(&wp, &td) == -1)
    {
      /* nothing completed, and ready available */
      if (has_ready) goto do_ready;
    }

  do_activation:
    /* assume no task pushed */
    has_pushed = 0;

    /* does the completed task activate others */
    if (!kaapi_activationlist_isempty(&td->list))
    {
      for (al = td->list.front; al != NULL; al = al->next)
      {
	if (kaapi_taskdescr_activated(al->td))
	{
	  td_top[local_beg--] = al->td;
	  has_pushed = 1;
	}
      }
    }

    /* do bcast after child execution */
    if (td->bcast != 0)
    {
      for (al = td->bcast->front; al != NULL; al = al->next)
      {
	/* bcast task are always ready */
	td_top[local_beg--] = al->td;
	has_pushed = 1;
      }
    }

    /* enqueue the pushed tasks */
    if (has_pushed)
    {
      has_ready = 1;
      kaapi_workqueue_push(&tl->wq_ready, 1 + local_beg);
    }

  } /* while_wait_port */

  /* execute tasks made ready */
  if (has_ready) goto do_ready;

 /* do_return: */

  /* todo: move in kproc */
  wait_port_destroy(&wp);

  /* todo_remove */
  cuCtxPopCurrent(&thread->proc->cuda_proc.ctx);

  /* pop frame */
  --thread->sfp;

  /* update executed tasks */
  tl->cnt_exectasks += cnt_exec;
  
  /* signal the end of the step for the thread
     - if no more recv (and then no ready task activated)
  */
  if (kaapi_tasklist_isempty(tl))
  {
    tl->context.chkpt = 0;
#if defined(KAAPI_DEBUG)
    tl->context.td = 0;
    tl->context.fp = 0;
#endif    
    return ECHILD;
  }
  tl->context.chkpt = 2;
  tl->context.td = 0;
  tl->context.fp = 0;
  return EWOULDBLOCK;
}
