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


/* notes on using event:
   if we dont use events, we use refcount.
   Querying refcount is faster than querying
   event BUT querying streams is much slower
   than querying events (at least with cuda4.0).
   If we use events, we dont need query streams.
   An additionnal reason for using event is if
   the user code queues more than one stuff.
 */
#define CONFIG_USE_EVENT 1


/* cuda task body */
typedef void (*cuda_task_body_t)(void*, CUstream);


/* wait fifo.
   cuda stream notifications dont pass
   data back to the user, so we need a
   way to associate a stream event with
   some app specific data.
   a refn is used since we may want to
   associate a node with 3 cuda event
   completion (see taskmove).
 */

typedef struct wait_node
{
  void* data;

#if CONFIG_USE_EVENT
  CUevent event;
#else
  unsigned int refn;
#endif

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

static void wait_fifo_push
(wait_fifo_t* fifo, wait_node_t* node)
{
  if (fifo->tail == NULL)
    fifo->tail = node;
  else
    fifo->head->prev = node;

  node->prev = NULL;
  node->next = fifo->head;
  fifo->head = node;
}

static void wait_fifo_destroy_node
(wait_fifo_t* fifo, wait_node_t* node)
{
  /* fifo unused */

#if CONFIG_USE_EVENT
  cuEventDestroy(node->event);
#endif

  free(node);
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

  wait_fifo_destroy_node(fifo, node);

  return data;
}

static wait_node_t* wait_fifo_create_node
#if CONFIG_USE_EVENT
(wait_fifo_t* fifo, void* data)
#else
(wait_fifo_t* fifo, void* data, unsigned int refn)
#endif
{
#if CONFIG_USE_EVENT
  CUresult res;
#endif

  wait_node_t* const node = malloc(sizeof(wait_node_t));
  if (node == NULL) return NULL;

#if CONFIG_USE_EVENT
  res = cuEventCreate(&node->event, CU_EVENT_DISABLE_TIMING);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamCreate", res);
    return NULL;
  }

  res = cuEventRecord(node->event, fifo->stream);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuEventRecord", res);
    cuEventDestroy(node->event);
    return NULL;
  }
#else
  node->refn = refn;
#endif

  node->data = data;

  node->prev = NULL;
  node->next = NULL;

  return node;
}

static int wait_fifo_create_push
#if CONFIG_USE_EVENT
(wait_fifo_t* fifo, void* data)
#else
(wait_fifo_t* fifo, void* data, unsigned int refn)
#endif
{
  /* create and push a node */

  wait_node_t* const node = wait_fifo_create_node
#if CONFIG_USE_EVENT
    (fifo, data);
#else
    (fifo, data, refn);
#endif
  if (node == NULL) return -1;

  wait_fifo_push(fifo, node);

  return 0;
}

static inline unsigned int wait_fifo_is_empty(wait_fifo_t* fifo)
{
  return fifo->head == NULL;
}

static inline unsigned int wait_fifo_is_ready(wait_fifo_t* fifo)
{
  /* return 1 if the top node is ready */
#if CONFIG_USE_EVENT
  const CUresult res = cuEventQuery(fifo->head->event);
#else
  const CUresult res = cuStreamQuery(fifo->stream);
#endif

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

#if (CONFIG_USE_EVENT == 0)
  if (--fifo->tail->refn) return NULL;
#endif

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


/* inlined internal task bodies
 */

static inline int allocate_device_mem(CUdeviceptr* devptr, size_t size)
{
  const CUresult res = cuMemAlloc(devptr, size);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemAlloc", res);
    return -1;
  }

  return 0;
}

static inline void free_device_mem(CUdeviceptr devptr)
{
  cuMemFree(devptr);
}


#if 0 /* UNUSED, block transfer */

static inline int memcpy_htod
(CUstream stream, CUdeviceptr devptr, const void* hostptr, size_t size)
{
  const CUresult res = cuMemcpyHtoDAsync
    (devptr, hostptr, size, stream);

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyHToDAsync", res);
    return -1;
  }
  
  return 0;
}

static inline int memcpy_dtoh_sync
(void* hostptr, CUdeviceptr devptr, size_t size)
{
  const CUresult res = cuMemcpyDtoH(hostptr, devptr, size);

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyDToHSync", res);
    return -1;
  }
  
  return 0;
}

static void block_transfer_example(void)
{
  const size_t hostsize = kaapi_memory_view_size(sview);

  /* start the async copy */
  size_t nblocks;
  size_t blocksize;
  size_t stridesize;
  size_t i;

  /* assume a 2d case */
  nblocks = hostsize / (sview->size[1] * sview->wordsize);
  blocksize = sview->size[1] * sview->wordsize;
  stridesize = sview->lda * sview->wordsize;

  for (i = 0; i < nblocks; ++i)
  {
    memcpy_htod
    (
     wp->input_fifo.stream,
     devptr + i * blocksize,
     (void*)((uintptr_t)hostptr + i * stridesize),
     blocksize
    );
  }

  /* add to completion port */
#if CONFIG_USE_EVENT
  wait_fifo_create_push(&wp->input_fifo, (void*)td);
#else
  wait_fifo_create_push(&wp->input_fifo, (void*)td, nblocks);
#endif
}

#endif /* UNUSED, block transfer */


#if 0 /* UNUSED, not working with 2d */

static void register_memory_example(void)
{

  static const uintptr_t page_size = 0x1000UL;
  static const uintptr_t lo_mask = 0x1000UL - 1UL;
  static const uintptr_t hi_mask = ~(0x1000UL - 1UL);

  uintptr_t aligned_addr = (uintptr_t)hostptr;
  size_t aligned_size = hostsize;

  /* addr may be misaligned */
  if (aligned_addr & lo_mask)
  {
    aligned_addr = (uintptr_t)hostptr & hi_mask;
    aligned_size = hostsize + (uintptr_t)hostptr - aligned_addr;
  }

  /* size may be misaligned */
  if (aligned_size & lo_mask)
    aligned_size = (aligned_size & hi_mask) + page_size;

  res = cuMemHostRegister
    ((void*)aligned_addr, aligned_size, CU_MEMHOSTREGISTER_PORTABLE);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemHostRegister", res);
    return -1;
  }

  m.srcMemoryType = CU_MEMORYTYPE_HOST;
  m.srcHost = (void*)aligned_addr;
  m.srcPitch = sview->lda * sview->wordsize;
  m.srcXInBytes = (uintptr_t)hostptr - (uintptr_t)aligned_addr;
  m.srcY = 0;
}
#endif /* UNUSED, not working with 2d */


/* TODO: replace xxx_ with kaapi_ */

static int xxx_memory_cast_view
(
 kaapi_memory_view_t* dview,
 const kaapi_memory_view_t* sview
)
{
  /* TODO: asid as arguments.
     assume dview on gpu, sview on cpu.
  */

  switch (sview->type)
  {
  case KAAPI_MEMORY_VIEW_1D:
    {
      dview->type = sview->type;
      dview->size[0] = sview->size[0];
      dview->lda = sview->size[0];
      dview->wordsize = sview->wordsize;
      break ;
    }

  case KAAPI_MEMORY_VIEW_2D:
    {
      dview->type = sview->type;
      dview->size[0] = sview->size[0];
      dview->size[1] = sview->size[1];
      dview->lda = sview->size[1];
      dview->wordsize = sview->wordsize;
      break ;
    }

  default:
    {
      /* TODO */
      kaapi_assert(0);
      return -1;
      break ;
    }
  }

  return 0;
}

typedef struct async_context
{
  /* context for gpu to host async copy */
  wait_port_t* wp;
  kaapi_taskdescr_t* td;
} async_context_t;

static int xxx_memory_async_copy
(
 kaapi_pointer_t dkptr,
 const kaapi_memory_view_t* dview,
 const void* sptr,
 const kaapi_memory_view_t* sview,
 void* opaak_data
)
{
  /* assume host to cpu transfer
   */

  /* assume dkptr allocated
     aussme dview initialized
   */

  async_context_t* const ac = opaak_data;
  const uintptr_t dptr = dkptr.ptr;
  wait_node_t* node;

  /* push before copying, so that failure is recoverable */

  node = wait_fifo_create_node
#if CONFIG_USE_EVENT
    (&ac->wp->input_fifo, (void*)ac->td);
#else
    (&ac->wp->input_fifo, (void*)ac->td, 1);
#endif
  if (node == NULL) return -1;

  switch (sview->type)
  {
  case KAAPI_MEMORY_VIEW_1D:
    {
      const size_t size = sview->size[0] * sview->wordsize;

      const CUresult res = cuMemcpyHtoDAsync
	((CUdevice)dptr, sptr, size, ac->wp->input_fifo.stream);
      if (res != CUDA_SUCCESS)
      {
	kaapi_cuda_error("cuMemcpy2DAsync", res);
	goto on_error;
      }

      break ;
    }

  case KAAPI_MEMORY_VIEW_2D:
    {
      /* TODO: optimize for contiguous case */

      CUresult res;
      CUDA_MEMCPY2D m;

      m.srcMemoryType = CU_MEMORYTYPE_HOST;
      m.srcHost = sptr;
      m.srcPitch = sview->lda * sview->wordsize;
      m.srcXInBytes = 0;
      m.srcY = 0;

      m.dstXInBytes = 0;
      m.dstY = 0;
      m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      m.dstDevice = dptr;
      m.dstPitch = sview->size[1] * sview->wordsize;

      m.WidthInBytes = sview->size[1] * sview->wordsize;
      m.Height = sview->size[0];

      res = cuMemcpy2DAsync(&m, ac->wp->input_fifo.stream);
      if (res != CUDA_SUCCESS)
      {
	kaapi_cuda_error("cuMemcpy2DAsync", res);
	goto on_error;
      }

      break ;
    }

    /* not supported */
  default:
    {
      kaapi_assert(0);
      goto on_error;
      break ;
    }
  }

  /* on success, pushes a new node in fifo */
  wait_fifo_push(&ac->wp->input_fifo, node);
  return 0;

 on_error:
  wait_fifo_destroy_node(&ac->wp->input_fifo, node);
  return -1;
}

static int taskmove_body
(
 kaapi_thread_context_t* thread,
 wait_port_t* wp,
 kaapi_taskdescr_t* td,
 void* sp, kaapi_thread_t* unused
)
{
  kaapi_move_arg_t* const arg = (kaapi_move_arg_t*)sp;

  kaapi_memory_view_t* const sview = &arg->src_data->view;
  kaapi_memory_view_t* const dview = &arg->dest->view;

  /* memory information */
  kaapi_metadata_info_t* mdi;
  kaapi_access_mode_t mode;

  /* for the copy */
  void* const sptr = (void*)arg->src_data->ptr.ptr;

  const size_t dsize = sview->lda * sview->size[1] * sview->wordsize;
  void* dptr;
  kaapi_pointer_t dkptr;

  async_context_t ac;

  /* cast the sview to dview */

  if (xxx_memory_cast_view(dview, sview))
    return -1;

  /* allocate destination memory */

#if 1 /* TODO: use memory layer */
  if (allocate_device_mem((CUdeviceptr*)&dptr, dsize))
    return -1;
#endif /* TODO */

  /* cast to kaapi pointer */

  dkptr.asid = thread->asid;
  dkptr.ptr = (uintptr_t)dptr;

  /* starts an async transfer */

  ac.wp = wp;
  ac.td = td;

  if (xxx_memory_async_copy(dkptr, dview, sptr, sview, (void*)&ac))
  {
#if 1 /* TODO: use memory layer */
    free_device_mem((CUdeviceptr)dptr);
#endif /* TODO */
    return -1;
  }

  /* update the task args */

  arg->dest->ptr = dkptr;
  arg->dest->mdi = arg->src_data->mdi;

  /* update memory information */

  mdi = arg->src_data->mdi;
  mode = mdi->version[0]->last_mode;

#if 0 /* TODO: revalidate as memory */
  if (KAAPI_ACCESS_IS_READ(mode))
  {
  }
#endif /* TODO */

  /* validate as */

  if (KAAPI_ACCESS_IS_WRITE(mode))
  {
    const kaapi_globalid_t gid =
      kaapi_memory_address_space_getgid(thread->asid);

    _kaapi_metadata_info_set_writer(mdi, thread->asid);
    mdi->data[gid].ptr.ptr = (uintptr_t)dptr;
    mdi->data[gid].ptr.asid = thread->asid;
    mdi->data[gid].view = *dview;
  }

  return 0;
}

static int taskalloc_body
(wait_port_t* wp, kaapi_taskdescr_t* td, void* sp, kaapi_thread_t* thread)
{
#if 0 /* TOREMOVE */
  printf("%s\n", __FUNCTION__);
#endif
  return 0;
}

static int taskfinalizer_body
(wait_port_t* wp, kaapi_taskdescr_t* td, void* sp, kaapi_thread_t* thread)
{
#if 0 /* TOREMOVE */
  printf("%s\n", __FUNCTION__);
#endif
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

#if 0 /* TOREMOVE */
  printf("%s\n", __FUNCTION__);
#endif /* TOREMOVE */

  /* to_remove: create a new address space
     so that data are considered to be present
     on the cpu (ie. ptr.asid == 0)
   */
  thread->asid = kaapi_memory_address_space_create
    (0x1, KAAPI_MEM_TYPE_CUDA, 0x100000000UL);

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

#if 0 /* TOREMOVE */
  {
    printf("-- readylist: (%lu, %lx)\n", tl->cnt_tasks, tl->td_ready);
    for (al = tl->readylist.front; al != NULL; al = al->next)
      printf("%lx, ", (uintptr_t)al);
    printf("\n");
  }
#endif /* TOREMOVE */

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

#if 0 /* TOREMOVE */
      free(tl->td_ready);
      tl->td_ready = 0;
#endif /* TOREMOVE */

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

#if 0
  printf("> do_ready()\n");
#endif

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

#if CONFIG_USE_EVENT
      err = wait_fifo_create_push(&wp.kernel_fifo, (void*)td);
#else
      err = wait_fifo_create_push(&wp.kernel_fifo, (void*)td, 1);
#endif
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
	taskmove_body(thread, &wp, td, pc->sp, (kaapi_thread_t*)thread->sfp);
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
#if 0
    printf("> wait_port_next()\n");
#endif

    /* event pump, wait for next to complete */
    while (wait_port_next(&wp, &td) == -1)
    {
      /* nothing completed, and ready available */
      if (has_ready) goto do_ready;
    }

  do_activation:

#if 0
    printf("> do_activation()\n");
#endif

    /* assume no task pushed */
    has_pushed = 0;

    /* does the completed task activate others */
    if (!kaapi_activationlist_isempty(&td->list))
    {
      for (al = td->list.front; al != NULL; al = al->next)
      {
#if 0
	printf("activating(%lx)\n", (uintptr_t)al->td);
#endif

	if (kaapi_taskdescr_activated(al->td))
	{
#if 0
	  printf("activated(%lx)\n", (uintptr_t)al->td);
#endif

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

#if 0 /* TOREMOVE */
	if ((al == td->bcast->back) && (al->next != NULL))
	{
	  /* bug here: end of list but next non null */
	  printf("%s: %d\n", __FUNCTION__, __LINE__);
	  break ;
	}
#endif /* TOREMOVE */
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

#if 0 /* TOREMOVE */
  free(tl->td_ready);
  tl->td_ready = 0;
#endif /* TOREMOVE */
  
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
