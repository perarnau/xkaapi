/*
** kaapi_cuda_execframe.c
** xkaapi
** 
** Created on Jul 2010
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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


#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include "kaapi_impl.h"
#include "kaapi_cuda_error.h"


/* memory management api */

#define KAAPI_MEM_MAX_SPACES 3

typedef uintptr_t kaapi_mem_laddr_t;

typedef struct kaapi_mem_laddrs
{
  struct kaapi_mem_laddrs* next;
  unsigned int bitmap;
  kaapi_mem_laddr_t laddrs[KAAPI_MEM_MAX_SPACES];
} kaapi_mem_laddrs_t;

static inline void kaapi_mem_laddrs_init
(kaapi_mem_laddrs_t* laddrs)
{
  laddrs->next = NULL;
  laddrs->bitmap = 0;
}

static inline void kaapi_mem_laddrs_set
(kaapi_mem_laddrs_t* laddrs, kaapi_mem_laddr_t laddr, unsigned int index)
{
  laddrs->bitmap |= 1 << index;
  laddrs->laddrs[index] = laddr;
}


/* a map translates between a device local addrs
 */

typedef struct kaapi_mem_map
{
  kaapi_mem_laddrs_t* head;
  unsigned int index;
} kaapi_mem_map_t;


static int kaapi_mem_map_initialize(kaapi_mem_map_t* map)
{
  map->head = NULL;
  return 0;
}

static void kaapi_mem_map_cleanup(kaapi_mem_map_t* map)
{
  kaapi_mem_laddrs_t* pos = map->head;

  while (pos != NULL)
  {
    kaapi_mem_laddrs_t* const tmp = pos;
    pos = pos->next;
    free(tmp);
  }

  map->head = NULL;
}

static int kaapi_mem_map_find_or_insert
(kaapi_mem_map_t* map, kaapi_mem_laddr_t laddr, kaapi_mem_laddrs_t** laddrs)
{
  kaapi_mem_laddrs_t* pos;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    if (pos->laddrs[map->index] == laddr)
      break;
  }

  if (pos == NULL)
  {
    pos = malloc(sizeof(kaapi_mem_laddr_t));
    if (pos == NULL)
      return -1;

    kaapi_mem_laddrs_init(pos);
    kaapi_mem_laddrs_set(pos, laddr, map->index);
  }

  *laddrs = pos;

  return 0;
}


/* device memory allocation */

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

/* copy memory to device */

static inline int copy_device_mem
(kaapi_processor_t* proc, CUdeviceptr devptr, void* hostptr, size_t size)
{
  const CUresult res = cuMemcpyHtoDAsync
    (devptr, hostptr, size, proc->cuda_proc.stream);

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyHToDAsync", res);
    return -1;
  }

  return 0;
}

/* prepare task args memory */

static kaapi_mem_map_t global_mem_map;

static void prepare_task
(kaapi_processor_t* proc, kaapi_task_t* task, kaapi_task_body_t body)
{
  kaapi_format_t* const format = kaapi_format_resolvebybody(body);
  kaapi_mem_map_t* map = &global_mem_map;

  kaapi_access_t* access;
  kaapi_mem_laddrs_t* laddrs;
  CUdeviceptr devptr;
  size_t size;
  unsigned int i;

  kaapi_assert_debug(format != NULL);

  /* to remove */
  {
    static unsigned int init_once = 0;
    if (init_once == 0)
    {
      init_once = 1;
      kaapi_mem_map_initialize(map);
    }
  }
  /* to remove */

  for (i = 0; i < format->count_params; ++i)
  {
    const kaapi_access_mode_t mode =
      KAAPI_ACCESS_GET_MODE(format->mode_params[i]);

    if (mode & KAAPI_ACCESS_MODE_V)
      continue ;

    access = (kaapi_access_t*)((uint8_t*)task->sp + format->off_params[i]);

    /* todo: handle error */
    kaapi_mem_map_find_or_insert
      (map, (kaapi_mem_laddr_t)access->data, &laddrs);

    if (KAAPI_ACCESS_IS_ONLYWRITE(mode))
    {
      /* allocate the memory on device */

      /* todo: size = format->param_size(i); */
      size = 50000 * sizeof(unsigned int);

      allocate_device_mem(&devptr, size);
      
      /* add devptr as a local addr */
#define KAAPI_MEM_GPU_INDEX 1
      kaapi_mem_laddrs_set(laddrs, devptr, KAAPI_MEM_GPU_INDEX);
    }
    else if (KAAPI_ACCESS_IS_READ(mode)) /* R or RW */
    {
      /* todo: size = format->param_size(i); */
      size = 50000 * sizeof(unsigned int);

      copy_device_mem(proc, devptr, access->data, size);
    }

    /* update param addr */
    access->data = (void*)(uintptr_t)devptr;
  }
}

/* execute a cuda task */

typedef void (*cuda_task_body_t)(CUstream, void*, kaapi_thread_t*);

static inline void execute_task
(
 kaapi_processor_t* proc,
 cuda_task_body_t body,
 void* args,
 kaapi_thread_t* thread
)
{
  body(proc->cuda_proc.stream, args, thread);
}


/* wait for stream completion */

static inline int synchronize_processor(kaapi_processor_t* proc)
{
  const CUresult res = cuStreamSynchronize(proc->cuda_proc.stream);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamSynchronize", res);
    return -1;
  }

  return 0;
}


/* exported */

int kaapi_cuda_execframe(kaapi_thread_context_t* thread)
{
  kaapi_processor_t* const proc = thread->proc;

  kaapi_task_t*              pc;
  kaapi_frame_t*             fp;
  kaapi_task_body_t          body;
  kaapi_frame_t*             eframe = thread->esfp;
#if defined(KAAPI_USE_PERFCOUNTER)
  kaapi_uint32_t             cnt_tasks = 0;
#endif

  kaapi_assert_debug(thread->sfp >= thread->stackframe);
  kaapi_assert_debug(thread->sfp < thread->stackframe+KAAPI_MAX_RECCALL);
  
push_frame:
  fp = (kaapi_frame_t*)thread->sfp;
  /* push the frame for the next task to execute */
  thread->sfp[1].sp_data = fp->sp_data;
  thread->sfp[1].pc = fp->sp;
  thread->sfp[1].sp = fp->sp;
  
  /* force previous write before next write */
  kaapi_writemem_barrier();

  /* update the current frame */
  ++thread->sfp;
  kaapi_assert_debug( thread->sfp - thread->stackframe <KAAPI_MAX_RECCALL);

#if 1/*(KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALCAS_METHOD) || (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)*/
begin_loop:
#endif
  /* stack of task growth down ! */
  while ((pc = fp->pc) != fp->sp)
  {
    kaapi_assert_debug( pc > fp->sp );

#if (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALCAS_METHOD)
    body = pc->body;
    kaapi_assert_debug( body != kaapi_exec_body);

    if (!kaapi_task_casstate( pc, pc->ebody, kaapi_exec_body)) 
    { 
      kaapi_assert_debug((pc->body == kaapi_suspend_body) || (pc->body == kaapi_aftersteal_body) );
      body = pc->body;
      if (body == kaapi_suspend_body)
        goto error_swap_body;
      /* else ok its aftersteal */
      body = kaapi_aftersteal_body;
      pc->body = kaapi_exec_body;
    }
#elif (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)
    /* wait thief get out pc */
    while (thread->thiefpc == pc)
      ;
    body = pc->body;
    kaapi_assert_debug( body != kaapi_exec_body);
    if (body == kaapi_suspend_body)
      goto error_swap_body;

    pc->body = kaapi_exec_body;
#else
#  error "Undefined steal task method"    
#endif

    prepare_task(proc, pc, body);
    execute_task
      (proc, (cuda_task_body_t)body, pc->sp, (kaapi_thread_t*)thread->sfp);

    /* todo: move in the end of frame execution */
    synchronize_processor(proc);
    
#if 0//!defined(KAAPI_CONCURRENT_WS)
    if (unlikely(thread->errcode)) goto backtrack_stack;
#endif
#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif

#if  0/*!defined(KAAPI_CONCURRENT_WS)*/
restart_after_steal:
#endif
    if (unlikely(fp->sp > thread->sfp->sp))
    {
      goto push_frame;
    }
#if defined(KAAPI_DEBUG)
    else if (unlikely(fp->sp < thread->sfp->sp))
    {
      kaapi_assert_debug_m( 0, "Should not appear: a task was popping stack ????" );
    }
#endif

    /* next task to execute */
    pc = fp->pc = pc -1;
    kaapi_writemem_barrier();
  } /* end of the loop */

  kaapi_assert_debug( fp >= eframe);
  kaapi_assert_debug( fp->pc == fp->sp );

  if (fp >= eframe)
  {
#if (KAAPI_USE_STEALFRAME_METHOD == KAAPI_STEALCAS_METHOD)
    /* here it's a pop of frame: we lock the thread */
    while (!KAAPI_ATOMIC_CAS(&thread->lock, 0, 1));
    while (fp > eframe) 
    {
      --fp;

      /* pop dummy frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        KAAPI_ATOMIC_WRITE(&thread->lock, 0);
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

    kaapi_writemem_barrier();
    KAAPI_ATOMIC_WRITE(&thread->lock, 0);
#elif (KAAPI_USE_STEALFRAME_METHOD == KAAPI_STEALTHE_METHOD)
    /* here it's a pop of frame: we use THE like protocol */
    while (fp > eframe) 
    {
      thread->sfp = --fp;
      kaapi_writemem_barrier();
      /* wait thief get out the frame */
      while (thread->thieffp > fp)
        ;

      /* pop dummy frame and the closure inside this frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

    kaapi_writemem_barrier();
#else
#  error "Bad steal frame method"    
#endif
  }
  thread->sfp = fp;
  
  /* end of the pop: we have finish to execute all the task */
  kaapi_assert_debug( fp->pc == fp->sp );
  kaapi_assert_debug( thread->sfp == eframe );

  /* note: the stack data pointer is the same as saved on enter */

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return 0;

#if (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALCAS_METHOD) || (KAAPI_USE_STEALTASK_METHOD == KAAPI_STEALTHE_METHOD)
error_swap_body:
  if (fp->pc->body == kaapi_aftersteal_body) goto begin_loop;
  kaapi_assert_debug(thread->sfp- fp == 1);
  /* implicityly pop the dummy frame */
  thread->sfp = fp;
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
#endif

#if 0
backtrack_stack:
#endif
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
#if 0 /*!defined(KAAPI_CONCURRENT_WS)*/
  if ((thread->errcode & 0x1) !=0) 
  {
    kaapi_sched_advance(thread->proc);
    thread->errcode = thread->errcode & ~0x1;
    if (thread->errcode ==0) goto restart_after_steal;
  }
#endif

  /* here back track the kaapi_stack_execframe until go out */
  return thread->errcode;
}
