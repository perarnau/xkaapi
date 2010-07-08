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

#define KAAPI_MEM_ID_CPU 0
#define KAAPI_MEM_ID_GPU 1
#define KAAPI_MEM_ID_MAX 2

typedef uintptr_t kaapi_mem_laddr_t;

typedef struct kaapi_mem_laddrs
{
  struct kaapi_mem_laddrs* next;
  unsigned int bitmap;
  kaapi_mem_laddr_t laddrs[KAAPI_MEM_ID_MAX];
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

static inline kaapi_mem_laddr_t kaapi_mem_laddrs_get
(kaapi_mem_laddrs_t* laddrs, unsigned int memid)
{
  return laddrs->laddrs[memid];
}

static inline unsigned int kaapi_mem_laddrs_isset
(const kaapi_mem_laddrs_t* laddrs, unsigned int memid)
{
  return laddrs->bitmap & (1 << memid);
}


/* a map translates between a device local addrs
 */

typedef struct kaapi_mem_map
{
  kaapi_mem_laddrs_t* head;
  unsigned int memid;
} kaapi_mem_map_t;


static int kaapi_mem_map_initialize
(kaapi_mem_map_t* map, unsigned int memid)
{
  map->memid = memid;
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
(
 kaapi_mem_map_t* map,
 unsigned int memid,
 kaapi_mem_laddr_t laddr,
 kaapi_mem_laddrs_t** laddrs
)
{
  /* laddr a host local address */

  kaapi_mem_laddrs_t* pos;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    if (pos->laddrs[memid] == laddr)
      break;
  }

  if (pos == NULL)
  {
    pos = malloc(sizeof(kaapi_mem_laddr_t));
    if (pos == NULL)
      return -1;

    kaapi_mem_laddrs_init(pos);
    kaapi_mem_laddrs_set(pos, laddr, memid);

    pos->next = map->head;
    map->head = pos;
  }

  *laddrs = pos;

  return 0;
}

static int kaapi_mem_map_find
(
 kaapi_mem_map_t* map,
 unsigned int memid,
 kaapi_mem_laddr_t laddr,
 kaapi_mem_laddrs_t** laddrs
)
{
  /* laddr a host local address */

  kaapi_mem_laddrs_t* pos;

  *laddrs = NULL;

  for (pos = map->head; pos != NULL; pos = pos->next)
  {
    if (pos->laddrs[map->memid] == laddr)
    {
      *laddrs = pos;
      return 0;
    }
  }

  return -1;
}


/* memory maps */

/* todo: move in kaapi_processor_t */
static kaapi_mem_map_t mem_maps[2];

static void init_maps_once(void)
{
  kaapi_mem_map_initialize(&mem_maps[0], 0);
  kaapi_mem_map_initialize(&mem_maps[1], 1);
}


/* get processor memory map */

static kaapi_mem_map_t* kaapi_processor_get_mem_map(kaapi_processor_t* proc)
{
  return &mem_maps[proc->kid];
}

static kaapi_mem_map_t* kaapi_get_self_mem_map(void)
{
  return kaapi_processor_get_mem_map(kaapi_get_current_processor());
}

static unsigned int kaapi_get_self_mem_id(void)
{
  return kaapi_get_current_processor()->kid;
}

static kaapi_mem_map_t* kaapi_mem_get_map(unsigned int memid)
{
  return &mem_maps[memid];
}

static unsigned int kaapi_processor_get_memid(kaapi_processor_t* proc)
{
  /* ok for now, processor should have a memid */
  return proc->kid;
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

/* copy from host to device */

static inline int memcpy_htod
(kaapi_processor_t* proc, CUdeviceptr devptr, void* hostptr, size_t size)
{
  printf("memcpy_htod(%llx, %llx)\n", devptr, hostptr);

#if 0 /* async version */
  const CUresult res = cuMemcpyHtoDAsync
    (devptr, hostptr, size, proc->cuda_proc.stream);
#else
  const CUresult res = cuMemcpyHtoD
    (devptr, hostptr, size);
#endif

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyHToDAsync", res);
    return -1;
  }

  return 0;
}


/* copy from device to host */

static inline int memcpy_dtoh
(kaapi_processor_t* proc, void* hostptr, CUdeviceptr devptr, size_t size)
{
#if 0 /* async version */
  const CUresult res = cuMemcpyDtoHAsync
    (hostptr, devptr, size, proc->cuda_proc.stream);
#else
  const CUresult res = cuMemcpyDtoH
    (hostptr, devptr, size);
#endif

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuMemcpyDToHAsync", res);
    return -1;
  }

  return 0;
}


void kaapi_mem_read_barrier(void* hostptr, size_t size)
{
  /* ensure everything past this point
     has been written to host memory.
     assumed to be called from host.
   */

  kaapi_processor_t* self_proc = kaapi_get_current_processor();
  kaapi_mem_map_t* const self_map = kaapi_get_self_mem_map();
  const unsigned int self_memid = kaapi_get_self_mem_id();

  CUdeviceptr devptr;
  unsigned int memid;
  kaapi_mem_laddrs_t* laddrs;

  /* assume no error */
  kaapi_mem_map_find
    (self_map, KAAPI_MEM_ID_CPU, (kaapi_mem_laddr_t)hostptr, &laddrs);

  for (memid = 0; memid < KAAPI_MEM_ID_MAX; ++memid)
  {
    /* find the first valid non identity mapping */
    if (memid == self_memid)
      continue ;
    if (!kaapi_mem_laddrs_isset(laddrs, memid))
      continue ;

    devptr = (CUdeviceptr)kaapi_mem_laddrs_get(laddrs, memid);
    memcpy_dtoh(self_proc, hostptr, devptr, size);

    /* done */
    break ;
  }
}


/* prepare task args memory */

static void prepare_task
(kaapi_processor_t* proc, kaapi_task_t* task, kaapi_format_t* format)
{
  kaapi_mem_map_t* cpu_map = kaapi_mem_get_map(KAAPI_MEM_ID_CPU);
  kaapi_mem_map_t* gpu_map = kaapi_mem_get_map(KAAPI_MEM_ID_GPU);

  kaapi_access_t* access;
  kaapi_mem_laddrs_t* laddrs;
  CUdeviceptr devptr;
  void* hostptr;
  size_t size;
  unsigned int i;

  /* to remove */
  {
    static unsigned int init_once = 0;
    if (init_once == 0)
    {
      init_once = 1;
      init_maps_once();
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
    hostptr = (void*)access->data;

    /* create a mapping on host if not exist */
    kaapi_mem_map_find_or_insert
      (cpu_map, KAAPI_MEM_ID_CPU, (kaapi_mem_laddr_t)hostptr, &laddrs);

    /* mapping does not yet exist */
    if (!kaapi_mem_laddrs_isset(laddrs, kaapi_processor_get_memid(proc)))
    {
      size = 50000 * sizeof(unsigned int);

      allocate_device_mem(&devptr, size);

      /* host -> gpu mapping */
      kaapi_mem_laddrs_set
	(laddrs, (kaapi_mem_laddr_t)devptr, kaapi_processor_get_memid(proc));

      /* gpu -> gpu mapping */
      kaapi_mem_map_find_or_insert
	(gpu_map, KAAPI_MEM_ID_GPU, (kaapi_mem_laddr_t)devptr, &laddrs);

      /* gpu -> host mapping */
      kaapi_mem_laddrs_set
	(laddrs, (kaapi_mem_laddr_t)hostptr, KAAPI_MEM_ID_CPU);
    }
    else
    {
      devptr = kaapi_mem_laddrs_get(laddrs, kaapi_processor_get_memid(proc));
    }

    /* update param addr */
    access->data = (void*)(uintptr_t)devptr;

    if (KAAPI_ACCESS_IS_READ(mode)) /* R or RW */
    {
      /* todo: size = format->param_size(i); */
      size = 50000 * sizeof(unsigned int);
      devptr = (CUdeviceptr)kaapi_mem_laddrs_get(laddrs, KAAPI_MEM_ID_GPU);
      memcpy_htod(proc, devptr, hostptr, size);
    }
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

/* finalize task args memory */

static void finalize_task
(kaapi_processor_t* proc, kaapi_task_t* task, kaapi_format_t* format)
{
  kaapi_mem_map_t* gpu_map = kaapi_processor_get_mem_map(proc);

  kaapi_access_t* access;
  kaapi_mem_laddrs_t* laddrs;
  CUdeviceptr devptr;
  void* hostptr;
  size_t size;
  unsigned int i;

  for (i = 0; i < format->count_params; ++i)
  {
    const kaapi_access_mode_t mode =
      KAAPI_ACCESS_GET_MODE(format->mode_params[i]);

    if (mode & KAAPI_ACCESS_MODE_V)
      continue ;

    if (!KAAPI_ACCESS_IS_WRITE(mode))
      continue ;

    access = (kaapi_access_t*)((uint8_t*)task->sp + format->off_params[i]);
    devptr = (CUdeviceptr)(uintptr_t)access->data;

    /* assume laddrs and laddrs[CPU] */
    kaapi_mem_map_find
      (gpu_map, (kaapi_mem_laddr_t)devptr, KAAPI_MEM_ID_GPU, &laddrs);
    hostptr = (void*)kaapi_mem_laddrs_get(laddrs, KAAPI_MEM_ID_CPU);
    memcpy_dtoh(proc, hostptr, devptr, size);
  }
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

  kaapi_format_t* format;
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

    format = kaapi_format_resolvebybody(body);
    kaapi_assert_debug(format != NULL);

    prepare_task(proc, pc, format);
    execute_task
      (proc, (cuda_task_body_t)body, pc->sp, (kaapi_thread_t*)thread->sfp);
    synchronize_processor(proc);
    finalize_task(proc, pc, format);
    
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
