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
#include "../../memory/kaapi_mem.h"


/* get processor memory map */

static inline kaapi_mem_map_t* get_proc_mem_map(kaapi_processor_t* proc)
{
  return &proc->mem_map;
}

static inline kaapi_mem_map_t* get_host_mem_map(void)
{
  return get_proc_mem_map(kaapi_all_kprocessors[0]);
}

static inline kaapi_processor_t* get_host_proc(void)
{
  return kaapi_all_kprocessors[0];
}

static inline kaapi_mem_asid_t get_proc_asid(kaapi_processor_t* proc)
{
  return proc->mem_map.asid;
}

static inline kaapi_mem_asid_t get_host_asid(void)
{
  return get_host_mem_map()->asid;
}

static kaapi_processor_t* get_proc_by_asid(kaapi_mem_asid_t asid)
{
  kaapi_processor_t** kproc = kaapi_all_kprocessors;
  size_t count = kaapi_count_kprocessors;

  while (count)
  {
    if ((*kproc)->mem_map.asid == asid)
      return (*kproc);
    --count;
    ++kproc;
  }

  return NULL;
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


/* cuda body prototype */

typedef void (*cuda_task_body_t)(CUstream, void*, kaapi_thread_t*);


/* prepare task args memory */

static void prepare_task
(kaapi_processor_t* proc, void* sp, kaapi_format_t* format)
{
  kaapi_mem_map_t* const host_map = get_host_mem_map();
  kaapi_mem_asid_t const host_asid = host_map->asid;
  kaapi_mem_asid_t const self_asid = get_proc_asid(proc);

  kaapi_access_t* access;
  kaapi_mem_mapping_t* mapping;
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

    access = (kaapi_access_t*)((uint8_t*)sp + format->off_params[i]);
    hostptr = access->data;

    /* get parameter size */
    size = format->get_param_size(format, i, sp);

    /* create a mapping on host if not exist */
    kaapi_mem_map_find_or_insert
      (host_map, (kaapi_mem_addr_t)hostptr, &mapping);

    /* no addr for this asid. allocate remote memory. */
    if (!kaapi_mem_mapping_has_addr(mapping, self_asid))
    {
      allocate_device_mem(&devptr, size);
      kaapi_mem_mapping_set_addr(mapping, self_asid, (kaapi_mem_addr_t)devptr);
      kaapi_mem_mapping_set_dirty(mapping, self_asid);
    }
    else
    {
      devptr = kaapi_mem_mapping_get_addr(mapping, self_asid);
    }

    /* read or readwrite, ensure remote memory valid */
    if (KAAPI_ACCESS_IS_READ(mode))
    {
      if (kaapi_mem_mapping_is_dirty(mapping, self_asid))
      {
	/* find a non dirty addr */
	const kaapi_mem_asid_t valid_asid =
	  kaapi_mem_mapping_get_nondirty_asid(mapping);

	/* dev to dev copy, validate host area first */
	if (valid_asid != host_asid)
	{
	  const kaapi_mem_addr_t raddr =
	    kaapi_mem_mapping_get_addr(mapping, valid_asid);

	  kaapi_processor_t* const rproc = get_proc_by_asid(valid_asid);

	  CUresult res = cuCtxPushCurrent(rproc->cuda_proc.ctx);
	  if (res != CUDA_SUCCESS)
	  { kaapi_cuda_error("cuCtxPushCurrent", res); exit(-1); }

	  printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		 valid_asid, (void*)raddr,
		 0, (void*)hostptr,
		 size);

	  memcpy_dtoh(proc, hostptr, (CUdeviceptr)raddr, size);

	  cuCtxPopCurrent(&rproc->cuda_proc.ctx);

	  kaapi_mem_mapping_clear_dirty(mapping, host_asid);
	}

	printf("memcpy_htod(%u:%p -> %u:%p, %lu)\n",
	       0, (void*)hostptr,
	       self_asid, (void*)(uintptr_t)devptr,
	       size);

	/* copy from host to device */
	memcpy_htod(proc, devptr, hostptr, size);

	/* validate remote memory */
	kaapi_mem_mapping_clear_dirty(mapping, self_asid);
      }
    }

    /* invalidate in other as if written */
    if (KAAPI_ACCESS_IS_WRITE(mode))
    {
      kaapi_mem_mapping_set_all_dirty_except(mapping, self_asid);
    }

    /* update param addr */
    access->data = (void*)(uintptr_t)devptr;
  }
}


/* prepare a task that is to be executed on the cpu of the gpu.
   in this case, tweaking the proc->mem_map.asid is needed since
   it describes on the gpu AS.
 */

static void __attribute__((unused)) prepare_task2
(void* sp, kaapi_format_t* format)
{
  kaapi_mem_map_t* const host_map = get_host_mem_map();
  kaapi_mem_asid_t const self_asid = host_map->asid;

  kaapi_access_t* access;
  kaapi_mem_mapping_t* mapping;
  void* hostptr;
  size_t size;
  unsigned int i;

  for (i = 0; i < format->count_params; ++i)
  {
    const kaapi_access_mode_t mode =
      KAAPI_ACCESS_GET_MODE(format->mode_params[i]);

    if (mode & KAAPI_ACCESS_MODE_V)
      continue ;

    access = (kaapi_access_t*)((uint8_t*)sp + format->off_params[i]);
    hostptr = access->data;

    /* get parameter size */
    size = format->get_param_size(format, i, sp);

    /* create a mapping on host if not exist */
    kaapi_mem_map_find_or_insert
      (host_map, (kaapi_mem_addr_t)hostptr, &mapping);

    /* read or readwrite, ensure remote memory valid */
    if (KAAPI_ACCESS_IS_READ(mode))
    {
      /* self AS mapping is dirty */
      if (kaapi_mem_mapping_is_dirty(mapping, self_asid))
      {
	/* validate the host AS mapping */
	kaapi_mem_synchronize3(mapping, size);
      }
    }

    /* invalidate in other as if written */
    if (KAAPI_ACCESS_IS_WRITE(mode))
    {
      kaapi_mem_mapping_set_all_dirty_except(mapping, self_asid);
    }
  }

}


/* finalize task args memory */

static void __attribute__((unused)) finalize_task
(kaapi_processor_t* proc, void* sp, kaapi_format_t* format)
{
  kaapi_mem_map_t* const host_map = get_host_mem_map();
  const kaapi_mem_asid_t host_asid = host_map->asid;

  kaapi_access_t* access;
  kaapi_mem_mapping_t* mapping;
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

    access = (kaapi_access_t*)((uint8_t*)sp + format->off_params[i]);
    devptr = *kaapi_data(CUdeviceptr, access);

    /* inverted search. assume a mapping exists. */
    kaapi_mem_map_find_inverse
      (host_map, (kaapi_mem_addr_t)devptr, &mapping);
    hostptr = (void*)kaapi_mem_mapping_get_addr(mapping, host_asid);
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


/* cuda device taskbcast body.
   this is the same as kaapi_taskbcast_body
   except it calls a cuda prototyped body
   instead of a cpu one.
 */

static void cuda_taskbcast_body
(CUstream stream, void* sp, kaapi_thread_t* thread)
{
  /* thread[-1]->pc is the executing task (pc of the upper frame) */
  /* kaapi_task_t*          self = thread[-1].pc;*/
  kaapi_taskbcast_arg_t* arg  = sp;
  kaapi_com_t* comlist;
  int i;

  /* the lookup could be avoided by wrapping differently */
  kaapi_task_body_t original_body = arg->common.original_body;
  if (original_body != NULL)
  {
    /* format and cuda_body known to be non null */
    kaapi_format_t* const format = kaapi_format_resolvebybody(original_body);
    cuda_task_body_t cuda_body = (cuda_task_body_t)
      format->entrypoint[KAAPI_PROC_TYPE_CUDA];
    cuda_body(stream, arg->common.original_sp, thread);
  }

  /* write memory barrier to ensure that other threads will view the data produced */
  kaapi_mem_barrier();

  /* signal all readers */
  comlist = &arg->head;
  while(comlist != 0) 
  {
    for (i=0; i<comlist->size; ++i)
    {
      kaapi_task_t* task = comlist->entry[i].task;
      kaapi_assert( (task->ebody == kaapi_taskrecv_body) || (task->ebody == kaapi_taskbcast_body) );
      kaapi_taskrecv_arg_t* argrecv = (kaapi_taskrecv_arg_t*)task->sp;
      
      void* newsp;
      kaapi_task_body_t newbody;
      if (task->ebody == kaapi_taskrecv_body)
      {
        newbody = argrecv->original_body;
        newsp   = argrecv->original_sp;
      }
      else 
      {
        newbody = task->ebody;
        newsp   = task->sp;
      }

#if 1 /* --- copy from local to remote memory --- */
      {
	/* a write task is done, thus has to validate the remote
	   mapping before signaling the remote task.
	   we assume the remote task will be executed
	   on CPU, since this is the only case this function is
	   needed. Otherwise, it is executed on gpu and the
	   prepare_task function takes care of validating the
	   data. Futhermore, in the case of a device to device
	   copy, a valid copy is needed on the host. Thus, the
	   only case this function does useless work (ie. useless
	   copy to the host) is when a task is executed on the gpu
	   the data is already present on.
	 */

      kaapi_processor_t* const lproc = kaapi_get_current_processor();
      kaapi_processor_t* const rproc = get_host_proc();

      kaapi_mem_map_t* const host_map = get_host_mem_map();
      const kaapi_mem_asid_t rasid = rproc->mem_map.asid;

      /* find the remote address mapping */
      kaapi_mem_addr_t host_ptr = (kaapi_mem_addr_t)comlist->entry[i].addr;
      kaapi_mem_mapping_t* mapping;
      kaapi_mem_map_find(host_map, host_ptr, &mapping);

      /* validate the dirty mapping */
      if (kaapi_mem_mapping_is_dirty(mapping, rasid))
      {
	const kaapi_mem_asid_t lasid = lproc->mem_map.asid;
	kaapi_mem_addr_t laddr = kaapi_mem_mapping_get_addr(mapping, lasid);

	/* size to copy is cached in each comlist entries */
	const size_t size = comlist->entry[i].size;

	/* actual copy, assume lproc->proc_type == KAAPI_PROC_TYPE_CUDA */
	if (rproc->proc_type == KAAPI_PROC_TYPE_CUDA)
	{
#if 0 /* we dont have the proc */
	  const kaapi_mem_asid_t host_asid = host_map->asid;
	  kaapi_mem_addr_t raddr;

	  /* dev to dev copy, validate host area first */
	  if (kaapi_mem_mapping_is_dirty(mapping, host_asid))
	  {
	    memcpy_dtoh(rproc, (void*)host_ptr, (CUdeviceptr)laddr, size);

	    printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		   lasid, (void*)laddr,
		   0, (void*)host_ptr,
		   size);

	    kaapi_mem_mapping_clear_dirty(mapping, host_asid);
	  }

	  /* map raddr in rasid if not yet done */
	  if (kaapi_mem_mapping_has_addr(mapping, rasid) == 0)
	  {
	    CUdeviceptr devptr;
	    allocate_device_mem(&devptr, size);
	    kaapi_mem_mapping_set_addr
	      (mapping, rasid, (kaapi_mem_addr_t)devptr);
	    kaapi_mem_mapping_set_dirty(mapping, rasid);
	  }

	  raddr = kaapi_mem_mapping_get_addr(mapping, rasid);

	  /* copy from host to device */
	  CUresult res = cuCtxPushCurrent(rproc->cuda_proc.ctx);
	  if (res != CUDA_SUCCESS)
	  { kaapi_cuda_error("cuCtxPushCurrent", res); exit(-1); }
	  memcpy_htod(rproc, (CUdeviceptr)raddr, (void*)host_ptr, size);
	  cuCtxPopCurrent(&rproc->cuda_proc.ctx);
#endif
	}
	else
	{
	  /* normal device to host copy */
	  memcpy_dtoh(rproc, (void*)host_ptr, (CUdeviceptr)laddr, size);

	  printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		 lasid, (void*)laddr,
		 0, (void*)host_ptr,
		 size);
	}

	/* valdiate remote mapping */
	kaapi_mem_mapping_clear_dirty(mapping, rasid);
      }

      } /* --- copy from local to remote memory --- */
#endif
      
      if (kaapi_threadgroup_decrcounter(argrecv) ==0)
      {
        /* task becomes ready */        
        task->sp = newsp;
        kaapi_task_setextrabody(task, newbody);

        /* see code in kaapi_taskwrite_body */
        if (task->pad != 0) 
        {
          kaapi_wc_structure_t* wcs = (kaapi_wc_structure_t*)task->pad;
          /* remove it from suspended queue */
          if (wcs->wccell !=0)
          {
            kaapi_thread_context_t* kthread = kaapi_wsqueuectxt_steal_cell( wcs->wclist, wcs->wccell );
            if (kthread !=0) 
            {

#if 0     /* push on the owner of the bcast */
              kaapi_processor_t* kproc = kaapi_get_current_processor();
#else     /* push on the owner of the suspended thread */
              kaapi_processor_t* kproc = kthread->proc;
#endif
              if (!kaapi_thread_hasaffinity(kthread->affinity, kproc->kid))
              {
                /* find the first kid with affinity */
                kaapi_processor_id_t kid;
                for ( kid=0; kid<kaapi_count_kprocessors; ++kid)
                  if (kaapi_thread_hasaffinity( kthread->affinity, kid)) break;
                kaapi_assert_debug( kid < kaapi_count_kprocessors );
                kproc = kaapi_all_kprocessors[ kid ];
              }

              /* move the thread in the ready list of the victim processor */
              kaapi_sched_lock( kproc );
              kaapi_task_setbody(task, newbody );
              kaapi_sched_pushready( kproc, kthread );

              /* bcast will activate a suspended thread */
              kaapi_sched_unlock( kproc );
            } else 
              kaapi_task_setbody(task, newbody);
          } else { /* wccell == 0 */
            kaapi_thread_context_t* kthread = wcs->thread;
            kaapi_processor_t* kproc = kthread->proc;

            kaapi_sched_lock( kproc );
            kaapi_task_setbody(task, newbody );
            kaapi_sched_pushready( kproc, kthread );
            kaapi_sched_unlock( kproc );
          }
        }
        else {
          /* thread is not suspended... */        
          /* may activate the task */
          kaapi_task_setbody(task, newbody);
        }
      }
    }
    comlist = comlist->next;
  }
}


/* do nothing, as in the original kaapi_taskrecv_body
 */

static void cuda_taskrecv_body
(CUstream stream, void* sp, kaapi_thread_t* thread)
{
}


/* unwrap a wrapped task
 */

static inline void unwrap_task
(cuda_task_body_t* cuda_body, kaapi_task_body_t* original_body, void** original_sp)
{
  /* cuda_body will point the adapted body if task is wrapped.
     original_body the current body. updated to point the original body.
     original_sp the current sp. updated to point the original sp.
   */

  if (*original_body == kaapi_taskbcast_body)
  {
    kaapi_taskbcast_arg_t* const arg = (kaapi_taskbcast_arg_t*)*original_sp;
    *original_body = arg->common.original_body;
    *original_sp = arg->common.original_sp;
    *cuda_body = cuda_taskbcast_body;
  }
  else if (*original_body == kaapi_taskrecv_body)
  {
    kaapi_taskrecv_arg_t* const arg = (kaapi_taskrecv_arg_t*)*original_sp;
    *original_body = arg->original_body;
    *original_sp = arg->original_sp;
    *cuda_body = cuda_taskrecv_body;
  }
  /* else, nonwrapped task */
}


#if 0 /* todo, remove */
static const char* get_body_name(kaapi_task_body_t body)
{
  const char* name = "unknown";

#define NAME_CASE(__body) if (body == __body) { name = #__body; }

  NAME_CASE(kaapi_nop_body);
  NAME_CASE(kaapi_taskstartup_body);
  NAME_CASE(kaapi_taskmain_body);
  NAME_CASE(kaapi_exec_body);
  NAME_CASE(kaapi_suspend_body);
  NAME_CASE(kaapi_taskbcast_body);
  NAME_CASE(kaapi_taskrecv_body);
  NAME_CASE(kaapi_tasksig_body);
  NAME_CASE(kaapi_tasksteal_body);
  NAME_CASE(kaapi_aftersteal_body);
  NAME_CASE(kaapi_tasksignalend_body);

  return name;
}
#endif /* todo, remove */


/* exported */

int kaapi_cuda_execframe(kaapi_thread_context_t* thread)
{
  kaapi_processor_t* const proc = thread->proc;
  CUresult res;

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

    /* handle wrapped task */
    kaapi_task_body_t original_body = body;
    void* original_sp = pc->sp;
    cuda_task_body_t cuda_body = NULL;

    unwrap_task(&cuda_body, &original_body, &original_sp);

    format = kaapi_format_resolvebybody(original_body);
    if ((format != NULL) && (format->entrypoint[KAAPI_PROC_TYPE_CUDA]))
    {
      /* the context is saved then restore during
	 the whole task execution. not doing so would
	 make it non floating, preventing another thread
	 to use it (ie. for kaapi_mem_synchronize2)
       */

      res = cuCtxPushCurrent(proc->cuda_proc.ctx);
      if (res == CUDA_SUCCESS)
      {
	/* this is not a wrapped task */
	if (cuda_body == NULL)
	  cuda_body = (cuda_task_body_t)format->entrypoint[KAAPI_PROC_TYPE_CUDA];

	/* prepare task memory */
	prepare_task(proc, original_sp, format);

	/* execute the cuda body */
	cuda_body(proc->cuda_proc.stream, pc->sp, (kaapi_thread_t*)thread->sfp);

	/* synchronize processor execution */
	synchronize_processor(proc);

	cuCtxPopCurrent(&proc->cuda_proc.ctx);
      }
      /* todo, remove */
      else { printf("cuCtxPushCurrent() error\n"); exit(-1); }
      /* todo, remove */
    }
    else
    {
      /* assume body points to the default cpu implementation */
      kaapi_assert_debug(pc == thread->sfp[-1].pc);

#if 0 /* todo, is this needed? */
      if (format != NULL)
      {
	/* prepare task arg for host */
	cuCtxPushCurrent(proc->cuda_proc.ctx);
	prepare_task2(original_sp, format);
	cuCtxPopCurrent(&proc->cuda_proc.ctx);
      }
#endif /* todo */

      body(pc->sp, (kaapi_thread_t*)thread->sfp);
    }

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
