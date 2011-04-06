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


/* exported */
kaapi_processor_t* get_proc_by_asid(kaapi_mem_asid_t asid)
{
  /* todo, asid_to_kproc[asid] -> kproc */

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


/* get processor memory map */

kaapi_mem_map_t* get_proc_mem_map(kaapi_processor_t* proc)
{
  return &proc->mem_map;
}

kaapi_mem_map_t* get_host_mem_map(void)
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

typedef void (*cuda_task_body_t)(void*, CUstream);


/* access data wrappers */

static inline kaapi_access_t* get_access_at
(kaapi_format_t* f, unsigned int i, void* p)
{ return (kaapi_access_t*)f->get_off_param(f, i, p); }

static inline void* get_access_data_at
(kaapi_format_t* f, unsigned int i, void* p)
{ return get_access_at(f, i, p)->data; }

static inline void set_access_data_at
(kaapi_format_t* f, unsigned int i, void* p, void* d)
{ get_access_at(f, i, p)->data = d; }


/* retrieve the ith parameter size */

static size_t get_size_param
(kaapi_format_t* f, unsigned int i, void* p)
{
  kaapi_memory_view_t kmv = f->get_view_param(f, i, p);
  const size_t size = kaapi_memory_view_size(&kmv);
  return size;
}


/* prepare task args memory */

static void prepare_task
(kaapi_processor_t* proc, void* sp, kaapi_format_t* format)
{
  kaapi_mem_map_t* const host_map = get_host_mem_map();
  kaapi_mem_asid_t const host_asid = host_map->asid;
  kaapi_mem_asid_t const self_asid = get_proc_asid(proc);

  const size_t param_count = format->get_count_params(format, sp);

  kaapi_access_t access;
  kaapi_mem_mapping_t* mapping;
  CUdeviceptr devptr;
  void* hostptr;
  size_t size;
  unsigned int i;

  for (i = 0; i < param_count; ++i)
  {
    const kaapi_access_mode_t mode = format->get_mode_param(format, i, sp);
    if (mode & KAAPI_ACCESS_MODE_V) continue ;

    /* we use get_access_param since the user may need
       to update the actual addr value. after that, we
       handle access directly in memory */
    access = format->get_access_param(format, i, sp);
    hostptr = access.data;

    /* get parameter size */
    size = get_size_param(format, i, sp);

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

	  pthread_mutex_lock(&rproc->cuda_proc.ctx_lock);
	  CUresult res = cuCtxPushCurrent(rproc->cuda_proc.ctx);
	  if (res != CUDA_SUCCESS)
	  { kaapi_cuda_error("cuCtxPushCurrent", res); exit(-1); }

#if defined (KAAPI_DEBUG)
	  printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		 valid_asid, (void*)raddr, 0, (void*)hostptr, size);
#endif

	  memcpy_dtoh(proc, hostptr, (CUdeviceptr)raddr, size);

	  cuCtxPopCurrent(&rproc->cuda_proc.ctx);
	  pthread_mutex_unlock(&rproc->cuda_proc.ctx_lock);

	  kaapi_mem_mapping_clear_dirty(mapping, host_asid);
	}

#if defined (KAAPI_DEBUG)
	printf("memcpy_htod(%u:%p -> %u:%p, %lu)\n",
	       0, (void*)hostptr, self_asid, (void*)(uintptr_t)devptr, size);
#endif

	/* copy from host to device */
	memcpy_htod(proc, devptr, hostptr, size);

	/* validate remote memory */
	kaapi_mem_mapping_clear_dirty(mapping, self_asid);
      }
    }

    /* invalidate in other as if written */
    if (KAAPI_ACCESS_IS_WRITE(mode))
      kaapi_mem_mapping_set_all_dirty_except(mapping, self_asid);

    /* update param addr (todo: use cached access) */
    access.data = (void*)(uintptr_t)devptr;
    format->set_access_param(format, i, sp, &access);
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

  for (i = 0; i < format->_count_params; ++i)
  {
    const kaapi_access_mode_t mode =
      KAAPI_ACCESS_GET_MODE(format->_mode_params[i]);

    if (mode & KAAPI_ACCESS_MODE_V)
      continue ;

    access = (kaapi_access_t*)((uint8_t*)sp + format->_off_params[i]);
    hostptr = access->data;

    /* get parameter size */
    size = get_size_param(format, i, sp);

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


/* sycnhronize write parameters host memory */

static inline unsigned int access_is_readonly
(kaapi_access_mode_t m)
{ return KAAPI_ACCESS_IS_READ(m) && !KAAPI_ACCESS_IS_WRITE(m); }

static void finalize_task
(kaapi_processor_t* proc, void* sp, kaapi_format_t* format)
{
  const size_t param_count = format->get_count_params(format, sp);

  kaapi_mem_addr_t devptr;
  size_t size;
  unsigned int i;

  for (i = 0; i < param_count; ++i)
  {
    kaapi_access_t access;

    const kaapi_access_mode_t mode = format->get_mode_param(format, i, sp);

    if (mode & KAAPI_ACCESS_MODE_V)
      continue ;

    access = format->get_access_param(format, i, sp);

    /* deallocate remote mem for readonly mappings */
    if (access_is_readonly(mode))
    {
      const kaapi_mem_asid_t self_asid = proc->mem_map.asid;
      kaapi_mem_map_t* const host_map = get_host_mem_map();

      kaapi_mem_mapping_t* mapping;
      devptr = (kaapi_mem_addr_t)access.data;

      kaapi_mem_map_find_with_asid
	(host_map, devptr, self_asid, &mapping);

      kaapi_assert_debug(mapping);
      kaapi_assert_debug(kaapi_mem_mapping_has_addr(mapping, self_asid));

      kaapi_mem_mapping_clear_addr(mapping, self_asid);

      /* context already acquired */
      free_device_mem(devptr);

      continue ;
    }

    if (!KAAPI_ACCESS_IS_WRITE(mode))
      continue ;

    /* assume every KAAPI_ACCESS_MODE_W has been dirtied */

    /* get data, size */
    devptr = (kaapi_mem_addr_t)access.data;
    size = get_size_param(format, i, sp);

#if defined(KAAPI_DEBUG)
    printf("kaapi_mem_synchronize(%lx, %lu)\n", (uintptr_t)devptr, size);
#endif

    /* sync host memory */
    kaapi_mem_synchronize(devptr, size);
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


#if 0 /* unused tasks */

/* cuda device taskbcast body.
   this is the same as kaapi_taskbcast_body
   except it calls a cuda prototyped body
   instead of a cpu one.
 */

static void cuda_taskbcast_body
(void* sp, CUstream stream)
{
#if 0 /* todo, wrapped task */

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
    cuda_body(arg->common.original_sp, stream);
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
      kaapi_taskrecv_arg_t* argrecv = (kaapi_taskrecv_arg_t*)task->sp;
      
      void* newsp;
      kaapi_task_body_t newbody;
      if (kaapi_task_getbody(task) == kaapi_taskrecv_body)
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

#if defined (KAAPI_DEBUG)
	    printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		   lasid, (void*)laddr, 0, (void*)host_ptr, size);
#endif

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
	  pthread_mutex_lock(&rproc->cuda_proc.ctx_lock);
	  CUresult res = cuCtxPushCurrent(rproc->cuda_proc.ctx);
	  if (res != CUDA_SUCCESS)
	  { kaapi_cuda_error("cuCtxPushCurrent", res); exit(-1); }
	  memcpy_htod(rproc, (CUdeviceptr)raddr, (void*)host_ptr, size);
	  cuCtxPopCurrent(&rproc->cuda_proc.ctx);
	  pthread_mutex_unlock(&rproc->cuda_proc.ctx_lock);
#endif
	}
	else
	{
	  /* normal device to host copy */
	  memcpy_dtoh(rproc, (void*)host_ptr, (CUdeviceptr)laddr, size);

#if defined (KAAPI_DEBUG)
	  printf("memcpy_dtoh(%u:%p -> %u:%p, %lu)\n",
		 lasid, (void*)laddr, 0, (void*)host_ptr, size);
#endif
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
#endif /* todo, wrapped task */

}

/* do nothing, as in the original kaapi_taskrecv_body
 */

static void cuda_taskrecv_body
(void* sp, CUstream stream)
{
}

#endif /* unused tasks */

/* unwrap a wrapped task
 */

static inline void unwrap_task
(cuda_task_body_t* cuda_body, kaapi_task_body_t* original_body, void** original_sp)
{
  /* cuda_body will point the adapted body if task is wrapped.
     original_body the current body. updated to point the original body.
     original_sp the current sp. updated to point the original sp.
   */

#if 0
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
#endif
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
  NAME_CASE(kaapi_tasksteal_body);
  NAME_CASE(kaapi_aftersteal_body);

  return name;
}
#endif /* todo, remove */

/* exported */
#if ((KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD) || (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD))
int kaapi_cuda_execframe(kaapi_thread_context_t* thread)
{
  kaapi_processor_t* const proc = thread->proc;

  kaapi_task_t*              pc;
  kaapi_frame_t*             fp;
  kaapi_task_body_t          body;
  uintptr_t	             state;
  kaapi_frame_t*             eframe = thread->esfp;
#if defined(KAAPI_USE_PERFCOUNTER)
  uint32_t                   cnt_tasks = 0;
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

  pc = fp->pc;

  /* stack of task growth down ! */
  while (pc != fp->sp)
  {
    kaapi_assert_debug( pc > fp->sp );

#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD)

#if 0 /* unimplemented */
    body = pc->body;

#if (__SIZEOF_POINTER__ == 4)
    state = pc->state;
#else
    state = kaapi_task_body2state(body);
#endif

    kaapi_assert_debug( body != kaapi_exec_body);
    pc->body = kaapi_exec_body;
    /* task execution */
    kaapi_assert_debug(pc == thread->sfp[-1].pc);
    kaapi_assert_debug( kaapi_isvalid_body( body ) );

    /* here... */
    body( pc->sp, (kaapi_thread_t*)thread->sfp );      
#endif /* unimplemented */

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)

    state = kaapi_task_orstate( pc, KAAPI_MASK_BODY_EXEC );

#if (__SIZEOF_POINTER__ == 4)
    body = pc->body;
#else
    body = kaapi_task_state2body( state );
#endif /* __SIZEOF_POINTER__ */

#endif /* KAAPI_USE_EXECTASK_METHOD */

    if (likely( kaapi_task_state_isnormal(state) ))
    {
      /* task execution */

      kaapi_format_t* format;

      kaapi_assert_debug(pc == thread->sfp[-1].pc);

      /* look for cuda task */
      format = kaapi_format_resolvebybody(body);
      if ((format != NULL) && format->entrypoint[KAAPI_PROC_TYPE_CUDA])
      {
	/* the context is saved then restore during
	 the whole task execution. not doing so would
	 make it non floating, preventing another thread
	 to use it (ie. for kaapi_mem_synchronize2)
	 */

	CUresult res;

	pthread_mutex_lock(&proc->cuda_proc.ctx_lock);
	res = cuCtxPushCurrent(proc->cuda_proc.ctx);
	if (res == CUDA_SUCCESS)
	{
	  const cuda_task_body_t cuda_body =
	    (cuda_task_body_t)format->entrypoint[KAAPI_PROC_TYPE_CUDA];

	  /* prepare task memory */
	  prepare_task(proc, pc->sp, format);

	  /* execute the cuda body */
	  cuda_body(pc->sp, proc->cuda_proc.stream);

	  /* synchronize processor execution */
	  synchronize_processor(proc);

	  /* revalidate the host memory */
	  finalize_task(proc, pc->sp, format);

	  cuCtxPopCurrent(&proc->cuda_proc.ctx);
	}
#if defined (KAAPI_DEBUG)
	else { printf("cuCtxPushCurrent() error\n"); exit(-1); }
#endif

	pthread_mutex_unlock(&proc->cuda_proc.ctx_lock);
      }
      else /* format == NULL || entry[cuda] == NULL */
      {
	body( pc->sp, (kaapi_thread_t*)thread->sfp );
      }
    }
    else
    {
      /* It is a special task: it means that before atomic or update, the body
         has already one of the special flag set (either exec, either suspend).
         Test the following case with THIS (!) order :
         - kaapi_task_body_isaftersteal(body) -> call aftersteal body
         - kaapi_task_body_issteal(body) -> error
         - else it means that the task has been executed by a thief, but it 
         does not require aftersteal body to merge results.
      */
      if ( kaapi_task_state_isaftersteal( state ) )
      {
        /* means that task has been steal & not yet terminated due
           to some merge to do
        */
        kaapi_assert_debug( kaapi_task_state_issteal( state ) );
        kaapi_aftersteal_body( pc->sp, (kaapi_thread_t*)thread->sfp );      
      }
      else if ( kaapi_task_state_isterm( state ) ){
        /* means that task has been steal */
        kaapi_assert_debug( kaapi_task_state_issteal( state ) );
      }
      else if ( kaapi_task_state_issteal( state ) ) /* but not terminate ! so swap */
      {
        goto error_swap_body;
      }
      else {
        kaapi_assert_debug(0);
      }
    }
#if defined(KAAPI_DEBUG)
    const uintptr_t debug_state = kaapi_task_orstate(pc, KAAPI_MASK_BODY_TERM );
    kaapi_assert_debug( !kaapi_task_state_isterm(debug_state) || (kaapi_task_state_isterm(debug_state) && kaapi_task_state_issteal(debug_state))  );
    kaapi_assert_debug( kaapi_task_state_isexec(debug_state) );
#endif    

#if defined(KAAPI_USE_PERFCOUNTER)
    ++cnt_tasks;
#endif

    /* post execution: new tasks created ??? */
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

    /* next task to execute, store pc in memory */
    fp->pc = --pc;
    
    kaapi_writemem_barrier();
  } /* end of the loop */

  kaapi_assert_debug( fp >= eframe);
  kaapi_assert_debug( fp->pc == fp->sp );

  if (fp >= eframe)
  {
#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_SEQ_METHOD)
    while (fp > eframe) 
    {
      --fp;
      /* pop dummy frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD)
    /* here it's a pop of frame: we lock the thread */
    kaapi_sched_lock(&thread->proc->lock);
    while (fp > eframe) 
    {
      --fp;

      /* pop dummy frame */
      --fp->pc;
      if (fp->pc > fp->sp)
      {
        kaapi_sched_unlock(&thread->proc->lock);
        thread->sfp = fp;
        goto push_frame; /* remains work do do */
      }
    } 
    fp = eframe;
    fp->sp = fp->pc;

    kaapi_sched_unlock(&thread->proc->lock);
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


#if (KAAPI_USE_EXECTASK_METHOD == KAAPI_CAS_METHOD) 
error_swap_body:
  kaapi_assert_debug(thread->sfp- fp == 1);
  /* implicityly pop the dummy frame */
  thread->sfp = fp;
#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif
  return EWOULDBLOCK;
#endif

#if defined(KAAPI_USE_PERFCOUNTER)
  KAAPI_PERF_REG(thread->proc, KAAPI_PERF_ID_TASKS) += cnt_tasks;
  cnt_tasks = 0;
#endif

  /* here back track the kaapi_thread_execframe until go out */
  return 0;
}
#elif (KAAPI_USE_EXECTASK_METHOD == KAAPI_THE_METHOD)
int kaapi_thread_execframe( kaapi_thread_context_t* thread )
{
  return 0;
}
#endif /* KAAPI_EXEC_METHOD */


#if 0 /* OLD_UNUSED */

int kaapi_cuda_exectask
(kaapi_thread_context_t* thread, void* data, kaapi_format_t* format)
{
  kaapi_processor_t* const kproc = thread->proc;
  int res = -1;

  printf("%s\n", __FUNCTION__);

  pthread_mutex_lock(&kproc->cuda_proc.ctx_lock);

  if (cuCtxPushCurrent(kproc->cuda_proc.ctx) == CUDA_SUCCESS)
  {
    const cuda_task_body_t cuda_body = (cuda_task_body_t)
      format->entrypoint[KAAPI_PROC_TYPE_CUDA];

    /* already prepared in case of partitioning */
    if (thread->the_thgrp == NULL)
      prepare_task(kproc, data, format);

    cuda_body(data, kproc->cuda_proc.stream);
    synchronize_processor(kproc);

    if (thread->the_thgrp == NULL)
      finalize_task(kproc, data, format);

    cuCtxPopCurrent(&kproc->cuda_proc.ctx);
    
    /* success */
    res = 0;
  }
#if defined (KAAPI_DEBUG)
  else
  { kaapi_cuda_error("cuCtxPushCurrent", res); exit(-1); }
#endif

  pthread_mutex_unlock(&kproc->cuda_proc.ctx_lock);

  return res;
}

#elif 0 /* NEW_WIP */

#include "kaapi_tasklist.h"

int kaapi_cuda_exectask
(kaapi_thread_context_t* thread, void* sp, kaapi_format_t* format)
{
  /* todo: refer to kaapi_thread_checkdeps.c
     . faire moteur basique: donnees ne sont pas sur gpu, pas
     besoin d une map. un for + check du mode suffi. pas d overlapping,
     inutile. les fonctions de copy + cast sont dispo dans le moteur
     d execution asynchrone.
     
     . make deps computing lighter: can be assumed the data
     are not on the gpu.
     . push an activation link
   */

  /* build a tasklist turn and call execframe_tasklist */

  size_t param_count;
  unsigned int counter;
  size_t i;
  int res;
  int prev_state;
  kaapi_frame_t prev_frame;
  kaapi_move_arg_t* ma;
  kaapi_taskdescr_t* td;
  kaapi_tasklist_t tl;
  kaapi_access_t access;
  kaapi_access_mode_t mode;
  kaapi_metadata_info_t* mdi;
  kaapi_memory_view_t view;

  /* make as unstealable and save states */
  kaapi_sched_lock(&thread->proc->lock);
  prev_state = thread->unstealable;
  thread->unstealable = 1;
  kaapi_sched_unlock(&thread->proc->lock);
  kaapi_thread_save_frame((kaapi_thread_t*)thread->sfp, &prev_frame);

  /* initialize the thread ready tasklist */
  res = kaapi_tasklist_init(&tl);
  kaapi_assert_debug(res == 0);
  thread->sfp->tasklist = &tl;

  /* allocate move tasks */
  param_count = format->get_count_params(format, sp);
  kaapi_assert_debug(param_count <= 32);
  for (counter = 0, i = 0; i < param_count; ++i)
  {
    mode = format->get_mode_param(format, i, sp);
    if (mode & KAAPI_ACCESS_MODE_V) continue ;

    td = kaapi_tasklist_allocate_td(tl, task);
    kaapi_assert_debug(td != NULL);

    /* it is an access */
    access = kaapi_format_get_access_param(format, i, sp);
    kaapi_assert_debug(access != NULL);

    mdi = kaapi_mem_findinsert_metadata(access.data);
    kaapi_assert_debug(mdi != NULL);

    if (!_kaapi_metadata_info_is_valid(mdi, thread->asid))
    {
      view = kaapi_format_get_view_param(format, i, task->sp);
      _kaapi_metadata_info_bind_data(mdi, thread->asid, access.data, &view);
      mdi->version[0] = kaapi_thread_newversion
	(mdi, thread->asid, access.data, &view);
    }

#if 0 /* unused date */
    kaapi_thread_computeready_date(mdi->version[0], td, mode);
#endif

    /* change the data in the task by the handle */
    access.data = version->handle;
    kaapi_format_set_access_param(task_fmt, i, task->sp, &access);

    /* push a move task */
    ma = kaapi_task_pushdata(sizeof(kaapi_move_arg_t));
    kaapi_task_init(kaapi_thread_toptask(thread), kaapi_taskmove_body, ma);

    ma->src_data->ptr.asid = 0; /* assume cpu asid */
    ma->src_data->ptr.ptr = access.data;
    ma->view = view;
    ma->mdi = mdi;

    ma->dest->ptr.ptr = (uintptr_t)NULL;
    ma->dest->ptr.asid = 0;
    ma->dest->view.type = -1;
    ma->dest->mdi = NULL;

    td = kaapi_tasklist_allocate_td(tl, mt);
    kaapi_assert_debug(td != NULL);

    kaapi_tasklist_pushback_ready(tl, td);

    /* incr activation counter */
    ++counter;
  }

  /* initialize the task descriptor */
  td->counter = counter;
  td->fmt = format;

  /* call the tasklist executor */
  res = kaapi_cuda_execframe_tasklist(thread);
  if (res == -1) goto on_error;
  kaapi_assert_debug(res == 0);

  /* fetch back and finalize data */
  kaapi_memory_synchronize();

 on_error:
  /* restore previous states */
  kaapi_thread_restore_frame((kaapi_thread_t*)thread->sfp, &prev_frame);
  kaapi_sched_lock(&thread->proc->lock);
  prev_state = thread->unstealable;
  thread->unstealable = 1;
  kaapi_sched_unlock(&thread->proc->lock);

  /* release tasklist */
  kaapi_tasklist_destroy(tl);

  return res;
}

#else /* NEW_WIP2 */

static int cast_memory_view
(kaapi_memory_view_t* dview, const kaapi_memory_view_t* sview)
{
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

enum copy_direction{ DIR_HTOD, DIR_DTOH };

static int viewcopy_xtox
(
 uintptr_t dptr, const kaapi_memory_view_t* dview,
 uintptr_t sptr, const kaapi_memory_view_t* sview,
 enum copy_direction dir
)
{
  /* synchronous copy from host to device.
     dptr, dview the destnation pair.
     sptr, sview the source pair.
     return -1 on failure.
   */

  CUresult res = CUDA_SUCCESS;
  size_t size;

  kaapi_assert_debug(sview->type == dview->type);
  kaapi_assert_debug(sview->wordsize == dview->wordsize);

  switch (dview->type)
  {
  case KAAPI_MEMORY_VIEW_1D:
    {
      size = dview->size[0] * dview->wordsize;

    view_1d_case:
      if (dir == DIR_HTOD)
	res = cuMemcpyHtoD((CUdeviceptr)dptr, (const void*)sptr, size);
      else
	res = cuMemcpyDtoH((void*)dptr, (CUdeviceptr)sptr, size);

      break ;
    }

  case KAAPI_MEMORY_VIEW_2D:
    {
      /* contiguous case, dont use 2D */
      if (sview->size[1] == sview->lda)
      {
	/* update size and go to 1d case */
	size = dview->size[0] * dview->size[1] * dview->wordsize;
	goto view_1d_case;
      }
      else /* non contiguous */
      {
	CUDA_MEMCPY2D m;

	if (dir == DIR_HTOD)
	{
	  m.srcMemoryType = CU_MEMORYTYPE_HOST;
	  m.srcHost = (void*)sptr;
	  m.srcPitch = sview->lda * sview->wordsize;

	  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	  m.dstDevice = (CUdeviceptr)dptr;
	  m.dstPitch = sview->size[1] * sview->wordsize;
	}
	else /* if (dir == DIR_DTOH) */
	{
	  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	  m.srcDevice = (CUdeviceptr)sptr;
	  m.srcPitch = sview->size[1] * sview->wordsize;

	  m.dstMemoryType = CU_MEMORYTYPE_HOST;
	  m.dstHost = (void*)dptr;
	  m.dstPitch = sview->lda * sview->wordsize;
	}

	m.srcXInBytes = 0;
	m.srcY = 0;
	m.dstXInBytes = 0;
	m.dstY = 0;

	m.WidthInBytes = sview->size[1] * sview->wordsize;
	m.Height = sview->size[0];

	res = cuMemcpy2D(&m);
	if (res != CUDA_SUCCESS) goto on_error;
      }

      break ;
    }

    /* not implemented */
  default:
    {
      kaapi_assert(0);
      goto on_error;
      break ;
    }
  }

  /* success */
  return 0;

 on_error:
  kaapi_cuda_error("cuMemcpyXtoX", res);
  return -1;
}

static inline int viewcopy_htod
(
 CUdeviceptr dptr, const kaapi_memory_view_t* dview,
 const void* sptr, const kaapi_memory_view_t* sview
)
{
  return viewcopy_xtox
    ((uintptr_t)dptr, dview, (uintptr_t)sptr, sview, DIR_HTOD);
}

static inline int viewcopy_dtoh
(
 const void* dptr, const kaapi_memory_view_t* dview,
 CUdeviceptr sptr, const kaapi_memory_view_t* sview
)
{
  return viewcopy_xtox
    ((uintptr_t)dptr, dview, (uintptr_t)sptr, sview, DIR_DTOH);
}

int kaapi_cuda_exectask
(kaapi_thread_context_t* thread, void* sp, kaapi_format_t* format)
{
  /* simple synchronous executor. a copy is made for each task
     argument, no matter if it is already present on the device.
   */

  kaapi_processor_t* const proc = thread->proc;

  int error;
  kaapi_access_mode_t mode;
  kaapi_access_t access;
  CUresult res;
  CUdeviceptr devptr;
  void* hostptr;
  size_t param_count;
  size_t alloc_count;
  size_t i;
  cuda_task_body_t body;
  kaapi_memory_view_t* sview;
  kaapi_memory_view_t dview;

  kaapi_memory_view_t saved_views[32];
  void* saved_hostptrs[32];
  CUdeviceptr devptrs[32];

  /* assume error */
  error = -1;

  /* track the device allocation count */
  alloc_count = 0;

  /* set default context */
  res = cuCtxPushCurrent(proc->cuda_proc.ctx);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuCtxPushCurrent", res);
    goto on_error;
  }

  /* send read params to device */
  param_count = format->get_count_params(format, sp);
  kaapi_assert_debug(param_count <= 32);
  for (i = 0; i < param_count; ++i)
  {
    mode = format->get_mode_param(format, i, sp);
    if (mode & KAAPI_ACCESS_MODE_V) continue ;

    access = format->get_access_param(format, i, sp);

    /* capture hostptr, view */
    saved_hostptrs[i] = access.data;
    saved_views[i] = format->get_view_param(format, i, sp);

    hostptr = saved_hostptrs[i];
    sview = &saved_views[i];

    /* cast view and allocate device memory */
    if (cast_memory_view(&dview, sview))
      goto on_error;

    /* allocate device memory */
    if (allocate_device_mem(&devptr, kaapi_memory_view_size(&dview)))
      return -1;

    /* track device allocated pointers */
    devptrs[alloc_count++] = devptr;

    /* assume no pointer exists on the device */
    if (KAAPI_ACCESS_IS_READ(mode))
    {
      /* synchronous copy */
      if (viewcopy_htod(devptr, &dview, hostptr, sview))
	goto on_error;
    }

    /* update param addr */
    access.data = (void*)(uintptr_t)devptr;
    format->set_access_param(format, i, sp, &access);

    /* update view param */
    format->set_view_param(format, i, sp, &dview);
  }

  /* execute the task */
  body = (cuda_task_body_t)format->entrypoint[KAAPI_PROC_TYPE_CUDA];
  kaapi_assert_debug(body != NULL);
  body(sp, proc->cuda_proc.stream);

  /* wait for kernel completion */
  res = cuStreamSynchronize(proc->cuda_proc.stream);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamSynchronize", res);
    goto on_error;
  }

  /* get write params from card */
  for (i = 0; i < param_count; ++i)
  {
    mode = format->get_mode_param(format, i, sp);
    if (mode & KAAPI_ACCESS_MODE_V) continue ;
    if (KAAPI_ACCESS_IS_WRITE(mode) == 0) continue ;

    /* retrieve, could be avoided */
    access = format->get_access_param(format, i, sp);
    devptr = (CUdeviceptr)access.data;
    dview = format->get_view_param(format, i, sp);
    hostptr = saved_hostptrs[i];
    sview = &saved_views[i];

    /* synchronous copy */
    if (viewcopy_dtoh(hostptr, sview, devptr, &dview))
      goto on_error;
  }

  /* here, success */
  error = 0;

 on_error:
  /* release device memory in a separate loop */
  for (i = 0; i < alloc_count; ++i)
    free_device_mem(devptrs[i]);

  /* pop current context */
  cuCtxPopCurrent(&proc->cuda_proc.ctx);

  return 0;
}

#endif /* NEW_WIP */
