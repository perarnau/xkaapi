/*
 ** kaapi_cuda.c
 ** 
 **
 ** Copyright 2009 INRIA.
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
#include <cuda.h>
#include <stdlib.h>
#include "kaapi_impl.h"
#include "kaapi_cuda.h"
#include "../common/kaapi_procinfo.h"

/* exported */
int kaapi_cuda_register_procs(kaapi_procinfo_list_t* kpl)
{
  const char* const gpuset_str = getenv("KAAPI_GPUSET");
  kaapi_procinfo_t* pos = kpl->tail;
  unsigned int kid = kpl->count;
  int devcount;
  int err;

  if (gpuset_str == NULL)
    return 0;

  if (cuInit(0) != CUDA_SUCCESS)
    return -1;

  if (cuDeviceGetCount(&devcount) != CUDA_SUCCESS)
    return -1;
  
  if (devcount == 0)
    return 0;
 
  err = kaapi_procinfo_list_parse_string
    (kpl, gpuset_str, KAAPI_PROC_TYPE_CUDA, (unsigned int)devcount);
  if (err) return -1;

  if (kpl->tail == NULL) return 0;

  /* affect kids */
  if (pos == NULL) pos = kpl->tail;
  else pos = pos->next;
  for (; pos; pos = pos->next, ++kid)
    pos->kid = kid;

  return 0;
}


/* todo: put in kaapi_sched_select_victim_with_tasks.c */
static inline void lock_proc(kaapi_processor_t* proc, kaapi_processor_id_t kid)
{
  while (1)
  {
    if (KAAPI_ATOMIC_CAS(&proc->lock, 0, 1 + kid))
      break;
  }
}

static inline void unlock_proc(kaapi_processor_t* proc)
{
  KAAPI_ATOMIC_WRITE(&proc->lock, 0);
}

static inline void lock_thread(kaapi_thread_context_t* thread)
{
  while (1)
  {
    if (KAAPI_ATOMIC_CAS(&thread->lock, 0, 1))
      break ;
  }
}

static inline void unlock_thread(kaapi_thread_context_t* thread)
{
  KAAPI_ATOMIC_WRITE(&thread->lock, 0);
}

#if 1 /* unused */
static inline int is_task_ready(const kaapi_task_t* task)
{
  if (kaapi_task_getbody(task) == kaapi_taskadapt_body)
    return 1;
  return kaapi_task_isstealable(task);
}
#endif /* unused */

static unsigned int __attribute__((unused)) count_tasks_by_type
(kaapi_thread_t* thread, unsigned int type)
{
  /* assume thread kproc locked */
  
  kaapi_task_t* const end = thread->sp;
  kaapi_task_t* pos;
  unsigned int count = 0;
  
  for (pos = thread->pc; pos != end; --pos)
  {
#if 0 /* unused */
    if (!is_task_ready(pos))
      continue ;
#endif /* unused */
    
    ++count;
  }
  
  return count;
}

#if 0 /* todo, unused */
static unsigned int __attribute__((unused)) has_task_by_proc_type
(kaapi_thread_t* thread, unsigned int proc_type)
{
  /* assume thread kproc locked */
  
#define CONFIG_DEBUG 0
  
  kaapi_task_t* const end = thread->sp;
  kaapi_task_t* pos;
  
  for (pos = thread->pc; pos != end; --pos)
  {
    const kaapi_format_t* const format =
    kaapi_format_resolvebybody(pos->ebody);
    
    if (format == NULL)
    {
#if CONFIG_DEBUG
      printf("format not found, body: %p, ebody: %p\n",
             (void*)pos->body, (void*)pos->ebody);
#endif
      continue ;
    }
    
    if (format->entrypoint[proc_type] == NULL)
    {
#if CONFIG_DEBUG
      printf("no entrypoint\n");
#endif
      continue ;
    }
    
    if (!is_task_ready(pos))
    {
#if CONFIG_DEBUG
      printf("not ready\n");
#endif
      continue ;
    }
    
#if CONFIG_DEBUG
    printf("found TASK\n");
#endif
    
    return 1;
  }
  
  return 0;
}
#endif /* todo, unused */

int kaapi_sched_select_victim_with_cuda_tasks
(
 kaapi_processor_t* kproc,
 kaapi_victim_t* victim,
 kaapi_selecvictim_flag_t flag
)
#if 1 /* disable worksealing */
{
  return kaapi_sched_select_victim_rand(kproc, victim, flag);
}
#else
{
  unsigned int has_task;
  int i;

  if (flag != KAAPI_SELECT_VICTIM) return 0;
  
  for (i = 0; i < kaapi_count_kprocessors; ++i)
  {
    kaapi_processor_t* const pos = kaapi_all_kprocessors[i];
    
    if ((pos == NULL) || (pos == kproc))
      continue ;
    
    has_task = 0;
    
    lock_proc(pos, kproc->kid);
    if (pos->thread != NULL)
    {
      lock_thread(pos->thread); /* useless since kproc locked? */
      kaapi_thread_t* const thread = kaapi_threadcontext2thread(pos->thread);
      if (pos->thread->unstealable == 0)
        has_task = has_task_by_proc_type(thread, KAAPI_PROC_TYPE_CUDA);
      unlock_thread(pos->thread); /* useless since kproc locked? */
    }
    unlock_proc(pos);
    
    /* it potentially has a cuda task */
    if (has_task)
    {
      victim->kproc = pos;
      victim->level = 0; /* unused? */
      return 0;
    }
  }
  
  return EINVAL;
}
#endif


void kaapi_exec_cuda_task
(kaapi_task_t* task, kaapi_thread_t* thread)
{
  kaapi_task_getbody(task)(task->sp, thread);
}
