/*
** kaapi_cuda.c
** 
** Created on Jun 23
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
  if (err)
    return -1;

  return 0;
}


/* todo: put in kaapi_sched_select_victim_with_tasks.c */

static inline void lock_kproc(kaapi_processor_t* kproc, kaapi_processor_id_t kid)
{
  while (1)
  {
    if (KAAPI_ATOMIC_CAS(&kproc->lock, 0, 1 + kid))
      break;
  }
}

static inline void unlock_kproc(kaapi_processor_t* kproc)
{
  KAAPI_ATOMIC_WRITE(&kproc->lock, 0);
}

#if 0 /* unused */
static inline int is_task_ready(const kaapi_task_t* task)
{
  if (task->body == kaapi_adapt_body)
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
    if (pos->proctype != type)
      continue ;

#if 0 /* unused */
    if (!is_task_ready(pos))
      continue ;
#endif /* unused */

    ++count;
  }

  return count;
}

static unsigned int has_task_by_type
(kaapi_thread_t* thread, unsigned int type)
{
  /* assume thread kproc locked */

  /* it is more complicated: a task has a
     format describing the possible implementations
     it has.
     for each task present in the thread,
     we must get the format and check if an
     implementation matching the kproc caps
     exists.
   */

  kaapi_task_t* const end = thread->sp;
  kaapi_task_t* pos;

  for (pos = thread->pc; pos != end; --pos)
  {
#if 0
    if (pos->proctype != type)
      continue ;
#endif

#if 0 /* unused */
    if (!is_task_ready(pos))
      continue ;
#endif /* unused */

    return 1;
  }

  return 0;
}

int kaapi_sched_select_victim_with_cuda_tasks
(kaapi_processor_t* kproc, kaapi_victim_t* victim)
{
  unsigned int has_task;
  int i;

  for (i = 0; i < kaapi_count_kprocessors; ++i)
  {
    kaapi_processor_t* const pos = kaapi_all_kprocessors[i];

    if ((pos == NULL) || (pos == kproc))
      continue ;

    lock_kproc(pos, kproc->kid);
    has_task = has_task_by_type
      (kaapi_threadcontext2thread(pos->thread), KAAPI_PROC_TYPE_CUDA);
    unlock_kproc(pos);

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


void kaapi_exec_cuda_task
(kaapi_task_t* task, kaapi_thread_t* thread)
{
  task->body(task->sp, thread);
}
