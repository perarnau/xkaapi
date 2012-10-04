/*
** kaapi_hws_pushtask.c
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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


/* push the task thread->pc on a given processor
 */
int kaapi_thread_distribute_task (
  kaapi_thread_t*      thread,
  kaapi_processor_id_t kid
)
{
#if 0
  kaapi_processor_t* kproc;
  if (kid >= kaapi_getconcurrency())
    kid %= kaapi_getconcurrency();
  kproc = kaapi_get_current_processor();
  if (kproc->kid == kid)
    return kaapi_thread_pushtask( kaapi_threadcontext2thread(kproc->thread) );

  kproc = kaapi_all_kprocessors[kid];
  
  /* the task is in the own queue of the thread but not yet visible to other threads */
  kaapi_task_t* local_task = kaapi_thread_toptask(thread);
  
  /* create a new steal_body task.
     task is allocated into local queue: once the local steal task is remotely finished,
     the local stack may be pop.
  */
  kaapi_task_withlink_t* remote_task    = 
      (kaapi_task_withlink_t*)kaapi_thread_pushdata_align(
                          thread, 
                          sizeof(kaapi_task_withlink_t), 
                          sizeof(uintptr_t)
      );
  
  kaapi_tasksteal_arg_t* argsteal = 
      (kaapi_tasksteal_arg_t*)kaapi_thread_pushdata_align(
                          thread, 
                          sizeof(kaapi_tasksteal_arg_t), 
                          sizeof(void*)
      );
  argsteal->origin_task           = local_task;
  argsteal->origin_body           = local_task->body;
  argsteal->origin_fmt            = 0;    /* should be computed by a thief */
  argsteal->war_param             = 0;    /* assume no war */
  argsteal->cw_param              = 0;    /* assume no cw mode */
  kaapi_task_init(  
      &remote_task->task, 
      kaapi_tasksteal_body, 
      argsteal
  );
  local_task->reserved             = &remote_task->task;
  argsteal->origin_body = kaapi_task_marksteal( local_task ) ? local_task->body : 0;
  kaapi_assert_debug( argsteal->origin_body != 0);
  
  /* ok here initial local_task is marked as steal and remote 
     task corresponds to a steal task 
     - push local_task into the own thread queue
     - push remote_task to the kproc' mailbox queue
  */
  kaapi_thread_pushtask(thread);
  
  /* mail box: FIFO push in tail */
  remote_task->next = 0;
  kaapi_atomic_lock(&kproc->lock);
  {
    if (kproc->mailbox.tail ==0)
      kproc->mailbox.tail = kproc->mailbox.head = remote_task;
    else
      kproc->mailbox.tail->next = remote_task;
    kproc->mailbox.tail = remote_task;
  }
  kaapi_atomic_unlock(&kproc->lock);
#else
  kaapi_thread_pushtask(thread);
#endif

  return 0;
}
