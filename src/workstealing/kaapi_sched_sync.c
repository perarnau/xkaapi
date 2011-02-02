/*
** kaapi_sched_sync.c
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
#include "kaapi_impl.h"


/** kaapi_sched_sync
    Here the stack frame is organised like this, task1 is the running task.

    thread->pc | task1  |
              ----------
               | task2  |
               | task3  |
               | task4  |
               | task5  |
          sp ->| ....   |
     
  
   We first push a retn task and execute all the task in the frame in [pc+1, ...,  sp). 
   The task retn serves to mark the stack structure in case of stealing (...) ? Sure ?
   
   On return, we leave the stack such that (the pushed tasks between frame_sp and sp)
   are poped :

    thread->pc | task1  |
               ----------
          sp ->| x x x  |
               | x x x  |
               | x x x  |
               | x x x  |
               | ....   |
*/
int kaapi_sched_sync(void)
{
  kaapi_thread_context_t* thread;
  kaapi_task_t*           savepc;
  int                     err;
  kaapi_cpuset_t          save_affinity;
  kaapi_frame_t*          save_esfp;
#if defined(KAAPI_DEBUG)
  kaapi_frame_t*          save_fp;
#endif

  /* If pure DFG frame and empty return */
  thread = kaapi_self_thread_context();
  if ((thread->sfp == 0) && kaapi_frame_isempty( thread->sfp ) ) 
    return 0;

  /* here affinity should be deleted (not scalable concept) 
     - use localkid to enforce execution into one specific
     kprocessor
  */
  kaapi_cpuset_copy(&save_affinity, &thread->affinity);
  kaapi_cpuset_clear(&thread->affinity);

  savepc = thread->sfp->pc;
#if defined(KAAPI_DEBUG)
  save_fp = (kaapi_frame_t*)thread->sfp;
#endif
  save_esfp = thread->esfp;
  thread->esfp = (kaapi_frame_t*)thread->sfp;

  /* write barrier in order to commit update */
  kaapi_mem_barrier();
  
redo:
  if (thread->sfp->tasklist == 0) 
    err = kaapi_thread_execframe(thread);
  else
    err = kaapi_threadgroup_execframe(thread);

  if (err == EWOULDBLOCK)
  {
    kaapi_sched_suspend( kaapi_get_current_processor() );
    kaapi_assert_debug( kaapi_self_thread_context() == thread );
    kaapi_assert_debug( thread->proc == kaapi_get_current_processor() );
    goto redo;
  }

  /* reset affinity flag if save_stick != 1 */
  kaapi_cpuset_copy(&thread->affinity, &save_affinity);
  
  if (err) /* but do not restore anyting */
    return err;

#if defined(KAAPI_DEBUG)
  kaapi_assert_debug(save_fp == thread->sfp);
#endif
  /* flush the stack of task */
  thread->sfp->pc = thread->sfp->sp = savepc;
  thread->esfp = save_esfp;

  return err;
}
