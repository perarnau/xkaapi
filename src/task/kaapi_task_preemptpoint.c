/*
** kaapi_task_preemptpoint.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
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

int kaapi_preemptpoint_before_reducer_call( kaapi_thread_t* thread, kaapi_task_t* task, void* arg_for_victim, int size )
{
#if 0
  kaapi_taskadaptive_t* ta = task->sp; /* do not use kaapi_task_getarg */

  /* lock stack in case of CONCURRENT WS in order to avoid stealing on this task */
#if 0//defined(KAAPI_CONCURRENT_WS)
  pthread_mutex_lock(&stack->_proc->lsuspend.lock);
  kaapi_task_unsetstealable(task);
  pthread_mutex_unlock(&stack->_proc->lsuspend.lock);
#endif
  
  /* push data to the victim and list of thief */
  if ((arg_for_victim !=0) && (size >0))
  {
    memcpy(ta->result->data, arg_for_victim, size );
  }
  if (ta->head !=0)
  { /* recall the list if double linked list */
    ta->result->rhead = ta->head;
    ta->head = 0;
    ta->result->rtail = ta->tail;
    ta->tail = 0;
  }

  /* mark the stack as preemption processed -> signal victim */
  stack->haspreempt = 0;
#endif

  return 0;
}

int kaapi_preemptpoint_after_reducer_call( kaapi_thread_t* thread, kaapi_task_t* task, int reducer_retval )
{
#if 0
  kaapi_taskadaptive_t* ta = task->sp; /* do not use kaapi_task_getarg */

  kaapi_writemem_barrier();   /* serialize previous line with next line */
  ta->result->thief_term = 1;
  kaapi_mem_barrier();
  ta->result = 0;
#endif

  return 1;
}
