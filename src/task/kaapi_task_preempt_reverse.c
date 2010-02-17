/*
** kaapi_task_preempt_reverse.c
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

int kaapi_preempt_nextthief_reverse_helper( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_to_thief )
{
  kaapi_assert_debug( task->flag & KAAPI_TASK_ADAPTIVE );
  kaapi_assert_debug( !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) );
  int retval = 1;
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp;
#if defined(KAAPI_USE_PERFCOUNTER)
  double t0, t1;
#endif
  
#if defined(KAAPI_CONCURRENT_WS)
  int flagsticky = kaapi_task_isstealable(task);
#  if defined(KAAPI_USE_PERFCOUNTER)
  t0 = kaapi_get_elapsedtime();
#  endif
  pthread_mutex_lock(&stack->_proc->lsuspend.lock);
  kaapi_task_unsetstealable(task);
  pthread_mutex_unlock(&stack->_proc->lsuspend.lock);
#  if defined(KAAPI_USE_PERFCOUNTER)
  t1 = kaapi_get_elapsedtime();
  stack->_proc->t_preempt += t1-t0;
/*    printf("[kaapi_preempt_nextthief_reverse_helper]: wait/lock:%f\n", t1-t0); */
#  endif
#endif

  kaapi_taskadaptive_result_t* athief = ta->tail;

  ta->current_thief= 0;
  /* no more thief to preempt */
  if (athief ==0)
  {
    retval =0;
    goto reset_return;
  }
  
  /* pass arg to the thief */
  *athief->parg_from_victim = arg_to_thief;  
  kaapi_mem_barrier();

  athief->req_preempt = 1;
  kaapi_mem_barrier();
  
  if (athief->thief_term)
  {
    /* thief has finished */
  }
  else 
  {
    /* send signal on the thief stack */  
    *athief->signal = 1;
    
#if defined(KAAPI_USE_PERFCOUNTER)
    t0 = kaapi_get_elapsedtime();
#endif
    /* wait thief receive preemption */
    while (!athief->thief_term) pthread_yield(); 
#if defined(KAAPI_USE_PERFCOUNTER)
    t1 = kaapi_get_elapsedtime();
    stack->_proc->t_preempt += t1-t0;
/*    printf("[kaapi_preempt_nextthief_reverse_helper]: wait thief:%f\n", t1-t0); */
#endif
  }

  /* push current preempted thief in current_thief: used kaapi_preempt_nextthief 
     to call the reducer with the thief args
  */
  ta->current_thief = ta->tail;
  kaapi_assert_debug( ta->current_thief == athief );

  /* pop current thief and push list of thiefs of the preempted thief in the from 
     of the local preemption list 
  */
  ta->tail = ta->tail->prev;
  if (ta->tail ==0)
    ta->head = 0;
  else
    ta->tail->next = 0;

#if defined(KAAPI_DEBUG)
  ta->current_thief->next = 0;
  ta->current_thief->prev = 0;
#endif  

  if (athief->rhead !=0)
  {
    kaapi_assert_debug( athief->rhead->prev ==0 );
    kaapi_assert_debug( athief->rtail->next ==0 );
    athief->rhead->prev = ta->tail;
    if (ta->tail ==0)
      ta->head = athief->rhead;
    else
      ta->tail->next = athief->rhead;
    
    ta->tail = athief->rtail;
  }

reset_return:
#if defined(KAAPI_CONCURRENT_WS)
  if (flagsticky)
      kaapi_task_setstealable(task);
#endif

  return retval;
}
