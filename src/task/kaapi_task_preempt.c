/*
** kaapi_task_preempt.c
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

int kaapi_preemptpoint_before_reducer_call( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_for_victim, int size )
{
  kaapi_taskadaptive_t* ta = task->sp; /* do not use kaapi_task_getarg */

  /* push data to the victim and signal it */
  if ((arg_for_victim !=0) && (size >0))
  {
    memcpy(ta->result->data, arg_for_victim, size );
  }
/*  printf("Kaapi_preempt: send:%p to victim, result: @=%p\n", arg_for_victim, (void*)ta->result);*/
  if (ta->head !=0)
  {
    ta->result->rhead = ta->head;
    ta->head = 0;
    ta->result->rtail = ta->tail;
    ta->tail = 0;
  }
  kaapi_mem_barrier();

  return 0;
}

int kaapi_preemptpoint_after_reducer_call( kaapi_stack_t* stack, kaapi_task_t* task, int reducer_retval )
{
  kaapi_taskadaptive_t* ta = task->sp; /* do not use kaapi_task_getarg */
  stack->haspreempt = 0;
  kaapi_writemem_barrier();
  ta->result->thief_term = 1;
  return 1;
}


int kaapi_preempt_nextthief_helper( kaapi_stack_t* stack, kaapi_task_t* task, void* arg_to_thief )
{
  kaapi_assert_debug( task->flag & KAAPI_TASK_ADAPTIVE );
  kaapi_assert_debug( !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) );

  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp;
  kaapi_taskadaptive_result_t* athief = ta->head;
  
  ta->current_thief= 0;
  /* no more thief to preempt */
  if (athief ==0) return 0;
  
  /* pass arg to the thief */
/*  printf("Kaapi_preempt_nextthief: send:%p to thief, result @=%p\n", arg_to_thief, (void*)athief);*/
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
    
    /* wait thief receive preemption */
    while (!athief->thief_term) ; 
  }

  /* push current preempt thief in current_thief: used by caller to make call */
  ta->current_thief = ta->head;
  
  /* pop current thief and push thiefs of the thief into the local preemption list */
  ta->head = ta->head->next;
  if (ta->head ==0) ta->tail = 0;

  if (athief->rhead !=0)
    athief->rtail->next = ta->head;
  ta->head = athief->rhead;
  if (ta->tail ==0) ta->tail = athief->rtail;
  return 1;
}
