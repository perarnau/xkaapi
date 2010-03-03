/*
** kaapi_mt_task_signal.c
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
#include <stdio.h>

/**
*/
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_tasksig_body( kaapi_task_t* task, kaapi_stack_t* stack)
#else
void kaapi_tasksig_body( kaapi_task_t* task, kaapi_stack_t* stack)
#endif
{
  /*
    printf("Thief end, @stack: 0x%p\n", stack);
    fflush( stdout );
  */
  kaapi_tasksig_arg_t* argsig;
  kaapi_task_t* task2sig;

  argsig = kaapi_task_getargst( task, kaapi_tasksig_arg_t);
  task2sig = argsig->task2sig;
  KAAPI_LOG(100, "signaltask: 0x%p -> task2signal: 0x%p\n", (void*)task, (void*)task2sig );

  if (!(argsig->flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL)) /* steal a whole task */
  {
    kaapi_task_setbody(task2sig, kaapi_aftersteal_body );
  }
  if ( !(argsig->flag & KAAPI_TASK_ADAPT_NOPREEMPT) ) /* required preemption */
  {
    /* mark result as produced */
    if (argsig->taskadapt->head !=0)
    { /* avoid remote write */
      argsig->result->rhead = argsig->taskadapt->head;
      argsig->result->rtail = argsig->taskadapt->tail;
    }
    argsig->result->thief_term = 1;
  }

  /* flush in memory all pending write and read ops */  
  kaapi_mem_barrier();

  if (!(argsig->flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL)) /* steal a whole task */
  {
  }
  else /* partial steal -> adaptive task */
  {
    if ( !(argsig->flag & KAAPI_TASK_ADAPT_NOSYNC) ) /* required synchronisation */
    {
      kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task2sig->sp;/* do not use kaapi_task_getargs !!! */
      kaapi_assert_debug( ta !=0 );
      KAAPI_ATOMIC_DECR( &ta->thievescount );
    } 
    if ( !(argsig->flag & KAAPI_TASK_ADAPT_NOPREEMPT) ) /* required also preemption */
    { /* mark result as term */

      if (!argsig->result->thief_term && argsig->result->req_preempt) /* remote read */
      {
         while (stack->haspreempt ==0) ;
      }
    }
  }
}

