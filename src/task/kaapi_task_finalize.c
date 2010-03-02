/*
** kaapi_task_finalize.c
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


/**
*/
#if defined(KAAPI_VERY_COMPACT_TASK)
void _kaapi_taskfinalize_body( kaapi_task_t* task, kaapi_stack_t* stack )
#else
void kaapi_taskfinalize_body( kaapi_task_t* task, kaapi_stack_t* stack )
#endif
{
  kaapi_taskadaptive_t* ta = task->sp;
  kaapi_assert_debug( ta !=0 );

#if defined(KAAPI_USE_PERFCOUNTER)
    double t0, t1;
#endif

  if (ta->mastertask ==0) /* I'm a master task, wait  */
  {
#if defined(KAAPI_USE_PERFCOUNTER)
    t0 = kaapi_get_elapsedtime();
#endif
    while (KAAPI_ATOMIC_READ( &ta->thievescount ) !=0) ;/* pthread_yield_np(); */
#if defined(KAAPI_USE_PERFCOUNTER)
    t1 = kaapi_get_elapsedtime();
    stack->_proc->t_sched += t1-t0;
#endif
    kaapi_readmem_barrier(); /* avoid read reorder before the barrier, for instance reading some data */
#if defined(KAAPI_DEBUG)
    kaapi_assert_debug( ta->thievescount._counter == 0);
#endif
  }
  else if (ta->result !=0) /* thief has been preempted or flagged as NOPREEMPT */
  {
    /* If I have something to write, write it */
    if ((ta->local_result_data !=0) && (ta->local_result_size !=0))
    {
      memcpy( ta->result->data, ta->local_result_data, ta->local_result_size );
    }
  }
}
