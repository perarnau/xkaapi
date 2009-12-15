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


static void reduce_adaptive_results
(
 kaapi_taskadaptive_t* ta,
 kaapi_reducer_t reducer,
 void* arg
)
{
  kaapi_taskadaptive_result_t* athief;

  for (athief = ta->head; athief != NULL; athief = athief->next)
  {
#if 0
    /* no need, thieves already terminated */
    while (!athief->thief_term) ;
#endif

    if (athief->result_data != NULL)
    {
      /* invoke the reducer */
      reducer(athief->result_data, arg);

      /* result data was allocated with malloc */
      free(athief->result_data);
      athief->result_data = NULL;
    }
  }
}


static int finalize_steal( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_reducer_t reducer_fn, void* reducer_arg, const void* result_data)
{
  if (kaapi_task_isadaptive(task) && !(task->flag & KAAPI_TASK_ADAPT_NOSYNC))
  {
    kaapi_taskadaptive_t* ta = task->sp;
    kaapi_assert_debug( ta !=0 );

    if (ta->mastertask ==0) /* I'm a master task, wait  */
    {
/*      double t0 = kaapi_get_elapsedtime();  */
      while (KAAPI_ATOMIC_READ( &ta->thievescount ) !=0) ;
/*      double t1 = kaapi_get_elapsedtime(); */
/*      printf("[finalize] wait for:%es\n", t1 -t0); */
      kaapi_readmem_barrier(); /* avoid read reorder before the barrier, for instance reading some data */
#if defined(KAAPI_DEBUG)
      kaapi_assert_debug( ta->thievescount._counter == 0);
#endif

      if (reducer_fn != NULL)
      {
	/* reduce the thief results */
	reduce_adaptive_results(ta, reducer_fn, reducer_arg);
      }
    }

    if (result_data != NULL)
    {
      /* copy thief result into the dedicated victim area.
	 the area was allocated in kaapi_mt_request_reply.
      */
      kaapi_assert_debug(ta->result_data != NULL);
      memcpy(ta->result_data, result_data, ta->result_size);
    }

  }

  return 0;
}


/** Return the number of splitted parts (here 1: only steal the whole task)
    Currently assume independent task only.
*/

int kaapi_finalize_steal(kaapi_stack_t* stack, kaapi_task_t* task)
{
  return finalize_steal(stack, task, NULL, NULL, NULL);
}


int kaapi_finalize_steal2(kaapi_stack_t* stack, kaapi_task_t* task, kaapi_reducer_t reducer_fn, void* reducer_arg, const void* result_data)
{
  return finalize_steal(stack, task, reducer_fn, reducer_arg, result_data);
}
