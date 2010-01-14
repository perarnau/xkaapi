/*
** kaapi_task_splitter_dfg.c
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

/** Return the number of splitted parts (at most 1 if the task may be steal)
*/
int kaapi_task_splitter_dfg(kaapi_stack_t* stack, kaapi_task_t* task, int count, struct kaapi_request_t* array)
{
  int i;
  int countparam;
  kaapi_request_t* request   = 0;
  kaapi_stack_t* thief_stack = 0;
  kaapi_task_t*  steal_task   = 0;

  kaapi_assert_debug (task !=0);
  kaapi_assert_debug (task->format !=0);
  
  KAAPI_LOG(50, "dfgsplitter task: 0x%p\n", (void*)task);

  kaapi_assert_debug( task->body !=0);
  kaapi_assert_debug( task->body !=kaapi_suspend_body);

  /* cas the state */
  if (!kaapi_task_casstate(task, KAAPI_TASK_S_INIT, KAAPI_TASK_S_STEAL )) return 0;
  
  /* find the first request in the list */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {
    if (kaapi_request_ok( &array[i] )) 
    {
      request = &array[i];
      break;
    }
  }

  if (request ==0) 
  {
    kaapi_task_setstate(task, KAAPI_TASK_S_INIT);
    return 0;
  }
    
  task->body = &kaapi_suspend_body;
  
  countparam = task->format->count_params;
    
  /* - create the task steal that will execute the stolen task
     The task stealtask stores:
       - the original stack
       - the original task pointer
       - the original body
       - the pointer to shared data with R / RW access data
       - and at the end it reserve enough space to store original task arguments
  */
  thief_stack = request->stack;
  
  steal_task = kaapi_stack_toptask( thief_stack );
  steal_task->flag = KAAPI_TASK_STICKY;
  kaapi_task_setbody( steal_task, &kaapi_tasksteal_body );
  kaapi_task_format_debug( steal_task );
  kaapi_task_setargs( steal_task, kaapi_stack_pushdata(thief_stack, sizeof(kaapi_tasksteal_arg_t)) );
  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( steal_task, kaapi_tasksteal_arg_t );
  arg->origin_stack          = stack;
  arg->origin_task           = task;
  arg->origin_fmt            = task->format;

  kaapi_stack_pushtask( thief_stack );

  /* ... and push continuation if w, cw or rw mode */
  if (1)
  {
    kaapi_task_t* task = kaapi_stack_toptask( thief_stack );
    task->flag = KAAPI_TASK_STICKY;
    kaapi_task_setbody( task, &kaapi_taskwrite_body );
    kaapi_task_format_debug( task );
    kaapi_task_setargs( task, arg ); /* keep the pointer as kaapi_tasksteal_body */
    kaapi_stack_pushtask( thief_stack );
  }

#if 0
  printf("Steal task:%p, stack:%p, fmt:%p ||| thiefstack:%p arg:%p, task:%p, fmt:%p\n", task, stack, task->format, thief_stack, arg, arg->origin_task, arg->origin_fmt );
/*
  printf("Victim stack:\n");
  kaapi_stack_print( 1, stack );
*/
  printf("Thief stack:\n");
  kaapi_stack_print( 1, thief_stack );
#endif
 
  /* do not decrement the counter */
  _kaapi_request_reply( stack, task, request, thief_stack, 0, 1 ); /* success of steal */
  return 1;
}
