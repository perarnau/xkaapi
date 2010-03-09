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
  kaapi_request_t* request    = 0;
  kaapi_stack_t* thief_stack  = 0;
  kaapi_task_t*  steal_task   = 0;

  kaapi_assert_debug (task !=0);
  
  kaapi_assert_debug( kaapi_task_getbody(task) ==kaapi_suspend_body);

  /* find the first request in the list */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
  {
    if (kaapi_request_ok( &array[i] )) 
    {
      request = &array[i];
      break;
    }
  }
  kaapi_assert(request !=0);

  /* - create the task steal that will execute the stolen task
     The task stealtask stores:
       - the original stack
       - the original task pointer
       - the pointer to shared data with R / RW access data
       - and at the end it reserve enough space to store original task arguments
     The original body is saved as the extra body of the original task data structure.
  */
  thief_stack = request->stack;

//  printf( "[splitter] task: @=%p, stack: @=%p, thief_stack: @=%p\n", task, stack, thief_stack);

  
  steal_task = kaapi_stack_toptask( thief_stack );
  kaapi_task_init( steal_task, kaapi_tasksteal_body, kaapi_stack_pushdata(thief_stack, sizeof(kaapi_tasksteal_arg_t)) );
  kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( steal_task, kaapi_tasksteal_arg_t );
  arg->origin_stack          = stack;
  arg->origin_task           = task;
  
  kaapi_stack_pushtask( thief_stack );

  /* do not decrement the counter */
  _kaapi_request_reply( stack->_proc, stack, task, request, thief_stack, 0, 1, 0 ); /* success of steal */
  return 1;
}
