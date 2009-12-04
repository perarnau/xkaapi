/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:00 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@imag.fr
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


/** Implementation note:
    - only the thief_stack + signal to the thief has to be port on the machine.
    - the creation of the task to signal back the end of computation must be keept.
    I do not want to split this function in two parts (machine dependent and machine independent)
    in order to avoid 2 function calls, BUT for maintenance an inline version in kaapi_impl.h
    would be peferable.
    
    For instance if no memory is shared between both -> communication of the memory stack.
    with translation of :
      - function body
      - function splitter
      - arguments in case of an adaptive task
      - all stack pointer of parameter that should be considered as offset.
    The format of the task should give all necessary information about types used in the
    data stack.
*/
int _kaapi_request_reply( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_request_t* request, kaapi_stack_t* thief_stack, int retval )
{
  kaapi_taskadaptive_result_t* result =0;
  kaapi_taskadaptive_t* ta =0;
  int flag;
  kaapi_assert_debug( stack != 0 );
  kaapi_assert_debug( request != 0 );
  
  flag = request->flag;
  request->flag = 0;
  
  if (retval)
  {
    kaapi_task_t* sig;
    kaapi_tasksig_arg_t* argsig;
    

    /* reply: several cases
       - if partial steal -> signal should decr the thieves counter (if task is not KAAPI_TASK_ADAPT_NOSYNC)
       - if complete steal of the task -> signal sould pass the body to aftersteal body
       If steal an
    */
    if (flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL)
    {
      ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
      kaapi_assert_debug( ta !=0 );
      kaapi_assert_debug( kaapi_task_isadaptive(task) );

      if ( !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) ) /* required preemption */
      {
        if (stack->pc == task) { /* current running task */
          result = (kaapi_taskadaptive_result_t*)kaapi_stack_pushdata(stack, sizeof(kaapi_taskadaptive_result_t));
          result->flag = KAAPI_RESULT_INSTACK;
        }
        else
        {
          result = (kaapi_taskadaptive_result_t*)malloc(sizeof(kaapi_taskadaptive_result_t));
          result->flag = KAAPI_RESULT_INHEAP;
        }
        result->signal          = &thief_stack->haspreempt;
        result->req_preempt     = 0;
        result->thief_term      = 0;
        result->arg_from_thief  = 0;
        result->parg_from_victim= 0;
        result->head            = 0;
        result->tail            = 0;
        /* link ressult */
        result->next            = ta->head;
        ta->head                = result->next;
        
        /* update ta of the first replied task in the stack */
        kaapi_task_t* thief_task = thief_stack->pc;
        if (kaapi_task_isadaptive(thief_task))
        {
          kaapi_taskadaptive_t* thief_ta = (kaapi_taskadaptive_t*)thief_task->sp;
          thief_ta->mastertask           = ( ta->mastertask == 0 ? ta : ta->mastertask );
          thief_ta->result               = result;
          result->parg_from_victim       = &thief_ta->arg_from_victim;
        }
      }
      else flag |= KAAPI_TASK_ADAPT_NOPREEMPT;

      if ( !(task->flag & KAAPI_TASK_ADAPT_NOSYNC) )
        KAAPI_ATOMIC_INCR( &ta->thievescount );
      else flag |= KAAPI_TASK_ADAPT_NOSYNC;
    }

    sig = kaapi_stack_toptask( thief_stack );
    sig->flag = KAAPI_TASK_STICKY;
    kaapi_task_setbody( sig, &kaapi_tasksig_body );
    kaapi_task_format_debug( sig );
    kaapi_task_setargs( sig, kaapi_stack_pushdata(thief_stack, sizeof(kaapi_tasksig_arg_t)));
    argsig           = kaapi_task_getargst( sig, kaapi_tasksig_arg_t);
    argsig->task2sig = task;
    argsig->flag     = flag;

    argsig->taskadapt= ta;
    argsig->result   = result;
    kaapi_stack_pushtask( thief_stack );

    request->status  = KAAPI_REQUEST_S_EMPTY;
    request->reply->data = thief_stack;
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_SUCCESS;
  }
  else 
  {
    request->status = KAAPI_REQUEST_S_EMPTY;
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_FAIL;
  }
  return 0;
}

/* This is the public function to be used with adaptive algorithm.
   Be carreful: to not use this function inside the library where reply count is accumulate
   before decremented to the counter.
*/
int kaapi_request_reply( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_request_t* request, kaapi_stack_t* thief_stack, int retval )
{
  request->flag |= KAAPI_REQUEST_FLAG_PARTIALSTEAL;
  _kaapi_request_reply( stack, task, request, thief_stack, retval);
  KAAPI_ATOMIC_DECR( (kaapi_atomic_t*)stack->hasrequest ); 
  kaapi_assert_debug( *stack->hasrequest >= 0 );
  return 0;
}
