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
  kaapi_assert_debug( stack != 0 );
  int flag = request->flag;
  
  if (retval)
  {
    kaapi_task_t* sig;
    kaapi_tasksig_arg_t* argsig;
    
    /* reply: several cases
       - if partial steal -> signal should decr the thieves counter (if task is not KAAPI_TASK_ADAPT_NOSYNC)
       - if complete steal of the task -> signal sould pass the body to aftersteal body
       If steal an
    */
    if (kaapi_task_isadaptive(task) && (flag & KAAPI_REQUEST_FLAG_PARTIALSTEAL))
    {
      kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task->sp; /* do not use kaapi_task_getargs !!! */
      kaapi_assert_debug( ta !=0 );
      if ( !(task->flag & KAAPI_TASK_ADAPT_NOPREEMPT) )
      {
        if (stack->pc == task) {
          result = (kaapi_taskadaptive_result_t*)kaapi_stack_pushdata(stack, sizeof(kaapi_taskadaptive_result_t));
          result->flag = KAAPI_RESULT_MASK_EXEC|KAAPI_RESULT_INSTACK;
        }
        else
        {
          result = (kaapi_taskadaptive_result_t*)malloc(sizeof(kaapi_taskadaptive_result_t));
          result->flag = KAAPI_RESULT_MASK_EXEC|KAAPI_RESULT_INHEAP;
        }
        result->signal          = &thief_stack->haspreempt;
        result->arg_from_thief  = 0;
        result->arg_from_victim = 0;
        result->next            = ta->head;
        ta->head                = result->next;
      }
      else flag |= KAAPI_TASK_ADAPT_NOPREEMPT;

      if ( !(task->flag & KAAPI_TASK_ADAPT_NOSYNC) )
        KAAPI_ATOMIC_INCR( &ta->thievescount );
      else flag |= KAAPI_TASK_ADAPT_NOSYNC;
    }
#if defined(KAAPI_DEBUG)
    if (kaapi_task_issync(task))
    {
      /* only in the current implementation:
         - optimization: only put kaapi_suspend_body on task that depend of parameters produced by stolen task
         - add handler (bit field ?) to update shared object as for kaapi_after_steal.
         - if not such task exist: and the shared by declared in the frame of the stolen task...
      */
      kaapi_assert_debug( task->body == &kaapi_suspend_body);
    }
#endif

    sig = kaapi_stack_toptask( thief_stack );
    sig->flag = KAAPI_TASK_STICKY;
    kaapi_task_setbody( sig, &kaapi_tasksig_body );
    kaapi_task_format_debug( sig );
    kaapi_task_setargs( sig, kaapi_stack_pushdata(thief_stack, sizeof(kaapi_tasksig_arg_t)));
    argsig = kaapi_task_getargst( sig, kaapi_tasksig_arg_t);
    argsig->task2sig = task;
    argsig->flag     = flag;
    argsig->result   = result;
    kaapi_stack_pushtask( thief_stack );

    request->status = KAAPI_REQUEST_S_EMPTY;
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
  _kaapi_request_reply( stack, task, request, thief_stack, retval);
  KAAPI_ATOMIC_DECR( (kaapi_atomic_t*)stack->hasrequest ); 
  kaapi_assert_debug( *stack->hasrequest >= 0 );
  return 0;
}
