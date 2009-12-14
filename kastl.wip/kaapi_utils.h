/*
 *  transform.cpp
 *  xkaapi
 *
 *  Created by FLM
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */


#ifndef _XKAAPI_KAAPI_UTILS_H
#define _XKAAPI_KAAPI_UTILS_H



#include <pthread.h>
#include "kaapi.h"



// TODO: should be in kaapi.h
static inline void kaapi_stack_popdata(kaapi_stack_t* stack, kaapi_uint32_t count)
{
  stack->sp_data -= count;
}



namespace kaapi_utils
{
  // prototypes

  void fail_requests(kaapi_stack_t*, kaapi_task_t*, int, kaapi_request_t*);

  // static function definistions

  template<typename SelfType>
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    SelfType* self_work = kaapi_task_getargst(task, SelfType);
    self_work->doit(task, stack);
  }

  template<typename SelfType>
  static int static_splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  {
    SelfType* const self_work = kaapi_task_getargst(task, SelfType);
    return self_work->splitter( victim_stack, task, count, request );
  }

  template<typename SelfType>
  static int foreach_request
  (
   kaapi_stack_t* victim_stack,
   kaapi_task_t* task,
   int count,
   kaapi_request_t* request,
   typename SelfType::request_handler_t& handler,
   SelfType* this_work
   )
  {
    // process the requests and fail
    // the one we did not reply to

    const int replied_count = count;
    SelfType* output_work = 0;

    for (; count > 0; ++request)
      {
	if (!kaapi_request_ok(request))
	  continue ;

	kaapi_stack_t* const thief_stack = request->stack;
	kaapi_task_t*  const thief_task  = kaapi_stack_toptask(thief_stack);
	void* const stack_data = kaapi_stack_pushdata(thief_stack, sizeof(SelfType));

	kaapi_task_initadaptive(thief_stack, thief_task, KAAPI_TASK_ADAPT_DEFAULT);
	kaapi_task_setbody(thief_task, &static_thiefentrypoint<SelfType>);
	kaapi_task_setargs(thief_task, stack_data);

	output_work = kaapi_task_getargst(thief_task, SelfType);

	// stop request processing
	if (handler(this_work, output_work) == false)
	  {
	    kaapi_stack_popdata(thief_stack, sizeof(SelfType));
	    break;
	  }

	kaapi_stack_pushtask(thief_stack);

	// reply ok (1) to the request
	kaapi_request_reply(victim_stack, task, request, thief_stack, 1);

	--count; 
      }

    fail_requests(victim_stack, task, count, request);

    return replied_count;
  }


  template<typename SelfType>
  static void start_adaptive_task(SelfType* work)
  {
    kaapi_stack_t* const stack = kaapi_self_stack();
    kaapi_task_t* const task = kaapi_stack_toptask(stack);

    kaapi_task_initadaptive( stack, task, KAAPI_TASK_ADAPT_DEFAULT);
    kaapi_task_setargs(task, work);
    kaapi_stack_pushtask(stack);

    work->doit(task, stack);

    kaapi_stack_poptask(stack);
  }

  static unsigned int __attribute__((unused)) self_id()
  {
    return (unsigned int)pthread_self();
  } 

#if 0 // TODO

  template<typename SelfType, typename IteratorType>
  static void split_requests
  (
   kaapi_stack_t* victim_stack, kaapi_task_t* task,
   int count, kaapi_request_t* request,
   IteratorType& begin, IteratorType& end,
   SelfType* self_instance
  )
  {
    const size_t size = end - begin;
    const int total_count = count;
    int replied_count = 0;
    size_t bloc;

    /* threshold should be defined (...) */
    if (size < 512)
      goto finish_splitter;

    bloc = size / (1 + count);

    if (bloc < 128) { count = size / 128 - 1; bloc = 128; }

    // iterate over requests
    {
      SelfType::request_handler_t handler(_iend, bloc_size);

      replied_count =
	kaapi_utils::foreach_request
	(
	 victim_stack, task,
	 count, request,
	 handler, self_instance
	 );

      // mute victim state after processing
      _iend = handler.local_end;

      kaapi_assert_debug(iend - _ibeg > 0);
    }

  finish_splitter:
    {
      // fail the remaining requests

      const int remaining_count = total_count - replied_count;

      if (remaining_count)
	{
	  kaapi_utils::fail_requests
	    (
	     victim_stack,
	     task,
	     remaining_count,
	     request + replied_count
	     );
	}
    }

    // all requests have been replied to
    return total_count;
  }

#endif // TODO

} // kaapi_utils namespace



#endif // ! _XKAAPI_KAAPI_UTILS_H
