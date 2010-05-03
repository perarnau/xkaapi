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

extern unsigned long get_clock(void);

#define kaapi_trace(fmt, ...)			\
do {						\
  unsigned long _clock = get_clock();		\
  fprintf					\
    (						\
     stderr,					\
     "[KAAPI_TRACE] %u %u " fmt "\n",		\
     (unsigned int)_clock,			\
     (unsigned int)pthread_self(),		\
     ##__VA_ARGS__				\
     );						\
} while (0)



namespace kaapi_utils
{
  void fail_requests(kaapi_stack_t*, kaapi_task_t*, int, kaapi_request_t*);

  // static function definistions

  template<typename SelfType>
  static void static_thiefentrypoint( kaapi_task_t* task, kaapi_stack_t* stack )
  {
    SelfType* self_work = kaapi_task_getargst(task, SelfType);
    self_work->doit(task, stack);
    kaapi_finalize_steal( stack, task, self_work, sizeof(SelfType) );
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
      
      kaapi_task_initadaptive(thief_stack, thief_task, NULL, NULL, KAAPI_TASK_ADAPT_DEFAULT);
      kaapi_task_setbody(thief_task, &static_thiefentrypoint<SelfType>);
      
      void* const stack_data = kaapi_thread_pushdata(thief_stack, sizeof(SelfType));
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
      kaapi_request_reply(victim_stack, task, request, thief_stack, sizeof(SelfType), 1);
      
      --count; 
    }
    
    fail_requests(victim_stack, task, count, request);
    
    return replied_count;
  }
  
  template<typename SelfType>
  static void static_mainentrypoint(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    SelfType* const self_work = kaapi_task_getargst(task, SelfType);
    self_work->doit(task, stack);
  }

  template<typename SelfType>
  static void start_adaptive_task(SelfType* work)
  {
    kaapi_stack_t* const stack = kaapi_self_frame();
    kaapi_task_t* const task = kaapi_stack_toptask(stack);

    kaapi_task_initadaptive( stack, task, static_mainentrypoint<SelfType>, (void*)work, KAAPI_TASK_ADAPT_DEFAULT);
    kaapi_stack_pushtask(stack);
    kaapi_finalize_steal(stack, task, NULL, 0);
    kaapi_sched_sync(stack);
  }

  static unsigned int __attribute__((unused)) self_id()
  {
    return (unsigned int)pthread_self();
  } 

} // kaapi_utils namespace


#endif // ! _XKAAPI_KAAPI_UTILS_H
