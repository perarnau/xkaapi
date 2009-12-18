#include "kaapi.h"
#include "kaapi_utils.h"


void kaapi_utils::fail_requests
(
 kaapi_stack_t* victim_stack,
 kaapi_task_t* task,
 int count,
 kaapi_request_t* requests
)
{
  for (; count; ++requests)
  {
    if (!kaapi_request_ok(requests))
      continue ;
    
    kaapi_request_reply(victim_stack, task, requests, 0, 0, 0);
    
    --count;
  }
}
