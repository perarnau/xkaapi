#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


static kaapi_ws_error_t pop
(
 void* p,
 kaapi_thread_context_t* thread,
 kaapi_request_t* req
)
{
  /* todo: kaapi_thread_execframe */

  kaapi_processor_t* const kproc = *(kaapi_processor_t**)p;

  printf("%s(%u)\n", __FUNCTION__, kproc->kid);

  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t steal
(
 void* p,
 kaapi_thread_context_t* thread,
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  /* todo: kaapi_sched_stealprocessor.c */

  kaapi_processor_t* const kproc = *(kaapi_processor_t**)p;

  printf("%s(%u)\n", __FUNCTION__, kproc->kid);

  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t push
(
 void* p,
 kaapi_task_body_t body,
 void* arg
)
{
  /* pushing in a non local queue is not allowed */
  kaapi_assert(0);
  return KAAPI_WS_ERROR_FAILURE;
}


/* exported
 */

kaapi_ws_queue_t* kaapi_ws_queue_create_kproc(kaapi_processor_t* kproc)
{
  /* points to the given kproc and use local function to steal, pop */

  kaapi_ws_queue_t* const wsq =
    kaapi_ws_queue_create(sizeof(kaapi_processor_t*));

  void* const aliasing_fix = (void*)wsq->data;
  *(void**)aliasing_fix = (void*)kproc;

  wsq->push = push;
  wsq->steal = steal;
  wsq->pop = pop;

  return wsq;
}
