#include "kaapi_impl.h"
#include "kaapi_ws_queue.h"


typedef struct lifo_queue
{
  unsigned int top; /* first avail */
  kaapi_task_t tasks[32];
} lifo_queue_t;


static kaapi_ws_error_t push(void* p, kaapi_task_body_t body, void* arg)
{
  /* assume q->top < sizeof(q->tasks) */

  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_task_t* const task = &q->tasks[q->top++];
  kaapi_task_initdfg(task, body, arg);
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
  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_request_t* req;

  req = kaapi_listrequest_iterator_get(lr, lri);
  while ((req != NULL) && (q->top))
  {
    /* refer to kaapi_task_splitter_dfg_single */

    kaapi_task_t* const task = &q->tasks[--q->top];
    kaapi_reply_t* const rep = kaapi_request_getreply(req);
    kaapi_task_body_t task_body = kaapi_task_getbody(task);
    kaapi_tasksteal_arg_t* const argsteal = (kaapi_tasksteal_arg_t*)rep->udata;

    argsteal->origin_thread = thread;
    argsteal->origin_task = task;
    argsteal->origin_fmt = kaapi_format_resolvebybody(task_body);
    argsteal->war_param = 0;
    argsteal->cw_param = 0;
    rep->u.s_task.body = kaapi_tasksteal_body;

    _kaapi_request_reply(req, KAAPI_REPLY_S_TASK);

    req = kaapi_listrequest_iterator_next(lr, lri);
  }

  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t pop(void* p)
{
  lifo_queue_t* const q = (lifo_queue_t*)p;
  return KAAPI_WS_ERROR_EMPTY;
}


static void destroy(void* p)
{
  lifo_queue_t* const q = (lifo_queue_t*)p;
}


/* exported
 */

kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void)
{
  kaapi_ws_queue_t* const wsq =
    kaapi_ws_queue_create(sizeof(lifo_queue_t));

  lifo_queue_t* const q = (lifo_queue_t*)wsq->data;

  wsq->push = push;
  wsq->steal = steal;
  wsq->pop = pop;
  wsq->destroy = destroy;

  q->top = 0;

  return wsq;
}
