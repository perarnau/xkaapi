#include "kaapi_impl.h"
#include "kaapi_ws_queue.h"


typedef struct lifo_queue
{
  unsigned int top; /* first avail */
  kaapi_atomic_t lock; /* toremove */
  kaapi_task_t tasks[32];
} lifo_queue_t;


static kaapi_ws_error_t push(void* p, kaapi_task_body_t body, void* arg)
{
  /* assume q->top < sizeof(q->tasks) */

  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_task_t* task;

  kaapi_sched_lock(&q->lock);
  task = &q->tasks[q->top++];
  kaapi_task_initdfg(task, body, arg);
  kaapi_sched_unlock(&q->lock);

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

  kaapi_sched_lock(&q->lock);

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

  kaapi_sched_unlock(&q->lock);

  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t pop
(
 void* p,
 kaapi_thread_context_t* thread,
 kaapi_request_t* req
)
{
  /* currently, the kproc request is passed even if not posted */

  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_ws_error_t error = KAAPI_WS_ERROR_EMPTY;

  kaapi_sched_lock(&q->lock);

  if (q->top)
  {
    kaapi_task_t* const task = &q->tasks[--q->top];
    kaapi_reply_t* const rep = kaapi_request_getreply(req);
    kaapi_task_body_t task_body = kaapi_task_getbody(task);
    kaapi_tasksteal_arg_t* const argsteal = (kaapi_tasksteal_arg_t*)rep->udata;

    error = KAAPI_WS_ERROR_SUCCESS;

    argsteal->origin_thread = thread;
    argsteal->origin_task = task;
    argsteal->origin_fmt = kaapi_format_resolvebybody(task_body);
    argsteal->war_param = 0;
    argsteal->cw_param = 0;
    rep->u.s_task.body = kaapi_tasksteal_body;

    _kaapi_request_reply(req, KAAPI_REPLY_S_TASK);
  }
  kaapi_sched_unlock(&q->lock);

  return error;
}


static unsigned int is_empty(void* p)
{
  lifo_queue_t* const q = (lifo_queue_t*)p;
  unsigned int is_empty;

  kaapi_sched_lock(&q->lock);
  is_empty = (q->top == 0);
  kaapi_sched_unlock(&q->lock);

  return is_empty;
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
  wsq->is_empty = is_empty;

  kaapi_sched_initlock(&q->lock);
  q->top = 0;

  return wsq;
}
