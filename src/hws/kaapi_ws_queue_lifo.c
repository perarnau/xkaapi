#include <stdint.h>
#include "kaapi_impl.h"

/* needed for config_hws_counters */
#include "kaapi_hws.h"

#include "kaapi_ws_queue.h"


typedef struct lifo_queue
{
  kaapi_ws_lock_t lock; /* toremove, use block lock */
  __attribute__((aligned)) unsigned int top; /* first avail */
#define CONFIG_QUEUE_SIZE 128
  kaapi_task_t tasks[CONFIG_QUEUE_SIZE];
} lifo_queue_t;


static kaapi_ws_error_t push(void* p, kaapi_task_body_t body, void* arg)
{
  /* assume q->top < sizeof(q->tasks) */

  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_task_t* task;

  kaapi_ws_lock_lock(&q->lock);
  task = &q->tasks[q->top++];
  kaapi_task_initdfg(task, body, arg);
  kaapi_ws_lock_unlock(&q->lock);

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
  unsigned int top;
  uintptr_t state;

  /* printf("[%u] %s\n", kaapi_get_self_kid(), __FUNCTION__); */

  /* avoid to take the lock */
  if (q->top == 0) return KAAPI_WS_ERROR_EMPTY;

  kaapi_ws_lock_lock(&q->lock);

  /* work on a local copy of top, never updated */
  top = q->top;

  req = kaapi_listrequest_iterator_get(lr, lri);
  while ((req != NULL) && top)
  {
    kaapi_task_t* const task = &q->tasks[--top];
    kaapi_reply_t* const rep = kaapi_request_getreply(req);
    kaapi_task_body_t task_body = kaapi_task_getbody(task);

    if (task_body == kaapi_hws_adapt_body)
    {
      /* adaptive task. refer to kaapi_sched_stealstack.c */

      kaapi_stealcontext_t* const sc =
	kaapi_task_getargst(task, kaapi_stealcontext_t);

      if (sc->header.flag & KAAPI_SC_INIT)
      {
	if (sc->splitter != NULL)
	{
	  state = kaapi_task_orstate(task, KAAPI_MASK_BODY_STEAL);
	  if (!kaapi_task_state_isterm(state))
	  {
	    kaapi_task_splitter_t splitter = sc->splitter;
	    void* const argsplitter = sc->argsplitter;

	    if (splitter != NULL)
	    {
	      const kaapi_ws_error_t err = kaapi_task_splitter_adapt
		(thread, task, splitter, argsplitter, lr, lri);
	      if (err == KAAPI_WS_ERROR_EMPTY)
	      {
		KAAPI_ATOMIC_DECR(&sc->header.msc->thieves.count);
		sc->splitter = NULL;
	      }

	      /* update request */
	      req = kaapi_listrequest_iterator_get(lr, lri);
	    }

	    kaapi_task_andstate(task, ~KAAPI_MASK_BODY_STEAL);
	  }
	}
      }
    }
    else
    {
      /* dfg task, refer to kaapi_task_splitter_dfg_single */

      /* todo: kaapi_sched_stealstack.c */

      kaapi_tasksteal_arg_t* argsteal;
      const kaapi_format_t* format;

      format = kaapi_format_resolvebybody(task_body);

      if (format == NULL)
      {
	printf("-- format == null\n");
	exit(-1);
	continue ;
      }

      state = kaapi_task_orstate(task, KAAPI_MASK_BODY_STEAL);
      if (kaapi_task_isstealable(task) == 0) continue ;
      if (kaapi_task_state_isstealable(state) == 0) continue ;

      argsteal = (kaapi_tasksteal_arg_t*)rep->udata;

#if CONFIG_HWS_COUNTERS
      kaapi_hws_inc_steal_counter(p, req->kid);
#endif

      argsteal->origin_thread = thread;
      argsteal->origin_task = task;
      argsteal->origin_fmt = format;
      argsteal->war_param = 0;
      argsteal->cw_param = 0;
      rep->u.s_task.body = kaapi_tasksteal_body;

      _kaapi_request_reply(req, KAAPI_REPLY_S_TASK);

      req = kaapi_listrequest_iterator_next(lr, lri);
    }
  }

  kaapi_ws_lock_unlock(&q->lock);

  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t pop
(
 void* p,
 kaapi_thread_context_t* thread,
 kaapi_request_t* req
)
{
  /* currently fallback to steal */

  const kaapi_processor_id_t kid =
    kaapi_get_current_processor()->kid;

  kaapi_listrequest_t lr;
  kaapi_listrequest_iterator_t lri;

#if CONFIG_HWS_COUNTERS
  kaapi_hws_inc_pop_counter(p);
#endif

  kaapi_bitmap_clear(&lr.bitmap);
  kaapi_bitmap_set(&lr.bitmap, kid);
  memcpy(&lr.requests[kid], req, sizeof(kaapi_request_t));
  kaapi_listrequest_iterator_init(&lr, &lri);

  return steal(p, thread, &lr, &lri);
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

  kaapi_ws_lock_init(&q->lock);
  q->top = 0;

  return wsq;
}
