#include "kaapi_ws_queue.h"


typedef struct lifo_queue
{
  /* todo */

  unsigned int fubar;

} lifo_queue_t;


static kaapi_ws_error_t push(void* p, xxx_kaapi_task_t* task)
{
  lifo_queue_t* const q = (lifo_queue_t*)p;
  return KAAPI_WS_ERROR_EMPTY;
}


static kaapi_ws_error_t stealn(void* p, xxx_kaapi_request_t* reqs)
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
  wsq->stealn = stealn;
  wsq->destroy = destroy;

  /* todo: initialize q */

  return wsq;
}
