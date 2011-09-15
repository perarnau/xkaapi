#ifndef KAAPI_WS_QUEUE_H_INCLUDED
# define KAAPI_WS_QUEUE_H_INCLUDED


#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#include "kaapi_impl.h"


typedef enum kaapi_ws_error
{
  KAAPI_WS_ERROR_SUCCESS = 0,
  KAAPI_WS_ERROR_EMPTY,
  KAAPI_WS_ERROR_FAILURE
} kaapi_ws_error_t;


typedef struct kaapi_ws_queue
{
  kaapi_ws_error_t (*push)(void*, kaapi_task_body_t, void*);
  kaapi_ws_error_t (*steal)(void*, kaapi_thread_context_t*, kaapi_listrequest_t*, kaapi_listrequest_iterator_t*);
  kaapi_ws_error_t (*pop)(void*);
  void (*destroy)(void*);

  unsigned char data[1];

} kaapi_ws_queue_t;


kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void);

static inline kaapi_ws_queue_t* kaapi_ws_queue_create(size_t size)
{
  const size_t total_size = offsetof(kaapi_ws_queue_t, data) + size;
  kaapi_ws_queue_t* const q = malloc(total_size);
  kaapi_assert(q);

  q->push = NULL;
  q->steal = NULL;
  q->pop = NULL;
  q->destroy = NULL;

  return q;
}

static inline kaapi_ws_error_t kaapi_ws_queue_push
(kaapi_ws_queue_t* q, kaapi_task_body_t body, void* arg)
{
  return q->push((void*)q->data, body, arg);
}

static inline kaapi_ws_error_t kaapi_ws_queue_steal
(
 kaapi_ws_queue_t* q,
 kaapi_thread_context_t* t,
 kaapi_listrequest_t* r,
 kaapi_listrequest_iterator_t* i
)
{
  return q->steal((void*)q->data, t, r, i);
}

static inline kaapi_ws_error_t kaapi_ws_queue_pop
(kaapi_ws_queue_t* q)
{
  return q->pop((void*)q->data);
}

static inline void kaapi_ws_queue_destroy(kaapi_ws_queue_t* q)
{
  q->destroy((void*)q->data);
  free(q);
}


#endif /* ! KAAPI_WS_QUEUE_H_INCLUDED */
