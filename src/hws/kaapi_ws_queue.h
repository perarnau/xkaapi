#ifndef KAAPI_WS_QUEUE_H_INCLUDED
# define KAAPI_WS_QUEUE_H_INCLUDED


#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#include "kaapi_impl.h"


typedef void* xxx_kaapi_task_t;
typedef void* xxx_kaapi_request_t;


typedef enum kaapi_ws_error
{
  KAAPI_WS_ERROR_SUCCESS = 0,
  KAAPI_WS_ERROR_EMPTY,
  KAAPI_WS_ERROR_FAILURE
} kaapi_ws_error_t;


typedef struct kaapi_ws_queue
{
  kaapi_ws_error_t (*push)(void*, xxx_kaapi_task_t*);
  kaapi_ws_error_t (*stealn)(void*, xxx_kaapi_request_t*);
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
  q->stealn = NULL;
  q->destroy = NULL;

  return q;
}

static inline kaapi_ws_error_t kaapi_ws_queue_push
(kaapi_ws_queue_t* q, xxx_kaapi_task_t* task)
{
  return q->push((void*)q->data, task);
}

static inline kaapi_ws_error_t kaapi_ws_queue_stealn
(kaapi_ws_queue_t* q, xxx_kaapi_request_t* req)
{
  return q->stealn((void*)q->data, req);
}

static inline void kaapi_ws_queue_destroy(kaapi_ws_queue_t* q)
{
  q->destroy((void*)q->data);
  free(q);
}


#endif /* ! KAAPI_WS_QUEUE_H_INCLUDED */
