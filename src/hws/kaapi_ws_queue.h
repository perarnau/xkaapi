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
  kaapi_ws_error_t (*push)(void*, void*, void*);
  kaapi_ws_error_t (*stealn)(void*, kaapi_listrequest_t*, kaapi_listrequest_iterator_t*);
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
  q->stealn = NULL;
  q->pop = NULL;
  q->destroy = NULL;

  return q;
}

static inline kaapi_ws_error_t kaapi_ws_queue_push
(kaapi_ws_queue_t* q, void* task, void* data)
{
  return q->push((void*)q->data, task, data);
}

static inline kaapi_ws_error_t kaapi_ws_queue_stealn
(kaapi_ws_queue_t* q, kaapi_listrequest_t* r, kaapi_listrequest_iterator_t* i)
{
  return q->stealn((void*)q->data, r, i);
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
