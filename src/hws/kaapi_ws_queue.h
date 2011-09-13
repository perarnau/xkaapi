#ifndef KAAPI_WS_QUEUE_H_INCLUDED
# define KAAPI_WS_QUEUE_H_INCLUDED


#include <stdlib.h>
#include <sys/types.h>


struct kaapi_task;
struct kaapi_request;


typedef struct kaapi_ws_queue
{
  int (*push)(void*, struct kaapi_task*);
  int (*stealn)(void*, struct kaapi_request*);
  void (*destroy)(void*);

  unsigned char data[1];

} kaapi_ws_queue_t;


kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void);

static inline kaapi_ws_queue_t* kaapi_ws_queue_create(size_t size)
{
  const size_t size = offsetof(kaapi_ws_queue_t, data);
  kaapi_ws_queue_t* const q = malloc(size);
  kaapi_assert(q);

  q->push = NULL;
  q->stealn = NULL;
  q->destroy = NULL;

  return q;
}

static inline int kaapi_ws_queue_push(kaapi_ws_queue_t* q)
{
  return q->push((void*)q->data);
}

static inline int kaapi_ws_queue_stealn(kaapi_ws_queue_t* q)
{
  return q->stealn((void*)q->data);
}

static inline int kaapi_ws_queue_destroy(kaapi_ws_queue_t* q)
{
  q->destroy((void*)q->data);
  free(q);
}


#endif /* ! KAAPI_WS_QUEUE_H_INCLUDED */
