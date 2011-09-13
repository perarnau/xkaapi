#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#include "kaapi_ws_queue.h"


typedef struct lifo_queue
{
  /* todo */

  unsigned int fubar;

} lifo_queue_t;


kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void)
{
  kaapi_ws_queue_t* const wsq =
    kaapi_ws_queue_create(sizeof(lifo_queue_t));

#if 0 /* todo */
  wsq->push = ;
  wsq->stealn = ;
  wsq->destroy = ;
#endif /* todo */

  return wsq;
}
