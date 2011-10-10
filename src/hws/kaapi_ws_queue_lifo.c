/*
 ** kaapi_ws_queue_lifo.c
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 ** 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 ** 
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 ** 
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 ** 
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 ** 
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 ** 
 */
#include <stdint.h>
#include "kaapi_impl.h"

/* needed for config_hws_counters */
#include "kaapi_hws.h"

#include "kaapi_ws_queue.h"

/* Current implementation is a bounded queue manage.
*/
typedef struct lifo_queue
{
  kaapi_ws_lock_t lock;                      /* toremove, use block lock */
  __attribute__((aligned)) volatile unsigned int top; /* first avail */
#define CONFIG_QUEUE_SIZE (4096 * 10)
  kaapi_task_t* tasks[CONFIG_QUEUE_SIZE];
} lifo_queue_t;


static kaapi_ws_error_t push(
    kaapi_ws_block_t* ws_bloc __attribute__((unused)),
    void* p, 
    kaapi_task_t* task
)
{
  /* assume q->top < sizeof(q->tasks) */
  lifo_queue_t* const q = (lifo_queue_t*)p;
  
  kaapi_ws_lock_lock(&q->lock);
  q->tasks[q->top++] = task;
  kaapi_ws_lock_unlock(&q->lock);
  
  return KAAPI_WS_ERROR_SUCCESS;
}

static void callback_empty( kaapi_task_t* task)
{
  kaapi_stealcontext_t* const sc = kaapi_task_getargst(task, kaapi_stealcontext_t);
  KAAPI_ATOMIC_DECR(&sc->header.msc->thieves.count);
  /* deallocate the stealcontext */
  free(sc);
  task->sp = NULL;
}


static kaapi_ws_error_t steal
(
    kaapi_ws_block_t* ws_bloc __attribute__((unused)),
    void* p,
    kaapi_listrequest_t* lr,
    kaapi_listrequest_iterator_t* lri
)
{
  lifo_queue_t* const q = (lifo_queue_t*)p;
  kaapi_request_t* req;

  /* avoid to take the lock */
  if (q->top == 0) return KAAPI_WS_ERROR_EMPTY;
  
  kaapi_ws_lock_lock(&q->lock);

  req = kaapi_listrequest_iterator_get(lr, lri);
  while ((req != NULL) && q->top)
  {
    kaapi_task_t* const task = q->tasks[--q->top];
    
    kaapi_task_body_t task_body = kaapi_task_getbody(task);
    if (task_body == kaapi_hws_adapt_body)
    {
      /* todo */
      kaapi_task_steal_adapt(0, task, lr, lri, &callback_empty);

      /* task is not empty */
      if (task->sp != NULL)
      {
	++q->top;
	break ;
      }

      /* the request may have been updated. reread it. */
      req = kaapi_listrequest_iterator_get(lr, lri);
    }
    else /* != kaapi_hws_adapt_body */
    {
      /* special case of kaapi_task_steal_dfg */
      req->thief_sp = task->sp;
      req->thief_task = task;
      kaapi_writemem_barrier();
      kaapi_request_replytask(req, KAAPI_REQUEST_S_OK);
      KAAPI_DEBUG_INST( kaapi_listrequest_iterator_countreply( lri ) );
      /* next request */
      req = kaapi_listrequest_iterator_next(lr, lri);
    }
  }

  /* q->top consistency */
  kaapi_writemem_barrier();

  kaapi_ws_lock_unlock(&q->lock);
  
  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t pop
(
    kaapi_ws_block_t* ws_bloc,
    void* p,
    kaapi_request_t* req
)
{
  /* currently fallback to steal */
  
  const kaapi_processor_id_t kid = kaapi_get_current_processor()->kid;
  
  kaapi_listrequest_t lr;
  kaapi_listrequest_iterator_t lri;
  
#if CONFIG_HWS_COUNTERS
  kaapi_hws_inc_pop_counter(p);
#endif
  
  kaapi_bitmap_clear(&lr.bitmap);
  kaapi_bitmap_set(&lr.bitmap, kid);
// NO NEEDED here: kid is emitting a dummy request with its own bitmap
//  memcpy(&lr.requests[kid], req, sizeof(kaapi_request_t));
  kaapi_listrequest_iterator_init(&lr, &lri);
  
  return steal(ws_bloc, p, &lr, &lri);
}


/* exported
 */
kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void)
{
  kaapi_ws_queue_t* const wsq = kaapi_ws_queue_alloc(sizeof(lifo_queue_t));
  
  lifo_queue_t* const q = (lifo_queue_t*)wsq->data;
  
  wsq->push = push;
  wsq->steal = steal;
  wsq->pop = pop;
  
  kaapi_ws_lock_init(&q->lock);
  q->top = 0;
  
  return wsq;
}
