/*
 ** kaapi_ws_queue_lifo.c
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
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

/* Current implementation is a bounded cicular queue manage
   with THE protocol using workqueue.
*/
typedef struct fifo_queue
{
  kaapi_atomic_t     pusher_lock;
  kaapi_workqueue_t  wq;
  kaapi_atomic_t     stealer_lock;
#define CONFIG_QUEUE_SIZE 128UL
  kaapi_task_t* tasks[CONFIG_QUEUE_SIZE];
} fifo_queue_t;


static kaapi_ws_error_t push(
    void* p, 
    kaapi_task_t* task
)
{
  /* assume q->top < sizeof(q->tasks) */
  fifo_queue_t* const q = (fifo_queue_t*)p;
  kaapi_ws_lock_lock(&q->pusher_lock);
  kaapi_workqueue_index_t beg = kaapi_workqueue_range_begin(&q->wq);
  kaapi_assert( 0 == kaapi_workqueue_push( &q->wq, beg-1 ) );
  q->tasks[ (unsigned long)(beg-1) % CONFIG_QUEUE_SIZE] = task;
  kaapi_ws_lock_unlock(&q->pusher_lock);
  
  return KAAPI_WS_ERROR_SUCCESS;
}


/** Try to steal nreq works into the workqueue
*/
static kaapi_ws_error_t steal
(
    void* p,
    kaapi_listrequest_t* lr,
    kaapi_listrequest_iterator_t* lri
)
{
  fifo_queue_t* const q = (fifo_queue_t*)p;
  kaapi_request_t* req;
  uintptr_t state;
  
  /* avoid to take the lock */
  if ( kaapi_workqueue_isempty(&q->wq) ) return KAAPI_WS_ERROR_EMPTY;
  
  /* take the stealer lock : SHOULD BE THE WSBLOC LOCK */
  kaapi_ws_lock_lock(&q->stealer_lock);
  
  nreq = kaapi_listrequest_iterator_count( lri );
  req = kaapi_listrequest_iterator_get(lr, lri);

  while ((req != NULL) && q->top)
  {
    kaapi_task_t* const task = q->tasks[--q->top];
    kaapi_task_body_t task_body = kaapi_task_getbody(task);
    
    if (task_body == kaapi_hws_adapt_body)
    {
      /* adaptive task. refer to kaapi_sched_stealstack.c */
      
      kaapi_stealcontext_t* const sc = kaapi_task_getargst(task, kaapi_stealcontext_t);
      if (sc == NULL) continue ;
      
      if (sc->header.flag & KAAPI_SC_INIT)
      {
        if (sc->splitter != NULL)
        {
          //	  state = kaapi_task_orstate(task, KAAPI_MASK_BODY_STEAL);
          //	  if (!kaapi_task_state_isterm(state))
          uintptr_t orig_state = kaapi_task_getstate(task);
          /* do not steal if terminated */
          if (    (orig_state == KAAPI_TASK_STATE_INIT) && (orig_state == KAAPI_TASK_STATE_EXEC)
               && likely(kaapi_task_casstate(task, orig_state, KAAPI_TASK_STATE_STEAL))
             )
          {
            kaapi_task_splitter_t splitter = sc->splitter;
            void* const argsplitter = sc->argsplitter;
            
            if (splitter != NULL)
            {
              const kaapi_ws_error_t err = kaapi_task_splitter_adapt(0, task, splitter, argsplitter, lr, lri);
              if (err == KAAPI_WS_ERROR_EMPTY)
              {
                KAAPI_ATOMIC_DECR(&sc->header.msc->thieves.count);
                
                /* deallocate the stealcontext */
                free(sc);
                task->sp = NULL;
              }
              
              /* update request */
              req = kaapi_listrequest_iterator_get(lr, lri);
            }
            /* reset initial state of the task */
            kaapi_task_setstate(task, orig_state);
          }
        }
      }
    }
    else
    {
#if CONFIG_HWS_COUNTERS
      kaapi_hws_inc_steal_counter(p, req->ident);
#endif
      req->thief_task = task;
      kaapi_request_replytask( req, KAAPI_REQUEST_S_OK); /* success of steal */
      
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
  
  return steal(p, &lr, &lri);
}


/* exported
 */
kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void)
{
  kaapi_ws_queue_t* const wsq = kaapi_ws_queue_alloc(sizeof(fifo_queue_t));
  
  fifo_queue_t* const q = (fifo_queue_t*)wsq->data;
  
  wsq->push = push;
  wsq->steal = steal;
  wsq->pop = pop;
  
  kaapi_ws_lock_init(&q->lock);
  q->top = 0;
  
  return wsq;
}
