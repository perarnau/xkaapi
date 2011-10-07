/*
** kaapi_ws_queue.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#ifndef KAAPI_WS_QUEUE_H_INCLUDED
# define KAAPI_WS_QUEUE_H_INCLUDED


#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>
#include "kaapi_impl.h"


#ifndef CONFIG_HWS_COUNTERS
# error "kaapi_hws_h_not_included"
#endif

/* fwd declaration */
struct kaapi_ws_block;


typedef struct kaapi_ws_queue
{
  /* TODO: use an union to isolate the ops in cache */
  /* TODO: allocation should be cache aligned */

  /* interface required to implement workstealing queues.
     the queue must be created with kaapi_ws_queue_alloc,
     which takes care of initializing default fields with
     correct values.
     the concrete queue size must be passed as an argument
     to kaapi_ws_queue_alloc, which takes care of allocating
     a buffer large enough for the concrete implementation.
     this buffer is then retrieved by casting the data field
     with the concrete implementation type.
     push, steal, and pop methods must points to the
     corresponding concrete implementations.
     destroy method is optionnal.
     refer to kaapi_ws_queue_create_lifo for a complete
     example of queue creation.
  */

  kaapi_ws_error_t (*push)(struct kaapi_ws_block*, void*, kaapi_task_t* );
  kaapi_ws_error_t (*steal)(struct kaapi_ws_block*, void*, kaapi_listrequest_t*, kaapi_listrequest_iterator_t*);
  kaapi_ws_error_t (*pop)(struct kaapi_ws_block*, void*, kaapi_request_t*);
  void (*destroy)(void*);

#if CONFIG_HWS_COUNTERS
  /* counters, one per remote kid */
  /* todo: put in the ws_block */
  kaapi_atomic_t steal_counters[KAAPI_MAX_PROCESSOR];
  kaapi_atomic_t pop_counter;
#endif

  unsigned char data[1];

} kaapi_ws_queue_t;


/** Create a LIFO queue of task
  \retval 0 in case of failure
  \retval a new lifo queue that manage tasks.
*/
kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void);

/** Create a queue attached to a given k-proc.
  The behavior of this queue is to steal task inside a K-processors,
  iterating through all possible locations where tasks are stored into k-processor.
  \retval 0 in case of failure
  \retval a new lifo queue that manage tasks.
*/
kaapi_ws_queue_t* kaapi_ws_queue_create_kproc(struct kaapi_processor_t*);

/** Default empty destroy function
*/
extern void kaapi_ws_queue_unimpl_destroy(void*);

/** Allocate a queue data structure that stores up to size byte.
    This function must be called in order to attach data with the kaapi_ws_queue_t
    data structure.
*/
static inline kaapi_ws_queue_t* kaapi_ws_queue_alloc(size_t size)
{
  const size_t total_size = offsetof(kaapi_ws_queue_t, data) + size;

  kaapi_ws_queue_t* q;
  const int err = posix_memalign((void**)&q, sizeof(void*), total_size);
  kaapi_assert(err == 0);

  q->push    = NULL;
  q->steal   = NULL;
  q->pop     = NULL;
  q->destroy = kaapi_ws_queue_unimpl_destroy;

#if CONFIG_HWS_COUNTERS
  memset(q->steal_counters, 0, sizeof(q->steal_counters));
  memset(&q->pop_counter, 0, sizeof(q->pop_counter));
#endif

  return q;
}


/** Push a task into the queue
    The queue does not make copies of pushed tasks.
    \param queue [IN/OUT] the queue that will store the newly pushed task
    \param task [IN] the task to push into the queue
*/
static inline kaapi_ws_error_t kaapi_ws_queue_push(
  struct kaapi_ws_block* ws_bloc,
  kaapi_ws_queue_t* queue, 
  kaapi_task_t* task
)
{
  return queue->push(ws_bloc, (void*)queue->data, task);
}


/** Push a task into the queue
    \param queue [IN/OUT] the queue where to pop task
    \param request [IN/OUT] where to store result of the pop operation
*/
static inline kaapi_ws_error_t kaapi_ws_queue_pop
(
  struct kaapi_ws_block* ws_bloc,
  kaapi_ws_queue_t* queue, 
  kaapi_request_t* request
)
{
  return queue->pop(ws_bloc, (void*)queue->data, request);
}


/** Steal a task into the queue.
    \param queue [IN/OUT] the queue to steal
    \param lrequests [IN/OUT] the list of requests
    \param iter_requests [IN/OUT] the iterator over the list of requests
*/
static inline kaapi_ws_error_t kaapi_ws_queue_steal
(
  struct kaapi_ws_block* ws_bloc,
  kaapi_ws_queue_t* queue,
  kaapi_listrequest_t* lrequests,
  kaapi_listrequest_iterator_t* iter_requests
)
{
  return queue->steal(ws_bloc, (void*)queue->data, lrequests, iter_requests);
}


/** Destroy a queue
    Depend on the trampoline function set at queue creation time.
*/
static inline void kaapi_ws_queue_destroy(kaapi_ws_queue_t* q)
{
  q->destroy((void*)q->data);
  free(q);
}


#endif /* ! KAAPI_WS_QUEUE_H_INCLUDED */
