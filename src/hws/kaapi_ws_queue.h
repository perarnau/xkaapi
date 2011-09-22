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


typedef enum kaapi_ws_error
{
  KAAPI_WS_ERROR_SUCCESS = 0,
  KAAPI_WS_ERROR_EMPTY,
  KAAPI_WS_ERROR_FAILURE
} kaapi_ws_error_t;


typedef struct kaapi_ws_queue
{
  /* TODO: use an union to isolate the ops in cache */
  /* TODO: allocation should be cache aligned */

  kaapi_ws_error_t (*push)(void*, kaapi_task_body_t, void*);
  kaapi_ws_error_t (*steal)(void*, kaapi_thread_context_t*, kaapi_listrequest_t*, kaapi_listrequest_iterator_t*);
  kaapi_ws_error_t (*pop)(void*, kaapi_thread_context_t*, kaapi_request_t*);
  void (*destroy)(void*);

#if CONFIG_HWS_COUNTERS
  /* counters, one per remote kid */
  /* todo: put in the ws_block */
  kaapi_atomic_t steal_counters[KAAPI_MAX_PROCESSOR];
  kaapi_atomic_t pop_counter;
#endif

  unsigned char data[1];

} kaapi_ws_queue_t;


kaapi_ws_queue_t* kaapi_ws_queue_create_lifo(void);

static void kaapi_ws_queue_unimpl_destroy(void* fu)
{
  /* destroy may be unimplemented */
  fu = fu;
}

static inline kaapi_ws_queue_t* kaapi_ws_queue_create(size_t size)
{
  const size_t total_size = offsetof(kaapi_ws_queue_t, data) + size;
  kaapi_ws_queue_t* const q = malloc(total_size);
  kaapi_assert(q);

  q->push = NULL;
  q->steal = NULL;
  q->pop = NULL;
  q->destroy = kaapi_ws_queue_unimpl_destroy;

#if CONFIG_HWS_COUNTERS
  memset(q->steal_counters, 0, sizeof(q->steal_counters));
  memset(&q->pop_counter, 0, sizeof(q->pop_counter));
#endif

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
(
 kaapi_ws_queue_t* q,
 kaapi_thread_context_t* t,
 kaapi_request_t* r
)
{
  return q->pop((void*)q->data, t, r);
}

static inline void kaapi_ws_queue_destroy(kaapi_ws_queue_t* q)
{
  q->destroy((void*)q->data);
  free(q);
}


#endif /* ! KAAPI_WS_QUEUE_H_INCLUDED */
