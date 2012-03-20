/*
** kaapi_hws.h
** xkaapi
** 
**
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
#include "kaapi_impl.h"


static kaapi_ws_error_t pop
(
    kaapi_ws_block_t* ws_bloc __attribute__((unused)),
    void* p,
    kaapi_request_t* req
)
{
  /* todo, kaapi_sched_idle.c, local wakeup first */
  /* kaapi_processor_t* const kproc = *(kaapi_processor_t**)p; */
  return KAAPI_WS_ERROR_EMPTY;
}


static kaapi_ws_error_t steal
(
    kaapi_ws_block_t* ws_bloc __attribute__((unused)),
    void* p,
    kaapi_listrequest_t* lr,
    kaapi_listrequest_iterator_t* lri
)
{
  kaapi_processor_t* const kproc = *(kaapi_processor_t**)p;
  const int saved_count = (int)kaapi_listrequest_iterator_count(lri);

  /* TODO: synchronize with flat workstealing, i.e. share the same lock */
  kaapi_sched_lock( &kproc->lock );
  kaapi_sched_stealprocessor(kproc, lr, lri);
  kaapi_sched_unlock(&kproc->lock);

  if (kaapi_listrequest_iterator_count(lri) == saved_count)
    return KAAPI_WS_ERROR_EMPTY;
  return KAAPI_WS_ERROR_SUCCESS;
}


static kaapi_ws_error_t push
(
    kaapi_ws_block_t* ws_bloc __attribute__((unused)),
    void* p,
    kaapi_task_t* task
)
{
  /* pushing in a non local queue is not allowed */
  kaapi_assert(0);
  return KAAPI_WS_ERROR_FAILURE;
}


/* exported
 */

kaapi_ws_queue_t* kaapi_ws_queue_create_kproc(kaapi_processor_t* kproc)
{
  /* points to the given kproc and use local function to steal, pop */

  kaapi_ws_queue_t* const wsq =
    kaapi_ws_queue_alloc(sizeof(kaapi_processor_t*));

  void* const aliasing_fix = (void*)wsq->data;
  *(void**)aliasing_fix = (void*)kproc;

  wsq->push  = push;
  wsq->steal = steal;
  wsq->pop   = pop;

  return wsq;
}
