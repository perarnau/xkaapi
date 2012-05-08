/*
** kaapi_cuda_data_async.h
** xkaapi
** 
** Created on Jan 2012
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** Joao.Lima@imag.fr
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

#ifndef KAAPI_CUDA_EVENT_H_INCLUDED
#define KAAPI_CUDA_EVENT_H_INCLUDED

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_event.h"

#if defined(KAAPI_USE_PERFCOUNTER)

static inline kaapi_event_buffer_t*
kaapi_cuda_event_push0(
    kaapi_event_buffer_t*   evb, 
    uint64_t                tclock,
    uint8_t                 eventno
)
{
#if defined(KAAPI_USE_CUPTI)
  tclock -= kaapi_default_param.cudastartuptime;
#else
  tclock -= kaapi_default_param.startuptime;
#endif
  kaapi_event_t* evt = &evb->buffer[evb->pos++];
  evt->evtno   = eventno;
#if defined(KAAPI_USE_CUDA)
  evt->type    = kaapi_processor_get_type(kaapi_get_current_processor());
#else
  evt->type    = 0;
#endif
  evt->kid     = evb->kid;
  evt->gid     = 0;
  evt->date    = tclock;

  if (evb->pos == KAAPI_EVENT_BUFFER_SIZE)
    return kaapi_event_flushbuffer(evb);
  return evb;
}

#  define KAAPI_CUDA_EVENT_PUSH0(kproc, kthread, eventno ) \
    if ((kproc->eventbuffer) )\
    {\
      kaapi_event_buffer_t* evb = kaapi_cuda_event_push0(kproc->eventbuffer, kaapi_event_date(), eventno ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }

#  define KAAPI_CUDA_EVENT_PUSH0_(kproc, tclock, eventno ) \
    if ((kproc->eventbuffer) )\
    {\
      kaapi_event_buffer_t* evb = kaapi_cuda_event_push0(kproc->eventbuffer, tclock, eventno ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }

#else

#define KAAPI_CUDA_EVENT_PUSH0(kproc, kthread, eventno)

#define KAAPI_CUDA_EVENT_PUSH0_(kproc, tclock, eventno) 

#endif

#endif /* KAAPI_CUDA_EVENT_H_INCLUDED */
