/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#ifndef _KAAPI_EVENT_RECORDER_H_
#define _KAAPI_EVENT_RECORDER_H_ 1

#include "config.h"
#include "kaapi_event.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** Flush the event buffer of the kproc
    \param kproc the kaapi_processor owner of the event buffer to flush
*/
extern void kaapi_event_flushbuffer( kaapi_processor_t* kproc );

/** Flush the event buffer and close the associated file descriptor.
*/
extern void kaapi_event_closebuffer( kaapi_processor_t* kproc );

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline void kaapi_event_push0_(
    kaapi_processor_t*      kproc, 
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  uint64_t tclock = kaapi_get_elapsedns();
  kaapi_event_t* evt = &kproc->eventbuffer->buffer[kproc->eventbuffer->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = kproc->kid;
  evt->gid     = 0;
  evt->date    = tclock;

  if (kproc->eventbuffer->pos == KAAPI_EVENT_BUFFER_SIZE)
    kaapi_event_flushbuffer(kproc);
#endif
}

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline void kaapi_event_push1_(
    kaapi_processor_t*      kproc, 
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno, 
    void*                   p0 
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  uint64_t tclock = kaapi_get_elapsedns();
  kaapi_event_t* evt = &kproc->eventbuffer->buffer[kproc->eventbuffer->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = kproc->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.p    = p0;

  if (kproc->eventbuffer->pos == KAAPI_EVENT_BUFFER_SIZE)
    kaapi_event_flushbuffer(kproc);
#endif
}

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline void kaapi_event_push2_(
    kaapi_processor_t*      kproc, 
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno, 
    void*                   p0, 
    void*                   p1
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  uint64_t tclock = kaapi_get_elapsedns();
  kaapi_event_t* evt = &kproc->eventbuffer->buffer[kproc->eventbuffer->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = kproc->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.p    = p0;
  evt->d1.p    = p1;

  if (kproc->eventbuffer->pos == KAAPI_EVENT_BUFFER_SIZE)
    kaapi_event_flushbuffer(kproc);
#endif
}

#if defined(KAAPI_USE_PERFCOUNTER)
#  define kaapi_event_push0(kproc, kthread, eventno ) \
    if (kproc->eventbuffer) kaapi_event_push0_(kproc, kthread, eventno )
#  define kaapi_event_push1(kproc, kthread, eventno, p1 ) \
    if (kproc->eventbuffer) kaapi_event_push1_(kproc, kthread, eventno, (void*)(p1))
#  define kaapi_event_push2(kproc, kthread, eventno, p1, p2 ) \
    if (kproc->eventbuffer) kaapi_event_push2_(kproc, kthread, eventno, (void*)(p1), (void*)(p2))
#else
#  define kaapi_event_push0(kproc, kthread, eventno ) 
#  define kaapi_event_push1(kproc, kthread, eventno, p1 )
#  define kaapi_event_push2(kproc, kthread, eventno, p1, p2 )
#endif


#if defined(__cplusplus)
}
#endif

#endif
