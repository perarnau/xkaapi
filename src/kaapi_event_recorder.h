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

/** Mask of events
    The mask is set at runtime to select events that will be registered
    to file.
    The bit i-th in the mask is 1 iff the event number i is registered.
    It means that not more than sizeof(kaapi_event_mask_type_t)*8 events 
    are available in Kaapi.
*/
extern uint64_t kaapi_event_mask;

/** Statup time for event recorder.
    Serve as the epoch for the event recorder sub library.
*/
extern uint64_t kaapi_event_startuptime;

/* the datation used for event */
static inline uint64_t kaapi_event_date(void)
{ return kaapi_get_elapsedns(); }

/** Initialize the event recorder sub library.
    Must be called before any other functions.
*/
void kaapi_eventrecorder_init(void);


/** Destroy the event recorder sub library.
*/
void kaapi_eventrecorder_fini(void);

/** Open a new buffer for kprocessor kid.
    Kid must be less than KAAPI_MAX_PROCESSOR.
    \param kid the identifier of a kprocessor. Must be unique between threads.
    \retval a new eventbuffer
*/
extern kaapi_event_buffer_t* kaapi_event_openbuffer(int kid);


/** Flush the event buffer evb and return and new buffer.
    \param evb the event buffer to flush
    \retval the new event buffer to use for futur records.
*/
extern kaapi_event_buffer_t* kaapi_event_flushbuffer( kaapi_event_buffer_t* evb );

/** Flush the event buffer and close the associated file descriptor.
*/
extern void kaapi_event_closebuffer( kaapi_event_buffer_t* evb );


/** Fence: write all flushed buffers and return
*/
void kaapi_event_fencebuffers(void);


/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline kaapi_event_buffer_t* kaapi_event_push0(
    kaapi_event_buffer_t*   evb, 
    uint64_t                tclock,
    uint8_t                 eventno
)
{
  tclock -= kaapi_event_startuptime;
  kaapi_event_t* evt = &evb->buffer[evb->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = evb->kid;
  evt->gid     = 0;
  evt->date    = tclock;

  if (evb->pos == KAAPI_EVENT_BUFFER_SIZE)
    return kaapi_event_flushbuffer(evb);
  return evb;
}

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline kaapi_event_buffer_t*  kaapi_event_push1(
    kaapi_event_buffer_t*   evb, 
    uint64_t                tclock,
    uint8_t                 eventno, 
    uintptr_t               p0 
)
{
  tclock -= kaapi_event_startuptime;
  kaapi_event_t* evt = &evb->buffer[evb->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = evb->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.i    = p0;

  if (evb->pos == KAAPI_EVENT_BUFFER_SIZE)
    return kaapi_event_flushbuffer(evb);
  return evb;
}

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline kaapi_event_buffer_t*  kaapi_event_push2(
    kaapi_event_buffer_t*   evb, 
    uint64_t                tclock,
    uint8_t                 eventno, 
    uintptr_t               p0, 
    uintptr_t               p1
)
{
  tclock -= kaapi_event_startuptime;
  kaapi_event_t* evt = &evb->buffer[evb->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = evb->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.i    = p0;
  evt->d1.i    = p1;

  if (evb->pos == KAAPI_EVENT_BUFFER_SIZE)
    return kaapi_event_flushbuffer(evb);
  return evb;
}

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline kaapi_event_buffer_t*  kaapi_event_push3(
    kaapi_event_buffer_t*   evb, 
    uint64_t                tclock,
    uint8_t                 eventno, 
    uintptr_t               p0, 
    uintptr_t               p1,
    uintptr_t               p2
)
{
  tclock -= kaapi_event_startuptime;
  kaapi_event_t* evt = &evb->buffer[evb->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = evb->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.i    = p0;
  evt->d1.i    = p1;
  evt->d2.i    = p2;

  if (evb->pos == KAAPI_EVENT_BUFFER_SIZE)
    return kaapi_event_flushbuffer(evb);
  return evb;
}


#if defined(KAAPI_USE_PERFCOUNTER)
#  define KAAPI_IFUSE_TRACE(kproc,inst) \
    if (kproc->eventbuffer) { inst; }
#  define KAAPI_EVENT_PUSH0(kproc, kthread, eventno ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
    {\
      kaapi_event_buffer_t* evb = kaapi_event_push0(kproc->eventbuffer, kaapi_event_date(), eventno ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }
#  define KAAPI_EVENT_PUSH1(kproc, kthread, eventno, p1 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
    {\
      kaapi_event_buffer_t* evb = kaapi_event_push1(kproc->eventbuffer, kaapi_event_date(), eventno, (uintptr_t)(p1) ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }
#  define KAAPI_EVENT_PUSH2(kproc, kthread, eventno, p1, p2 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
    {\
      kaapi_event_buffer_t* evb = kaapi_event_push2(kproc->eventbuffer, kaapi_event_date(), eventno, (uintptr_t)(p1), (uintptr_t)(p2) ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }
#  define KAAPI_EVENT_PUSH3(kproc, kthread, eventno, p1, p2, p3 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
    {\
      kaapi_event_buffer_t* evb = kaapi_event_push3(kproc->eventbuffer, kaapi_event_date(), eventno, (uintptr_t)(p1), (uintptr_t)(p2), (uintptr_t)(p3) ); \
      if (evb != kproc->eventbuffer) kproc->eventbuffer = evb;\
    }

/* push new event with given date (value return by kaapi_event_date()) */
#  define KAAPI_EVENT_PUSH0_AT(kproc, tclock, kthread, eventno ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH0_(kproc, tclock, kthread, eventno )
#  define KAAPI_EVENT_PUSH1_AT(kproc, tclock, kthread, eventno, p1 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH1_(kproc, tclock, kthread, eventno, (void*)(p1))
#  define KAAPI_EVENT_PUSH2_AT(kproc, tclock, kthread, eventno, p1, p2 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH2_(kproc, tclock, kthread, eventno, (void*)(p1), (void*)(p2))
#  define KAAPI_EVENT_PUSH3_AT(kproc, tclock, kthread, eventno, p1, p2, p3 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH3_(kproc, tclock, kthread, eventno, (void*)(p1), (void*)(p2), (void*)(p3))
#else
#  define KAAPI_IFUSE_TRACE(kproc,inst)
#  define KAAPI_EVENT_PUSH0(kproc, kthread, eventno ) 
#  define KAAPI_EVENT_PUSH1(kproc, kthread, eventno, p1 )
#  define KAAPI_EVENT_PUSH2(kproc, kthread, eventno, p1, p2 )
#  define KAAPI_EVENT_PUSH3(kproc, kthread, eventno, p1, p2, p3 )
#  define KAAPI_EVENT_PUSH0_AT(kproc, tclock, kthread, eventno ) 
#  define KAAPI_EVENT_PUSH1_AT(kproc, tclock, kthread, eventno, p1 )
#  define KAAPI_EVENT_PUSH2_AT(kproc, tclock, kthread, eventno, p1, p2 )
#  define KAAPI_EVENT_PUSH3_AT(kproc, tclock, kthread, eventno, p1, p2, p3 )
#endif


#if defined(__cplusplus)
}
#endif

#endif
