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

/** Size of the event mask 
*/
typedef uint32_t kaapi_event_mask_type_t;

/** Mask of events
    The mask is set at runtime to select events that will be registered
    to file.
    The bit i-th in the mask is 1 iff the event number i is registered.
    It means that not more than sizeof(kaapi_event_mask_type_t)*8 events 
    are available in Kaapi.
*/
extern uint64_t kaapi_event_mask;

/** Heler for creating mask from an event
*/
#define KAAPI_EVT_MASK(eventno) \
  ((kaapi_event_mask_type_t)1 << eventno)
  
/** Standard mask' sets of event */

/* The following set is always in the mask
*/
#define KAAPI_EVT_MASK_STARTUP \
    (  KAAPI_EVT_MASK(KAAPI_EVT_KPROC_START) \
     | KAAPI_EVT_MASK(KAAPI_EVT_KPROC_STOP) \
    )

#define KAAPI_EVT_MASK_COMPUTE \
    (  KAAPI_EVT_MASK(KAAPI_EVT_TASK_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_TASK_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_STATIC_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_STATIC_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_STATIC_TASK_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_STATIC_TASK_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_FOREACH_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_FOREACH_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_FOREACH_STEAL) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_CPU_HTOD_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_CPU_HTOD_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_CPU_DTOH_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_CPU_DTOH_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_SYNC_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_SYNC_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_MEM_ALLOC_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_CUDA_MEM_ALLOC_END) \
    )

#define KAAPI_EVT_MASK_IDLE \
    (  KAAPI_EVT_MASK(KAAPI_EVT_SCHED_IDLE_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SCHED_IDLE_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SCHED_SUSPEND_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SCHED_SUSPEND_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SCHED_SUSPWAIT_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SCHED_SUSPWAIT_END) \
    )

#define KAAPI_EVT_MASK_STEALOP \
    (  KAAPI_EVT_MASK(KAAPI_EVT_REQUESTS_BEG) \
     | KAAPI_EVT_MASK(KAAPI_EVT_REQUESTS_END) \
     | KAAPI_EVT_MASK(KAAPI_EVT_STEAL_OP) \
     | KAAPI_EVT_MASK(KAAPI_EVT_SEND_REPLY) \
     | KAAPI_EVT_MASK(KAAPI_EVT_RECV_REPLY) \
    )


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
static inline void KAAPI_EVENT_PUSH0_(
    kaapi_processor_t*      kproc, 
    uint64_t                tclock,
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  tclock -= kaapi_default_param.startuptime;
  kaapi_event_t* evt = &kproc->eventbuffer->buffer[kproc->eventbuffer->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = kproc->kid;
  evt->ktype   = kproc->proc_type;
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
static inline void KAAPI_EVENT_PUSH1_(
    kaapi_processor_t*      kproc, 
    uint64_t                tclock,
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno, 
    void*                   p0 
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  tclock -= kaapi_default_param.startuptime;
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
static inline void KAAPI_EVENT_PUSH2_(
    kaapi_processor_t*      kproc, 
    uint64_t                tclock,
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno, 
    void*                   p0, 
    void*                   p1
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  tclock -= kaapi_default_param.startuptime;
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

/** Push a new event into the eventbuffer of the kprocessor.
    Assume that the event buffer was allocated into the kprocessor.
    Current implementation only work if library is compiled 
    with KAAPI_USE_PERFCOUNTER flag.
*/
static inline void KAAPI_EVENT_PUSH3_(
    kaapi_processor_t*      kproc, 
    uint64_t                tclock,
    kaapi_thread_context_t* thread, 
    uint8_t                 eventno, 
    void*                   p0, 
    void*                   p1,
    void*                   p2
)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  tclock -= kaapi_default_param.startuptime;
  kaapi_event_t* evt = &kproc->eventbuffer->buffer[kproc->eventbuffer->pos++];
  evt->evtno   = eventno;
  evt->type    = 0;
  evt->kid     = kproc->kid;
  evt->gid     = 0;
  evt->date    = tclock;
  evt->d0.p    = p0;
  evt->d1.p    = p1;
  evt->d2.p    = p2;

  if (kproc->eventbuffer->pos == KAAPI_EVENT_BUFFER_SIZE)
    kaapi_event_flushbuffer(kproc);
#endif
}

/* the datation used for event */
static inline uint64_t kaapi_event_date(void)
{ return kaapi_get_elapsedns(); }

#if defined(KAAPI_USE_PERFCOUNTER)
#  define KAAPI_IFUSE_TRACE(kproc,inst) \
    if (kproc->eventbuffer) { inst; }
#  define KAAPI_EVENT_PUSH0(kproc, kthread, eventno ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH0_(kproc, kaapi_event_date(), kthread, eventno )
#  define KAAPI_EVENT_PUSH1(kproc, kthread, eventno, p1 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH1_(kproc, kaapi_event_date(), kthread, eventno, (void*)(p1))
#  define KAAPI_EVENT_PUSH2(kproc, kthread, eventno, p1, p2 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH2_(kproc, kaapi_event_date(), kthread, eventno, (void*)(p1), (void*)(p2))
#  define KAAPI_EVENT_PUSH3(kproc, kthread, eventno, p1, p2, p3 ) \
    if ((kproc->eventbuffer) && (kaapi_event_mask & KAAPI_EVT_MASK(eventno)))\
      KAAPI_EVENT_PUSH3_(kproc, kaapi_event_date(), kthread, eventno, (void*)(p1), (void*)(p2), (void*)(p3))

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
