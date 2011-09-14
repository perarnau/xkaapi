/*
** kaapi_staticsched.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
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
#ifndef _KAAPI_EVENT_H_
#define _KAAPI_EVENT_H_

#include "kaapi_impl.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** Definition of internal KAAPI event.
    Not that any extension or modification of the events list should
    be reflected in the utility bin/kaapi_event_reader file.
*/
#define KAAPI_EVT_TASK_BEG        1     /* begin execution of task */
#define KAAPI_EVT_TASK_END        2     /* end execution of task */

#define KAAPI_EVT_FRAME_TL_BEG    3     /* begin execution of frame tasklist */
#define KAAPI_EVT_FRAME_TL_END    4     /* end execution of frame tasklist */

#define KAAPI_EVT_STATIC_BEG      5     /* begin of static schedule computation */
#define KAAPI_EVT_STATIC_END      6     /* end of static schedule computation */

#define KAAPI_EVT_STATIC_TASK_BEG 7     /* begin of sub task exec for static schedule */
#define KAAPI_EVT_STATIC_TASK_END 8     /* end of sub task exec for static schedule */

#define KAAPI_EVT_SCHED_IDLE_BEG  9     /* begin when k-processor starts to be idle */
#define KAAPI_EVT_SCHED_IDLE_END  10    /* end when k-processor starts to be idle */

#define KAAPI_EVT_NUMBER          10    /* total number of events ! */

/* ........................................ Implementation notes ........................................*/

/* entry of event. 3 entry per event.
*/
typedef union {
    void*     p;
    uintptr_t i; 
} kaapi_event_data_t;

/* Event 
*/
typedef struct kaapi_event_t {
  uint8_t     evtno;      /* event number */
  uint8_t     type;       /* event number */
  uint16_t    kid;        /* processor identifier */
  uint32_t    gid;        /* global identifier */
  uint64_t    date;       /* nano second */
  kaapi_event_data_t d0;
  kaapi_event_data_t d1;
  kaapi_event_data_t d2;
} kaapi_event_t;

/** Event Buffer 
*/
#define KAAPI_EVENT_BUFFER_SIZE 1024
typedef struct kaapi_event_buffer_t {
  uint32_t      pos;
  int           fd;
  kaapi_event_t buffer[KAAPI_EVENT_BUFFER_SIZE];
} kaapi_event_buffer_t;


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

/* Signal handler attached to:
    - SIGINT
    - SIGQUIT
    - SIGABRT
    - SIGTERM
    - SIGSTOP
  when the library is configured with --with-perfcounter in order to flush some counters.
*/
extern void _kaapi_signal_dump_counters(int);


#if defined(__cplusplus)
}
#endif

#endif /* _KAAPI_EVENT_H_ */
