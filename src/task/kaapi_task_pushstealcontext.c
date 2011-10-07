/*
** kaapi_task_pushstealcontext.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#include "kaapi_impl.h"


static void acquire_kproc_lock(kaapi_processor_t* kproc)
{
#if defined(KAAPI_SCHED_LOCK_CAS)
  while (!kaapi_sched_trylock( kproc ))
    kaapi_slowdown_cpu();
#else /* cannot rely on kaapi_sched_trylock... */
  while (1)
  {
    if (KAAPI_ATOMIC_DECR(&kproc->lock) ==0)
      break;
    while (KAAPI_ATOMIC_READ(&kproc->lock) <=0)
      kaapi_slowdown_cpu();
  }
#endif
}

static inline int get_request_count(kaapi_listrequest_t* klr)
{
  /* todo: better way to count request */
  return kaapi_bitmap_count(*(volatile kaapi_bitmap_value_t*)&klr->bitmap);
}

static void wait_for_aggregation(kaapi_processor_t* kproc)
{
  /* assuming nearly all processors take part to the algorithm */
  const int concurrency = kaapi_getconcurrency();

  kaapi_listrequest_t* const klr = &kproc->hlrequests;

  int prev_count;
  int iter;
  int delay;

  for (iter = 0; iter < concurrency; ++iter)
  {
    prev_count = get_request_count(klr);
    for (delay = 2000; delay; --delay)
    {
      if (get_request_count(klr) != prev_count)
	break ;
      kaapi_slowdown_cpu();
    }

    /* the count didnot change, we are done */
    if (delay == 0) break;
  }
}

static void do_initial_split(kaapi_processor_t* kproc)
{
  kaapi_listrequest_t* const klr = &kproc->hlrequests;
  kaapi_listrequest_iterator_t kli;

  kaapi_listrequest_iterator_init(klr, &kli);
  if (!kaapi_listrequest_iterator_empty(&kli))
  {
    kaapi_sched_stealprocessor(kproc, klr, &kli);

    /* reply failed for all others requests */
    kaapi_request_t* kreq = kaapi_listrequest_iterator_get(klr, &kli);
    while (kreq != 0)
    {
#warning TODO HERE
#if 0
      _kaapi_request_reply(kreq, KAAPI_REPLY_S_NOK);
      kreq = kaapi_listrequest_iterator_next(klr, &kli);
#endif
    }
  }
}

/**
*/
kaapi_stealcontext_t* kaapi_task_begin_adaptive
(
  kaapi_thread_t*       thread,
  int                   flag,
  kaapi_task_splitter_t splitter,
  void*                 argsplitter
)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();

  kaapi_stealcontext_t*   sc;
  kaapi_thread_context_t* self_thread;
  kaapi_task_t*           task;
  kaapi_frame_t           frame;
  
  kaapi_thread_save_frame(thread, &frame);

  self_thread = kaapi_self_thread_context();
  
  /* todo: should be pushed cacheline aligned */
  sc = (kaapi_stealcontext_t*)kaapi_thread_pushdata_align
    (thread, sizeof(kaapi_stealcontext_t), sizeof(void*));
  kaapi_assert_debug(sc != 0);

#warning "TO DO HERE"
//  sc->preempt           = &self_thread->static_reply.preempt;
  sc->splitter          = splitter;
  sc->argsplitter       = argsplitter;
  sc->header.flag       = flag;
  sc->header.msc        = sc; /* self pointer to detect master */
  sc->header.ktr	    = 0;

  if (flag & KAAPI_SC_PREEMPTION)
  {
    /* if preemption, thief list used ... */
    KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
    sc->thieves.list.head = 0;
    sc->thieves.list.tail = 0;
  }
  else
  {
    /* ... otherwise thief count */
    KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);
  }

  sc->frame                 = frame;
  sc->save_splitter         = 0;
  sc->save_argsplitter      = 0;

  task = kaapi_thread_toptask(thread);
  sc->ownertask = task;

  /* ok is initialized */
  sc->header.flag       |= KAAPI_SC_INIT;

  /* split before publishing the task */
  if (flag & KAAPI_SC_HWS_SPLITTER)
  {
    /* todo: levelid should be an argument */
    static const kaapi_hws_levelid_t levelid = KAAPI_HWS_LEVELID_NUMA;
    kaapi_assert_debug(!(flag & KAAPI_SC_AGGREGATE));
    kaapi_hws_splitter(sc, splitter, argsplitter, levelid);
  }

  if (flag & KAAPI_SC_AGGREGATE)
    acquire_kproc_lock(kproc);

  /* change our execution state before pushing the task.
     this is needed for the assumptions made by the
     kaapi_task_lock_steal rouines, see for comments.
   */
  kaapi_task_init(task, kaapi_adapt_body, sc);

  /* barrier done by kaapi_thread_pushtask */
  kaapi_thread_pushtask(thread);

  if (flag & KAAPI_SC_AGGREGATE)
  {
    wait_for_aggregation(kproc);
    do_initial_split(kproc);
    kaapi_sched_unlock( &kproc->lock );
  }

  return sc;
}
