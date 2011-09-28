/*
** kaapi_hws_sched_sync.c
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


#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


/* toremove */

static kaapi_atomic_t hws_sync;

void kaapi_hws_sched_init_sync(void)
{
  KAAPI_ATOMIC_WRITE(&hws_sync, 0);
}

void kaapi_hws_sched_inc_sync(void)
{
  KAAPI_ATOMIC_INCR(&hws_sync);
}

void kaapi_hws_sched_dec_sync(void)
{
  KAAPI_ATOMIC_DECR(&hws_sync);
}

/* to remove */


void kaapi_hws_sched_sync(void)
{
  /* todo: q->push must push in local stack too,
     so that syncing is equivalent to sched_sync
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* ctxt;
  kaapi_thread_context_t* thread;
  int err;

  while (1)
  {
    ctxt = kproc->thread;

    thread = kaapi_hws_emitsteal(kproc);
    if (thread == NULL)
    {
      __sync_synchronize();
      if (KAAPI_ATOMIC_READ(&hws_sync) == 0)
      {
	/* no more tasks in queue to sync with */
	break ;
      }

      continue ;
    }

    if (thread != ctxt)
    {
      /* also means ctxt is empty, so push ctxt into the free list */
      kaapi_setcontext( kproc , 0);
      /* wait end of thieves before releasing a thread */
      kaapi_sched_lock(&kproc->lock);
      kaapi_lfree_push( kproc, ctxt );
      kaapi_sched_unlock(&kproc->lock);
    }
    kaapi_setcontext(kproc, thread);

    if (kproc->thread->stack.sfp->tasklist == 0)
      err = kaapi_thread_execframe(kproc->thread);
    else
      err = kaapi_thread_execframe_tasklist( kproc->thread );

    if (err == EWOULDBLOCK)
      kaapi_sched_suspend(kproc);
  }

  /* sync local stack */
  kaapi_sched_sync();
}


void kaapi_hws_sched_sync_once(void)
{
  /* todo: q->push must push in local stack too,
     so that syncing is equivalent to sched_sync
   */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* ctxt;
  kaapi_thread_context_t* thread;
  int err;

  ctxt = kproc->thread;

  thread = kaapi_hws_emitsteal(kproc);
  if (thread == NULL) return ;
  if (thread != ctxt)
  {
    /* also means ctxt is empty, so push ctxt into the free list */
    kaapi_setcontext( kproc , 0);
    /* wait end of thieves before releasing a thread */
    kaapi_sched_lock(&kproc->lock);
    kaapi_lfree_push( kproc, ctxt );
    kaapi_sched_unlock(&kproc->lock);
  }
  kaapi_setcontext(kproc, thread);

  if (kproc->thread->stack.sfp->tasklist == 0)
    err = kaapi_thread_execframe(kproc->thread);
  else
    err = kaapi_thread_execframe_tasklist( kproc->thread );

  if (err == EWOULDBLOCK) kaapi_sched_suspend(kproc);
}
