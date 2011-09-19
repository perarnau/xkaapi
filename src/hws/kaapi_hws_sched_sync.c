#include <unistd.h> /* toremove, usleep */

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

    if (kproc->thread->sfp->tasklist == 0)
      err = kaapi_thread_execframe(kproc->thread);
    else
      err = kaapi_thread_execframe_tasklist( kproc->thread );

    if (err == EWOULDBLOCK)
      kaapi_sched_suspend(kproc);
  }

  /* sync local stack */
  kaapi_sched_sync();
}
