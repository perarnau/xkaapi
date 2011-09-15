#include <unistd.h> /* toremove, usleep */

#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


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
#if 1 /* toremove */
      /* are all the levels empty */
      kaapi_ws_block_t* block;
      kaapi_hws_level_t* level;
      kaapi_hws_levelid_t levelid;
      unsigned int i;

      for (levelid = 0; levelid < hws_level_count; ++levelid)
      {
	if (!kaapi_hws_is_levelid_set(levelid)) continue ;

	level = &hws_levels[levelid];
	for (i = 0; i < level->block_count; ++i)
	{
	  block = &level->blocks[i];
	  if (!kaapi_ws_queue_is_empty(block->queue)) break;
	}
	if (i != level->block_count) break ;
      }
      /* all level are empty */
      if (levelid == hws_level_count)
      {
	usleep(1000);
	return ;
      }
#endif /* toremove */

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
}
