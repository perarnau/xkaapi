#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


static kaapi_listrequest_t hws_requests;


static inline kaapi_ws_block_t* get_self_ws_block
(kaapi_processor_t* self, unsigned int level)
{
  /* return the ws block for the kproc at given level */
  /* assume self, level dont overflow */

  return hws_levels[level].kid_to_block[self->kid];
}


static int select_block_victim
(
 kaapi_processor_t* kproc,
 kaapi_victim_t* victim,
 kaapi_selecvictim_flag_t flag
)
{
  /* select a victim in the block */
  /* assume block->kid_count > 1 */

  kaapi_ws_block_t* block;
  unsigned int kid;

  if (flag != KAAPI_SELECT_VICTIM) return 0;

  block = kproc->fnc_selecarg[0];

 redo_rand:
  kid = block->kids[rand() % block->kid_count];
  if (kid == kproc->kid) goto redo_rand;

  victim->kproc = kaapi_all_kprocessors[kid];
  victim->level = 0;

  return 0;
}


static void fail_requests
(
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  kaapi_request_t* req = kaapi_listrequest_iterator_get(lr, lri);

  while (req != NULL)
  {
    _kaapi_request_reply(req, KAAPI_REPLY_S_NOK);
    req = kaapi_listrequest_iterator_next(lr, lri);
  }
}


static kaapi_thread_context_t* steal_block
(
 kaapi_processor_t* kproc,
 kaapi_reply_t* reply,
 kaapi_ws_block_t* block,
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  while (!kaapi_sched_trylock(&block->lock))
  {
    if (kaapi_reply_test(reply))
      goto on_request_replied;
  }

  /* got the lock: reply and unlock */

  kaapi_ws_queue_steal(block->queue, kproc->thread, lr, lri);

  kaapi_sched_unlock(&block->lock);

 on_request_replied:
  kaapi_replysync_data(reply);

  switch (kaapi_reply_status(reply))
  {
  case KAAPI_REPLY_S_TASK_FMT:
    {
      kaapi_format_t* const format =
	kaapi_format_resolvebyfmit(reply->u.s_taskfmt.fmt);
      reply->u.s_task.body = format->entrypoint[kproc->proc_type];
      kaapi_assert_debug(reply->u.s_task.body);

    } /* KAAPI_REPLY_S_TASK_FMT */

  case KAAPI_REPLY_S_TASK:
    {
      kaapi_thread_t* const self_thread =
	kaapi_threadcontext2thread(kproc->thread);

      kaapi_task_init
      (
       kaapi_thread_toptask(self_thread),
       reply->u.s_task.body,
       (void*)(reply->udata + reply->offset)
      );

      kaapi_thread_pushtask(self_thread);

#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
      return kproc->thread;

    } /* KAAPI_REPLY_S_TASK */

  case KAAPI_REPLY_S_THREAD:
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
      return reply->u.s_thread;

    } /* KAAPI_REPLY_S_THREAD */

  default: break ;
  }

  return NULL;
}


static kaapi_thread_context_t* steal_level
(
 kaapi_processor_t* kproc,
 kaapi_reply_t* reply,
 kaapi_hws_level_t* level,
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  kaapi_thread_context_t* thread = NULL;
  unsigned int i;

  for (i = 0; i < level->block_count; ++i)
  {
    thread = steal_block(kproc, reply, &level->blocks[i], lr, lri);
    if (thread != NULL) break ;
  }

  return thread;
}


static kaapi_thread_context_t* pop_ws_block
(
 kaapi_processor_t* kproc,
 kaapi_ws_block_t* block
)
{
  return NULL;
}


static kaapi_reply_t* post_request(kaapi_processor_t* kproc)
{
  kaapi_request_t* const req = &hws_requests.requests[kproc->kid];
  kaapi_reply_t* const rep = &kproc->thread->static_reply;

  /* from kaapi_mt_machine.h/kaapi_request_post */
  
  req->kid = kproc->kid;
  req->reply = rep;

  rep->offset = 0;
  rep->preempt = 0;
  rep->status = KAAPI_REQUEST_S_POSTED;
  kaapi_writemem_barrier();
  kaapi_bitmap_set(&hws_requests.bitmap, kproc->kid);

  return rep;
}

kaapi_thread_context_t* kaapi_hws_emitsteal(kaapi_processor_t* kproc)
{
  kaapi_thread_context_t* thread = NULL;
  kaapi_ws_block_t* block;
  unsigned int level = 0;
  kaapi_reply_t* reply;
  kaapi_listrequest_iterator_t lri;

  kproc->issteal = 1;

  reply = post_request(kproc);

  for (; level < hws_level_count; ++level)
  {
    block = get_self_ws_block(kproc, level);

    /* steal in the level queue first */
    {
    }
    /* steal in the level queue first */

#if 0
    /* randomly steal in a random kproc local queue */
    if (block->kid_count >= 2)
    {
      kaapi_selectvictim_fnc_t saved_fn = kproc->fnc_select;
      void* saved_arg = kproc->fnc_selecarg[0];

      kproc->fnc_select = select_block_victim;
      kproc->fnc_selecarg[0] = block;
      thread = kaapi_sched_emitsteal(kproc);
      kproc->fnc_select = saved_fn;
      kproc->fnc_selecarg[0] = saved_arg;

      if (thread != NULL)
      {
	goto on_replied;
      }
    }
    /* randomly steal in a random kproc local queue */
#endif

    /* wait for lock or reply */
    while (1)
    {
      if (kaapi_sched_trylock(&block->lock))
      {
	/* got the lock, reply all. if there is no
	   task to extract for this level, go next.
	 */

	kaapi_ws_queue_t* const queue = block->queue;
	const kaapi_ws_error_t error =
	  kaapi_ws_queue_steal(queue, kproc->thread, &hws_requests, &lri);
	if (error == KAAPI_WS_ERROR_EMPTY)
	{
	  goto next_level;
	}
      }

      if (kaapi_reply_test(reply))
      {
	/* request got replied */
	goto on_replied;
      }

    }

  next_level: ;
  }

 on_replied:
  /* unlock all levels < level */
  if (level > 0)
  {
    printf("[%u] unlocking from %u\n", kproc->kid, level);
    for (; level; --level)
    {
      block = get_self_ws_block(kproc, level - 1);
      if (block->kid_count <= 1) continue ;
      kaapi_sched_unlock(&block->lock);
    }
  }

  return thread;
}
