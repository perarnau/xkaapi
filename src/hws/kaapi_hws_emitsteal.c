/* todo: kaapi_hws_level_iterator_t
 */

#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


static kaapi_listrequest_t hws_requests;


static inline kaapi_ws_block_t* get_self_ws_block
(kaapi_processor_t* self, kaapi_hws_levelid_t levelid)
{
  /* return the ws block for the kproc at given level */
  /* assume self, level dont overflow */

  return hws_levels[levelid].kid_to_block[self->kid];
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
 kaapi_ws_block_t* block,
 kaapi_processor_t* kproc,
 kaapi_reply_t* reply,
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

  kaapi_listrequest_iterator_update(lr, lri, &block->kid_mask);

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


static kaapi_thread_context_t* steal_block_leaves
(
 kaapi_ws_block_t* block,
 kaapi_processor_t* kproc,
 kaapi_reply_t* reply,
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  /* steal randomly amongst the block leaves */

  kaapi_processor_id_t kid;
  kaapi_ws_block_t* leaf_block;

  /* actually block->kid_count == 1 */
  if (block->kid_count <= 1) return NULL;

 redo_rand:
  kid = block->kids[rand() % block->kid_count];
  if (kid == kproc->kid) goto redo_rand;

  /* get the leaf block (ie. block at flat level) */
  leaf_block = hws_levels[KAAPI_HWS_LEVELID_FLAT].kid_to_block[kid];

  return steal_block(leaf_block, kproc, reply, lr, lri);
}


static kaapi_thread_context_t* steal_level
(
 kaapi_hws_level_t* level,
 kaapi_processor_t* kproc,
 kaapi_reply_t* reply,
 kaapi_listrequest_t* lr,
 kaapi_listrequest_iterator_t* lri
)
{
  kaapi_thread_context_t* thread = NULL;
  unsigned int i;

  for (i = 0; i < level->block_count; ++i)
  {
    thread = steal_block(&level->blocks[i], kproc, reply, lr, lri);
    if (thread != NULL) break ;
  }

  return thread;
}


static kaapi_thread_context_t* pop_block
(
 kaapi_ws_block_t* block,
 kaapi_processor_t* kproc
)
{
  /* not a real steal operation, dont actually post */
  kaapi_request_t* const req = &hws_requests.requests[kproc->kid];
  kaapi_reply_t* const rep = &kproc->thread->static_reply;
  kaapi_ws_error_t err;

  req->kid = kproc->kid;
  req->reply = rep;

  rep->offset = 0;
  rep->preempt = 0;
  rep->status = KAAPI_REQUEST_S_POSTED;

  err = kaapi_ws_queue_pop(block->queue, kproc->thread, req);
  if (err != KAAPI_WS_ERROR_SUCCESS) return NULL;
   
  switch (kaapi_reply_status(rep))
  {
  case KAAPI_REPLY_S_TASK_FMT:
    {
      kaapi_format_t* const format =
	kaapi_format_resolvebyfmit(rep->u.s_taskfmt.fmt);
      rep->u.s_task.body = format->entrypoint[kproc->proc_type];
      kaapi_assert_debug(rep->u.s_task.body);

    } /* KAAPI_REPLY_S_TASK_FMT */

  case KAAPI_REPLY_S_TASK:
    {
      kaapi_thread_t* const self_thread =
	kaapi_threadcontext2thread(kproc->thread);

      kaapi_task_init
      (
       kaapi_thread_toptask(self_thread),
       rep->u.s_task.body,
       (void*)(rep->udata + rep->offset)
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
      return rep->u.s_thread;

    } /* KAAPI_REPLY_S_THREAD */

  default: break ;
  }

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
  kaapi_hws_levelid_t child_levelid;
  kaapi_hws_levelid_t levelid = 0;
  kaapi_reply_t* reply;
  kaapi_listrequest_iterator_t lri;

  /* dont fail_request with an uninitialized bitmap */
  kaapi_listrequest_iterator_clear(&lri);

  /* pop locally without emitting request */
  /* todo: kaapi_ws_queue_pop should fit the steal interface */
  block = get_self_ws_block(kproc, KAAPI_HWS_LEVELID_FLAT);
  thread = pop_block(block, kproc);
  if (thread != NULL) return thread;

  /* post the stealing request */
  kproc->issteal = 1;
  reply = post_request(kproc);
  kaapi_thread_reset(kproc->thread);

  /* foreach parent level, pop. if pop failed, steal in level children. */
  for (levelid = KAAPI_HWS_LEVELID_FIRST; levelid < hws_level_count; ++levelid)
  {
    if (!(kaapi_hws_is_levelid_set(levelid))) continue ;

    block = get_self_ws_block(kproc, levelid);

    /* dont steal at flat level during ascension */
    if (levelid != KAAPI_HWS_LEVELID_FLAT)
    {
      /* todo: this is a pop, not a steal */
      /* todo: dont rely upon thread for termination condition */
      thread = steal_block(block, kproc, reply, &hws_requests, &lri);
      if (thread != NULL)
      {
	/* something replied, we are done */
	goto on_done;
      }

      /* popping failed at this level, steal in level children */
      for (child_levelid = levelid - 1; child_levelid >= 0; --child_levelid)
      {
	kaapi_hws_level_t* const child_level = &hws_levels[child_levelid];
	if (!kaapi_hws_is_levelid_set(child_levelid)) continue ;

	thread = steal_level(child_level, kproc, reply, &hws_requests, &lri);
	if (thread != NULL) goto on_done;
      }

    } /* levelid != KAAPI_HWS_LEVELID_FLAT */

    /* child level stealing failed, steal in block leaf local queues */
    thread = steal_block_leaves(block, kproc, reply, &hws_requests, &lri);
    if (thread != NULL) goto on_done;

    /* next level */
  }

 on_done:
  fail_requests(&hws_requests, &lri);

  kproc->issteal = 0;

  return thread;
}
