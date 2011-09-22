#include <string.h>
#include <stdint.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


int kaapi_hws_splitter
(
 kaapi_stealcontext_t* sc,
 kaapi_task_splitter_t splitter,
 void* args,
 kaapi_hws_levelid_t levelid
)
{
  /* split equivalently among all the nodes of a given level */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();

  /* todo: dynamic allocation */
  kaapi_request_t reqs[KAAPI_MAX_PROCESSOR];
  kaapi_reply_t reps[KAAPI_MAX_PROCESSOR];

  int retval;
  kaapi_hws_level_t* level;
  unsigned int count;
  unsigned int i;

  kaapi_assert_debug(kaapi_is_levelid_set(levelid));
  kaapi_assert_debug(count < KAAPI_MAX_PROCESSOR);

  level = &hws_levels[levelid];
  count = level->block_count;

  /* generate a request array and call the splitter */
  for (i = 0; i < count; ++i)
  {
    kaapi_request_t* const req = &reqs[i];
    kaapi_reply_t* const rep = &reps[i];

    rep->offset = 0;
    rep->preempt = 0;
    rep->status = KAAPI_REQUEST_S_POSTED;

    req->kid = (kaapi_processor_id_t)i;
    req->mapping = 0;
    req->reply = rep;
    req->ktr = NULL;
  }

  retval = splitter(sc, count, reqs, args);

  /* foreach replied requests, push in queues */
  for (i = 0; i < count; ++i)
  {
    kaapi_request_t* const req = &reqs[i];
    kaapi_reply_t* const rep = &reps[i];
    kaapi_ws_queue_t* queue;

    if (rep->status == KAAPI_REQUEST_S_POSTED) continue ;

    /* extract the task and push in the correct queue */
    queue = hws_levels[levelid].blocks[req->kid].queue;

    switch (kaapi_reply_status(rep))
    {
    case KAAPI_REPLY_S_TASK_FMT:
      {
	kaapi_format_t* const format =
	  kaapi_format_resolvebyfmit(rep->u.s_taskfmt.fmt);
	rep->u.s_task.body = format->entrypoint[kproc->proc_type];
      } /* KAAPI_REPLY_S_TASK_FMT */

    case KAAPI_REPLY_S_TASK:
      {
	/* data is stored in the first word of udata */
	void* const dont_break_aliasing = (void*)rep->udata;
	void* const data = *(void**)dont_break_aliasing;
	kaapi_ws_queue_push(queue, rep->u.s_task.body, data);
	break ;

      } /* KAAPI_REPLY_S_TASK */

    default: break ;
    }
  }

  return retval;
}


int kaapi_hws_get_splitter_info
(
 kaapi_stealcontext_t* sc,
 kaapi_hws_levelid_t* levelid
)
{
  /* return -1 if this is not the hws splitter.
     otherwise, levelid is set to the correct
     level and 0 is returned.
   */

  /* todo: currently hardcoded. find a way to pass
     information between xkaapi and the user splitter
   */

  if (!(sc->header.flag & KAAPI_SC_HWS_SPLITTER))
    return -1;

  *levelid = KAAPI_HWS_LEVELID_NUMA;

  return 0;
}


void kaapi_hws_clear_splitter_info(kaapi_stealcontext_t* sc)
{
  sc->header.flag &= ~KAAPI_SC_HWS_SPLITTER;
}


unsigned int kaapi_hws_get_request_nodeid(const kaapi_request_t* req)
{
  return (unsigned int)req->kid;
} 


unsigned int kaapi_hws_get_node_count(kaapi_hws_levelid_t levelid)
{
  kaapi_assert_debug(kaapi_hws_is_levelid_set(levelid));
  return hws_levels[levelid].block_count;
}
