#include "kaapi_impl.h"
#include "kaapi_hws.h"


int kaapi_hws_splitter
(
 kaapi_stealcontext_t* sc,
 kaapi_task_splitter_t splitter,
 void* args,
 kaapi_hws_levelid_t levelid
)
{
  /* split equivalently among all the nodes of a given level */

  /* todo: dynamic allocation */
  kaapi_request_t reqs[KAAPI_MAX_PROCESSOR];
  kaapi_reply_t reps[KAAPI_MAX_PROCESSOR];

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

  return splitter(sc, count, reqs, args);
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


unsigned int kaapi_hws_get_request_nodeid(const kaapi_request_t* req)
{
  return (unsigned int)req->kid;
} 


unsigned int kaapi_hws_get_node_count(kaapi_hws_levelid_t levelid)
{
  kaapi_assert_debug(kaapi_hws_is_levelid_set(levelid));
  return hws_levels[levelid].block_count;
}
