#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"

/* push a task at a given hierarchy level
 */

int kaapi_hws_pushtask
(kaapi_task_body_t body, void* data, kaapi_hws_levelid_t levelid)
{
  /* kaapi_assert(kaapi_hws_is_levelid_set(levelid)); */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  kaapi_ws_block_t* const block = hws_levels[levelid].kid_to_block[kproc->kid];
  kaapi_ws_queue_t* const queue = block->queue;

  /* toremove */
  kaapi_hws_sched_inc_sync();
  /* toremove */

  kaapi_ws_queue_push(queue, body, data);

  return 0;
}
