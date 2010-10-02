#include "kaapi_impl.h"


/* ready list routines
 */

kaapi_thread_context_t* kaapi_sched_stealready
(kaapi_processor_t* kproc, kaapi_processor_id_t tid)
{
  /* if the owner call this method then it
     should protect itself again thieves by
     using sched_lock & sched_unlock
  */

  kaapi_lready_t* const list = &kproc->lready;

  kaapi_thread_context_t* pos;
  kaapi_thread_context_t* pre;

  if (list->_front == NULL)
    return NULL;

  pre = list->_front;
  if (kaapi_thread_hasaffinity(pre->affinity, tid))
  {
    /* unlink the thread */
    pos = list->_front;
    list->_front = list->_front->_next;
    return pos;
  }

  pos = pre->_next;
  while (pos != NULL) 
  {
    if (kaapi_thread_hasaffinity(pos->affinity, tid))
    {
      /* unlink the thread */
      pre->_next = pos->_next;
      return pos;
    }

    pre = pos;
    pos = pos->_next;
  }

  /* todo: should return NULL (?) */
  return pos;
}


void kaapi_sched_pushready
(kaapi_processor_t* kproc, kaapi_thread_context_t* node)
{
  /* push back version. prev not updated.
   */

  kaapi_lready_t* const list = &kproc->lready;

  node->_next = NULL;

  if (list->_back != NULL)
    list->_back->_next = node;
  else
    list->_front = node;

  list->_back = node;
}
