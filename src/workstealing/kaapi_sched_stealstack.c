/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:18:04 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributor :
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
#if defined(KAAPI_DEBUG_LOURD)
#include <unistd.h>
#endif

static inline void _kaapi_bitmapkid2cpuset( kaapi_cpuset_t* cpusetrequest, kaapi_bitmap_value_t* bitmap )
{
  int kid;
  kaapi_bitmap_value_t save = *bitmap;
  kaapi_cpuset_clear(cpusetrequest);
  
  while ( (kid = kaapi_bitmap_first1_and_zero( &save )) != 0)
    kaapi_cpuset_set(cpusetrequest, kaapi_all_kprocessors[kid]->cpuid);
}


/* find and clear the first request whose emiter matches the numaid
 */
__attribute__((unused)) 
static kaapi_request_t* _kaapi_matching_request 
(
 kaapi_listrequest_t*          lr,
 kaapi_listrequest_iterator_t* lri,
 kaapi_cpuset_t*               mapping
 ) 
{
  kaapi_cpuset_t cpusetrequest;
  _kaapi_bitmapkid2cpuset( &cpusetrequest, &lri->bitmap );
  kaapi_cpuset_and( &cpusetrequest, mapping);
  
  /* take the first bit == 1 in cpusetrequest, if ==0 return 0 
   else mark as setted the i-th bit in the iterator
   */
  if (kaapi_cpuset_empty(&cpusetrequest) == 0) 
    return 0;
  int reqid = kaapi_bitmap_first1_and_zero_128( (kaapi_bitmap_value128_t*)&cpusetrequest );
  kaapi_assert_debug(reqid != 0);
  --reqid;
  
  /* get the kprod id which has cpuset == reqid */
  kaapi_assert_debug( reqid < (int)kaapi_default_param.cpucount );
  int kid = kaapi_default_param.cpu2kid[reqid];
  
  /* return the request attached to kid and set its bit to 0 and push the 
   iterator to the next value 
   */
  return kaapi_listrequest_iterator_getkid_andnext( lr, lri, kid );
}


/** Steal task in the frame [frame->pc:frame->sp)
 */
static int kaapi_sched_stealframe
(
  kaapi_thread_context_t*       thread, 
  kaapi_frame_t*                frame, 
  kaapi_hashmap_t*              map, 
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_task_body_t     task_body;
  kaapi_task_t*         task_top;
  kaapi_task_t*         task_sp;
  
  /* suppress history of the previous frame ! */
  kaapi_hashmap_clear( map );
  task_top   = frame->pc;
  task_sp    = frame->sp;
  
  /** HERE TODO: if they are enough tasks in the frame (up to a threshold which may depend
   of the number of requests (it not interesting to do it if #tasks == #request), then 
   it is interesting to compute the tasklist viewed as a speeding data structure
   for next steal requests. 
   */
  while ( !kaapi_listrequest_iterator_empty(lrrange) && (task_top > task_sp))
  {
    task_body = kaapi_task_getbody(task_top);
    
    /* its an adaptive task !!! */
    if (task_body == kaapi_adapt_body || task_body == kaapi_hws_adapt_body)
    {
      kaapi_task_steal_adapt(
        thread, 
        task_top,
        lrequests,
        lrrange,
        0     /* no callback if the adaptive task was into a queue of a thread */
      );
      
      --task_top;
      continue;
    }

    /* else try to steal a DFG task using history of accesses stored in map 
       On return the request iterator has been advanced to the next request if steal successed
    */
    kaapi_task_steal_dfg( map, thread, task_top, lrequests, lrrange );
    --task_top;
  }
  
  return 0;
}


/** Steal ready task in tasklist 
 */
static void kaapi_sched_steal_tasklist( 
                                       kaapi_thread_context_t*       thread, 
                                       kaapi_frame_t*                frame, 
                                       kaapi_listrequest_t*          lrequests, 
                                       kaapi_listrequest_iterator_t* lrrange
                                       )
{
  int                     err;
  kaapi_tasklist_t*       tasklist;
  kaapi_taskdescr_t**     steal_td_beg;
  kaapi_taskdescr_t**     steal_td_end;
  size_t size_steal;
  
  tasklist= frame->tasklist;  
  kaapi_workqueue_index_t count_req 
  = (kaapi_workqueue_index_t)kaapi_listrequest_iterator_count( lrrange );
#if 0
  kaapi_workqueue_index_t size_ws;
  size_ws = kaapi_workqueue_size( &tasklist->wq_ready );
  if (size_ws ==0) return;
  //printf("%i::[steal] victim'queue size:%i, #req=%i\n", kaapi_get_self_kid(), (int)size_ws, (int)count_req );
  /* try to steal ready one task for each stealers */
  if (count_req+1 > size_ws) 
    count_req = size_ws;
  
  size_steal = (count_req*size_ws) / (count_req + 1);
  if (size_steal ==0) size_steal = 1;
#else
  
  /* else try to steal count_req */
  size_steal = count_req;
#endif
  
  //#warning "Only for debug here"
  while (size_steal >0)
  {
    kaapi_assert_debug( kaapi_sched_islocked( &thread->stack.proc->lock ) );
    err = kaapi_thread_tasklistready_steal( &tasklist->rtl, &steal_td_beg, &steal_td_end, size_steal);
    if (err ==0)
    {
      //printf("%i::[steal] steal size:%i\n", kaapi_get_self_kid(), (int)(steal_end - steal_beg) );
      /* steal ok: reply */
      kaapi_task_splitter_readylist( 
                                    thread,
                                    tasklist,
                                    steal_td_beg,
                                    steal_td_end,
                                    lrequests, 
                                    lrrange,
                                    count_req
                                    );
      return;
    }
    
    /* else try with half less requests */
    size_steal /= 2;
  }
}


/** Steal task in the stack from the bottom to the top.
     This signature MUST BE the same as a splitter function.
 */
int kaapi_sched_stealstack  
( 
  kaapi_thread_context_t*       thread, 
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_frame_t*           top_frame;  
  kaapi_hashmap_t          access_to_gd;
  kaapi_hashentries_bloc_t stackbloc;
  
  if ((thread ==0) || (thread->unstealable != 0)) 
    return 0;
  
  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &access_to_gd, &stackbloc );
  
  kaapi_sched_lock(&thread->stack.lock);
  
  /* try to steal in each frame */
  for (  top_frame =thread->stack.stackframe; 
       (top_frame <= thread->stack.sfp) && !kaapi_listrequest_iterator_empty(lrrange); 
       ++top_frame)
  {
    /* void frame ? */
    if (top_frame->tasklist == 0)
    {
      thread->stack.thieffp = top_frame;
      /* classical steal */
      if (top_frame->pc == top_frame->sp) continue;
      kaapi_sched_stealframe( thread, top_frame, &access_to_gd, lrequests, lrrange );
    } else 
    /* */
      kaapi_sched_steal_tasklist( thread, top_frame, lrequests, lrrange );
  }
  thread->stack.thieffp = 0;
  kaapi_sched_unlock(&thread->stack.lock);

  kaapi_hashmap_destroy( &access_to_gd );
  
  return 0;
}
