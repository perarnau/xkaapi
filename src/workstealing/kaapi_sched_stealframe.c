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


/** Steal task in the frame [frame->pc:frame->sp)
 */
int kaapi_sched_stealframe
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
    kaapi_sched_steal_task( map, thread, task_top, lrequests, lrrange );
    --task_top;
  }
  
  return 0;
}
