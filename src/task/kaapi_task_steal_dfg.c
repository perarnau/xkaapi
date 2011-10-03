/*
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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

void kaapi_task_steal_dfg
(
  kaapi_hashmap_t*              map, 
  kaapi_thread_context_t*       thread, 
  kaapi_task_t*                 task,
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  const kaapi_format_t*         task_fmt = kaapi_format_resolvebybody( task->body );
  if (task_fmt !=0)
  {
    unsigned int war_param = 0;
    unsigned int cw_param = 0;
    size_t wc;
    if (map !=0)
      wc = kaapi_task_computeready( 
              task, kaapi_task_getargs(task), 
              task_fmt, 
              &war_param, 
              &cw_param, map );
    else 
      wc = 0;
    if ((wc ==0) && kaapi_task_isstealable(task))
    {
      kaapi_task_body_t body = kaapi_task_marksteal( task );
      if (likely( body ) ) /* success */
      {
        kaapi_request_t* request = kaapi_listrequest_iterator_get( lrequests, lrrange );
        ((kaapi_task_t* volatile)task)->reserved = request->thief_task;
        kaapi_writemem_barrier();
        kaapi_task_setstate( task, KAAPI_TASK_STATE_STEAL);
        
        /* - create the task steal that will execute the stolen task
           The task stealtask stores:
             - the original thread
             - the original task pointer
             - the pointer to shared data with R / RW access data
             - and at the end it reserve enough space to store original task arguments
        */
        kaapi_tasksteal_arg_t* argsteal = request->thief_sp;
        argsteal->origin_thread         = thread;
        argsteal->origin_task           = task;
        argsteal->origin_body           = body;
        argsteal->origin_fmt            = task_fmt;
        argsteal->war_param             = war_param;  
        argsteal->cw_param              = cw_param;
        request->thief_task->body       = kaapi_tasksteal_body; /* TODO MUST be always this task no write */
        kaapi_request_replytask( request, KAAPI_REQUEST_S_OK);  /* success of steal */
        
        /* advance to the next request */
        kaapi_listrequest_iterator_next( lrequests, lrrange );
      }
    }
  } /* if fmt != 0 */
}
