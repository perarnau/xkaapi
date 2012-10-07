/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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

int kaapi_sched_steal_task
(
  kaapi_hashmap_t*              map, 
  const kaapi_format_t*         task_fmt,
  kaapi_task_t*                 task,
  kaapi_listrequest_t*          lrequests, 
  kaapi_listrequest_iterator_t* lrrange
)
{
  kaapi_task_t* tasksteal;
  unsigned int war_param;
  unsigned int cw_param;
  kaapi_request_t* request;  
  size_t wc;

  if (task_fmt == 0) 
    return ENOENT;

  war_param = 0;
  cw_param = 0;
  wc = 0;

  /* if map == history of visited access, then compute readiness 
     - even if the task is not stealable, then it is necessary to
     propagate data flow constraints.
  */
  if (map !=0)
  {
#if 0
    wc = 0; /* simulate indepent task */
#else
    wc = kaapi_task_computeready( 
      task,
      kaapi_task_getargs(task), 
      task_fmt, 
      &war_param, 
      &cw_param,
      map
    );
#endif
  }

  if (wc !=0)
    return EACCES;
    
  if (!kaapi_task_isstealable(task))
    return EPERM;

  int retval = kaapi_task_marksteal( task );
  if (unlikely( !retval ) ) 
    return EBUSY;

  request = kaapi_listrequest_iterator_get( lrequests, lrrange );
  
  /* To be solved before marking the task as theft */
  kaapi_assert_debug( 
      kaapi_task_has_arch(task, 
          kaapi_processor_get_type( kaapi_all_kprocessors[request->ident]) 
      )
  );

  /* - create the task steal that will execute the stolen task
     The task stealtask stores:
     - the original task pointer
     - the pointer to shared data with R / RW access data
     - and at the end it reserve enough space to store original task arguments
  */
  kaapi_tasksteal_arg_t* argsteal 
    = (kaapi_tasksteal_arg_t*)kaapi_request_pushdata(request, sizeof(kaapi_tasksteal_arg_t));
  argsteal->origin_task           = task;
  argsteal->origin_body           = task->body;
  argsteal->origin_fmt            = task_fmt;
  argsteal->war_param             = war_param;  
  argsteal->cw_param              = cw_param;
  
  tasksteal = kaapi_request_toptask(request);
  kaapi_task_init( 
    tasksteal,
    kaapi_tasksteal_body,
    argsteal
  );
  ((kaapi_task_t* volatile)task)->reserved = tasksteal;
  kaapi_request_pushtask(request,0);

  /* success of steal */
  kaapi_request_replytask( request, KAAPI_REQUEST_S_OK);
  KAAPI_DEBUG_INST( kaapi_listrequest_iterator_countreply( lrrange ) );

  /* advance to the next request */
  kaapi_listrequest_iterator_next( lrequests, lrrange );

  return 0;
}
