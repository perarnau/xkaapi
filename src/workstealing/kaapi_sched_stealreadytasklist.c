/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributor :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imag.fr
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


/** Steal ready task in tasklist 
 */
void kaapi_sched_stealreadytasklist( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_processor_t*            kproc,
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
)
{
  int                         err;
  kaapi_readytasklist_t*      rtl;
  kaapi_taskdescr_t*          td;
  kaapi_task_t*               tasksteal;
  kaapi_request_t*            request;
  kaapi_taskstealready_arg_t* argsteal;
    
  request = kaapi_listrequest_iterator_get( lrequests, lrrange );
  while( request !=0 )
  {
#if 1
    if (kproc->mailbox.head != 0 )
    {
      kaapi_task_withlink_t* taskwl;

      /* pop the first item at kproc->mailbox.head
         lock free MPSC FIFO queue must be used here.
      */
      taskwl = kproc->mailbox.head;
      if (kproc->mailbox.tail == taskwl)
        kproc->mailbox.tail = 0;
      kproc->mailbox.head = taskwl->next;
      
      /* push the task into local queue */
      *kaapi_request_toptask(request) = taskwl->task;
      kaapi_request_pushtask(request, 0);
      kaapi_request_replytask( request, KAAPI_REQUEST_S_OK); /* success of steal */
      KAAPI_DEBUG_INST( kaapi_listrequest_iterator_countreply( lrrange ) );
      /* update next request to process */
      kaapi_listrequest_iterator_next( lrequests, lrrange );    
      request = kaapi_listrequest_iterator_get( lrequests, lrrange );
      continue;
    }
#endif

    rtl = kproc->rtl;
    err = kaapi_readylist_steal( kproc, rtl, &td );
    if( err == 0 )
    {
      /* - create the task steal that will execute the stolen task
        The task stealtask stores:
           - the original thread
           - the original task pointer
           - the pointer to shared data with R / RW access data
           - and at the end it reserve enough space to store original task arguments
        The original body is saved as the extra body of the original task data structure.
      */
      argsteal
        = (kaapi_taskstealready_arg_t*)kaapi_request_pushdata(request, sizeof(kaapi_taskstealready_arg_t));

      argsteal->master_tasklist = td->tasklist;
      argsteal->td              = td; /* recopy of the pointer, need synchro if range */

      tasksteal = kaapi_request_toptask(request);
      kaapi_task_init(  tasksteal, kaapi_taskstealready_body, argsteal );
      kaapi_request_pushtask(request, 0);

      kaapi_request_replytask( request, KAAPI_REQUEST_S_OK); /* success of steal */
      KAAPI_DEBUG_INST( kaapi_listrequest_iterator_countreply( lrrange ) );

      /* update next request to process */
      kaapi_listrequest_iterator_next( lrequests, lrrange );    
      request = kaapi_listrequest_iterator_get( lrequests, lrrange );
      continue;
    }
    
    rtl = kproc->rtl_remote;
    err = kaapi_readylist_steal( kproc, rtl, &td );
    if( err == 0 )
    {
      /* - create the task steal that will execute the stolen task
       The task stealtask stores:
       - the original thread
       - the original task pointer
       - the pointer to shared data with R / RW access data
       - and at the end it reserve enough space to store original task arguments
       The original body is saved as the extra body of the original task data structure.
       */
      argsteal
      = (kaapi_taskstealready_arg_t*)kaapi_request_pushdata(request, sizeof(kaapi_taskstealready_arg_t));
      
      argsteal->master_tasklist = td->tasklist;
      argsteal->td              = td; /* recopy of the pointer, need synchro if range */
      
      tasksteal = kaapi_request_toptask(request);
      kaapi_task_init(  tasksteal, kaapi_taskstealready_body, argsteal );
      kaapi_request_pushtask(request, 0);
      
      kaapi_request_replytask( request, KAAPI_REQUEST_S_OK); /* success of steal */
      KAAPI_DEBUG_INST( kaapi_listrequest_iterator_countreply( lrrange ) );
      
      /* update next request to process */
      kaapi_listrequest_iterator_next( lrequests, lrrange );
      request = kaapi_listrequest_iterator_get( lrequests, lrrange );
    }
    else
      break; 
  }
}

