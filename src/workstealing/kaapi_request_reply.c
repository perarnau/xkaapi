/*
** xkaapi
** 
** Created on Tue Mar 31 15:21:00 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@imag.fr
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

int kaapi_request_reply( kaapi_stack_t* stack, kaapi_task_t* task, kaapi_request_t* request, kaapi_stack_t* thief_stack, int retval )
{
  kaapi_assert_debug( stack != 0 );
  if (retval)
  {
    kaapi_task_t* sig;
    
    if (kaapi_task_isadaptive(task))
    {
      kaapi_taskadaptive_t* ta = task->sp;
      kaapi_assert_debug( ta !=0 );
      KAAPI_ATOMIC_INCR( &ta->thievescount );
    }
    else {
      kaapi_assert_debug( task->body == &kaapi_suspend_body);
    }
    sig = kaapi_stack_toptask( thief_stack );
    kaapi_task_init(thief_stack, sig, KAAPI_TASK_STICKY | (kaapi_task_isadaptive(task) ? KAAPI_TASK_ADAPTIVE : 0U) );
    kaapi_task_setbody( sig, &kaapi_tasksig_body );
    kaapi_task_setargs( sig, task );
    kaapi_stack_pushtask( thief_stack );

    request->status = KAAPI_REQUEST_S_EMPTY;
    request->reply->data = thief_stack;
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_SUCCESS;
  }
  else {
    request->status = KAAPI_REQUEST_S_EMPTY;
    kaapi_writemem_barrier();
    request->reply->status = KAAPI_REQUEST_S_FAIL;
  }
  return 0;
}
