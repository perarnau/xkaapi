/*
** kaapi_task_reqreply.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
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

/** Args for tasksignal
*/
typedef struct kaapi_tasksig_arg_t {
  kaapi_taskadaptive_t*                 ta;         /* the victim or the master */
  volatile kaapi_taskadaptive_result_t* result;
  volatile int			                    inuse;
} kaapi_tasksig_arg_t;



/**
*/
void kaapi_tasksig_body( void* taskarg, kaapi_thread_t* thread)
{
  kaapi_tasksig_arg_t* arg = (kaapi_tasksig_arg_t*)taskarg;

  kaapi_writemem_barrier();

  if (arg->result !=0)
  {
    arg->result->thief_term = 1;
    arg->result->is_signaled = 1;
  }

  KAAPI_ATOMIC_DECR( &arg->ta->thievescount );
}


/*
*/
int kaapi_request_reply(
    kaapi_stealcontext_t*               stc,
    kaapi_request_t*                    request, 
    kaapi_taskadaptive_result_t*        result,
    int                                 flag
)
{
  kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)stc;
  kaapi_taskadaptive_t* ta_master;
  kaapi_task_t* tasksig;
  
  kaapi_assert_debug( (flag == KAAPI_REQUEST_REPLY_HEAD) || (flag == KAAPI_REQUEST_REPLY_TAIL) );
  
  if ((result ==0) && (stc ==0))
  {
    return _kaapi_request_reply(request, 0);
  }
  
  if (result !=0)
  {
    /* lock the ta result list */
    while (!KAAPI_ATOMIC_CAS(&ta->lock, 0, 1)) 
      ;

    /* insert in head or tail */
    if (ta->head ==0)
      ta->tail = ta->head = result;
    else if ((flag & 0x1) == KAAPI_REQUEST_REPLY_HEAD) 
    { 
      result->next   = ta->head;
      ta->head->prev = result;
      ta->head       = result;
    } 
    else 
    {
      result->prev   = ta->tail;
      ta->tail->next = result;
      ta->tail       = result;
    }

    KAAPI_ATOMIC_WRITE( &ta->lock, 0 );

    /* link result to the stc */
    result->master = ta;
  }

#if 0 // TODO  
  if ((stc->flag & 0x1) == KAAPI_STEALCONTEXT_LINKED) 
  { 
    ta_master = ta->origin_master;
    if (ta_master ==0) ta_master = ta;
  }
  else {
    ta_master = ta;
  }
  KAAPI_ATOMIC_INCR( &ta_master->thievescount );

  /* add task to tell to the master that this disappear */
  tasksig = kaapi_thread_toptask( request->thread );
  kaapi_tasksig_arg_t* const arg = kaapi_thread_pushdata( request->thread, sizeof(kaapi_tasksig_arg_t));
  kaapi_task_init(tasksig, kaapi_tasksig_body, arg);
  arg->ta     = ta_master;
  arg->result = result;
  kaapi_thread_pushtask(request->thread);
#endif
  return _kaapi_request_reply( request, 1 );
}
