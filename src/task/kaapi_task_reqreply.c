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


/* adaptive task body
 */

extern volatile unsigned int g;

typedef struct athief_taskarg
{
  kaapi_taskadaptive_t* mta; /* master ta */
  volatile kaapi_taskadaptive_result_t* result;
  kaapi_task_body_t ubody; /* user body */
  unsigned char udata[1]; /* user data */
} athief_taskarg_t;

static void athief_body(void* arg, kaapi_thread_t* thread)
{
  athief_taskarg_t* const ata = (athief_taskarg_t*)arg;

  /* execute the original task */
  ata->ubody((void*)ata->udata, thread);

  /* signal the remote master */
  kaapi_writemem_barrier();
  if (ata->result != 0)
  {
    ata->result->thief_term = 1;
    ata->result->is_signaled = 1;
  }

  if (ata->mta != NULL)
    KAAPI_ATOMIC_DECR(&ata->mta->thievescount);
}

void* kaapi_reply_pushtask
(
 kaapi_stealcontext_t* msc,
 kaapi_request_t* req,
 kaapi_task_body_t body
)
{
  /* athief task body */
  req->reply->u.s_task.body = (kaapi_task_bodyid_t)athief_body;

  /* athief task args */
  athief_taskarg_t* const ata = (athief_taskarg_t*)
    req->reply->u.s_task.data;
  ata->mta = (kaapi_taskadaptive_t*)msc;
  ata->result = 0;
  ata->ubody = body;

  /* user put args in this area */
  return (void*)ata->udata;
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
  
  kaapi_assert_debug
    ((flag == KAAPI_REQUEST_REPLY_HEAD) || (flag == KAAPI_REQUEST_REPLY_TAIL));
  
  if ((result ==0) && (stc ==0))
    return _kaapi_request_reply(request, KAAPI_REPLY_S_NOK);
  
  if (result !=0)
  {
    /* lock the ta result list */
    while (!KAAPI_ATOMIC_CAS(&ta->lock, 0, 1)) 
      kaapi_slowdown_cpu();

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

    /* set athief result to signal task end */
    athief_taskarg_t* const ata = (athief_taskarg_t*)
      request->reply->u.s_task.data;
    ata->result = result;
  }

  /* increment master thief count */
  athief_taskarg_t* const ata = (athief_taskarg_t*)
    request->reply->u.s_task.data;

  if (ata->mta != NULL)
    KAAPI_ATOMIC_INCR( &ata->mta->thievescount );

  return _kaapi_request_reply( request, KAAPI_REPLY_S_TASK );
}
