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
** fabien.lementec@imag.fr
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


/* common reply internal function
 */
int kaapi_request_reply(
  kaapi_stealcontext_t* sc, 
  kaapi_request_t*      req, 
  int                   headtail_flag
)
{
  /* sc the stolen stealcontext */

  /* if there is preemption, link to thieves */
  if (sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    kaapi_taskadaptive_result_t* const ktr = req->ktr;
#if defined(KAAPI_DEBUG)
#warning TODO HERE
#if 0
    kaapi_assert_debug( ktr != 0 );
    {
      kaapi_stealheader_t* stch = (kaapi_stealheader_t*)(req->reply->udata+req->reply->offset);
      kaapi_assert_debug( stch->ktr == ktr );
    }
#endif
#endif

    kaapi_task_lock_adaptive_steal(sc);

    /* insert in head or tail */
    if (sc->thieves.list.head == 0)
    {
      sc->thieves.list.tail = ktr;
      sc->thieves.list.head = ktr;
    }
    else if (headtail_flag == KAAPI_REQUEST_REPLY_HEAD)
    { 
      ktr->next = sc->thieves.list.head;
      sc->thieves.list.head->prev = ktr;
      sc->thieves.list.head = ktr;
    } 
    else 
    {
      ktr->prev = sc->thieves.list.tail;
      sc->thieves.list.tail->next = ktr;
      sc->thieves.list.tail = ktr;
    }

    kaapi_task_unlock_adaptive_steal(sc);
  }
  else
  {
    /* non preemptive algorithm, inc the root master thiefcount */
    KAAPI_ATOMIC_INCR(&sc->header.msc->thieves.count);
  }

#warning TODO HERE
//  _kaapi_request_reply(req, KAAPI_REPLY_S_TASK);
  return 0; 
}


void kaapi_request_reply_failed(kaapi_request_t* req)
{
  /* sc the stolen stealcontext */
  kaapi_request_replytask(req, KAAPI_REQUEST_S_NOK);
}
