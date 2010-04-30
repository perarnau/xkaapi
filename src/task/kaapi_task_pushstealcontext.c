/*
** kaapi_task_pushstealcontext.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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

/**
*/
kaapi_stealcontext_t* kaapi_thread_pushstealcontext( 
  kaapi_thread_t*       thread,
  int                   flag,
  kaapi_task_splitter_t spliter,
  void*                 argsplitter,
  kaapi_stealcontext_t* master
)
{
  kaapi_frame_t frame;
  kaapi_taskadaptive_t* ta;
  kaapi_thread_save_frame(thread, &frame);
  
  kaapi_mem_barrier();
  
  ta = (kaapi_taskadaptive_t*) kaapi_thread_pushdata(thread, sizeof(kaapi_taskadaptive_t));
  kaapi_assert_debug( ta !=0 );

  ta->sc.ctxtthread         = _kaapi_self_thread();
  ta->sc.thread             = thread;
  ta->sc.splitter           = spliter;
  ta->sc.argsplitter        = argsplitter;
  ta->sc.flag               = flag;
  ta->sc.hasrequest         = 0;
  ta->sc.requests           = ta->sc.ctxtthread->proc->hlrequests.requests;
  KAAPI_ATOMIC_WRITE(&ta->sc.is_there_thief, 0);

  KAAPI_ATOMIC_WRITE(&ta->lock, 0);
  KAAPI_ATOMIC_WRITE(&ta->thievescount, 0);
  ta->head                  = 0;
  ta->tail                  = 0;
  ta->frame                 = frame;
  ta->sc.ownertask          = kaapi_thread_toptask(thread);
  
  /* link two contexts together (master -> {thief*}) relation, ie thief B of a thief A has the same master as the thief A */
  if ((master !=0) && ((flag & 0x1) == KAAPI_STEALCONTEXT_LINKED))
  {
    ta->origin_master        = ((kaapi_taskadaptive_t*)master)->origin_master;
    if (ta->origin_master ==0) ta->origin_master = (kaapi_taskadaptive_t*)master;
  }
  else
    ta->origin_master        = 0;
  
  ta->save_splitter         = 0;
  ta->save_argsplitter      = 0;
  kaapi_task_init(ta->sc.ownertask, kaapi_adapt_body, ta);
  kaapi_thread_pushtask(thread);
  return &ta->sc;
}
