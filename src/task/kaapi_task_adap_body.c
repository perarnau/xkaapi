/*
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

#include <stdint.h>
#include <stddef.h>
#include "kaapi_impl.h"


static inline kaapi_adaptive_reply_data_t*
stealheader_to_reply_data(kaapi_stealheader_t* header)
{
  return (kaapi_adaptive_reply_data_t*)
    ((uintptr_t)header - offsetof(kaapi_adaptive_reply_data_t, header));
}

/* adaptive task body
 */
void kaapi_adapt_body(void* arg, kaapi_thread_t* thread)
{
  kaapi_thread_context_t* self_thread;
  kaapi_stealcontext_t* sc;
  kaapi_adaptive_reply_data_t* adata;
  
  /* 2 cases to handle:
     . either we are in the master task created with
     kaapi_task_begin_adpative. the argument is the
     master stealcontext and sc->msc = sc. we return
     without further processing since the sequential
     code is assumed to run by itself.
     . otherwise, we have been forked during a steal.
     the stealcontext is a partial sc and we have to
     build a full stealcontext. then call the user body.
   */

  self_thread = kaapi_self_thread_context();
  sc = (kaapi_stealcontext_t*)&self_thread->sc;

  /* this is the master task, return */
  if (sc->header.msc == sc) return ;

  /* retrieve the adaptive reply data */
  adata = stealheader_to_reply_data(arg);

  /* todo: save the sp and sync if changed during
     the call (ie. wait for tasks forked)
  */  

  /* header flag, msc, ktr init by remote write */
  sc->preempt          = &self_thread->static_reply.status;
  sc->save_splitter    = 0;
  sc->save_argsplitter = 0;
  sc->ownertask = kaapi_thread_toptask(thread);

  /* finalize the stealcontext creation */
  kaapi_thread_save_frame(thread, &sc->frame);

  /* execute the user task entrypoint */
  kaapi_assert_debug(adata->ubody != 0);
  adata->ubody((void*)adata->udata, thread, sc);

  if (!(adata->header.flag & KAAPI_SC_PREEMPTION))
  {
    /* non preemptive algorithm decrement the
       thievecount. this is the only way for
       the master to sync on algorithm term.
       
       HERE TODO: store a pointer to the thieves_count
    */
    KAAPI_ATOMIC_DECR(&adata->header.msc->thieves.count);
  }
  else /* if (sc->ktr != 0) */
  {
    kaapi_assert_debug(adata->header.ktr);

    /* preemptive algorithms need to inform
       they are done so they can be reduced.
    */
    adata->header.ktr->thief_term = 1;
  }

  kaapi_thread_restore_frame(thread, &sc->frame);

  kaapi_writemem_barrier();
}
