/*
** kaapi_hws_adaptive.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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

#include <string.h>
#include <sys/types.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"


void* kaapi_hws_init_adaptive_task
(
 kaapi_stealcontext_t* parent_sc,
 kaapi_request_t* req,
 kaapi_task_body_t body,
 size_t arg_size,
 kaapi_task_splitter_t splitter,
 kaapi_taskadaptive_result_t* ktr
)
{
  /* initialize an adpative task that is to be
     replied to req.
     assume splitter and task args the same.
     return the allocated space.
   */

  size_t total_size;
  kaapi_stealcontext_t* sc;
  kaapi_taskadaptive_user_taskarg_t* adata;
  kaapi_reply_t* rep;

  /* toremove */
  kaapi_assert(!(parent_sc->header.flag & KAAPI_SC_PREEMPTION));
  /* toremove */

  kaapi_assert(arg_size <= KAAPI_REPLY_USER_DATA_SIZE_MAX);

  /* allocate sc and data */
  total_size =
    sizeof(kaapi_stealcontext_t) + sizeof(kaapi_taskadaptive_user_taskarg_t);
  sc = malloc(total_size);
  kaapi_assert(sc);

  /* user data follow the steal context */
  adata = (kaapi_taskadaptive_user_taskarg_t*)
    ((uintptr_t)sc + sizeof(kaapi_stealcontext_t));

  /* sc->header */
  sc->header.flag = parent_sc->header.flag | KAAPI_SC_INIT;
  sc->header.msc = parent_sc->header.msc;
  sc->header.ktr = ktr;
  req->ktr = ktr;

  sc->preempt = NULL;
  sc->splitter = splitter;
  sc->argsplitter = (void*)adata->udata;
  sc->ownertask = 0;
  sc->save_splitter = 0;
  sc->save_argsplitter = 0;
  sc->data_victim = 0;
  sc->sz_data_victim = 0;

  if (sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
    sc->thieves.list.head = 0;
    sc->thieves.list.tail = 0;
  }
  else
  {
    KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);
  }

  adata->ubody = (kaapi_adaptive_thief_body_t)body;

  rep = req->reply;
  rep->u.s_task.body = kaapi_hws_adapt_body;

  {
    void* const dont_break_aliasing = (void*)rep->udata;
    *(void**)dont_break_aliasing = (void*)sc;
  }

  return (void*)adata->udata;
}


void kaapi_hws_adapt_body(void* arg, kaapi_thread_t* thread)
{
  /* from kaapi_adapt_body */

  kaapi_stealcontext_t* sc;
  kaapi_taskadaptive_user_taskarg_t* adata;

  /* retrieve the adaptive reply data */
  sc = (kaapi_stealcontext_t*)arg;

  adata = (kaapi_taskadaptive_user_taskarg_t*)
    ((uintptr_t)sc + sizeof(kaapi_stealcontext_t));

  /* finalize the stealcontext creation */
  kaapi_thread_save_frame(thread, &sc->frame);
  
#if defined(KAAPI_USE_CUDA)
  kaapi_processor_t* const kproc = self_thread->proc;
  if (kproc->proc_type == KAAPI_PROC_TYPE_CUDA)
  {
    /* has the task a cuda implementation */
    kaapi_format_t* const format =
      kaapi_format_resolvebybody((kaapi_task_bodyid_t)adata->ubody);
    if (format == NULL) goto execute_ubody;
    if (format->entrypoint[KAAPI_PROC_TYPE_CUDA] == NULL) goto execute_ubody;
    kaapi_cuda_exectask(self_thread, adata->udata, format);
  }
  else
  {
  execute_ubody:
#endif /* KAAPI_USE_CUDA */
  adata->ubody((void*)adata->udata, thread, sc);
#if defined(KAAPI_USE_CUDA)
  }
#endif

  if (!(sc->header.flag & KAAPI_SC_PREEMPTION))
  {
    KAAPI_ATOMIC_DECR(&sc->header.msc->thieves.count);
  }
  /* otherwise, SC_PREEMPTION but not preempted */
  else if (sc->header.ktr != 0)
  {
    kaapi_task_body_t body;
    int retval;

    /* preemptive algorithms need to inform
     they are done so they can be reduced.
     */
    kaapi_taskadaptive_result_t* const ktr = sc->header.ktr;

    /* prevent ktr insertion race by steal syncing */
    kaapi_synchronize_steal(sc);
    ktr->rhead = sc->thieves.list.head;
    ktr->rtail = sc->thieves.list.tail;

#warning TODO HERE
#if 0
//    state = kaapi_task_orstate(&ktr->state, KAAPI_MASK_BODY_TERM);
//    if (state & KAAPI_MASK_BODY_PREEMPT)
    do {
      body = kaapi_task_getbody(&ktr->state);
      retval = kaapi_task_casstate(&ktr->state, body, kaapi_term_body);
    } while (!retval);
    if (body == kaapi_preempt_body)
    {
      /* wait for the preemption status to be seen, otherwise we
       have a race where this thread leaves this code, emits a
       request and see the KAAPI_TASK_S_PREEMPTED reply.
       */
      
      while (*ktr->preempt == 0)
        kaapi_slowdown_cpu();
    }
#endif
  }

  kaapi_thread_restore_frame(thread, &sc->frame);

  kaapi_writemem_barrier();

  /* deallocate the stealcontext if not master */
  if (sc != sc->header.msc) free(sc);
}


void kaapi_hws_reply_adaptive_task
(kaapi_stealcontext_t* sc, kaapi_request_t* req)
{
  kaapi_request_reply(sc, req, 0);
}


void kaapi_hws_end_adaptive(kaapi_stealcontext_t* sc)
{
  if (sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    /* todo */

    /* while there is ktr, preempt, reduce and continue.
       if the task has not been executed, execute without
       merging the results.
       the reducer should be passed as an argument.
     */

    kaapi_assert(0);
  }
  else
  {
    while (KAAPI_ATOMIC_READ(&sc->thieves.count))
      kaapi_hws_sched_sync_once();
  }

  /* todo: deallocate sc, allocated with pushdata_align */
}
