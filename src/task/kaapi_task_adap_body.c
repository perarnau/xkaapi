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


extern void kaapi_synchronize_steal(kaapi_stealcontext_t*);

#if defined(KAAPI_USE_CUDA)
#include "../machine/common/kaapi_procinfo.h"
extern int kaapi_cuda_exectask(kaapi_thread_context_t*, void*, kaapi_format_t*);
#endif


/* adaptive task body
 */
void kaapi_adapt_body(void* arg, kaapi_thread_t* thread)
{
  kaapi_thread_context_t* self_thread;
  kaapi_stealcontext_t* sc;
  kaapi_taskadaptive_user_taskarg_t* adata;
  
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
 
  /* retrieve the adaptive reply data */
  sc = (kaapi_stealcontext_t*)arg;
  adata = (kaapi_taskadaptive_user_taskarg_t*)self_thread->static_reply.udata;
  
  
  /* this is the master task, return */
  if (sc->header.msc == sc) return ;
  
  /* todo: save the sp and sync if changed during
   the call (ie. wait for tasks forked)
   */
  
  /* here is more or less like in function kaapi_task_begin_adaptive
     * header flag, msc, ktr init by remote write 
     * remains to initialize other field
  */
  sc->preempt          = &self_thread->static_reply.preempt;
  sc->splitter		= 0;
  sc->argsplitter	= 0;
  sc->ownertask	       = kaapi_thread_toptask(thread) + 1;

  if (sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    /* if preemption, thief list used ... */
    KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
    sc->thieves.list.head = 0;
    sc->thieves.list.tail = 0;
  }
  else
  {
    /* ... otherwise thief count */
    KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);
  }
  sc->save_splitter         = 0;
  sc->save_argsplitter      = 0;

  /* ok from here steal may occurs */
  kaapi_writemem_barrier();
  sc->header.flag     |= KAAPI_SC_INIT;
  
  /* finalize the stealcontext creation */
  kaapi_thread_save_frame(thread, &sc->frame);
  
  /* execute the user task entrypoint */
  kaapi_assert_debug(adata->ubody != 0);

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
    /* non preemptive algorithm decrement the
     thievecount. this is the only way for
     the master to sync on algorithm term.
     
     HERE TODO: store a pointer to the thieves_count
     */
    KAAPI_ATOMIC_DECR(&sc->header.msc->thieves.count);
  }
  /* otherwise, SC_PREEMPTION but not preempted */
  else if (sc->header.ktr != 0)
  {
    /* preemptive algorithms need to inform
     they are done so they can be reduced.
     */
    
    kaapi_taskadaptive_result_t* const ktr = sc->header.ktr;
    uintptr_t state;

    /* prevent ktr insertion race by steal syncing */
    kaapi_synchronize_steal(sc);
    ktr->rhead = sc->thieves.list.head;
    ktr->rtail = sc->thieves.list.tail;
    
    state = kaapi_task_orstate(&ktr->state, KAAPI_MASK_BODY_TERM);
    if (state & KAAPI_MASK_BODY_PREEMPT)
    {
      /* wait for the preemption status to be seen, otherwise we
       have a race where this thread leaves this code, emits a
       request and see the KAAPI_TASK_S_PREEMPTED reply.
       */
      
      while (*ktr->preempt == 0)
        kaapi_slowdown_cpu();
    }
  }

  kaapi_thread_restore_frame(thread, &sc->frame);
  
  kaapi_writemem_barrier();
}
