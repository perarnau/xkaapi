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
 kaapi_task_splitter_t splitter
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
  if (parent_sc->header.flag & KAAPI_SC_PREEMPTION)
  {
    printf("[!] KAAPI_SC_PREEMPTION\n");
    fflush(stdout);
    while (1) ;
  }
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
  sc->header.ktr = NULL;

  sc->preempt = NULL;
  sc->splitter = splitter;
  sc->argsplitter = (void*)adata->udata;

  sc->ownertask = 0;

  sc->save_splitter = 0;
  sc->save_argsplitter = 0;

  sc->data_victim = 0;
  sc->sz_data_victim = 0;

  KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);

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
  while (KAAPI_ATOMIC_READ(&sc->thieves.count))
    kaapi_hws_sched_sync_once();

  /* todo: deallocate sc, allocated with pushdata_align */
}
