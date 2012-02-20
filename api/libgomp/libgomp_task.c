/*
 ** xkaapi
 ** 
 **
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
#include "libgomp.h"

//#define USE_WORKLOAD 1

typedef struct GOMP_spawn_task_arg {
  int                       numthreads;
  int                       threadid;
  void                     (*fn) (void *);
  void*                     data;
} GOMP_spawn_task_arg_t;

static void GOMP_trampoline_task(
  void* voidp, kaapi_thread_t* thread
)
{
#if defined(USE_WORKLOAD)
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_set_workload( kproc, 
    kproc->thread->stack.sfp - kproc->thread->stack.stackframe
  );
#endif

  GOMP_spawn_task_arg_t* taskarg = (GOMP_spawn_task_arg_t*)voidp;
  kaapi_libkompctxt_t* ctxt = komp_get_ctxt();

  int num_threads  = taskarg->numthreads;
  int thread_id    = taskarg->threadid;
  
  ctxt->numthreads = num_threads;
  ctxt->threadid   = thread_id;
  taskarg->fn(taskarg->data);
  ctxt->numthreads = num_threads;
  ctxt->threadid   = thread_id;
}

KAAPI_REGISTER_TASKFORMAT(GOMP_task_format,
    "GOMP/Task",
    GOMP_trampoline_task,
    sizeof(GOMP_spawn_task_arg_t),
    4,
    (kaapi_access_mode_t[]){ 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V 
    },
    (kaapi_offset_t[])     { 
        offsetof(GOMP_spawn_task_arg_t, numthreads), 
        offsetof(GOMP_spawn_task_arg_t, threadid), 
        offsetof(GOMP_spawn_task_arg_t, fn), 
        offsetof(GOMP_spawn_task_arg_t, data)
    },
    (kaapi_offset_t[])     { 0, 0, 0, 0 },
    (const struct kaapi_format_t*[]) {
        kaapi_int_format, 
        kaapi_int_format,
        kaapi_voidp_format, 
        kaapi_voidp_format
      },
    0
)

void GOMP_task(
               void (*fn) (void *), 
               void *data, 
               void (*cpyfn) (void *, void *),
               long arg_size, 
               long arg_align, 
               bool if_clause,
               unsigned flags __attribute__((unused))
               )
{
#if 0// EXPERIMENTAL defined(KAAPI_USE_PERFCOUNTER)
  if (if_clause) 
  {
    /* try to force sequential degeneration is no steal request */
    kaapi_processor_t* kproc = kaapi_get_current_processor();
    int seqdeg = 0;
    kaapi_perf_counter_t rcntsi = KAAPI_PERF_REG_READALL(kproc, KAAPI_PERF_ID_STEALIN);
    if (rcntsi == kproc->lastcounter)
    {
      seqdeg = 1;
      kaapi_push_frame(&kproc->thread->stack);
    }
    else
      kproc->lastcounter = rcntsi;
  }
  if (!if_clause || seqdeg) 
#else
  if (!if_clause) 
#endif
  {
    if (cpyfn)
    {
      char buf[arg_size + arg_align - 1];
      char *arg = (char *) (((uintptr_t) buf + arg_align - 1)
                            & ~(uintptr_t) (arg_align - 1));
      cpyfn (arg, data);
      fn (arg);
    }
    else
      fn (data);
#if 0// EXPERIMENTAL//defined(KAAPI_USE_PERFCOUNTER)
    kaapi_pop_frame(&kproc->thread->stack);
#endif
    return;
  }
#if 1// EXPERIMENTAL//!defined(KAAPI_USE_PERFCOUNTER)
  kaapi_processor_t* kproc = kaapi_get_current_processor();
#endif
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  kaapi_thread_t* thread =  kaapi_threadcontext2thread(kproc->thread);
  kaapi_task_t* task = kaapi_thread_toptask(thread);
  kaapi_task_init( 
      task, 
      GOMP_trampoline_task, 
      kaapi_thread_pushdata(thread, sizeof(GOMP_spawn_task_arg_t)) 
  );
  void* userarg = kaapi_thread_pushdata_align( thread, arg_size, arg_align);
  if (cpyfn)
    cpyfn(userarg, data);
  else
    memcpy(userarg, data, arg_size);

  GOMP_spawn_task_arg_t* arg = kaapi_task_getargst( task, GOMP_spawn_task_arg_t );
  arg->numthreads = ctxt->numthreads;
  arg->threadid   = ctxt->threadid;
  arg->fn         = fn;
  arg->data       = userarg;
  kaapi_thread_pushtask(thread);
}

void GOMP_taskwait (void)
{
  kaapic_sync();
}

