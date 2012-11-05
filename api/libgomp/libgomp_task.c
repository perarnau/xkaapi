/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
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


/* This version pass to each created task a pointer to the parent context.
   It is correct because:
   - the parent thread will always waits for completion of the child tasks,
   - a context has a scope at least equal to the task that creates it.
*/
typedef struct GOMP_trampoline_task_arg {
  kompctxt_t*               parentctxt;
  void                     (*fn) (void *);
  void*                     data;
} GOMP_trampoline_task_arg;

static void GOMP_trampoline_task(
  void* voidp, kaapi_thread_t* thread
)
{
  GOMP_trampoline_task_arg* taskarg = (GOMP_trampoline_task_arg*)voidp;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  kompctxt_t* new_ctxt;

  /* save context information: allocate new context in the caller stack */
  /* ideally this context should be created only on steal operation */
  new_ctxt = 
    (kompctxt_t*)kaapi_thread_pushdata(
        thread, 
        sizeof(kompctxt_t)
  );
  
  /* init workshared construct */
  new_ctxt->workshare          = 0;
  new_ctxt->teaminfo           = taskarg->parentctxt->teaminfo;
  
  /* initialize master context: nextnum thread is inherited */
  new_ctxt->icv.thread_id       = taskarg->parentctxt->icv.thread_id;
  new_ctxt->icv.next_numthreads = taskarg->parentctxt->icv.next_numthreads; /* WARNING: spec ?*/
  
  new_ctxt->inside_single      = 0;
  new_ctxt->save_ctxt          = ctxt;

  /* swap context: until end_parallel, new_ctxt becomes the current context */
  kproc->libkomp_tls = new_ctxt;

  taskarg->fn(taskarg->data);

  /* restore the initial context */
  kproc->libkomp_tls = ctxt;
}

KAAPI_REGISTER_TASKFORMAT(GOMP_task_format,
    "GOMP/Task",
    GOMP_trampoline_task,
    sizeof(GOMP_trampoline_task_arg),
    3,
    (kaapi_access_mode_t[]){ 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V 
    },
    (kaapi_offset_t[])     { 
        offsetof(GOMP_trampoline_task_arg, parentctxt), 
        offsetof(GOMP_trampoline_task_arg, fn), 
        offsetof(GOMP_trampoline_task_arg, data)
    },
    (kaapi_offset_t[])     { 0, 0, 0 },
    (const struct kaapi_format_t*[]) {
        kaapi_voidp_format, 
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
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_t* thread =  kaapi_threadcontext2thread(kproc->thread);
  kompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  if (!if_clause) 
  {
    kompctxt_t* new_ctxt;

    /* save context information: allocate new context in the caller stack */
    /* ideally this context should be created only on steal operation */
    new_ctxt = 
      (kompctxt_t*)kaapi_thread_pushdata(
          thread, 
          sizeof(kompctxt_t)
    );
    
    /* init workshared construct */
    *new_ctxt = *ctxt;
    new_ctxt->save_ctxt = ctxt;

    /* swap context: until end_parallel, new_ctxt becomes the current context */
    kproc->libkomp_tls = new_ctxt;

    if (!cpyfn)
      fn (data);
    else
    {
      char buf[arg_size + arg_align - 1];
      char *arg = (char *) (((uintptr_t) buf + arg_align - 1)
                            & ~(uintptr_t) (arg_align - 1));
      cpyfn (arg, data);
      fn (arg);
    }

    /* restore the initial context */
    kproc->libkomp_tls = ctxt;
    return;
  }

  kaapi_task_t* task = kaapi_thread_toptask(thread);
  kaapi_task_init( 
      task, 
      GOMP_trampoline_task, 
      kaapi_thread_pushdata(thread, sizeof(GOMP_trampoline_task_arg)) 
  );
  void* userarg = kaapi_thread_pushdata_align( thread, (int)arg_size, arg_align);
  if (cpyfn)
    cpyfn(userarg, data);
  else
    memcpy(userarg, data, arg_size);

  GOMP_trampoline_task_arg* arg = kaapi_task_getargst( task, GOMP_trampoline_task_arg );
  arg->parentctxt     = ctxt;
  arg->fn             = fn;
  arg->data           = userarg;
  kaapi_thread_pushtask(thread);
}

void GOMP_taskwait (void)
{
  kaapic_sync();
}

