/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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


typedef struct komp_parallel_task_arg {
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  komp_teaminfo_t*          teaminfo;
  int                       nextnumthreads;
  int                       nestedlevel;
  int                       nestedparallel;
} komp_parallel_task_arg_t;

/*
*/
static void komp_trampoline_task_parallel
(
  void*           voidp, 
  kaapi_thread_t* thread
)
{
  kaapi_processor_t* kproc  = kaapi_get_current_processor();
  komp_parallel_task_arg_t* taskarg = (komp_parallel_task_arg_t*)voidp;
  kompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  kompctxt_t* new_ctxt;
  
  /* save context information: allocate new context in the caller stack */
  new_ctxt = 
    (kompctxt_t*)kaapi_thread_pushdata(
        thread, 
        sizeof(kompctxt_t)
  );

  /* init workshared construct */
  new_ctxt->workshare          = 0;
  new_ctxt->teaminfo           = taskarg->teaminfo;
  
  /* initialize master context: nextnum thread is inherited */
  new_ctxt->icv.thread_id       = taskarg->threadid;
  new_ctxt->icv.next_numthreads = taskarg->nextnumthreads; /* WARNING: spec ?*/
  new_ctxt->icv.nested_level    = 1+taskarg->nestedlevel;
  new_ctxt->icv.nested_parallel = taskarg->nestedparallel;
  
  new_ctxt->inside_single      = 0;
  new_ctxt->save_ctxt          = ctxt;
  
  /* swap context: until end_parallel, new_ctxt becomes the current context */
  kproc->libkomp_tls = new_ctxt;

  /* GCC compiled code */
  taskarg->fn(taskarg->data);

  /* restore the initial context */
  kproc->libkomp_tls = ctxt;
}


KAAPI_REGISTER_TASKFORMAT( komp_parallel_task_format,
    "KOMP/Parallel Task",
    komp_trampoline_task_parallel,
    sizeof(komp_parallel_task_arg_t),
    7,
    (kaapi_access_mode_t[]){ 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V 
    },
    (kaapi_offset_t[])     { 
        offsetof(komp_parallel_task_arg_t, threadid), 
        offsetof(komp_parallel_task_arg_t, fn), 
        offsetof(komp_parallel_task_arg_t, data),
        offsetof(komp_parallel_task_arg_t, teaminfo),
        offsetof(komp_parallel_task_arg_t, nextnumthreads),
        offsetof(komp_parallel_task_arg_t, nestedlevel),
        offsetof(komp_parallel_task_arg_t, nestedparallel)
    },
    (kaapi_offset_t[])     { 0, 0, 0, 0, 0, 0, 0 },
    (const struct kaapi_format_t*[]) { 
        kaapi_int_format, 
        kaapi_voidp_format,
        kaapi_voidp_format, 
        kaapi_voidp_format,
        kaapi_int_format, 
        kaapi_int_format, 
        kaapi_int_format
      },
    0
)


komp_teaminfo_t*  
komp_init_parallel_start (
  kaapi_processor_t* kproc,
  unsigned num_threads
)
{
  kompctxt_t* new_ctxt;
  kaapi_thread_t* thread;
  komp_teaminfo_t* teaminfo;

  /* do not save the ctxt, assume just one top level ctxt */
  kompctxt_t* ctxt = komp_get_ctxtkproc(kproc);

  /* pseudo OpenMP spec algorithm to compute the number of threads */
  if ( (!ctxt->icv.nested_parallel && (ctxt->icv.nested_level >0)) 
    ||  (ctxt->icv.nested_level >= omp_max_active_levels)
  )
    num_threads = 1;
  else {
    if (num_threads == 0)
      num_threads = ctxt->icv.next_numthreads;
    if (num_threads > kaapi_getconcurrency())
      num_threads = kaapi_getconcurrency();
  }

  thread = kaapi_threadcontext2thread(kproc->thread);
  
  /* allocate new context in the caller stack */
  new_ctxt = 
    (kompctxt_t*)kaapi_thread_pushdata(
        thread, 
        sizeof(kompctxt_t)
  );
  
  /* init team information */
  teaminfo = 
    (komp_teaminfo_t*)kaapi_thread_pushdata_align(
        thread, 
        sizeof(komp_teaminfo_t), 
        8
  );
  /* lock for ??? */
  kaapi_atomic_initlock(&teaminfo->lock);

  /* barrier for the team */
  komp_barrier_init (&teaminfo->barrier, num_threads);

  teaminfo->single_data        = 0;
  teaminfo->numthreads         = num_threads;
  teaminfo->gwork              = 0;
  teaminfo->serial             = 0;

  /* init workshared construct */
  new_ctxt->workshare          = 0;
  new_ctxt->teaminfo           = teaminfo;
  
  /* initialize master context: nextnum thread is inherited */
  new_ctxt->icv.thread_id       = 0;
  new_ctxt->icv.next_numthreads = ctxt->icv.next_numthreads; /* WARNING: spec ? */
  new_ctxt->icv.nested_level    = 1+ctxt->icv.nested_level; 
  new_ctxt->icv.nested_parallel = ctxt->icv.nested_parallel; /* WARNING: spec ? */
  
  new_ctxt->inside_single       = 0;
  new_ctxt->save_ctxt           = ctxt;
  
  /* swap context: until end_parallel, new_ctxt becomes the current context */
  kproc->libkomp_tls = new_ctxt;

  return teaminfo;
}


void 
komp_parallel_start (
  void   (*fn) (void *), 
  void*    data, 
  unsigned num_threads
)
{
  kaapi_processor_t* kproc  = kaapi_get_current_processor();
  kompctxt_t* ctxt;
  kaapi_thread_t* thread;
  komp_teaminfo_t* teaminfo;
  kaapi_task_t* task;
  komp_parallel_task_arg_t* arg;
  komp_parallel_task_arg_t* allarg;
    
  /* begin parallel region: also push a new frame that will be pop
     during call to kaapic_end_parallel
  */
  kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);

  /* init the new context with team information and workshare construct 
     the method push a new context in the caller Kaapi' stack 
     and a call to komp_get_ctxtkproc must be done to retreive the new ctxt.
  */
  teaminfo = komp_init_parallel_start( kproc, num_threads );
  thread = kaapi_threadcontext2thread(kproc->thread);
  
  ctxt = komp_get_ctxtkproc(kproc);
  num_threads = teaminfo->numthreads;
  
  /* allocate in the caller stack the tasks for the parallel region */
  allarg = kaapi_thread_pushdata(thread, num_threads * sizeof(komp_parallel_task_arg_t));


#if 1  /* OLD CODE: push locally all tasks that may be steal by any thread */
  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  task = kaapi_thread_toptask(thread);
  for (int i = 1; i < num_threads; i++)
  {
    kaapi_task_init( 
        task, 
        komp_trampoline_task_parallel, 
        allarg+i
    );
    arg = kaapi_task_getargst( task, komp_parallel_task_arg_t );
    arg->threadid       = i;
    arg->fn             = fn;
    arg->data           = data;
    arg->teaminfo       = teaminfo;
    /* WARNING: see spec: nextnum threads is inherited ? */
    arg->nextnumthreads = ctxt->icv.next_numthreads;
    arg->nestedlevel    = ctxt->icv.nested_level;
    arg->nestedparallel = ctxt->icv.nested_parallel;

    task = kaapi_thread_nexttask(thread, task);
  }
  kaapi_thread_push_packedtasks(thread, num_threads-1);

#else 
  /* push the task for the i-th kprocessor queue... 
     - work fine for 1rst level parallel region.
     - else may introduce deadlock because threads are not reused (...)
     If thread i (kprocessor i) is waiting on a barrier while and other
     thread push a task into its mailbox, then thread-i is unable to execute
     the task (...)
  */

  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  for (int i = 1; i < num_threads; i++)
  {
    task = kaapi_thread_toptask(thread);
    kaapi_task_init( 
        task, 
        komp_trampoline_task_parallel, 
        allarg+i
    );
    arg = kaapi_task_getargst( task, komp_parallel_task_arg_t );
    arg->threadid       = i;
    arg->fn             = fn;
    arg->data           = data;
    arg->teaminfo       = teaminfo;
    /* WARNING: see spec: nextnum threads is inherited ? */
    arg->nextnumthreads = ctxt->icv.next_numthreads;
    arg->nestedlevel    = ctxt->icv.nested_level;
    arg->nestedparallel = ctxt->icv.nested_parallel;

    kaapi_thread_distribute_task( thread, i );
  }
#endif  
}


void 
GOMP_parallel_start (
  void (*fn) (void *), 
  void *data, 
  unsigned num_threads
)
{
  komp_parallel_start( fn, data, num_threads );
}


void 
GOMP_parallel_end (void)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;
  kompctxt_t* old_ctxt;

  /* implicit sync + implicit pop fame */
  ctxt->teaminfo->gwork = 0;
  ctxt->teaminfo = 0;
  old_ctxt = ctxt->save_ctxt;
  kaapic_end_parallel (KAAPI_SCHEDFLAG_DEFAULT);
  
  /* restore old context */
  kproc->libkomp_tls = old_ctxt;

  /* free shared resource */
  kaapi_atomic_destroylock(&teaminfo->lock);
  komp_barrier_destroy(&teaminfo->barrier);
}
