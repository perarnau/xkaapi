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

// ==1 to use task based begin_parallel
// else ==0 to use foreach based begin_parallel
#define KAAPI_GOMP_USE_TASK 1


#if (KAAPI_GOMP_USE_TASK == 1)

typedef struct GOMP_parallel_task_arg {
  int                       numthreads;
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  kaapi_libkomp_teaminfo_t* teaminfo;
} GOMP_parallel_task_arg_t;

static void GOMP_trampoline_task_parallel
(
  void*           voidp, 
  kaapi_thread_t* thread
)
{
  GOMP_parallel_task_arg_t* taskarg = (GOMP_parallel_task_arg_t*)voidp;
  kaapi_libkompctxt_t* ctxt = komp_get_ctxt();
  
  /* save context information */
  int save_numthreads = ctxt->numthreads;
  int save_threadid = ctxt->threadid;
  kaapi_libkomp_teaminfo_t* save_teaminfo = ctxt->teaminfo;

  ctxt->numthreads         = taskarg->numthreads;
  ctxt->threadid           = taskarg->threadid;
  ctxt->inside_single      = 0;
  ctxt->teaminfo           = taskarg->teaminfo;

  taskarg->fn(taskarg->data);

  /* Restore the initial context values. */
  ctxt->numthreads         = save_numthreads;
  ctxt->threadid           = save_threadid;
  ctxt->teaminfo           = save_teaminfo;
}

KAAPI_REGISTER_TASKFORMAT( GOMP_parallel_task_format,
    "GOMP/Parallel Task",
    GOMP_trampoline_task_parallel,
    sizeof(GOMP_parallel_task_arg_t),
    5,
    (kaapi_access_mode_t[]){ 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V 
    },
    (kaapi_offset_t[])     { 
        offsetof(GOMP_parallel_task_arg_t, numthreads), 
        offsetof(GOMP_parallel_task_arg_t, threadid), 
        offsetof(GOMP_parallel_task_arg_t, fn), 
        offsetof(GOMP_parallel_task_arg_t, data),
        offsetof(GOMP_parallel_task_arg_t, teaminfo)
    },
    (kaapi_offset_t[])     { 0, 0, 0, 0, 0 },
    (const struct kaapi_format_t*[]) { 
        kaapi_int_format, 
        kaapi_int_format,
        kaapi_voidp_format,
        kaapi_voidp_format, 
        kaapi_voidp_format
      },
    0
)

#else

static void GOMP_trampoline_task_parallel
(
  int32_t                   i, 
  int32_t                   j, 
  int32_t                   tid,
  int                       numthreads,
  kaapi_libkomp_teaminfo_t* teaminfo,
  void                    (*fn) (void *),
  void*                     data
)
{
  kaapi_libkompctxt_t* ctxt= komp_get_ctxt();
  
  ctxt->numthreads         = numthreads;
  KAAPI_ATOMIC_WRITE(&ctxt->workshare.init, 0);
  ctxt->teaminfo           = teaminfo;
  
  for (int32_t k = i; k<j; ++k)
  {
    ctxt->threadid = k;
    fn(data);
  }
}

#endif

kaapi_libkomp_teaminfo_t* 
komp_init_parallel_start (
  kaapi_processor_t* kproc,
  unsigned num_threads
)
{
  kaapi_thread_t* thread;

  if (num_threads == 0)
    num_threads = gomp_nthreads_var;

  /* do not save the ctxt, assume just one top level ctxt */
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc(kproc);

  thread = kaapi_threadcontext2thread(kproc->thread);
  
  /* init team information */
  kaapi_libkomp_teaminfo_t* teaminfo = 
    (kaapi_libkomp_teaminfo_t*)kaapi_thread_pushdata_align(
        thread, 
        sizeof(kaapi_libkomp_teaminfo_t), 
        8
  );
  /* lock for ??? */
  kaapi_atomic_initlock(&teaminfo->lock);

  /* barrier for the team */
  gomp_barrier_init (&teaminfo->barrier, num_threads);

  teaminfo->numthreads   = num_threads;
  KAAPI_ATOMIC_WRITE(&teaminfo->single_state, 0);

  /* init workshared construct, assume just one top level ctxt */
  ctxt->workshare.workload  = 0;
  ctxt->teaminfo            = teaminfo;

  /* initialize master context */
  ctxt->numthreads = num_threads;
  ctxt->threadid   = 0;

  return teaminfo;
};


void 
GOMP_parallel_start (
  void (*fn) (void *), 
  void *data, 
  unsigned num_threads
)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_t* thread;

  if (num_threads == 0)
    num_threads = gomp_nthreads_var;

  kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);

  kaapi_libkomp_teaminfo_t* teaminfo = 
    komp_init_parallel_start( kproc, num_threads );
  
  thread = kaapi_threadcontext2thread(kproc->thread);

#if (KAAPI_GOMP_USE_TASK == 1)

  kaapi_task_t* task;
  GOMP_parallel_task_arg_t* arg;
  GOMP_parallel_task_arg_t* allarg;
  
  
  allarg = kaapi_thread_pushdata(thread, num_threads * sizeof(GOMP_parallel_task_arg_t));

  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  task = kaapi_thread_toptask(thread);
  for (int i = 1; i < num_threads; i++)
  {
    kaapi_task_init( 
        task, 
        GOMP_trampoline_task_parallel, 
        allarg+i
    );
    arg = kaapi_task_getargst( task, GOMP_parallel_task_arg_t );
    arg->numthreads = num_threads;
    arg->threadid   = i;
    arg->fn         = fn;
    arg->data       = data;
    arg->teaminfo   = teaminfo; /* this is the master workshare of the team... */

    task = kaapi_thread_nexttask(thread, task);
  }
  kaapi_thread_push_packedtasks(thread, num_threads-1);

#else

  /* ici : il faut un kaapic_foreach asynchrone 
     - creation de la tache adaptative
     - mais on continue localement.
     Le thread courant ne doit pas prendre d'element dans la sequence
     pour les executer.
  */
  kaapic_foreach_attr_t attr;
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains( &attr, 1, 1 );
  
  kaapic_foreach(1, num_threads, &attr, 
    4, GOMP_trampoline_task_parallel,
    num_threads,
    teaminfo,
    fn,
    data
  );

#endif
}


void 
GOMP_parallel_end (void)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();

  /* implicit sync + pop frame */
  kaapic_end_parallel (KAAPI_SCHEDFLAG_DEFAULT);

  /* free shared resource */
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  kaapi_atomic_destroylock(&ctxt->teaminfo->lock);

  ctxt->teaminfo = 0;
}
