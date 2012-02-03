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



typedef struct GOMP_parallel_task_arg {
  int                       numthreads;
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  kaapi_libgomp_teaminfo_t* teaminfo;
} GOMP_parallel_task_arg_t;

static void GOMP_trampoline_spawn(
  void* voidp, kaapi_thread_t* thread
)
{
  GOMP_parallel_task_arg_t* taskarg = (GOMP_parallel_task_arg_t*)voidp;
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxt();
  
  ctxt->numthreads         = taskarg->numthreads;
  ctxt->threadid           = taskarg->threadid;
  KAAPI_ATOMIC_WRITE(&ctxt->workshare.init, 0);
  ctxt->workshare.master   = taskarg->teaminfo->localinfo[0];
  ctxt->teaminfo           = taskarg->teaminfo;

  /* register thread to the workshare structure */
  KAAPI_ATOMIC_WRITE_BARRIER(&ctxt->workshare.init, 0);
  taskarg->teaminfo->localinfo[ctxt->threadid] = &ctxt->workshare;

  taskarg->fn(taskarg->data);
}

KAAPI_REGISTER_TASKFORMAT( GOMP_parallel_task_format,
    "GOMP/Parallel Task",
    GOMP_trampoline_spawn,
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


void 
GOMP_parallel_start (void (*fn) (void *), void *data, unsigned num_threads)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_t* thread;
  kaapi_task_t* task;
  GOMP_parallel_task_arg_t* arg;
  GOMP_parallel_task_arg_t* allarg;
  
  if (num_threads == 0)
    num_threads = gomp_nthreads_var;

  /* do not save the ctxt, assume just one top level ctxt */
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxtkproc(kproc);

  thread = kaapi_threadcontext2thread(kproc->thread);

  /* save frame */
  kaapi_thread_save_frame( thread, &ctxt->frame );
  
  /* init team information */
  kaapi_libgomp_teaminfo_t* teaminfo = 
    (kaapi_libgomp_teaminfo_t*)kaapi_thread_pushdata_align(
        thread, 
        sizeof(kaapi_libgomp_teaminfo_t), 
        8
  );
  /* lock for ??? */
  kaapi_atomic_initlock(&teaminfo->lock);

  /* barrier for the team */
  gomp_barrier_init (&teaminfo->barrier, num_threads);

  teaminfo->numthreads   = num_threads;
  KAAPI_ATOMIC_WRITE(&teaminfo->single_state, 0);
  memset( teaminfo->localinfo, 0, num_threads*sizeof(kaapi_libgompworkshared_t*) );
  teaminfo->localinfo[0] = &ctxt->workshare;
  
  /* init workshared construct, assume just one top level ctxt */
  KAAPI_ATOMIC_WRITE(&ctxt->workshare.init, 0);
  ctxt->workshare.workload  = 0;
  ctxt->workshare.master    = &ctxt->workshare;
  ctxt->teaminfo            = teaminfo;


  allarg = kaapi_thread_pushdata(thread, num_threads * sizeof(GOMP_parallel_task_arg_t));

  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  task = kaapi_thread_toptask(thread);
  for (int i = 1; i < num_threads; i++)
  {
    kaapi_task_init( 
        task, 
        GOMP_trampoline_spawn, 
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

#if 0 /* previous lines are equivalents to : */
      kaapic_spawn (4, 
       GOMP_trampoline_spawn,
       KAAPIC_MODE_V, num_threads, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, i, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, fn, 1, KAAPIC_TYPE_PTR,
       KAAPIC_MODE_V, data, 1, KAAPIC_TYPE_PTR
    );
#endif

  kaapic_begin_parallel();

  /* initialize master context */
  ctxt->numthreads = num_threads;
  ctxt->threadid   = 0;
}

void 
GOMP_parallel_end (void)
{
  kaapi_processor_t* kproc = kaapi_get_current_processor();

  /* implicit sync */
  kaapic_end_parallel (0);
  //kaapic_sync ();

  /* restore frame */
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxtkproc(kproc);
  kaapi_atomic_destroylock(&ctxt->teaminfo->lock);

  kaapi_thread_restore_frame( kaapi_threadcontext2thread(kproc->thread), &ctxt->frame);
  ctxt->teaminfo = 0;
}
