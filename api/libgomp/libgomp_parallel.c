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


kaapi_libgompctxt_t* GOMP_get_ctxtkproc( kaapi_processor_t* kproc )
{
  if (kproc->libgomp_tls == 0)
  {
    kaapi_libgompctxt_t* ctxt = (kaapi_libgompctxt_t*)malloc(sizeof(kaapi_libgompctxt_t));
    ctxt->threadid   = 0;
    ctxt->numthreads = 1;
    kproc->libgomp_tls = ctxt;
    return ctxt;
  }
  return (kaapi_libgompctxt_t*)kproc->libgomp_tls;
}


kaapi_libgompctxt_t* GOMP_get_ctxt()
{
  return GOMP_get_ctxtkproc(kaapi_get_current_processor());
}


int
omp_get_num_threads (void)
{
  return GOMP_get_ctxt()->numthreads;
}

int
omp_get_thread_num (void)
{
  return GOMP_get_ctxt()->threadid;
}


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
  ctxt->workshare.teaminfo = taskarg->teaminfo;

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
  GOMP_parallel_task_arg_t* arg;
  
  kaapic_begin_parallel();

  if (num_threads == 0)
    num_threads = kaapic_get_concurrency();
  
  thread = kaapi_threadcontext2thread(kproc->thread);

  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxtkproc(kproc);

  /* save frame */
  kaapi_thread_save_frame( thread, &ctxt->frame );
  
  /* init team information */
  kaapi_libgomp_teaminfo_t* teaminfo = 
    (kaapi_libgomp_teaminfo_t*)kaapi_thread_pushdata_align(
        thread, 
        sizeof(kaapi_libgomp_teaminfo_t), 
        8
  );
  kaapi_atomic_initlock(&teaminfo->lock);
  teaminfo->numthreads   = num_threads;
  memset( teaminfo->localinfo, 0, num_threads*sizeof(kaapi_libgompworkshared_t*) );
  teaminfo->localinfo[0] = &ctxt->workshare;
  
  
  /* init workshared construct, assume just one top level ctxt */
  KAAPI_ATOMIC_WRITE(&ctxt->workshare.init, 0);
  ctxt->workshare.workload  = 0;
  ctxt->workshare.master    = &ctxt->workshare;
  ctxt->workshare.teaminfo  = teaminfo;

  /* init team context */
  KAAPI_ATOMIC_WRITE(&global_single, 0);
  gomp_barrier_init (&global_barrier, num_threads);


  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  for (int i = 1; i < num_threads; i++)
  {
    kaapi_task_t* task = kaapi_thread_toptask(thread);
    kaapi_task_init( 
        task, 
        GOMP_trampoline_spawn, 
        kaapi_thread_pushdata(thread, sizeof(GOMP_parallel_task_arg_t)) 
    );
    arg = kaapi_task_getargst( task, GOMP_parallel_task_arg_t );
    arg->numthreads = num_threads;
    arg->threadid   = i;
    arg->fn         = fn;
    arg->data       = data;
    arg->teaminfo   = teaminfo; /* this is the master workshare of the team... */
    kaapi_thread_pushtask(thread);
  }

#if 0 /* previous lines are equivalents to : */
      kaapic_spawn (4, 
       GOMP_trampoline_spawn,
       KAAPIC_MODE_V, num_threads, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, i, 1, KAAPIC_TYPE_INT,
       KAAPIC_MODE_V, fn, 1, KAAPIC_TYPE_PTR,
       KAAPIC_MODE_V, data, 1, KAAPIC_TYPE_PTR
    );
#endif

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

  /* restore frame */
  kaapi_libgompctxt_t* ctxt = GOMP_get_ctxtkproc(kproc);
  kaapi_thread_restore_frame( kaapi_threadcontext2thread(kproc->thread), &ctxt->frame);
}
