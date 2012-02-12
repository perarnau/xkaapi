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


#define KAAPI_GOMP_USE_TASK 1


/* General comment on: http://gcc.gnu.org/onlinedocs/libgomp/Implementing-FOR-construct.html

4.11 Implementing FOR construct

       #pragma omp parallel for
       for (i = lb; i <= ub; i++)
         body;
becomes

       void subfunction (void *data)
       {
         long _s0, _e0;
         while (GOMP_loop_static_next (&_s0, &_e0))
         {
           long _e1 = _e0, i;
           for (i = _s0; i < _e1; i++)
             body;
         }
         GOMP_loop_end_nowait ();
       }
     
       GOMP_parallel_loop_static (subfunction, NULL, 0, lb, ub+1, 1, 0);
       subfunction (NULL);
       GOMP_parallel_end ();
       
       
       #pragma omp for schedule(runtime)
       for (i = 0; i < n; i++)
         body;
becomes

       {
         long i, _s0, _e0;
         if (GOMP_loop_runtime_start (0, n, 1, &_s0, &_e0))
           do {
             long _e1 = _e0;
             for (i = _s0, i < _e0; i++)
               body;
           } while (GOMP_loop_runtime_next (&_s0, _&e0));
         GOMP_loop_end ();
       }
Note that while it looks like there is trickiness to propagating a non-constant STEP, there isn't really. We're explicitly allowed to evaluate it as many times as we want, and any variables involved should automatically be handled as PRIVATE or SHARED like any other variables. So the expression should remain evaluable in the subfunction. We can also pull it into a local variable if we like, but since its supposed to remain unchanged, we can also not if we like.

If we have SCHEDULE(STATIC), and no ORDERED, then we ought to be able to get away with no work-sharing context at all, since we can simply perform the arithmetic directly in each thread to divide up the iterations. Which would mean that we wouldn't need to call any of these routines.

There are separate routines for handling loops with an ORDERED clause. Bookkeeping for that is non-trivial...

*/
bool GOMP_loop_dynamic_start (
  long start, 
  long end, 
  long incr, 
  long chunk_size,
	long *istart, 
  long *iend
)
{  
//  printf("%s:: \n", __FUNCTION__);
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  kaapi_libkompworkshared_t* workshare = &ctxt->workshare;
  kaapi_libkomp_teaminfo_t* teaminfo = ctxt->teaminfo;

  long ka_start = start;
  long ka_end   = (end-start+incr-1)/incr;
  workshare->incr = incr;

  if (ctxt->threadid ==0)
  {
    kaapic_foreach_attr_t attr;
    kaapic_foreach_attr_init(&attr);
    kaapic_foreach_attr_set_grains( &attr, chunk_size, chunk_size );
        
    /* initialize the master if not already done */
    workshare->lwork = kaapic_foreach_workinit(self_thread, 
          ka_start, 
          ka_end, 
          &attr, /* attr */
          0,     /* body */
          0      /* arg */
      );
    
    kaapi_writemem_barrier();
    teaminfo->gwork = workshare->lwork->global;
  }
  else {
    while (teaminfo->gwork ==0)
      kaapi_slowdown_cpu();
    kaapi_readmem_barrier();
    
    workshare->lwork = kaapic_foreach_local_workinit( 
                            self_thread,
                            teaminfo->gwork );    
  }

  /* pop next slice */
  if (kaapic_foreach_worknext(
        workshare->lwork, 
        istart,
        iend)
      )
  {
    *istart *= incr;
    *iend *= incr;
    return 1;
  }
  return 0;
}

bool GOMP_loop_dynamic_next (long *istart, long *iend)
{
//  printf("%s:: \n", __FUNCTION__);

  kaapi_processor_t*   kproc = kaapi_get_current_processor();
  kaapi_libkompctxt_t* ctxt  = komp_get_ctxtkproc( kproc );

  if (kaapic_foreach_worknext(
        ctxt->workshare.lwork, 
        istart,
        iend)
  )
  {
    *istart *= ctxt->workshare.incr;
    *iend *= ctxt->workshare.incr;
    return 1;
  }
  return 0;
}



#if (KAAPI_GOMP_USE_TASK == 1)
typedef struct komp_parallelfor_task_arg {
  int                       numthreads;
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  kaapi_libkomp_teaminfo_t* teaminfo;
  
  long                      incr;
} komp_parallelfor_task_arg_t;

static void komp_trampoline_task_parallelfor
(
  void*           voidp, 
  kaapi_thread_t* thread
);
#endif


/* TODO:
   - this form may be used to avoid initial synchronisation at 
   the expense of a longer critical path...
   Only the main thread call it. All thread uses first 
   GOMP_loop_dynamic_next to get initial range.
*/
void GOMP_parallel_loop_dynamic_start (
          void (*fn) (void *), 
          void *data,
				  unsigned num_threads, 
          long start, 
          long end, 
          long incr, 
          long chunk_size
)
{
  if (num_threads == 0)
    num_threads = gomp_nthreads_var;

//  printf("%s:: numthread:%i \n", __FUNCTION__, num_threads);

  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kaapi_thread_t* thread;

  kaapi_libkomp_teaminfo_t* teaminfo = 
      komp_init_parallel_start( kproc, num_threads );

  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  kaapi_libkompworkshared_t* workshare = &ctxt->workshare;

  long ka_start = start;
  long ka_end   = (end-start+incr-1)/incr;
  workshare->incr = incr;

  kaapi_assert_debug(ctxt->threadid ==0)

  kaapic_foreach_attr_t attr;
  kaapic_foreach_attr_init(&attr);
  
  /* initialize the master work */
  teaminfo->gwork = kaapic_foreach_global_workinit(
      self_thread, 
      ka_start, 
      ka_end, 
      &attr, 
      0,     /* body */
      0      /* arg */
  );

  workshare->lwork = kaapic_foreach_local_workinit(self_thread, teaminfo->gwork );
  /* initialize the local workqueue with the first poped state */

  if (kaapic_global_work_pop(teaminfo->gwork, kproc->kid, &start, &end))
  {    
#if defined(USE_KPROC_LOCK)
    kaapi_workqueue_init_with_lock(
      &workshare->lwork->cr,
      start, end,
      &kaapi_all_kprocessors[kproc->kid]->lock
    );
#else
    kaapi_atomic_initlock(&workshare->lwork->lock);
    kaapi_workqueue_init_with_lock(
      &workshare->lwork->cr, 
      start, end,
      &workshare->lwork->lock
    );
#endif
    workshare->lwork->global     = teaminfo->gwork;
    workshare->lwork->tid        = kproc->kid;
    teaminfo->gwork->lwork[kproc->kid] = workshare->lwork;
  }

  /* create each task, as in GOMP_parallel_start 
     + arguments to initialize local context
  */
#if (KAAPI_GOMP_USE_TASK == 1)
  kaapi_task_t* task;
  komp_parallelfor_task_arg_t* arg;
  komp_parallelfor_task_arg_t* allarg;
  
  thread = kaapi_threadcontext2thread(kproc->thread);
  allarg = kaapi_thread_pushdata(thread, 
            num_threads * sizeof(komp_parallelfor_task_arg_t)
  );

  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  task = kaapi_thread_toptask(thread);
  for (int i = 1; i < num_threads; i++)
  {
    kaapi_task_init( 
        task, 
        komp_trampoline_task_parallelfor, 
        allarg+i
    );
    arg = kaapi_task_getargst( task, komp_parallelfor_task_arg_t );
    arg->numthreads = num_threads;
    arg->threadid   = i;
    arg->fn         = fn;
    arg->data       = data;
    arg->teaminfo   = teaminfo; /* this is the master workshare of the team... */
    arg->incr       = incr;
    
    task = kaapi_thread_nexttask(thread, task);
  }
  kaapi_thread_push_packedtasks(thread, num_threads-1);
#else
#error "Not yet implemented"
#endif
}



static void komp_trampoline_task_parallelfor
(
  void*           voidp, 
  kaapi_thread_t* thread
)
{
//printf("In %s\n", __PRETTY_FUNCTION__ );
  komp_parallelfor_task_arg_t* taskarg = (komp_parallelfor_task_arg_t*)voidp;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kaapi_libkompctxt_t* ctxt = komp_get_ctxtkproc(kproc);
  kaapi_workqueue_index_t start, end;
  
  /* save context information */
  int save_numthreads = ctxt->numthreads;
  int save_threadid = ctxt->threadid;
  kaapi_libkomp_teaminfo_t* save_teaminfo = ctxt->teaminfo;
  
  ctxt->numthreads         = taskarg->numthreads;
  ctxt->threadid           = taskarg->threadid;
  ctxt->inside_single      = 0;
  ctxt->teaminfo           = taskarg->teaminfo;

  kaapi_assert_debug(ctxt->threadid !=0);

  kaapi_libkompworkshared_t* workshare = &ctxt->workshare;
  workshare->incr = taskarg->incr;
  /* only main thread of the team has initialized global work */
  workshare->lwork = kaapic_foreach_local_workinit( 
                          kproc->thread,
                          ctxt->teaminfo->gwork );    

  if (kaapic_global_work_pop(ctxt->teaminfo->gwork, kproc->kid, &start, &end))
  {    
#if defined(USE_KPROC_LOCK)
    kaapi_workqueue_init_with_lock(
      &workshare->lwork->cr,
      start, end,
      &kaapi_all_kprocessors[kproc->kid]->lock
    );
#else
    kaapi_atomic_initlock(&workshare->lwork->lock);
    kaapi_workqueue_init_with_lock(
      &workshare->lwork->cr, 
      start, end,
      &workshare->lwork->lock
    );
#endif
    workshare->lwork->global     = ctxt->teaminfo->gwork;
    workshare->lwork->tid        = kproc->kid;
    ctxt->teaminfo->gwork->lwork[kproc->kid] = workshare->lwork;
  }

//printf("In %s: begin call user code \n", __PRETTY_FUNCTION__ );
  /* GCC compiled code */
  taskarg->fn(taskarg->data);
//printf("In %s: end call user code \n", __PRETTY_FUNCTION__ );

  if (ctxt->threadid !=0)
  {
    kaapic_foreach_local_workend( ctxt->workshare.lwork );
    
    /* Restore the initial context values. */
    ctxt->numthreads         = save_numthreads;
    ctxt->threadid           = save_threadid;
    ctxt->teaminfo           = save_teaminfo;
  }
}

KAAPI_REGISTER_TASKFORMAT( komp_parallelfor_task_format,
    "GOMP/Parallel Task",
    komp_trampoline_task_parallelfor,
    sizeof(komp_parallelfor_task_arg_t),
    5,
    (kaapi_access_mode_t[]){ 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V, 
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V,
        KAAPI_ACCESS_MODE_V 
    },
    (kaapi_offset_t[])     { 
        offsetof(komp_parallelfor_task_arg_t, numthreads), 
        offsetof(komp_parallelfor_task_arg_t, threadid), 
        offsetof(komp_parallelfor_task_arg_t, fn), 
        offsetof(komp_parallelfor_task_arg_t, data),
        offsetof(komp_parallelfor_task_arg_t, teaminfo),
        offsetof(komp_parallelfor_task_arg_t, incr)
    },
    (kaapi_offset_t[])     { 0, 0, 0, 0, 0,0 },
    (const struct kaapi_format_t*[]) { 
        kaapi_int_format, 
        kaapi_int_format,
        kaapi_voidp_format,
        kaapi_voidp_format, 
        kaapi_voidp_format,
        kaapi_long_format
      },
    0
)


bool GOMP_loop_ordered_dynamic_start (
          long start, 
          long end, 
          long incr,
          long chunk_size, 
          long *istart, 
          long *iend
)
{
  printf("%s:: \n", __FUNCTION__);
  return 0;
}

bool GOMP_loop_ordered_dynamic_next (long *istart, long *iend)
{
  printf("%s:: \n", __FUNCTION__);
  return 0;
}



