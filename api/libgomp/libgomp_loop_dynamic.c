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

/*
*/
static inline komp_workshare_t*  komp_loop_dynamic_start_init(
  kaapi_processor_t* kproc,
  long               start, 
  long               incr
)
{
  kaapi_thread_t* thread = kaapi_threadcontext2thread(kproc->thread);
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;
  komp_workshare_t* workshare = ctxt->workshare;

  /* initialize the work share data: reuse previous allocated workshare if !=0 */
  if (workshare ==0)
  {
    workshare = kaapi_thread_pushdata(thread, sizeof(komp_workshare_t) );
    ctxt->workshare = workshare;
  }
  workshare->start  = start;
  workshare->incr   = incr;
  workshare->serial = ++teaminfo->serial;
  return workshare;
}

/*
*/
static inline void komp_loop_dynamic_start_master(
  kaapi_processor_t* kproc,
  komp_workshare_t* workshare,
  long start, 
  long end, 
  long incr, 
  long chunk_size
)
{
  kaapi_thread_context_t* const self_thread = kproc->thread;
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;

  long ka_start = 0;
  long ka_end   = (end-start+incr-1)/incr;
  
  /* TODO: automatic adaptation on the chunksize here
     or (better) automatic adaptation in libkaapic
  */
  kaapic_foreach_attr_t attr;
  kaapic_foreach_attr_init(&attr);
  if (chunk_size == -1) 
  {
    chunk_size=(ka_end-ka_start)/1024*kaapi_getconcurrency();
    if (chunk_size ==0) chunk_size = 1;
  }
  kaapic_foreach_attr_set_grains( &attr, chunk_size, 1 );
  kaapic_foreach_attr_set_threads( &attr, teaminfo->numthreads );
      
  /* initialize the master if not already done */
  workshare->lwork = kaapic_foreach_workinit(self_thread, 
        ka_start, 
        ka_end, 
        &attr, /* attr */
        0,     /* body */
        0      /* arg */
    );

  /* publish the global work information */
  kaapi_writemem_barrier();
  teaminfo->gwork = workshare->lwork->global;
}


/*
*/
static inline void komp_loop_dynamic_start_slave(
  kaapi_processor_t* kproc,
  komp_workshare_t*  workshare,
  kaapic_global_work_t* gwork
)
{
  long start, end;

  /* wait global work becomes ready */
  kaapi_assert_debug(gwork !=0);
        
  /* get own slice */
  if (!kaapic_global_work_pop( gwork, kproc->kid, &start, &end))
    start = end = 0;
  else
    printf("%i::Slave -> [%li,%li)\n", kaapi_get_self_kid(), start, end); fflush(stdout);

  workshare->lwork = kaapic_foreach_local_workinit( 
                          &gwork->lwork[kproc->kid],
                          start, end );
}


/* This method is called by each task of the parallel region
   The master thread publish the global work information in 
   the teaminfo data structure.
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
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  kompctxt_t* ctxt = komp_get_ctxtkproc( kproc );
  komp_teaminfo_t* teaminfo = ctxt->teaminfo;
  kaapic_global_work_t* gwork;

  komp_workshare_t* workshare = 
    komp_loop_dynamic_start_init( kproc, start, incr );

  if (ctxt->icv.thread_id ==0)
  {
    komp_loop_dynamic_start_master(
      kproc,
      workshare,
      start,
      end,
      incr,
      chunk_size
    );
    gwork = teaminfo->gwork;
  }
  else 
  {
    /* wait global work becomes ready */
    while ( (gwork = teaminfo->gwork) ==0)
      kaapi_slowdown_cpu();
    kaapi_readmem_barrier();

    komp_loop_dynamic_start_slave(
      kproc,
      workshare,
      gwork
    );
  }

  /* pop next range and start execution (on return...) */
  if (kaapic_foreach_worknext(
        workshare->lwork, 
        istart,
        iend)
      )
  {
    *istart = ctxt->workshare->start + *istart * ctxt->workshare->incr;
    *iend   = ctxt->workshare->start + *iend   * ctxt->workshare->incr;
    return 1;
  }
  return 0;
}


/*
*/
bool GOMP_loop_dynamic_next (long *istart, long *iend)
{
  kaapi_processor_t*   kproc = kaapi_get_current_processor();
  kompctxt_t* ctxt  = komp_get_ctxtkproc( kproc );

  if (kaapic_foreach_worknext(
        ctxt->workshare->lwork, 
        istart,
        iend)
  )
  {
    *istart = ctxt->workshare->start + *istart * ctxt->workshare->incr;
    *iend   = ctxt->workshare->start + *iend   * ctxt->workshare->incr;
    return 1;
  }
  return 0;
}


/* fwd decl */
typedef struct komp_parallelfor_task_arg_t {
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  komp_teaminfo_t*          teaminfo;
  int                       nextnumthreads;
  long                      start;
  long                      incr;
} komp_parallelfor_task_arg_t;

static void komp_trampoline_task_parallelfor
(
  void*           voidp, 
  kaapi_thread_t* thread
);


/* Same as GOMP_parallel_start except that:
   - workshare is initialized before creating sub task
   - the created task is a komp_trampoline_task_parallelfor
   that execute the second branch of GOMP_loop_dynamic_start
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
  kaapi_processor_t* kproc  = kaapi_get_current_processor();
  kompctxt_t* ctxt;
  kaapi_thread_t* thread;
  komp_teaminfo_t* teaminfo;
  komp_workshare_t* workshare;
  kaapi_task_t* task;
  komp_parallelfor_task_arg_t* arg;
  komp_parallelfor_task_arg_t* allarg;
    
  /* begin parallel region: also push a new frame that will be pop
     during call to kaapic_end_parallel
  */
  kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);

  /* init the new context with team information and workshare construct 
     the method push a new context in the caller Kaapi' stack 
     and a call to komp_get_ctxtkproc must be done to retreive the new ctxt.
  */
  teaminfo = komp_init_parallel_start( kproc, num_threads );
  
  ctxt = komp_get_ctxtkproc(kproc);
  num_threads = teaminfo->numthreads;

  /* initialize master workshare construct */
  workshare = komp_loop_dynamic_start_init( kproc, start, incr );
  komp_loop_dynamic_start_master(
    kproc,
    workshare,
    start,
    end,
    incr,
    chunk_size
  );
  
  /* allocate in the caller stack the tasks for the parallel region */
  thread = kaapi_threadcontext2thread(kproc->thread);
  allarg = kaapi_thread_pushdata(thread, 
     num_threads * sizeof(komp_parallelfor_task_arg_t)
  );
  
  /* The master thread (id 0) calls fn (data) directly. That's why we
     start this loop from id = 1.*/
  task = kaapi_thread_toptask(thread);
  for (int i = 1; i < num_threads; i++)
  {
    arg = allarg+i;
    arg->threadid   = i;
    arg->fn         = fn;
    arg->data       = data;
    arg->teaminfo   = teaminfo;
    /* WARNING: see spec: nextnum threads is inherited ? */
    arg->nextnumthreads = ctxt->icv.next_numthreads;
    arg->start      = start;
    arg->incr       = incr;

    kaapi_task_init( 
        task, 
        komp_trampoline_task_parallelfor, 
        arg
    );
    task = kaapi_thread_nexttask(thread, task);
  }
  kaapi_thread_push_packedtasks(thread, num_threads-1);
}



static void komp_trampoline_task_parallelfor
(
  void*           voidp, 
  kaapi_thread_t* thread
)
{
  komp_parallelfor_task_arg_t* taskarg = (komp_parallelfor_task_arg_t*)voidp;
  kaapi_processor_t* kproc = kaapi_get_current_processor();
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
  
  new_ctxt->inside_single      = 0;
  new_ctxt->save_ctxt          = ctxt;
  
  /* swap context: until end_parallel, new_ctxt becomes the current context */
  kproc->libkomp_tls = new_ctxt;

  /* initialize the workshare construct */
  komp_workshare_t* workshare = 
    komp_loop_dynamic_start_init( kproc, taskarg->start, taskarg->incr );

  kaapi_assert_debug(new_ctxt->icv.thread_id !=0);
  kaapi_assert_debug(new_ctxt->teaminfo->gwork !=0);

  komp_loop_dynamic_start_slave(
    kproc,
    workshare,
    new_ctxt->teaminfo->gwork
  );

  /* GCC compiled code */
  taskarg->fn(taskarg->data);

  /* restore the initial context */
  kproc->libkomp_tls = ctxt;
}

KAAPI_REGISTER_TASKFORMAT( komp_parallelfor_task_format,
    "KOMP/ParallelFor Task",
    komp_trampoline_task_parallelfor,
    sizeof(komp_parallelfor_task_arg_t),
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
        offsetof(komp_parallelfor_task_arg_t, threadid), 
        offsetof(komp_parallelfor_task_arg_t, fn), 
        offsetof(komp_parallelfor_task_arg_t, data),
        offsetof(komp_parallelfor_task_arg_t, teaminfo),
        offsetof(komp_parallelfor_task_arg_t, nextnumthreads), 
        offsetof(komp_parallelfor_task_arg_t, start), 
        offsetof(komp_parallelfor_task_arg_t, incr)
    },
    (kaapi_offset_t[])     { 0, 0, 0, 0, 0, 0, 0 },
    (const struct kaapi_format_t*[]) { 
        kaapi_int_format,
        kaapi_voidp_format,
        kaapi_voidp_format, 
        kaapi_voidp_format,
        kaapi_int_format, 
        kaapi_long_format,
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



