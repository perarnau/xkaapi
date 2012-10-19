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

double kaapi_komp_start_parallel;
int kaapi_komp_start_parallel_count;
double kaapi_komp_end_parallel;
int kaapi_komp_end_parallel_count;

typedef struct komp_parallel_task_arg {
  int                       threadid;
  void                    (*fn) (void *);
  void*                     data;
  komp_teaminfo_t*          teaminfo;
  int                       nextnumthreads;
  int                       nestedlevel;
  int                       nestedparallel;
  int                       active_level;
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
  new_ctxt->icv.nested_level    = taskarg->nestedlevel;
  new_ctxt->icv.nested_parallel = taskarg->nestedparallel;
  new_ctxt->icv.active_level   = taskarg->active_level;
  
  new_ctxt->icv.run_sched           = omp_sched_dynamic;
  new_ctxt->icv.chunk_size          = 0; /* default */
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  kaapic_foreach_attr_init( &new_ctxt->icv.attr );
#endif
  
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
                          8,
                          (kaapi_access_mode_t[]){ 
                            KAAPI_ACCESS_MODE_V, 
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
                            offsetof(komp_parallel_task_arg_t, nestedparallel),
                            offsetof(komp_parallel_task_arg_t, active_level)
                          },
                          (kaapi_offset_t[])     { 0, 0, 0, 0, 0, 0, 0, 0 },
                          (const struct kaapi_format_t*[]) { 
                            kaapi_int_format, 
                            kaapi_voidp_format,
                            kaapi_voidp_format, 
                            kaapi_voidp_format,
                            kaapi_int_format, 
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
      ||  (ctxt->icv.nested_level >= omp_max_active_levels) )
    num_threads = 1;
  else {
    if (num_threads == 0)
      num_threads = (komp_env_nthreads != 0) ? komp_env_nthreads : ctxt->icv.next_numthreads;
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
  
  /* barrier for the team */
  komp_barrier_init (&teaminfo->barrier, num_threads);
  
  teaminfo->single_data        = 0;
  teaminfo->numthreads         = num_threads;
  teaminfo->gwork              = 0;
  teaminfo->serial             = 0;
  teaminfo->previous_team      = ctxt->teaminfo;
  teaminfo->father_id          = ctxt->icv.thread_id;
  
  /* init workshared construct */
  new_ctxt->workshare          = 0;
  new_ctxt->teaminfo           = teaminfo;
  
  /* initialize master context: nextnum thread is inherited */
  new_ctxt->icv.thread_id       = 0;
  new_ctxt->icv.next_numthreads = ctxt->icv.next_numthreads; /* WARNING: spec ? */
  new_ctxt->icv.nested_level    = 1+ctxt->icv.nested_level; 
  new_ctxt->icv.nested_parallel = ctxt->icv.nested_parallel; /* WARNING: spec ? */
  new_ctxt->icv.active_level    = num_threads == 1 ? ctxt->icv.active_level : ctxt->icv.active_level + 1;
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  new_ctxt->icv.attr            = ctxt->icv.attr;            /* WARNING: spec ? */
#endif
  
  new_ctxt->inside_single       = 0;
  new_ctxt->save_ctxt           = ctxt;
  
  /* swap context: until end_parallel, new_ctxt becomes the current context */
  kproc->libkomp_tls = new_ctxt;
  
  return teaminfo;
}

static void
komp_task_prepare (kaapi_task_t *task, 
                   komp_parallel_task_arg_t *allarg,
                   kaapi_thread_t *thread,
                   void (*fn) (void *),
                   void *data,
                   komp_teaminfo_t *teaminfo,
                   kompctxt_t* ctxt,
                   int task_logical_id)
{
  komp_parallel_task_arg_t *arg = NULL;
  kaapi_task_init( 
                  task, 
                  komp_trampoline_task_parallel, 
                  allarg + task_logical_id
                  );
  arg = kaapi_task_getargst( task, komp_parallel_task_arg_t );
  arg->threadid       = task_logical_id;
  arg->fn             = fn;
  arg->data           = data;
  arg->teaminfo       = teaminfo;
  arg->nextnumthreads = ctxt->icv.next_numthreads;
  arg->nestedlevel    = ctxt->icv.nested_level;
  arg->nestedparallel = ctxt->icv.nested_parallel;
  arg->active_level   = ctxt->icv.active_level;
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
  komp_parallel_task_arg_t* allarg;
  
  /* begin parallel region: also push a new frame that will be pop
   during call to kaapic_end_parallel
   */
  
#if KAAPI_KOMP_TRACE
  double t0, t1;
  t0 = kaapic_get_time();
#endif

  kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);
  
  /* init the new context with team information and workshare construct 
   the method push a new context in the caller Kaapi' stack 
   and a call to komp_get_ctxtkproc must be done to retreive the new ctxt.
   */
#if KAAPI_KOMP_TRACE
  t0 = kaapic_get_time();
#endif
  teaminfo = komp_init_parallel_start( kproc, num_threads );
  thread = kaapi_threadcontext2thread(kproc->thread);
  
  ctxt = komp_get_ctxtkproc(kproc);
  num_threads = teaminfo->numthreads;
  
  /* allocate in the caller stack the tasks for the parallel region */
  allarg = kaapi_thread_pushdata(thread, num_threads * sizeof(komp_parallel_task_arg_t));
  
  if (!ctxt->icv.nested_parallel)
  {
    /* push the task for the i-th kprocessor queue... 
     - work fine for 1rst level parallel region.
     - else may introduce deadlock because threads are not reused (...)
     If thread i (kprocessor i) is waiting on a barrier while and other
     thread push a task into its mailbox, then thread-i is unable to execute
     the task (...)
     */
    
    int nb_worker_threads = kaapi_getconcurrency ();
    int tasks_per_thread[nb_worker_threads];
    int chunk_size = num_threads / nb_worker_threads;
    int remaining_tasks = num_threads - (nb_worker_threads * chunk_size);
    
    for (int i = 0; i < nb_worker_threads; i++)
      tasks_per_thread[i] = chunk_size;
    
    int thread_id = 0;
    while (remaining_tasks != 0)
    {
      tasks_per_thread[thread_id]++;
      thread_id = (thread_id + 1) % nb_worker_threads;
      remaining_tasks--;
    }
    
    int task_id = 1;
    /* Distribute the num_threads tasks over the nb_worker_threads workers. */
    for (int i = 0; i < nb_worker_threads; i++)
    {
      int nb_pushed_tasks = (i == 0) ? 1 : 0; /* The master thread calls fn (data) directly. */
      
#if 0
      task = kaapi_thread_toptask(thread);
      while (nb_pushed_tasks < tasks_per_thread[i])
      {
        komp_task_prepare (task, allarg, thread, fn, data, teaminfo, ctxt, task_id++);
        kaapi_thread_distribute_task (thread, i);
        task = kaapi_thread_nexttask(thread, task);      
        nb_pushed_tasks++;
      }
#else
      while (nb_pushed_tasks < tasks_per_thread[i])
      {
        task = kaapi_thread_toptask(thread);
        komp_task_prepare (task, allarg, thread, fn, data, teaminfo, ctxt, task_id++);
        kaapi_thread_pushtask(thread);
        nb_pushed_tasks++;
      }
#endif

    }
  } 
  else 
  { 
    /* Nested parallel region, push all nested tasks in the queue of the calling thread. */
    task = kaapi_thread_toptask(thread);
    for (int i = 1; i < num_threads; i++)
    {
      komp_task_prepare (task, allarg, thread, fn, data, teaminfo, ctxt, i);
      task = kaapi_thread_nexttask(thread, task);
    }
    kaapi_thread_push_packedtasks(thread, num_threads-1);
  }

#if KAAPI_KOMP_TRACE
  t1 = kaapic_get_time();
  kaapi_komp_start_parallel += t1 - t0;
  ++kaapi_komp_start_parallel_count;
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

#if KAAPI_KOMP_TRACE
  double t0, t1; 
  t0 = kaapic_get_time();
#endif
  
  /* implicit sync + implicit pop fame */
  ctxt->teaminfo->gwork = 0;
  ctxt->teaminfo = 0;
  old_ctxt = ctxt->save_ctxt;
  
  kaapic_end_parallel (KAAPI_SCHEDFLAG_DEFAULT);

  /* restore old context */
  kproc->libkomp_tls = old_ctxt;
  
  /* free shared resource */
  komp_barrier_destroy(&teaminfo->barrier);
#if KAAPI_KOMP_TRACE
  t1 = kaapic_get_time();
  kaapi_komp_end_parallel += t1-t0;
  ++kaapi_komp_end_parallel_count;
#endif
}
