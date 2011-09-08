/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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

/* see README for info related to the papi usage in kaapi
   see examples/for_each for general code information
 */

#include "kaapi.h"
#include <string.h>
#include <math.h>

typedef struct work
{
  kaapi_workqueue_t cr;
  void (*op)(double*);
  double* array;
} work_t;

typedef struct thief_work_t {
  void (*op)(double*);
  double* beg;
  double* end;
} thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*);


/* parallel work splitter */
static int splitter
(
 kaapi_stealcontext_t* sc, 
 int nreq, kaapi_request_t* req, 
 void* args
)
{
  /* victim work */
  work_t* const vw = (work_t*)args;
  
  /* stolen range */
  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;
  
  /* reply count */
  int nrep = 0;
  
  /* size per request */
  kaapi_workqueue_index_t unit_size;
  
redo_steal:
  /* do not steal if range size <= PAR_GRAIN */
#define CONFIG_PAR_GRAIN 128
  range_size = kaapi_workqueue_size(&vw->cr);
  if (range_size <= CONFIG_PAR_GRAIN)
    return 0;
  
  /* how much per req */
  unit_size = range_size / (nreq + 1);
  if (unit_size == 0)
  {
    nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
    unit_size = CONFIG_PAR_GRAIN;
  }
  
  /* perform the actual steal. if the range
   changed size in between, redo the steal
   */
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, nreq * unit_size))
    goto redo_steal;
  
  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
    ( sc, req, thief_entrypoint, sizeof(thief_work_t), 0 );
    tw->op  = vw->op;
    tw->beg = vw->array+j-unit_size;
    tw->end = vw->array+j;
    
    kaapi_reply_push_adaptive_task(sc, req);
  }
  
  return nrep;
}


/** seq work extractor 
 */
static int extract_seq(work_t* w, double** pos, double** end)
{
  /* extract from range beginning */
  kaapi_workqueue_index_t i, j;
  
#define CONFIG_SEQ_GRAIN 256
  if (kaapi_workqueue_pop(&w->cr, &i, &j, CONFIG_SEQ_GRAIN)) return 1;
  
  *pos = w->array + i;
  *end = w->array + j;
  
  return 0; /* success */
}


/** thief entrypoint 
 */
static void inline common_entry
(double* beg, double* end, void (*op)(double*))
{
  for (; beg != end; ++beg) op(beg);
}

static void thief_entrypoint(void* args, kaapi_thread_t* thread)
{
  /* process the work */
  thief_work_t* const thief_work = (thief_work_t*)args;
  
  /* range to process */
  double* const beg = thief_work->beg;
  double* const end = thief_work->end;
  
  /* apply w->op foreach item in [pos, end[ */
  common_entry(beg, end, thief_work->op);
}


/* For each main function */
static void for_each( double* array, size_t size, void (*op)(double*) )
{
  const int sc_flags = KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION;

  kaapi_thread_t* thread;
  kaapi_stealcontext_t* sc;
  work_t  work;
  double* pos;
  double* end;
  
  /* get the self thread */
  thread = kaapi_self_thread();
  
  /* initialize work */
  kaapi_workqueue_init(&work.cr, 0, (kaapi_workqueue_index_t)size);
  work.op    = op;
  work.array = array;
  
  /* push an adaptive task */
  sc = kaapi_task_begin_adaptive
    (thread, sc_flags, splitter, &work);
  
  /* while there is sequential work to do*/
  while (!extract_seq(&work, &pos, &end))
    common_entry(pos, end, op);
  
  /* wait for thieves */
  kaapi_task_end_adaptive(sc);
}


/**
 */
static void apply_cos( double* v )
{
  *v += cos(*v);
}


/**
 */
int main(int ac, char** av)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;

  /* at most 3 papi counters */
  kaapi_perf_idset_t perfids;
  kaapi_perf_counter_t counters[3] = {0, 0, 0};
  
#define ITEM_COUNT 100000
  static double array[ITEM_COUNT];
  
  /* initialize the runtime */
  kaapi_init(1, &ac, &av);

  /* create performance counter id set. use the
     KAAPI_PERF_PAPIES=a,b,c environ variable to
     bind KAAPI_PERF_ID_PAPI_N. man papi_avail(3)
     to see how to list available counters.
   */
  kaapi_perf_idset_zero(&perfids);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_0);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_1);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_2);
  
  for (iter = 0; iter < 100; ++iter)
  {
    /* initialize, apply, check */
    for (i = 0; i < ITEM_COUNT; ++i)
      array[i] = 0.f;

    t0 = kaapi_get_elapsedns();
    for_each( array, ITEM_COUNT, apply_cos );
    t1 = kaapi_get_elapsedns();
    sum += (t1-t0)/1000; /* ms */

    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] != 1.f)
      {
        printf("invalid @%lu == %lf\n", i, array[i]);
        break ;
      }
  }

  printf("done: %lf (ms)\n", sum / 100);

  /* accumulate all the processors counters and report
     counters[i] where i in KAAPI_PERF_PAPIES[i]
   */
  kaapi_perf_accum_counters(&perfids, counters);
  printf("perf_counters: %lu, %lu, %lu\n",
	 counters[0], counters[1], counters[2]);

  /* finalize the runtime */
  kaapi_finalize();
  
  return 0;
}
