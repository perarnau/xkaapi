/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include "kaapi.h"
#define __USE_BSD 1
#include <sys/time.h>


typedef struct work
{
  kaapi_workqueue_t cr;
  const double* iarray;
  double* oarray;
  double prefix;
} work_t;


typedef struct thief_work_t {
  const double* ibeg;
  const double* iend;
  double* obeg;
  double prefix;
  unsigned int is_reduced;
} thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


/* choose between parallel and sequential reduction */
#define CONFIG_PARALLEL_REDUCE 0

#if CONFIG_PARALLEL_REDUCE

typedef struct reduce_work_t
{
  double* pos;
  double* end;
  double prefix;
} reduce_work_t;

static void reduce_entrypoint(void* arg, kaapi_thread_t* thread)
{
  reduce_work_t* const work = (reduce_work_t*)arg;

  /* apply prefix over [work->pos, work->end[ */
  for (; work->pos != work->end; ++work->pos)
    *work->pos += work->prefix;
}

#endif /* CONFIG_PARALLEL_REDUCE */


static void common_reducer(work_t* vw, thief_work_t* tw)
{
  /* thief range continuation */
  kaapi_workqueue_index_t beg, end;
  beg = (kaapi_workqueue_index_t)(tw->ibeg - vw->iarray);
  end = (kaapi_workqueue_index_t)(tw->iend - vw->iarray);

  /* apply the victim prefix over the thief processed
     range, either sequentially or in parallel
  */

#if (CONFIG_PARALLEL_REDUCE == 0) /* sequential reduction */

  kaapi_workqueue_index_t pos = kaapi_workqueue_range_begin(&vw->cr);
  for (; pos < beg; ++pos)
    vw->oarray[pos] += vw->prefix;

#else /* parallel reduction */

  /* todo: for on the thread that touched the data. in
     the case of a thief reduction, that thread is the
     current one, inactive. but if we are reducing on
     the victim, there is a problem since we cannot
     push a task in a thread that is active, and we
     dont know the thief thread state.
   */

  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_task_t* const task = kaapi_thread_toptask(thread);

  reduce_work_t* const rw = kaapi_thread_pushdata_align
    (thread, sizeof(reduce_work_t), 8);
  rw->pos = vw->oarray + kaapi_workqueue_range_begin(&vw->cr);
  rw->end = vw->oarray + beg;
  rw->prefix = vw->prefix;

  kaapi_task_init(task, reduce_entrypoint, (void*)rw);
  kaapi_thread_pushtask(thread);

#endif

  /* continue the thief work */
  vw->prefix += tw->prefix;
#warning "Now may be not safe for concurrent exec"
  kaapi_workqueue_set(&vw->cr, beg, end);
}


static int thief_reducer
(kaapi_taskadaptive_result_t* ktr, void* varg, void* targ)
{
  /* called from the thief upon victim preemption request */

  common_reducer(varg, (thief_work_t*)ktr->data);

  /* inform the victim we did the reduction */
  ((thief_work_t*)ktr->data)->is_reduced = 1;

  return 0;
}


static int victim_reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* called from the victim to reduce a thief result */

  if (((thief_work_t*)tdata)->is_reduced == 0)
  {
    /* not already reduced */
    common_reducer(varg, tdata);
  }

  return 0;
}


/* parallel work splitter */
static int splitter
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
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
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(thief_work_t), NULL);

    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), ktr );
    tw->ibeg = vw->iarray+j-unit_size;
    tw->iend = vw->iarray+j;
    tw->obeg = vw->oarray+j-unit_size;
    tw->prefix = 0.f;
    tw->is_reduced = 0;

    /* initialize ktr task may be preempted before entrypoint */
    ((thief_work_t*)ktr->data)->ibeg = tw->ibeg;
    ((thief_work_t*)ktr->data)->iend = tw->iend;
    ((thief_work_t*)ktr->data)->obeg = tw->obeg;
    ((thief_work_t*)ktr->data)->prefix = 0.f;
    ((thief_work_t*)ktr->data)->is_reduced = 0;

    /* reply head, preempt head */
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }

  return nrep;
}


/* seq work extractor */
static int extract_seq
(work_t* w, const double** ipos, const double** iend, double** opos)
{
  int err;

  /* extract from range beginning */
  kaapi_workqueue_index_t i, j;

#define CONFIG_SEQ_GRAIN 128
  if ((err =kaapi_workqueue_pop(&w->cr, &i, &j, CONFIG_SEQ_GRAIN)) !=0) return 1;

  *ipos = w->iarray + i;
  *opos = w->oarray + i;
  *iend = w->iarray + j;

  return 0;
}


/* entrypoint */
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* input work */
  thief_work_t* const work = (thief_work_t*)args;

  /* resulting work */
  thief_work_t* const res_work = kaapi_adaptive_result_data(sc);

  /* oarray[pos] = sumof(iarray[0:pos[) */
  while (work->ibeg != work->iend)
  {
    work->prefix += *work->ibeg;
    *work->obeg = work->prefix;

    /* update prior reducing */
    ++work->ibeg;
    ++work->obeg;

    const unsigned int is_preempted = kaapi_preemptpoint
      (sc, thief_reducer, NULL, (void*)work, sizeof(thief_work_t), NULL);
    if (is_preempted) return ;
  }

  /* we are finished, update results. */
  res_work->ibeg = work->ibeg;
  res_work->iend = work->iend;
  res_work->obeg = work->obeg;
  res_work->prefix = work->prefix;
}


/* algorithm main function */
static void prefix(const double* iarray, double* oarray, size_t size)
{
  /* self thread, task */
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;

  /* sequential work */
  const double* ipos, *iend;
  double* opos;

  /* initialize work */
  work_t work;
  kaapi_workqueue_init(&work.cr, 0, (kaapi_workqueue_index_t)size);
  work.iarray = iarray;
  work.oarray = oarray;
  work.prefix = 0.f;

  /* push an adaptive task. set the preemption flag. */
  sc = kaapi_task_begin_adaptive(
        thread, 
        KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION, 
        splitter, 
        &work     /* arg for splitter = work to split */
    );

  /* while there is sequential work to do */
 continue_work:
  while (!extract_seq(&work, &ipos, &iend, &opos))
  {
    for (; ipos != iend; ++ipos, ++opos)
    {
      work.prefix += *ipos;
      *opos = work.prefix;
    }
  }

  /* preempt and reduce thieves */
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief(sc, ktr, (void*)&work, victim_reducer, (void*)&work);
    goto continue_work;
  }

#if CONFIG_PARALLEL_REDUCE
  /* wait for the parallel reduction tasks */
  kaapi_sched_sync();
#endif

  kaapi_task_end_adaptive(thread, sc);

  /* wait for thieves */
  kaapi_sched_sync();
}


/* unit */

int main(int ac, char** av)
{
  struct timeval tms[3];
  double sum = 0.f;

#define ITEM_COUNT 100000
  static double iarray[ITEM_COUNT];
  static double oarray[ITEM_COUNT];

  /* initialize the runtime */
  kaapi_init(1, &ac, &av);

  for (ac = 0; ac < 100; ++ac)
  {
    /* initialize array */
    size_t i;
    for (i = 0; i < ITEM_COUNT; ++i)
    {
      iarray[i] = 1.f;
      oarray[i] = 42.f;
    }

    gettimeofday(&tms[0], NULL);
    prefix(iarray, oarray, ITEM_COUNT);
    gettimeofday(&tms[1], NULL);
    timersub(&tms[1], &tms[0], &tms[2]);
    const double diff = (double)(tms[2].tv_sec * 1000000 + tms[2].tv_usec);
    sum += diff;

    /* check */
    double sum = 0.f;
    for (i = 0; i < ITEM_COUNT; ++i)
    {
      sum += iarray[i];
      if (oarray[i] != sum)
      {
	printf("invalid @%lu %lf != %lf\n", i, oarray[i], sum);
	break ;
      }
    }
  }

  printf("done: %lf\n", sum / 1000);

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}
