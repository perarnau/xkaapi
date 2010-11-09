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
#include "kaapi.h"
#include "../common/conc_range.h"


typedef struct work
{
  conc_range_t cr;
  const double* array;
  double res;
} work_t;


typedef struct thief_work_t {
  const double* beg;
  const double* end;
  double res;
} thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


/* result reducer */
static int reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* victim work */
  work_t* const vw = (work_t*)varg;

  /* thief work */
  thief_work_t* const tw = (thief_work_t*)tdata;

  /* thief range continuation */
  conc_size_t beg, end;

  /* accumulate */
  vw->res += tw->res;

  /* retrieve the range */
  beg = (conc_size_t)(tw->beg - vw->array);
  end = (conc_size_t)(tw->end - vw->array);

  conc_range_set(&vw->cr, beg, end);

  return 0;
}


/* parallel work splitter */
static int splitter
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
{
  /* victim work */
  work_t* const vw = (work_t*)args;

  /* stolen range */
  conc_size_t i, j;
  conc_size_t range_size;

  /* reply count */
  int nrep = 0;

  /* size per request */
  conc_size_t unit_size;

 redo_steal:
  /* do not steal if range size <= PAR_GRAIN */
#define CONFIG_PAR_GRAIN 128
  range_size = conc_range_size(&vw->cr);
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
  if (!conc_range_pop_back(&vw->cr, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(thief_work_t), NULL);

    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), ktr );
    tw->beg = vw->array+j-unit_size;
    tw->end = vw->array+j;
    tw->res = 0.f;

    /* initialize ktr task may be preempted before entrypoint */
    ((thief_work_t*)ktr->data)->beg = tw->beg;
    ((thief_work_t*)ktr->data)->end = tw->end;
    ((thief_work_t*)ktr->data)->res = 0.f;

    /* reply head, preempt head */
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }

  return nrep;
}


/* seq work extractor */
static int extract_seq(work_t* w, const double** pos, const double** end)
{
  /* extract from range beginning */

  conc_size_t i, j;

#define CONFIG_SEQ_GRAIN 128
  if (conc_range_pop_front_max(&w->cr, &i, &j, CONFIG_SEQ_GRAIN) == 0)
    return -1; /* failure */

  *pos = w->array + i;
  *end = w->array + j;

  return 0; /* success */
}


/* entrypoint */
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* input work */
  thief_work_t* const work = (thief_work_t*)args;

  /* resulting work */
  thief_work_t* const res_work = kaapi_adaptive_result_data(sc);

  /* res += [work->beg, work->end[ */
  while (work->beg != work->end)
  {
    work->res += *work->beg;

    /* update prior reducing */
    ++work->beg;

    const unsigned int is_preempted = kaapi_preemptpoint
      (sc, NULL, NULL, (void*)work, sizeof(thief_work_t), NULL);
    if (is_preempted) return ;
  }

  /* we are finished, update results. */
  res_work->beg = work->beg;
  res_work->end = work->end;
  res_work->res = work->res;
}


/* algorithm main function */
static double accumulate(const double* array, size_t size)
{
  /* self thread, task */
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;

  /* sequential work */
  const double* pos, *end;

  /* initialize work */
  work_t work;
  conc_range_init(&work.cr, 0, (conc_size_t)size);
  work.array = array;
  work.res = 0.f;

  /* push an adaptive task. set the preemption flag. */
  sc = kaapi_task_begin_adaptive(
        thread, 
        KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION, 
        splitter, 
        &work     /* arg for splitter = work to split */
    );

  /* while there is sequential work to do */
 redo_work:
  while (extract_seq(&work, &pos, &end) != -1)
  {
    /* res += [pos, end[ */
    for (; pos != end; ++pos)
      work.res += *pos;
  }

  /* preempt and reduce thieves */
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief(sc, ktr, NULL, reducer, (void*)&work);
    goto redo_work;
  }

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);

  return work.res;
}


/* unit */

int main(int ac, char** av)
{
#define ITEM_COUNT 1000000
  static double array[ITEM_COUNT];

  /* initialize the runtime */
  kaapi_init();

  /* initialize array */
  for (size_t i = 0; i < ITEM_COUNT; ++i)
    array[i] = 2.f;

  for (ac = 0; ac < 100; ++ac)
  {
    double res = accumulate(array, ITEM_COUNT);
    if (res != (double)(2 * ITEM_COUNT))
    {
      printf("invalid: %lf != %lf\n", res, 2.f * ITEM_COUNT);
      break ;
    }
  }

  printf("done\n");

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}
