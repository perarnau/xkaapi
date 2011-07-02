#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "modp.h"
#include "kaapi.h"


/* this example implements arbitrary degree
   polynom evaluation in a given point. it
   uses the horner scheme and a prefix algorithm
   for parallelization. sequential codes are
   provided for testing purposes.
 */


/* degree to index translation routines
 */

inline static unsigned long to_index
(unsigned long n, unsigned long polynom_degree)
{ return polynom_degree - n; }

inline static unsigned long to_degree
(unsigned long i, unsigned long polynom_degree)
{ return polynom_degree - i; }


/* xkaapi adaptive horner
 */

typedef struct horner_work
{
  /* the range to process. since thieves
     are not made stealable, they dont
     need a workqueue.
   */
  union
  {
    kaapi_workqueue_t wq;

    struct
    {
      unsigned long i, j;
    } pair;

  } range;

  /* polynom */
  const unsigned long* a;
  unsigned long n;

  /* the point to evaluate */
  unsigned long x;

  /* the result */
  unsigned long res;

} horner_work_t;

typedef struct horner_res
{
  /* processed range */
  unsigned long i, j;

  /* computed result */
  unsigned long res;

  /* needed to avoid reducing twice */
  unsigned long is_reduced;

} horner_res_t;


/* forward declarations.
 */

static void thief_entrypoint
(void*, kaapi_thread_t*, kaapi_stealcontext_t*);

static unsigned long horner_seq_hilo
(unsigned long, const unsigned long*, unsigned long,
 unsigned long, unsigned long, unsigned long);


/* reduction.
   the runtime may execute it either on the victim or thief. it
   depends if the victim has finished or is still being executed
   at the time of preemption. this is why the is_reduced boolean
   is needed.
 */

static void common_reducer(horner_work_t* vw, horner_res_t* tw)
{
  /* how much has been processed by the thief */
  const unsigned long n = tw->i - (unsigned long)vw->range.wq.end;

  /* vw->res = tw->res + vw->res * x^n; */
  vw->res = axnb_modp(vw->res, vw->x, n, tw->res);

  /* continue the thief work */
  kaapi_workqueue_set(&vw->range.wq, tw->i, tw->j);
}

static int thief_reducer
(kaapi_taskadaptive_result_t* ktr, void* varg, void* targ)
{
  /* called from the thief upon victim preemption request */

  common_reducer(varg, (horner_res_t*)ktr->data);

  /* inform the victim we did the reduction */
  ((horner_res_t*)ktr->data)->is_reduced = 1;

  return 0;
}

static int victim_reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* called from the victim to reduce a thief result */

  if (((horner_res_t*)tdata)->is_reduced == 0)
  {
    /* not already reduced */
    common_reducer(varg, tdata);
  }

  return 0;
}


/* parallel work splitter
 */

static int splitter
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
{
  /* victim work */
  horner_work_t* const vw = (horner_work_t*)args;

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
  range_size = kaapi_workqueue_size(&vw->range.wq);
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
  if (kaapi_workqueue_steal(&vw->range.wq, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(horner_res_t), NULL);

    horner_work_t* const tw = kaapi_reply_init_adaptive_task
      (sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(horner_work_t), ktr);

    /* initialize the thief work */
    tw->a = vw->a;
    tw->n = vw->n;
    tw->x = vw->x;
    tw->range.pair.i = j - unit_size;
    tw->range.pair.j = j;

    /* initialize ktr task may be preempted before entrypoint */
    ((horner_res_t*)ktr->data)->i = tw->range.pair.i;
    ((horner_res_t*)ktr->data)->j = tw->range.pair.j;
    ((horner_res_t*)ktr->data)->res = 0;
    ((horner_res_t*)ktr->data)->is_reduced = 0;
    
    /* reply head, preempt head */
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }

  return nrep;
}


/* extract sequential work
 */

static inline int extract_seq
(horner_work_t* w, unsigned long* i, unsigned long* j)
{
  /* sequential size to extract */
  static const unsigned long seq_size = 128;
  const int err = kaapi_workqueue_pop
    (&w->range.wq, (long*)i, (long*)j, seq_size);
  return err ? -1 : 0;
}

/* thief task entrypoint
 */

static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* input work */
  horner_work_t* const work = (horner_work_t*)args;

  /* resulting work */
  horner_res_t* const res = kaapi_adaptive_result_data(sc);

  unsigned long hi = to_degree(work->range.pair.i, work->n);
  const unsigned long lo = to_degree(work->range.pair.j, work->n);

  for (; hi > lo; --hi)
  {
    const unsigned long i = to_index(hi - 1, work->n);

    res->res = axb_modp(res->res, work->x, work->a[i]);

    /* update work indices */
    res->i = i;

    const unsigned int is_preempted = kaapi_preemptpoint
      (sc, thief_reducer, NULL, NULL, 0, NULL);
    if (is_preempted) return ;
  }
}


/* parallel horner
 */

static unsigned long horner_par
(unsigned long x, const unsigned long* a, unsigned long n)
{
  /* stealcontext flags */
  static const unsigned long sc_flags =
    KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION;

  /* self thread, task */
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;

  /* sequential indices */
  unsigned long i, j;

  horner_work_t work;

  /* initialize horner work */
  work.x = x;
  work.a = a;
  work.n = n;
  work.res = a[to_index(n, n)];
  kaapi_workqueue_init(&work.range.wq, 0, n);

  /* enter adaptive section */
  sc = kaapi_task_begin_adaptive(thread, sc_flags, splitter, &work);

 continue_work:
  while (extract_seq(&work, &i, &j) != -1)
  {
    const unsigned long hi = to_degree(i, n);
    const unsigned long lo = to_degree(j, n);
    work.res = horner_seq_hilo(x, a, n, work.res, hi, lo);
  }

  /* preempt and reduce thieves */
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief(sc, ktr, (void*)&work, victim_reducer, (void*)&work);
    goto continue_work;
  }

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);

  return work.res;
}


/* sequential horner
 */

static unsigned long horner_seq_hilo
(
 unsigned long x, const unsigned long* a, unsigned long n,
 unsigned long res, unsigned long hi, unsigned long lo
)
{
  /* the degree in [hi, lo[ being evaluated */
  unsigned long i;

  for (i = hi; i > lo; --i)
    res = axb_modp(res, x, a[to_index(i - 1, n)]);
  return res;
}

static inline unsigned long horner_seq
(unsigned long x, const unsigned long* a, unsigned long n)
{
  unsigned long res = a[to_index(n, n)];
  return horner_seq_hilo(x, a, n, res, n, 0);
}


/* sequential naive implementation
 */

__attribute__((unused))
static unsigned long naive_seq
(unsigned long x, const unsigned long* a, unsigned long n)
{
  unsigned long res;
  unsigned long i;

  for (res = 0, i = 0; i <= n; ++i)
    res = axnb_modp(a[to_index(i, n)], x, i, res);

  return res;
}


/* generate a random polynom of degree n
 */

static unsigned long* make_rand_polynom(unsigned long n)
{
  unsigned long* const a = malloc((n + 1) * sizeof(unsigned long));
  size_t i;
  for (i = 0; i <= n; ++i) a[i] = modp(rand());
  return a;
}


/* main
 */

int main(int ac, char** av)
{
  /* polynom */
  static const unsigned long n = 1024 * 1024;
  unsigned long* const a = make_rand_polynom(n);

  /* the point to evaluate */
  static const unsigned long x = 2;

  /* testing */
  volatile unsigned long sum;

  /* timing */
  uint64_t start, stop;
  double seq_time, par_time;
  size_t iter;

  kaapi_init(1, &ac, &av);

  start = kaapi_get_elapsedns();
  for (sum = 0, iter = 0; iter < 100; ++iter)
    sum += horner_par(x, a, n);
  stop = kaapi_get_elapsedns();
  par_time = (double)(stop - start) / (1E6 * 100.);

  start = kaapi_get_elapsedns();
  for (sum = 0, iter = 0; iter < 100; ++iter)
    sum += horner_seq(x, a, n);
  stop = kaapi_get_elapsedns();
  seq_time = (double)(stop - start) / (1E6 * 100.);

  printf("%lf, %lf\n", seq_time, par_time);

  kaapi_finalize();

  free(a);

  return 0;
}
