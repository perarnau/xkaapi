#include "kaapi.h"


typedef struct work
{
#if CONFIG_CONCURRENT_SPLIT
  volatile long lock;
#endif

  unsigned int a;
  volatile unsigned int b;

#if CONFIG_REDUCE_RESULT
  unsigned int res;
  kaapi_taskadaptive_result_t* ktr;
#endif

} work_t;


/* fwd decl */
static void entry(void*, kaapi_thread_t*);


#if CONFIG_CONCURRENT_SPLIT

/* locking */
static void lock_work(work_t* w)
{
  while (!__sync_bool_compare_and_swap(&w->lock, 0, 1))
    ;
}

static void unlock_work(work_t* w)
{
  __sync_fetch_and_and(&w->lock, 0);
}

#endif /* CONFIG_CONCURRENT_SPLIT */


#if CONFIG_REDUCE_RESULT
static int reduce
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* victim work */
  work_t* const vw = (work_t*)varg;

  /* thief work */
  work_t* const tw = (work_t*)tdata;

  vw->res += tw->res;
  vw->b = tw->b;

  return 0;
}
#endif


/* splitter */
static int split
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
{
  /* victim work */
  work_t* const vw = (work_t*)args;

  /* reply count */
  int nrep = 0;

  /* size per request */
  unsigned int unit_size;

  /* extract work */
#if CONFIG_CONCURRENT_SPLIT
  /* concurrent with victim */
  lock_work(vw);
#endif

  /* how much per req */
#define CONFIG_PAR_GRAIN 128
  unit_size = 0;
  if (vw->b > CONFIG_PAR_GRAIN)
  {
    unit_size = vw->b / (nreq + 1);
    if (unit_size == 0)
    {
      nreq = (vw->b / CONFIG_PAR_GRAIN) - 1;
      unit_size = CONFIG_PAR_GRAIN;
    }

    vw->b -= unit_size * nreq;
  }

#if CONFIG_CONCURRENT_SPLIT
  unlock_work(vw);
#endif

  if (unit_size == 0)
    return 0;

  for (; nreq; --nreq, ++req, ++nrep)
  {
    /* thief work */
    work_t* const tw = kaapi_reply_pushtask(sc, req, entry);

#if CONFIG_CONCURRENT_SPLIT
    tw->lock = 0;
#endif

    tw->a = vw->a;
    tw->b = unit_size;

#if CONFIG_REDUCE_RESULT
    /* allocate and clear thief result space */
    tw->ktr = kaapi_allocate_thief_result(sc, sizeof(work_t), NULL);
    ((work_t*)tw->ktr->data)->b = 0;
    ((work_t*)tw->ktr->data)->res = 0;

    tw->res = 0;

    kaapi_request_reply_head(sc, req, tw->ktr);
#else
    kaapi_request_reply_head(sc, req, NULL);
#endif

  }

  return nrep;
}


static unsigned int extract_seq(work_t* w)
{
#define CONFIG_SEQ_GRAIN 10
  unsigned int b = CONFIG_SEQ_GRAIN;

#if CONFIG_CONCURRENT_SPLIT
  lock_work(w);
#endif

  if (b > w->b)
    b = w->b;
  w->b -= b;

#if CONFIG_CONCURRENT_SPLIT
  unlock_work(w);
#endif

  return b;
}


/* entrypoint */
static void entry
(void* args, kaapi_thread_t* thread)
{
  /* adaptive stealcontext flags */
  int flags = 0;

#if CONFIG_CONCURRENT_SPLIT
  flags |= KAAPI_SC_CONCURRENT;
#else
  flags |= KAAPI_SC_COOPERATIVE;
#endif

#if CONFIG_REDUCE_RESULT
  flags |= KAAPI_SC_PREEMPTION;
#else
  flags |= KAAPI_SC_NOPREEMPTION;
#endif

  /* push an adaptive task */
  kaapi_stealcontext_t* const sc =
    kaapi_task_begin_adaptive(thread, flags, split, args);

  /* process the work */
  work_t* const w = (work_t*)args;

#if CONFIG_REDUCE_RESULT
 redo_work:
#endif
  while (1)
  {
    /* extract sequential work */
    const unsigned int b = extract_seq(w);

    /* no more work */
    if (b == 0) break;

#if (CONFIG_CONCURRENT_SPLIT == 0) /* cooperative */
    kaapi_stealpoint(sc, split, (void*)w);
#endif

    /* process the seq work */
    const unsigned int res = w->a * b;
    __asm__ __volatile__ (""::"m"(res));

#if CONFIG_REDUCE_RESULT
    w->res += res;

    /* look for preemption */
    if (w->ktr != NULL)
    {
      const unsigned int is_preempted = kaapi_preemptpoint
	(w->ktr, sc, NULL, NULL, (void*)w, sizeof(work_t), NULL);
      if (is_preempted)
	return ;
    }
#endif
  }

#if CONFIG_REDUCE_RESULT
  /* preempt and reduce thieves */
  kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
  if (ktr != NULL)
  {
    kaapi_preempt_thief(sc, ktr, NULL, reduce, (void*)w);
    goto redo_work;
  }
#endif

#if CONFIG_REDUCE_RESULT
  if (w->ktr != NULL)
  {
    /* not preempted, update ktr */
    work_t* const res_work = (work_t*)w->ktr->data;
    res_work->b = 0;
    res_work->res = w->res;
  }
#endif

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);
}


static unsigned int adaptive_mul
(unsigned int a, unsigned int b)
{
  /* self thread, task */
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_task_t* const task = kaapi_thread_toptask(thread);

  kaapi_frame_t frame;
  kaapi_thread_save_frame(thread, &frame);

  /* work */
  work_t* const w = kaapi_alloca_align(64, sizeof(work_t));

  /* initialize work */
#if CONFIG_CONCURRENT_SPLIT
  w->lock = 0;
#endif
  w->a = a;
  w->b = b;
#if CONFIG_REDUCE_RESULT
  w->res = 0;
  w->ktr = NULL;
#endif

  /* fork root task */
  kaapi_task_init(task, entry, (void*)w);
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();

  kaapi_thread_restore_frame(thread, &frame);

#if CONFIG_REDUCE_RESULT
  return w->res;
#else
  return 0;
#endif
}


int main(int ac, char** av)
{
  /* initialize the runtime */
  kaapi_init();

  for (ac = 0; ac < 10000; ++ac)
  {
#define A 1
#define B 100000
#if CONFIG_REDUCE_RESULT
    const unsigned int res = adaptive_mul(A, B);
    if (res != (A * B))
    {
      printf("error: %d %u != %u\n", ac, res, A * B);
      break ;
    }
#else
    adaptive_mul(A, B);
#endif
  }

  printf("done\n");

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}
