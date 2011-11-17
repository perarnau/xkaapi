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
#include "kaapi_impl.h"
#include "kaapic_impl.h"


#define CONFIG_FOREACH_STATS 0
#if CONFIG_FOREACH_STATS
static double foreach_time = 0;
#endif

/* counter to fix termination bug */
#define CONFIG_TERM_COUNTER 0

/* for implm requiring bound tids */
#define CONFIG_MAX_TID 0


/* missing prototypes */
extern void kaapi_lock_self_kproc(void);
extern void kaapi_unlock_self_kproc(void);
extern kaapi_atomic_t* kaapi_get_kproc_lock(kaapi_processor_id_t);


#define FATAL()						\
do {							\
  printf("fatal error @ %s::%d\n", __FILE__, __LINE__);	\
  exit(-1);						\
} while(0)


#if defined(KAAPI_DEBUG)
# define PRINTF(__s, ...) printf(__s, __VA_ARGS__)
#else
# define PRINTF(__s, ...)
#endif


#if CONFIG_MAX_TID
/* maximum thread number */
static volatile unsigned int xxx_max_tid;
#endif

/* parallel and sequential grains */
static volatile long xxx_par_grain;
static volatile long xxx_seq_grain;



/* work array. allow for random access. */

typedef struct work_array
{
  kaapi_bitmap_value_t map;
  long off;
  long scale;
} work_array_t;


static void work_array_init
(
  work_array_t* wa,
  long off,
  long scale,
  const kaapi_bitmap_value_t* map
)
{
  wa->off = off;
  wa->scale = scale;
  kaapi_bitmap_value_copy(&wa->map, map);
}


static void work_array_pop
(
  work_array_t* wa,
  long pos, 
  long* i, long* j
)
{
  /* pop the task at pos in [*i, *j[ */
  const long k = wa->off + pos * wa->scale;

  *i = k;
  *j = k + wa->scale;

  kaapi_bitmap_value_unset(&wa->map, (unsigned int)pos);
}


static inline unsigned int work_array_is_empty(const work_array_t* wa)
{
  return kaapi_bitmap_value_empty(&wa->map);
}


static inline unsigned int work_array_is_set(const work_array_t* wa, long pos)
{
  return kaapi_bitmap_value_get(&wa->map, (unsigned int)pos);
}


static inline long work_array_first(const work_array_t* wa)
{
  /* return the first available task position */
  /* assume there is at least one bit set */
  return kaapi_bitmap_first1(&wa->map);
}


/* work container */
typedef struct work_info
{
#if CONFIG_MAX_TID
  /* maximum thread index */
  unsigned int max_tid;
#endif

  /* grains */
  long par_grain;
  long seq_grain;

} work_info_t;

typedef struct work
{
  kaapi_workqueue_t cr;

#if CONFIG_TERM_COUNTER
  /* global work counter */
  kaapi_atomic_t* counter;
#endif

  /* split_root_task */
  work_array_t* wa;

  /* infos */
  const work_info_t* wi;

  /* work routine */
  kaapic_foreach_body_t body_f;
  void*                 body_args;
} work_t;

typedef work_t thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t* );


static int split_common(
  struct kaapi_task_t*                 victim_task,
  void*                                args,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri,
  unsigned int                         do_root_task
);


static int split_leaf_task
(
  struct kaapi_task_t*                 task,
  void*                                arg,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri
)
{
  return split_common(task, arg, lr, lri, 0);
}


/* main splitter */
static int split_root_task
(
  struct kaapi_task_t*                 task,
  void*                                arg,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri
)
{
  return split_common(task, arg, lr, lri, 1);
}


/* parallel work splitters */

static int split_common(
  struct kaapi_task_t*                 victim_task,
  void*                                args,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri,
  unsigned int                         do_root_task
)
{
  /* victim and thief works */
  work_t* const vw = (work_t*)args;
  thief_work_t* tw;

  /* stolen range */
  kaapi_workqueue_index_t i;
  kaapi_workqueue_index_t j = 0;
  kaapi_workqueue_index_t p;
  kaapi_workqueue_index_t q;
  kaapi_workqueue_index_t range_size;

  /* size per request */
  kaapi_workqueue_index_t unit_size = 0;

  /* max workqueue pop count failure */
  unsigned int trials = 2;

  /* requests served from work array */
  unsigned int root_count = 0;

  /* requests served from workqueue */
  unsigned int leaf_count = 0;

  /* root tasks stored in work array */
  work_array_t* wa = vw->wa;

  /* root tasks stored in work array */
  const work_info_t* const wi = vw->wi;

  int nreq = kaapi_listrequest_iterator_count(lri);

  /* skip root task */
  if ((do_root_task == 0) || work_array_is_empty(wa))
    goto skip_work_array;

  /* count requests that will be served by root tasks */
  kaapi_listrequest_iterator_t cpy_lri;
  /* make a copy of the iterator state */
  cpy_lri = *lri;
  for (; !kaapi_listrequest_iterator_empty(&cpy_lri); ) 
  {
    kaapi_request_t* req = kaapi_listrequest_iterator_get(lr, &cpy_lri);
    if (work_array_is_set(wa, (long)req->ident))
      ++root_count;
  }

skip_work_array:

  /* todo: request preprocessing should be done in one pass.
     this will be no longer necessary with new splitter interface
   */

#if CONFIG_MAX_TID
  /* exclude req->kid > max_tid from requests. note that a
     root request cannot be in the > max_tid request set
  */
  for (k = nreq - 1; k >= 0; --k)
  {
    if (req[k].kid > wi->max_tid)
      --nreq;
  }
#endif /* CONFIG_MAX_TID */

  if (root_count == nreq) 
    goto skip_workqueue;

redo_steal:

  /* do not steal if range size <= par_grain */
  range_size = kaapi_workqueue_size(&vw->cr);
  if (range_size <= wi->seq_grain)
  {
    leaf_count = 0;
    goto skip_workqueue;
  }

  leaf_count = nreq - root_count;

  /* how much per non root req */
  unit_size = range_size / (leaf_count + 1);
  if (unit_size == 0)
  {
    leaf_count = (range_size / wi->par_grain) - 1;
    unit_size = wi->par_grain;
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
  */
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, leaf_count * unit_size))
  {
    if ((trials--) == 0)
    {
      leaf_count = 0;
      goto skip_workqueue;
    }

    goto redo_steal;
  }

skip_workqueue:
  for ( /* void */; 
        !kaapi_listrequest_iterator_empty(lri) && (root_count || leaf_count); 
        kaapi_listrequest_iterator_next(lr, lri)
      )
  {
    kaapi_request_t* req = kaapi_listrequest_iterator_get(lr, lri);
    
#if CONFIG_MAX_TID
    if (req->kid > wi->max_tid)
      /* skip kid > max_tid requests */
      continue;
#endif /* CONFIG_MAX_TID */

    if (root_count && work_array_is_set(wa, (long)req->ident))
    {
      /* serve from work array */
      tw = kaapi_request_pushdata(req, sizeof(thief_work_t) );
      work_array_pop(wa, (long)req->ident, &p, &q);
      --root_count;
    }
    else if (leaf_count)
    {
      /* serve from the workqueue */
      tw = kaapi_request_pushdata(req, sizeof(thief_work_t) );

      /* stolen indices */
      p = j - unit_size;
      q = j;

      j -= unit_size;

      --leaf_count;
    }
    else
    {
      /* dont reply, neither root nor leaf */
      continue ;
    }

    /* finish work init and reply the request */
    kaapi_workqueue_init_with_lock
      (&tw->cr, p, q, &kaapi_all_kprocessors[req->ident]->lock);
    tw->body_f    = vw->body_f;
    tw->body_args = vw->body_args;
    tw->wi = wi;
#if CONFIG_TERM_COUNTER
    tw->counter = vw->counter;
#endif
    kaapi_task_init(kaapi_request_toptask(req), thief_entrypoint, tw);
    kaapi_reply_pushtask_adaptive_tail( req, victim_task, split_leaf_task );
    kaapi_request_committask(req);
  }

  return 0;
}


/* thief entrypoint */
static void thief_entrypoint(
  void*                 args, 
  kaapi_thread_t*       thread
)
{
  /* range to process */
  kaapi_workqueue_index_t i;
  kaapi_workqueue_index_t j;

  /* process the work */
  thief_work_t* thief_work = (thief_work_t*)args;

  /* work info */
  const work_info_t* const wi = thief_work->wi;

  /* retrieve tid */
  const int tid = kaapi_get_self_kid();

#if CONFIG_TERM_COUNTER
  unsigned long counter = 0;
#endif

  /* while there is sequential work to do */
  while (kaapi_workqueue_pop(&thief_work->cr, &i, &j, wi->seq_grain) ==0)
  {
    /* apply w->f on [i, j[ */
    thief_work->body_f((int)i, (int)j, (int)tid, thief_work->body_args);
#if CONFIG_TERM_COUNTER
    counter += j - i;
#endif
  }

#if CONFIG_TERM_COUNTER
  KAAPI_ATOMIC_SUB(thief_work->counter, counter);
#endif
}


/* exported foreach interface */
void kaapic_foreach_common
(
  int32_t               first, 
  int32_t               last,
  kaapic_foreach_body_t body_f,
  void*                 body_args
)
{
  /* is_format true if called from kaapif_foreach_with_format */

  const int tid = kaapi_get_self_kid();
  kaapi_thread_t* const thread = kaapi_threadcontext2thread(
    kaapi_all_kprocessors[tid]->thread
  );
  void* context;
  kaapi_frame_t frame;

  /* warning: interval includes j */
  kaapi_workqueue_index_t i = first;
  kaapi_workqueue_index_t j = last;

  /* work */
  work_t w;
  
  /* mapping */
  work_array_t wa;

#if CONFIG_MAX_TID
  /* concurrency cannot be more than (maxtid + 1) threads */
  unsigned long concurrency = (unsigned long)(xxx_max_tid + 1);
#else
  unsigned long concurrency = kaapi_getconcurrency();
#endif

  /* work info */
  work_info_t wi;

  /* work array, for reserved task */
  kaapi_workqueue_index_t range_size;
  long off;
  long pos;
  long scale;
  kaapi_bitmap_value_t map;

#if CONFIG_FOREACH_STATS
  const double time = kaapif_get_time_();
#endif

#if CONFIG_TERM_COUNTER
  /* termination counter */
  kaapi_atomic_t counter;
  unsigned long local_counter = 0;
#endif

  /* save frame */
  kaapi_thread_save_frame(thread, &frame);

  /* initialize work array */

  /* alignment constraint ? sizeof(void*) */

#if CONFIG_TERM_COUNTER
  /* initialize work counter before changing range_size */
  KAAPI_ATOMIC_WRITE(&counter, range_size);
  w.counter = &counter;
#endif

  /* map has one bit per core and excludes the master */
  range_size = j - i;

  /* handle concurrency too high case */
  if (range_size < concurrency) concurrency = 1;

  /* round range to be multiple of concurrency 
     master will get the biggest part
  */
  off = range_size % concurrency;
  range_size -= off;
  scale = range_size / concurrency;

  /* set all except the master */
  kaapi_bitmap_value_clear(&map);
  kaapi_bitmap_set_low_bits(&map, concurrency);
  kaapi_bitmap_value_unset(&map, 0);

  /* allocate and init work array */
  work_array_init(&wa, i + off, scale, &map);

  /* master has to adjust to include off. bypass first pop. */
  j = i + off + scale;

  /* initialize the workqueu */
  kaapi_workqueue_init(&w.cr, i, j);
  w.wa        = &wa;
  w.body_f    = body_f;
  w.body_args = body_args;

  /* capture and setup work info */
#if CONFIG_MAX_TID
  wi.max_tid = xxx_max_tid;
#endif
  wi.par_grain = xxx_par_grain;
  wi.seq_grain = xxx_seq_grain;
  w.wi = &wi;

  /* start adaptive region */
  context = kaapi_task_begin_adaptive(
     thread, 
     KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION,
     split_root_task,
     &w
  );

#if CONFIG_MAX_TID
  /* dont process if we are excluded from tid set */
  if (tid > wi.max_tid) goto end_adaptive;
#endif

  /* process locally */
continue_work:
  while (kaapi_workqueue_pop(&w.cr, &i, &j, wi.seq_grain) == 0)
  {
    /* apply w->f on [i, j[ */
    body_f((int)i, (int)j, (int)tid, body_args);

#if CONFIG_TERM_COUNTER
    local_counter += j - i;
#endif
  }

  if (!work_array_is_empty(&wa))
  {
    kaapi_atomic_lock( &kaapi_all_kprocessors[tid]->lock );

    /* refill the workqueue from reseved task and continue */
    pos = work_array_first(&wa);
    work_array_pop(&wa, pos, &i, &j);

    kaapi_workqueue_init
      (&w.cr, (kaapi_workqueue_index_t)i, (kaapi_workqueue_index_t)j);

    kaapi_atomic_unlock( &kaapi_all_kprocessors[tid]->lock );

    goto continue_work;
  }

end_adaptive:
#if CONFIG_TERM_COUNTER
  KAAPI_ATOMIC_SUB(&counter, local_counter);
#endif

  /* wait for thieves */
  kaapi_task_end_adaptive(context);

#if CONFIG_TERM_COUNTER
  /* wait for work counter */
  while (KAAPI_ATOMIC_READ(&counter)) ;
#endif

#if CONFIG_FOREACH_STATS
  foreach_time += kaapif_get_time_() - time;
#endif
}


