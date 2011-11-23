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

#warning "TODO: based dependencies on kaapi.h only"
#include "kaapic_impl.h"

//#define BIG_DEBUG_MACOSX 1

#define CONFIG_FOREACH_STATS 0
#if CONFIG_FOREACH_STATS
static double foreach_time = 0;
#endif

/* counter to fix termination bug */
#define CONFIG_TERM_COUNTER 0

/* for implm requiring bound tids */
#define CONFIG_MAX_TID 0


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
  kaapi_assert_debug( pos >= 0 );
  kaapi_bitmap_value_unset(&wa->map, (unsigned int)pos);
}


static inline unsigned int work_array_is_empty(const work_array_t* wa)
{
  return kaapi_bitmap_value_empty(&wa->map);
}


static inline unsigned int work_array_is_set(const work_array_t* wa, long pos)
{
  kaapi_assert_debug( pos >= 0 );
  return kaapi_bitmap_value_get(&wa->map, (unsigned int)pos);
}


static inline long work_array_first(const work_array_t* wa)
{
  /* return the first available task position */
  /* assume there is at least one bit set */
  return kaapi_bitmap_first1(&wa->map) -1;
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
  kaapi_workqueue_t cr __attribute__((aligned(64)));

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
  kaapic_body_arg_t*    body_args;
} kaapic_work_t;

typedef kaapic_work_t kaapic_thief_work_t;


/* fwd decl */
static void _kaapic_thief_entrypoint(void*, kaapi_thread_t*,  kaapi_task_t* );


static int _kaapic_split_common(
  struct kaapi_task_t*                 victim_task,
  void*                                args,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri,
  unsigned int                         do_root_task
);


static int _kaapic_split_leaf_task
(
  struct kaapi_task_t*                 task,
  void*                                arg,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri
)
{
  return _kaapic_split_common(task, arg, lr, lri, 0);
}


/* main splitter */
static int _kaapic_split_root_task
(
  struct kaapi_task_t*                 task,
  void*                                arg,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri
)
{
  return _kaapic_split_common(task, arg, lr, lri, 1);
}


/* parallel work splitters */

static int _kaapic_split_common(
  struct kaapi_task_t*                 victim_task,
  void*                                args,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri,
  unsigned int                         do_root_task
)
{
  /* victim and thief works */
  kaapic_work_t* const vw = (kaapic_work_t*)args;
  kaapic_thief_work_t* tw;

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
    kaapi_listrequest_iterator_next(lr, &cpy_lri);
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
  if (unit_size < wi->par_grain)
  {
    leaf_count = (range_size / wi->par_grain) - 1;
    unit_size = wi->par_grain;
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
  */
  kaapi_assert_debug( vw->cr.lock 
    == &kaapi_get_current_processor()->victim_kproc->lock );
  kaapi_assert_debug( kaapi_atomic_assertlocked(vw->cr.lock) );
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, leaf_count * unit_size))
  {
    if ((trials--) == 0)
    {
      leaf_count = 0;
      goto skip_workqueue;
    }

    goto redo_steal;
  }
#if defined(BIG_DEBUG_MACOSX)
    kaapi_workqueue_index_t beg,end;
    beg = kaapi_workqueue_range_begin(&vw->cr);
    end = kaapi_workqueue_range_end(&vw->cr);
    printf("%i:: WS/#%i Steal @%p=[%i,%i[ remainder [%i,%i[ usz:%i rsz:%i\n",//"|lock=%i\n", 
      (int)kaapi_get_current_processor()->victim_kproc->kid,
      (int)leaf_count, 
      (void*)&vw->cr,
      (int)i, (int)j,
      (int)beg, (int)end,
      (int)unit_size,
      (int)range_size
    );
//      (int)KAAPI_ATOMIC_READ(vw->cr.lock)
    fflush(stdout);
#endif


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
      tw = kaapi_request_pushdata(req, sizeof(kaapic_thief_work_t) );
      work_array_pop(wa, (long)req->ident, &p, &q);
      kaapi_assert_debug( q-p > 0 );
      --root_count;
    }
    else if (leaf_count)
    {
      /* serve from the workqueue */
      tw = kaapi_request_pushdata(req, sizeof(kaapic_thief_work_t) );

      /* stolen indices */
      p = j - unit_size;
      q = j;

      j -= unit_size;

      kaapi_assert_debug( unit_size >= 1 );
      kaapi_assert_debug( q-p > 0 );
      --leaf_count;
    }
    else
    {
      /* dont reply, neither root nor leaf */
      continue ;
    }

    /* finish work init and reply the request */
#if defined(BIG_DEBUG_MACOSX)
    printf("%i:: WS/ Steal %i @%p[%i,%i[\n", 
      (int)req->ident, 
      (int)kaapi_all_kprocessors[req->ident]->victim_kproc->kid,
      (void*)&tw->cr,
      (int)p, (int)q);
    fflush(stdout);
#endif
    kaapi_workqueue_init_with_lock
      (&tw->cr, p, q, &kaapi_all_kprocessors[req->ident]->lock);
    tw->body_f    = vw->body_f;
    tw->body_args = vw->body_args;
    tw->wi = wi;
#if CONFIG_TERM_COUNTER
    tw->counter = vw->counter;
#endif
    kaapi_task_init_with_flag(
      kaapi_request_toptask(req), 
      (kaapi_task_body_t)_kaapic_thief_entrypoint, 
      tw,
      KAAPI_TASK_UNSTEALABLE
    );
    kaapi_reply_pushtask_adaptive_tail( 
      req, 
      victim_task,
      _kaapic_split_leaf_task 
    );
    kaapi_request_committask(req);
  }

  return 0;
}

#if defined(BIG_DEBUG_MACOSX)
static volatile int version = 0;
static volatile int arraytid[1+1000 * 48];
static volatile int arraytidmyself[4][1+1000 * 48];
#endif

/* thief entrypoint */
static void _kaapic_thief_entrypoint(
  void*                 args, 
  kaapi_thread_t*       thread,
  kaapi_task_t*         pc
)
{
  /* range to process */
  kaapi_workqueue_index_t i;
  kaapi_workqueue_index_t j;

  /* process the work */
  kaapic_thief_work_t* thief_work = (kaapic_thief_work_t*)args;

  /* work info */
  const work_info_t* const wi = thief_work->wi;

  /* retrieve tid */
  const int tid = kaapi_get_self_kid();

#if CONFIG_TERM_COUNTER
  unsigned long counter = 0;
#endif

#if defined(KAAPI_DEBUG)
  kaapi_processor_t* kproc = kaapi_get_current_processor();
#endif
  kaapi_assert_debug( &kproc->lock == thief_work->cr.lock );
  kaapi_assert_debug( kproc->kid == tid );

  /* while there is sequential work to do */
#if defined(BIG_DEBUG_MACOSX)
  kaapi_workqueue_index_t first_i = -1;
  kaapi_workqueue_index_t last_j = -1;
  kaapi_workqueue_t savewq = thief_work->cr;
  kaapi_workqueue_t beforewq = savewq;
#endif

  while (kaapi_workqueue_pop(&thief_work->cr, &i, &j, wi->seq_grain) ==0)
  {
#if defined(BIG_DEBUG_MACOSX)
    printf("%i:: WS/Pop @%p[%i,%i[\n", 
      tid,
      (void*)&thief_work->cr,
      (int)i, (int)j);
    fflush(stdout);
#endif
    kaapi_assert_debug( &kproc->lock == thief_work->cr.lock );
#if defined(BIG_DEBUG_MACOSX)
    if (first_i == -1) first_i = i;
    last_j = j;
    beforewq = thief_work->cr;
#endif
    kaapi_assert_debug( &kaapi_get_current_processor()->lock == thief_work->cr.lock );
    kaapi_assert_debug( i < j );
//    printf("%i:: WS/S_pop [%i,%i[\n", kaapi_get_self_kid(), (int)i, (int)j);

#if defined(BIG_DEBUG_MACOSX)
    /* shift -1 to match fortran definition... */
    for (int k = i; k<j; ++k)
    {
      if (arraytid[k] !=0) 
        kaapi_abort();
      arraytid[k] = tid;
      arraytidmyself[tid][k] = version;
    }
#endif
    /* apply w->f on [i, j[ */
    thief_work->body_f((int)i, (int)j, (int)tid, thief_work->body_args);
//    savewq = thief_work->cr;
#if CONFIG_TERM_COUNTER
    counter += j - i;
#endif
  }
#if 0//defined(BIG_DEBUG_MACOSX)
  if (last_j != -1)
    printf("%i:: WS/S_pop [%i,%i[\n", kaapi_get_self_kid(), (int)first_i, (int)last_j);
  else
    printf("%i:: WS/S_pop [%i,%i[\n", kaapi_get_self_kid(), (int)first_i, (int)last_j);
  fflush(stdout);
#endif

#if CONFIG_TERM_COUNTER
  KAAPI_ATOMIC_SUB(thief_work->counter, counter);
#endif
}


/* exported foreach interface */
int kaapic_foreach_common
(
  int32_t                first, 
  int32_t                last,
  kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_t  body_f,
  kaapic_body_arg_t*     body_args
)
{
#if defined(BIG_DEBUG_MACOSX)
  ++version;
  for (int k = 0; k<1+1000*48; ++k)
  {
    arraytid[k] = 0;
    arraytidmyself[0][k] = 0;
    arraytidmyself[1][k] = 0;
    arraytidmyself[2][k] = 0;
    arraytidmyself[3][k] = 0;
  }
//  kaapi_mem_barrier();
//  usleep(10000);
#endif

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

  /* master work */
  kaapic_work_t w;
  
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

  /* initialize the workqueue */
  kaapi_workqueue_init_with_lock(
    &w.cr, 
    i, j,
    &kaapi_all_kprocessors[tid]->lock
  );
  w.wa        = &wa;
  w.body_f    = body_f;
  w.body_args = body_args;

  /* capture and setup work info */
#if CONFIG_MAX_TID
  wi.max_tid = xxx_max_tid;
#endif
  wi.par_grain = (attr ==0 ? 1 : attr->p_grain);
  wi.seq_grain = (attr ==0 ? 1 : attr->s_grain);
  w.wi = &wi;

  /* start adaptive region */
  context = kaapi_task_begin_adaptive(
     thread, 
     KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION,
     _kaapic_split_root_task,
     &w
  );

#if CONFIG_MAX_TID
  /* dont process if we are excluded from tid set */
  if (tid > wi.max_tid) goto end_adaptive;
#endif

  /* process locally */
#if defined(KAAPI_DEBUG)
  kaapi_processor_t* kproc = kaapi_get_current_processor();
#endif
  kaapi_assert_debug( kproc->kid == tid );

#if defined(BIG_DEBUG_MACOSX)
  kaapi_workqueue_t savewq;
  kaapi_workqueue_t beforewq;
  kaapi_workqueue_index_t last_refill_i;
  kaapi_workqueue_index_t last_refill_j;
#endif

continue_work:
#if defined(BIG_DEBUG_MACOSX)
  savewq   = w.cr;
  beforewq = savewq;
  last_refill_i  = -1;
  last_refill_j  = -1;
#endif

  kaapi_assert_debug( &kproc->lock == w.cr.lock );
  while (kaapi_workqueue_pop(&w.cr, &i, &j, wi.seq_grain) == 0)
  {
#if defined(BIG_DEBUG_MACOSX)
    beforewq = savewq;
#endif

    kaapi_assert_debug( &kproc->lock == w.cr.lock );
#if 0//defined(BIG_DEBUG_MACOSX)
    printf("WS/M_pop [%i,%i[\n", (int)i, (int) j);
    fflush(stdout);
#endif
    /* apply w->f on [i, j[ */
    body_f((int)i, (int)j, (int)tid, body_args);

#if CONFIG_TERM_COUNTER
    local_counter += j - i;
#endif
  }
  kaapi_assert_debug( kaapi_workqueue_isempty(&w.cr) );

  _kaapi_workqueue_lock( &w.cr );
  if (work_array_is_empty(&wa))
  {
    _kaapi_workqueue_unlock( &w.cr );
    goto end_adaptive;
  }

  /* refill the workqueue from reseved task and continue */
  pos = work_array_first(&wa);
  kaapi_assert_debug( pos >0 );
  work_array_pop(&wa, pos, &i, &j);

#if defined(BIG_DEBUG_MACOSX)
  last_refill_i  = i;
  last_refill_j  = j;
#endif
  kaapi_workqueue_init(
    &w.cr, 
    (kaapi_workqueue_index_t)i, (kaapi_workqueue_index_t)j
  );
  _kaapi_workqueue_unlock( &w.cr );

  goto continue_work;

end_adaptive:
#if CONFIG_TERM_COUNTER
  KAAPI_ATOMIC_SUB(&counter, local_counter);
#endif

  /* wait for thieves */
  kaapi_task_end_adaptive(context);

  /* restore frame */
  kaapi_thread_context_t* const self_thread = kaapi_all_kprocessors[tid]->thread;
  kaapi_sched_lock( &self_thread->stack.lock );
  kaapi_thread_restore_frame(thread, &frame);
  kaapi_sched_unlock( &self_thread->stack.lock );
#if CONFIG_TERM_COUNTER
  /* wait for work counter */
  while (KAAPI_ATOMIC_READ(&counter)) ;
#endif

#if CONFIG_FOREACH_STATS
  foreach_time += kaapif_get_time_() - time;
#endif
  
  return 0;
}


