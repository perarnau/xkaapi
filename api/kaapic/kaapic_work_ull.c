/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
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
 
/* Note on the implementation.
   This file must be automatically generated from kaapi_work.c
   T.G.
*/
 
#include "kaapi_impl.h"
#include "kaapic_impl.h"


/* set to 0 to disable workload */
#define CONFIG_USE_WORKLOAD 1

#if CONFIG_USE_WORKLOAD
extern void kaapi_set_self_workload(unsigned long);
#define KAAPI_SET_SELF_WORKLOAD(__w)		\
do {						\
  kaapi_set_self_workload(__w);			\
} while (0)
#else 
#define KAAPI_SET_SELF_WORKLOAD(__w)
#endif


#define CONFIG_FOREACH_STATS 0
#if CONFIG_FOREACH_STATS
static double foreach_time = 0;
#endif


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


static int kaapic_foreach_globalwork_next_ull(
  kaapic_local_work_t*         lwork,
  kaapi_workqueue_index_ull_t* first,
  kaapi_workqueue_index_ull_t* last
);


/* return !=0 iff successful steal op
 */
static int kaapic_local_work_steal_ull
(
  kaapic_local_work_t*         lwork,
  kaapi_processor_t*           kproc, 
  kaapi_workqueue_index_ull_t* i, 
  kaapi_workqueue_index_ull_t* j,
  kaapi_workqueue_index_ull_t  unit_size
)
{
  int retval;
  _kaapi_workqueue_lock(&lwork->cr);
  retval = kaapi_workqueue_steal_ull(&lwork->cr, i, j, unit_size) ==0;
  _kaapi_workqueue_unlock(&lwork->cr);
  kaapi_assert_debug( (retval==0) || (*i < *j) );
  return retval;
}


/*
*/
static int kaapic_global_getwork_ull
(
  kaapic_global_work_t*        gw,
  kaapi_processor_id_t         tid, 
  kaapi_workqueue_index_ull_t* i, 
  kaapi_workqueue_index_ull_t* j
)
{
  kaapi_assert_debug(tid<KAAPI_MAX_PROCESSOR);

  /* Here, because work may have been finished  
  */
  if (KAAPI_ATOMIC_READ(&gw->workremain) ==0)
  {
    *i = *j = 0;
    return 0;
  }

  int pos = gw->wa.tid2pos[tid];
  if (pos == (uint8_t)-1) 
  {
    *i = *j = 0;
    return 0;
  }
  kaapi_assert_debug( (pos > 0) && (pos<kaapi_getconcurrency()) );
  
  *i = gw->wa.startindex_ull[pos];
  *j = gw->wa.startindex_ull[pos+1];
  return 1;
}


/*
*/
int kaapic_global_work_pop_ull
(
  kaapic_global_work_t* gw,
  kaapi_processor_id_t tid, 
  kaapi_workqueue_index_ull_t* i, 
  kaapi_workqueue_index_ull_t* j
)
{
  int retval;
  int pos;
  int idx;

  kaapi_assert_debug(tid<KAAPI_MAX_PROCESSOR);

  pos = gw->wa.tid2pos[tid];
  if (pos == (uint8_t)-1) 
  {
repop_any:
    do {
      /* no reserved slice: steal one */
      idx = kaapi_bitmap_first1_and_zero(&gw->wa.map);
      if (idx !=0)
      {
        pos = idx-1;
        *i = gw->wa.startindex_ull[pos];
        *j = gw->wa.startindex_ull[pos+1];
        return 1;    
      }
    } while (!kaapi_bitmap_empty(&gw->wa.map));
    *i = *j = 0;
    return 0;
  }
  kaapi_assert_debug( pos >= 0 );
  kaapi_assert_debug( pos<kaapi_getconcurrency() );
  
  retval = (0 == kaapi_bitmap_unset(&gw->wa.map, pos));
  if (retval ==0)
  {
    pos = -1;
    goto repop_any;
  }

  *i = gw->wa.startindex_ull[pos];
  *j = gw->wa.startindex_ull[pos+1];
  
  /* Here, because work may have been finished  
  */
  if (KAAPI_ATOMIC_READ(&gw->workremain) ==0)
  {
    *i = *j = 0;
    return 0;
  }

  return retval;
}


/* return !=0 iff successful steal op */
static int kaapic_global_work_steal_ull
(
  kaapic_global_work_t*        gwork,
  kaapi_processor_t*           kproc, 
  kaapi_workqueue_index_ull_t* i, 
  kaapi_workqueue_index_ull_t* j
)
{
  kaapi_workqueue_index_ull_t unit_size;
  kaapi_workqueue_index_ull_t range_size;

#if defined(KAAPI_USE_PERFCOUNTER)
  ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQ);
#endif

  /* here to do / to test:
     - any try to pop a slice closed to the tid of the thread
     - only 0 can pop a non poped slice
  */
#if 1
  /* caller has already pop and finish its slice, if it is 0 then may pop
     the next non null entry
  */
#if 0
  int tid = kproc->kid;
  if (tid == 0)
  {
    kaapi_assert_debug(tid<KAAPI_MAX_PROCESSOR);
#else
  if (1)
  {
#endif
    int tidpos = kaapi_bitmap_first1( &gwork->wa.map );
    if ((tidpos !=0) && kaapic_global_work_pop_ull(gwork, tidpos-1, i, j ))
    {
#if defined(KAAPI_USE_PERFCOUNTER)
      ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
      kaapi_assert_debug( *i < *j );

      /* success */
      return 1;
    }
  }
#endif
    
  /* try to steal from a random selected local workqueue */
  /* the code here is closed to the emitsteal code except we bypass the request mechanism */
  kaapi_victim_t  victim;
  int             err;

redo_select:
  /* warning here the method does not return except: work is found or total work is done
  */
  if (KAAPI_ATOMIC_READ(&gwork->workremain) ==0) 
    return 0;
  kaapi_slowdown_cpu();

  /* select the victim processor */
  err = (*kproc->fnc_select)( kproc, &victim, KAAPI_SELECT_VICTIM );
  //err = kaapi_sched_select_victim_hwsn( kproc, &victim, KAAPI_SELECT_VICTIM );
  if (unlikely(err !=0)) 
    goto redo_select;

  /* never pass by this function for a processor to steal itself */
  if (kproc == victim.kproc) 
    goto redo_select;
    
  /* do not steal if range size <= par_grain */
  kaapic_local_work_t* lwork = &gwork->lwork[victim.kproc->kid];

  /* because object lwork may exist without beging initialized, return if not */
  if (lwork->init == 0)
    goto redo_select;

  /* try to steal the local work */
  const kaapic_work_info_t* const wi = &gwork->wi;

  range_size = kaapi_workqueue_size(&lwork->cr);
  if (range_size <= wi->rep.ull.par_grain)
    goto redo_select;

  unit_size = range_size / 2;

  if (kaapic_local_work_steal_ull(
            lwork,
            kproc, 
            i,
            j,
            unit_size
    ))
  {
#if defined(KAAPI_USE_PERFCOUNTER)
    ++KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_STEALREQOK);
#endif
    return 1;
  }

  goto redo_select;
}


/* fwd decl */
static void _kaapic_thief_entrypoint_ull(void*, kaapi_thread_t*,  kaapi_task_t* );


/* Used to start parallel region if required */
extern unsigned int kaapic_do_parallel;


/* Initial work distribution wa.
   - self_tid is the thread that initial the globalwork
   - cpuset is the mask of kid on which work will be distributed
   - concurrency is the concurrency level
   - [first,last) the slice to distribute
*/
static void _kaapic_foreach_initwa_ull(
  kaapic_work_distribution_t* wa,
  int                         self_tid,
  const kaapi_bitmap_value_t* cpuset,
  int                         nthreads,
  kaapi_workqueue_index_ull_t first, 
  kaapi_workqueue_index_ull_t last
)
{
  kaapi_bitmap_value_t mask;
  kaapi_workqueue_index_ull_t range_size;
  int concurrency = kaapi_getconcurrency();
  long off;
  long scale;
  int sizemap, finalsize;
  int i;

  kaapi_assert_debug(KAAPI_MAX_PROCESSOR < (uint8_t)~0 );

  /* compute the map of kid where to pre-reserve local work. Exclude the calling kproc */
  kaapi_bitmap_value_clear(&mask);
  kaapi_bitmap_value_set_low_bits(&mask, nthreads);

  /* mask with specified cpuset */
  kaapi_bitmap_value_and(&mask, cpuset);
  kaapi_bitmap_init( &wa->map, &mask );

  /* concurrency now = size of the set */
  finalsize = sizemap = kaapi_bitmap_value_count(&mask);
  /* split the range in equal slices */
  range_size = last - first;

  /* handle concurrency too high case */
  if (range_size < sizemap) 
  {
    finalsize = (int)range_size;
    kaapi_bitmap_value_set_low_bits(&mask, finalsize);
    kaapi_bitmap_init( &wa->map, &mask );
  }

  /* round range to be multiple of concurrency 
     tid with indexes 0 will get the biggest part
     Here it is an uniform block distribution. An other distribution should be used.
  */
  off = range_size % finalsize;
  range_size -= off;
  scale = range_size / finalsize;
  
  /* init logical mapping from tid to [0, ..., n] such that localtid is attached to 0 if
     is in the set.
     Initialize also all the localwork to empty (but lock is initialized !)
  */
  uint16_t localcount = 0;
  kaapi_assert_debug( (self_tid>=0) && (self_tid < concurrency));
  if (kaapi_bitmap_value_get(&mask, self_tid))
    wa->tid2pos[self_tid] = localcount++;

  for (i=0; i<concurrency; ++i)
  {
    if (i == self_tid) 
      continue;
      
    if (kaapi_bitmap_value_get(&mask, i)  && (localcount < finalsize))
      wa->tid2pos[i] = localcount++;
    else 
      wa->tid2pos[i] = (uint8_t)-1; /* not in the set */
  }
  
  /* fill the start indexes: here it should be important to
     allocate slices depending of the futur thread id... ?
  */
  wa->startindex_ull[0] = first;
  wa->startindex_ull[1] = first+off+scale;
  for (i=1; i<finalsize; ++i)
    wa->startindex_ull[i+1] = wa->startindex_ull[i]+scale;
  for (i=finalsize; i<concurrency; ++i)
    wa->startindex_ull[i+1] = wa->startindex_ull[finalsize];

  kaapi_assert_debug(wa->startindex_ull[finalsize] == last);
}


/* Initialize the global work:
   - push a frame of the calling thread
   - allocate the global work into the stack
*/
kaapic_global_work_t* kaapic_foreach_global_workinit_ull
(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_ull_t  first, 
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
)
{
  kaapic_global_work_t* gwork;
  kaapic_local_work_t* lwork;
  int concurrency;
  unsigned int nthreads;
  int i;

  /* if empty gwork 
  */
  if (first >= last) 
    return 0;

  /* push a new frame.
     will be poped at the terminaison 
  */
  kaapi_thread_push_frame_( self_thread );
  
  /* is_format true if called from kaapif_foreach_with_format */
  kaapi_thread_t* const thread = kaapi_threadcontext2thread(self_thread);
  int localtid = self_thread->stack.proc->kid;

  /* allocate the gwork */
  gwork = kaapi_thread_pushdata(thread, sizeof(kaapic_global_work_t));
  kaapi_assert_debug(gwork !=0);
  
  /* work array, reserve range in [first,last) for each thief */
  if (attr == 0) attr = &kaapic_default_attr;

  concurrency = kaapi_getconcurrency();
  nthreads = attr->nthreads;
  if (nthreads == (unsigned int)-1)
    nthreads = concurrency;

  /* initialize work distribution 
     - this function should a policy of the foreach attribut
  */
  _kaapic_foreach_initwa_ull(
      &gwork->wa, 
      localtid, 
      (kaapi_bitmap_value_t*)&attr->threadset, 
      nthreads,
      first, last
  );

  /* reset the work remain/wokerdone field */
  KAAPI_ATOMIC_WRITE(&gwork->workremain, last - first);
  KAAPI_ATOMIC_WRITE(&gwork->workerdone, 0);

  /* Initialize all the localwork to empty (but lock is initialized !)
  */
  for (i=0; i<concurrency; ++i)
  {
    lwork = &gwork->lwork[i];
  /* initialize the lwork */
#if defined(KAAPIC_USE_KPROC_LOCK)
    kaapi_workqueue_init_with_lock(
      &lwork->cr,
      0, 0,
      &kaapi_all_kprocessors[i]->lock
    );
#else
    kaapi_atomic_initlock(&lwork->lock);
    kaapi_workqueue_init_with_lock(
      &lwork->cr, 
      0, 0,
      &lwork->lock
    );
#endif
    lwork->context  = 0;
    lwork->global   = gwork;
    lwork->workdone = 0;
    lwork->tid      = i;
    lwork->init     = 0;
  }
  
  gwork->wi.rep.ull.par_grain = attr->rep.li.p_grain;
  gwork->wi.rep.ull.seq_grain = attr->rep.li.s_grain;

  /* Initialize the information about distribution of iteration */
  gwork->wi.nthreads  = nthreads;
  gwork->wi.threadset = attr->threadset;
  gwork->wi.itercount = last-first;
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  gwork->wi.dist   = attr->datadist;
#endif

  gwork->body_f_ull = body_f;
  gwork->body_args  = body_args;

  KAAPI_SET_SELF_WORKLOAD(last-first);

  return gwork;
}  



/* Initialize the local work + the global work
   Return a pointer to the local work to execute. 
   Do not require that reset is protected:
   - steal cannot occurs on local work while ->init ==0. 
*/
kaapic_local_work_t* kaapic_foreach_local_workinit_ull
(
  kaapic_local_work_t*        lwork,
  kaapi_workqueue_index_ull_t first,
  kaapi_workqueue_index_ull_t last
)
{  
  kaapi_assert_debug(lwork !=0);
  kaapi_assert_debug(lwork->workdone == 0);
  kaapi_assert_debug(lwork->init == 0);

  kaapi_workqueue_reset_ull(
    &lwork->cr, 
    first,
    last
  );

  /* publish new local work */
  kaapi_writemem_barrier();
  lwork->init        = 1;

  return lwork;
}  



/* task splitter: interface between scheduler and kaapic works. 
   This function is called when an idle kproc emits a request.
   If the sequence has as many slices, one for each core, then
   each kproc only pass one through this function
*/
static int _kaapic_split_task_ull
(
  struct kaapi_task_t*                 victim_task,
  void*                                arg,
  struct kaapi_listrequest_t*          lr,
  struct kaapi_listrequest_iterator_t* lri
)
{
  /* call by a thief to steal work */
  int tid;
  kaapic_local_work_t* const lwork = (kaapic_local_work_t*)arg;
  kaapic_global_work_t* const gwork = lwork->global;
  kaapi_bitmap_value_t mask;
  kaapi_bitmap_value_t intersect;
  kaapi_bitmap_value_t negmask;
  kaapi_bitmap_value_t replymask;  
  kaapi_workqueue_index_ull_t first, last;
  kaapi_request_t* req;
  kaapic_local_work_t* tw;

  if (KAAPI_ATOMIC_READ(&gwork->workremain) ==0) 
    return 0;

  /* no attached body: return (typically case of OpenMP)*/
  if (gwork->body_f ==0)
    return 0;
  
  /* count requests that will be served by root tasks.
  */
#if defined(KAAPI_DEBUG)
  kaapi_listrequest_iterator_t save_lri __attribute__((unused)) = *lri;
#endif
  kaapi_listrequest_iterator_t cpy_lri;
  cpy_lri = *lri;
  kaapi_bitmap_value_clear( &mask );
  for (; !kaapi_listrequest_iterator_empty(&cpy_lri); ) 
  {
    kaapi_request_t* req = kaapi_listrequest_iterator_get(lr, &cpy_lri);
    int pos = gwork->wa.tid2pos[ req->ident ];
    if (pos != (uint8_t)-1)
      kaapi_bitmap_value_set( &mask, pos );
    
    kaapi_listrequest_iterator_next(lr, &cpy_lri);
  }
  kaapi_bitmap_value_neg( &negmask, &mask );

  intersect = mask;
  kaapi_bitmap_value_and( &intersect, (kaapi_bitmap_value_t*)&gwork->wa.map );

  if (kaapi_bitmap_value_empty(&intersect))
    goto skip_global;
    
  /* Steal toplevel request: mark to 0 all entries of a incomming request */
  kaapi_bitmap_and( &replymask, &gwork->wa.map, &negmask );
  
  /* reply to all requests in original_mask */
  kaapi_bitmap_value_and(&replymask, &mask);
  
  /* reply for first time to thread with reserved request (if !=0) */
  while ((tid = kaapi_bitmap_value_first1_and_zero(&replymask)) != 0)
  {
    --tid;

    /* use only get: the bitmap gwork->wa.map was updated atomically above */
    if (kaapic_global_getwork_ull( gwork, tid, &first, &last ))
    {
      req = &kaapi_global_requests_list[tid];
      
      KAAPI_ATOMIC_INCR(&gwork->workerdone);
      tw = &gwork->lwork[tid];
      kaapic_foreach_local_workinit(tw, first, last);

      kaapi_task_init_with_flag(
        kaapi_request_toptask(req), 
        (kaapi_task_body_t)_kaapic_thief_entrypoint_ull, 
        tw,
        KAAPI_TASK_UNSTEALABLE
      );
#if 1 /* comment this line if you do not want thief to be thief by other */
      kaapi_request_pushtask_adaptive_tail( 
        req, 
        victim_task,
        _kaapic_split_task_ull 
      );
#else // unstealable task 
      kaapi_request_pushtask(
        req,
        victim_task
      );
#endif
      kaapi_request_committask(req);
      kaapi_listrequest_iterator_unset_at( lri, tid );
    } /* else: no work to done, will be reply failed by the runtime */
  }

  if (kaapi_listrequest_iterator_empty(lri))
    return 0;


skip_global:  
  /* because object lwork may exist without thread, test if initialized */
  if (lwork->init == 0)
    return 0;
    
  /* */
  kaapi_workqueue_index_ull_t range_size, unit_size;

  /* else: remaining requests in lri was already steal their replied  
     here is code to reply to thread that do not have reserved slice
  */
  range_size = kaapi_workqueue_size(&lwork->cr);
  if (range_size <= gwork->wi.rep.ull.par_grain)
    /* no enough work: stop stealing this task */
    return 0;

  /* how much per non root req */
  int nreq = kaapi_listrequest_iterator_count(lri);
  unit_size = range_size / (nreq + 1);
  if (unit_size < gwork->wi.rep.ull.par_grain)
  {
    nreq = (int)((range_size / gwork->wi.rep.ull.par_grain) - 1);
    unit_size = gwork->wi.rep.ull.par_grain;
    if (nreq ==0)
      return 0;
  }

#if defined(KAAPIC_USE_KPROC_LOCK)
  kaapi_assert_debug( lwork->cr.lock 
    == &kaapi_get_current_processor()->victim_kproc->lock );
  kaapi_assert_debug( kaapi_atomic_assertlocked(lwork->cr.lock) );
#else
  _kaapi_workqueue_lock(&lwork->cr);
#endif

  if (kaapi_workqueue_steal_ull(&lwork->cr, &first, &last, nreq * unit_size))
  {
#if defined(KAAPIC_USE_KPROC_LOCK)
#else
    _kaapi_workqueue_unlock(&lwork->cr);
#endif
    return 0;
  }
#if defined(KAAPIC_USE_KPROC_LOCK)
#else
  _kaapi_workqueue_unlock(&lwork->cr);
#endif

  kaapi_assert_debug(first < last);  
  kaapi_assert_debug(unit_size*nreq == last-first);  
#if defined(KAAPI_DEBUG)
  int sfirst __attribute__((unused)) = first;
  int slast __attribute__((unused))  = last;
#endif

  for ( /* void */; 
        !kaapi_listrequest_iterator_empty(lri) && (nreq >0);
        kaapi_listrequest_iterator_next(lr, lri), --nreq
      )
  { 
    req = kaapi_listrequest_iterator_get(lr, lri);
    tw = &gwork->lwork[req->ident];
    kaapic_foreach_local_workinit(tw, last-unit_size, last);

    kaapi_task_init_with_flag(
      kaapi_request_toptask(req), 
      (kaapi_task_body_t)_kaapic_thief_entrypoint_ull, 
      tw,
      KAAPI_TASK_UNSTEALABLE
    );
#if 1 /* comment this line if you do not want thief to be thief by other */
    kaapi_request_pushtask_adaptive_tail( 
      req, 
      victim_task,
      _kaapic_split_task_ull
    );
#else // unstealable task 
    kaapi_request_pushtask(
      req,
      victim_task
    );
#endif
    kaapi_request_committask(req);
    last -= unit_size;
  }
  kaapi_assert_debug( last == first );

  return 0;
}


/* thief entrypoint, lwork is already initialized 
*/
static void _kaapic_thief_entrypoint_ull(
  void*                 arg, 
  kaapi_thread_t*       thread,
  kaapi_task_t*         pc
)
{
  /* range to process */
  kaapi_workqueue_index_ull_t i;
  kaapi_workqueue_index_ull_t j;

  kaapi_processor_t* kproc = kaapi_get_current_processor();

  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_FOREACH_BEG );

  /* process the work */
  kaapic_local_work_t* const lwork = (kaapic_local_work_t*)arg;
  kaapic_global_work_t* const gwork = lwork->global;
  
  /* asserts */
  kaapi_assert_debug(lwork->workdone == 0);
  kaapi_assert_debug(kaapi_get_self_kid() == lwork->tid);
#if defined(KAAPIC_USE_KPROC_LOCK)
  kaapi_assert_debug( &kproc->lock == lwork->cr.lock );
#else
#endif
  kaapi_assert_debug( kproc->kid == lwork->tid );
  
#if 0 // OLD LOOP
  /* work info */
  const kaapic_work_info_t* const wi = &gwork->wi;

  /* while there is sequential work to do in local work */
  while (kaapi_workqueue_pop_ull(&lwork->cr, &i, &j, wi->rep.ull.seq_grain) ==0)
  {
    KAAPI_SET_SELF_WORKLOAD(kaapi_workqueue_size(&lwork->cr));
    lwork->workdone += j-i;
redo_local_work:
    kaapi_assert_debug( i < j );
    /* apply w->f on [i, j[ */
    gwork->body_f((int)i, (int)j, (int)lwork->tid, gwork->body_args);
  }
  lwork->init = 0;
#endif

  /* while there is sequential work to do in local work */
  while (kaapic_foreach_worknext_ull(lwork, &i, &j) !=0)
  {
redo_local_work:
    kaapi_assert_debug( i < j );
    /* apply w->f on [i, j[ */
    gwork->body_f((int)i, (int)j, (int)lwork->tid, gwork->body_args);
  }
  kaapi_assert_debug( kaapi_workqueue_isempty(&lwork->cr) );

  /* */
  KAAPI_SET_SELF_WORKLOAD(0);

#if defined(KAAPI_DEBUG)
  kaapi_assert(lwork->workdone >=0);
  uint64_t gwr = KAAPI_ATOMIC_READ(&gwork->workremain);
  kaapi_assert( gwr >= lwork->workdone );
#endif

  if (KAAPI_ATOMIC_SUB(&gwork->workremain, lwork->workdone) ==0) 
    goto return_label;

  kaapi_assert_debug(KAAPI_ATOMIC_READ(&gwork->workremain) >=0);
  lwork->workdone = 0;

  /* finish: nothing to steal */
  kaapi_writemem_barrier();

  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
  int retval = kaapic_foreach_globalwork_next_ull( lwork, &i, &j );
  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_SCHED_IDLE_END );
  if (retval)
    goto redo_local_work;
  
return_label:
  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_FOREACH_END );

  lwork->workdone = 0;
  lwork->init = 0;

  /* suppress lwork reference */
  kaapi_task_unset_splittable(pc);
  kaapi_synchronize_steal_thread(kproc->thread);

  /* bypass scheduler: do the same things (unset/synch/membarrier) */
  kaapi_writemem_barrier();
  
  KAAPI_ATOMIC_DECR(&gwork->workerdone);
  
  /* no destruction of the local work: maintained by the master */
}


/* Initialize the local work + the global work
   Return a pointer to the local work to execute. 
*/
kaapic_local_work_t* kaapic_foreach_workinit_ull
(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_ull_t  first, 
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
)
{
  kaapic_global_work_t* gwork;
  kaapic_local_work_t*  lwork = 0;
  const int tid = self_thread->stack.proc->kid;

  gwork = kaapic_foreach_global_workinit_ull(
      self_thread, 
      first, 
      last, 
      attr, 
      body_f, 
      body_args
  );
  kaapi_assert_debug(gwork !=0);

  /* initialize the local workqueue with the first poped state */
  if (kaapic_global_work_pop_ull( gwork, tid, &first, &last))
  {    
    /* */
    lwork = kaapic_foreach_local_workinit_ull( 
                    &gwork->lwork[tid],
                    first,
                    last
    );
    /* from here lwork is visible to be steal, but it cannot until
       the adaptive task is spawned 
    */
  }
  kaapi_assert_debug(lwork !=0);

  /* start adaptive region */
  lwork->context = kaapi_task_begin_adaptive(
     kaapi_threadcontext2thread(self_thread), 
     KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION,
     _kaapic_split_task_ull,
     lwork
  );
#if 0
  KAAPI_EVENT_PUSH1(
      self_thread->stack.proc,
      self_thread,
      KAAPI_EVT_TASK_BEG,
      lwork->context
  );
#endif

  /* begin a parallel region */
  if (kaapic_do_parallel) kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);

  return lwork;
}  


/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute.
  The function try to steal from registered lwork in the global work.
  The local workqueue is fill by poped range.
*/
static int kaapic_foreach_globalwork_next_ull(
  kaapic_local_work_t*         lwork,
  kaapi_workqueue_index_ull_t* first,
  kaapi_workqueue_index_ull_t* last
)
{
  kaapic_global_work_t* gwork = lwork->global;
  kaapi_processor_t* kproc = kaapi_get_current_processor();

  while (KAAPI_ATOMIC_READ(&gwork->workremain) !=0)
  {
    kaapi_assert_debug(KAAPI_ATOMIC_READ(&gwork->workremain) >=0);

    if (kaapic_global_work_steal_ull(
          gwork,
          kproc,
          first,last)
       )
    {
      if (*last - *first <= gwork->wi.rep.ull.seq_grain) 
        goto retval1;

      /* refill the global work data structure without seq_grain */
      _kaapi_workqueue_lock( &lwork->cr );
      kaapic_foreach_local_workinit_ull( 
          lwork,
          *first+gwork->wi.rep.ull.seq_grain, 
          *last
      );
      _kaapi_workqueue_unlock( &lwork->cr );
      KAAPI_SET_SELF_WORKLOAD(
          kaapi_workqueue_size_ull(&lwork->cr)
      );
      *last = *first + gwork->wi.rep.ull.seq_grain;
      goto retval1;
    }
    kaapi_slowdown_cpu();
  }
  return 0; /* means global is terminated */

retval1:
  lwork->workdone += *last - *first;
  return 1;
}


/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute.
  This method is dedicated to be used
*/
int kaapic_foreach_worknext_ull(
  kaapic_local_work_t*         lwork,
  kaapi_workqueue_index_ull_t* first,
  kaapi_workqueue_index_ull_t* last
)
{
  int retval;
  kaapic_global_work_t* gwork = lwork->global;
  int iszero = (KAAPI_ATOMIC_READ(&gwork->workremain) ==0);
  kaapi_workqueue_index_ull_t sgrain = gwork->wi.rep.ull.seq_grain;

  if ( iszero || (sgrain == 0))
  {
    KAAPI_DEBUG_INST(*first = *last = 0);
    return 0;
  }

  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), 0, KAAPI_EVT_SCHED_IDLE_BEG );

#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  if (kaapic_local_workqueue_isempty_ull(&lwork->local_cr))
  {
    if (kaapi_workqueue_pop_ull(&lwork->cr, first, last, sgrain) == 0)
    {
      KAAPI_SET_SELF_WORKLOAD(
          kaapi_workqueue_size(&lwork->cr)
      );
      lwork->workdone += *last-*first; /* even if work is not yet performed, the poped range is considered to be sequentially executed */
      kaapic_local_workqueue_set_ull( &lwork->local_cr, *first, *last );
    }
    else
      goto fail_pop;
  }
  kaapi_assert( kaapic_local_workqueue_pop_withdatadistribution_ull( &lwork->local_cr, &gwork->wi, first, last, sgrain ) == 0 );
  retval = 1;
  goto return_value;

#else

  if (kaapi_workqueue_pop_ull(&lwork->cr, first, last, sgrain) == 0)
  {
    KAAPI_SET_SELF_WORKLOAD(
        kaapi_workqueue_size(&lwork->cr)
    );

    lwork->workdone += *last-*first;
    retval = 1;
    goto return_value;
  }
#endif

#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
fail_pop:
#endif
  kaapi_assert_debug( kaapi_workqueue_isempty_ull(&lwork->cr) );
  lwork->init = 0;

  /* Empty local work: try to steal from global work 
     After updating the workremain of the global work
  */
#if defined(KAAPI_DEBUG)
  kaapi_assert(lwork->workdone >=0);
  uint64_t gwr = KAAPI_ATOMIC_READ(&gwork->workremain);
  kaapi_assert( gwr >= lwork->workdone );
#endif
  KAAPI_ATOMIC_SUB(&gwork->workremain, lwork->workdone);
  kaapi_assert_debug(KAAPI_ATOMIC_READ(&gwork->workremain) >=0);
  lwork->workdone = 0;

  retval = kaapic_foreach_globalwork_next_ull( lwork, first, last );

return_value:
  KAAPI_EVENT_PUSH0(kaapi_get_current_processor(), 0, KAAPI_EVT_SCHED_IDLE_END );
  return retval;
}



/* exported foreach interface */
int kaapic_foreach_common_ull
(
  kaapi_workqueue_index_ull_t  first,
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
)
{
  kaapic_global_work_t* gwork;
  kaapic_local_work_t*  lwork;

  /* is_format true if called from kaapif_foreach_with_format */
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  kaapi_thread_context_t* const self_thread = kproc->thread;
  const int tid = self_thread->stack.proc->kid;

  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_FOREACH_BEG );

  lwork = kaapic_foreach_workinit_ull(
      self_thread,
      first,
      last,
      attr, 
      body_f,
      body_args
  );
  kaapi_assert_debug(lwork !=0);
  
  /* process locally */
#if defined(KAAPI_DEBUG)
  kaapi_assert_debug( kproc->kid == tid );
#endif

  gwork = lwork->global;

#if 0 //OLD LOOP
  long seq_grain = gwork->wi.rep.ull.seq_grain;
  
  /* while there is sequential work to do in local work */
  while (kaapi_workqueue_pop_ull(&lwork->cr, &first, &last, seq_grain) ==0)
  {
    KAAPI_SET_SELF_WORKLOAD(kaapi_workqueue_size(&lwork->cr));
    lwork->workdone += last-first;
redo_local_work:
    kaapi_assert_debug( first < last );
    /* apply w->f on [i, j[ */
    body_f(first, last, (int)tid, body_args);
  }
  lwork->init = 0;
#endif

  /* while there is sequential work to do in local work */
  while (kaapic_foreach_worknext_ull(lwork, &first, &last) !=0)
  {
redo_local_work:
    kaapi_assert_debug( first < last );
    /* apply w->f on [i, j[ */
    body_f((int)first, (int)last, (int)tid, body_args);
  }

  kaapi_assert_debug( kaapi_workqueue_isempty(&lwork->cr) );

  /* */
  KAAPI_SET_SELF_WORKLOAD(0);

#if defined(KAAPI_DEBUG)
  kaapi_assert(lwork->workdone >=0);
  uint64_t gwr = KAAPI_ATOMIC_READ(&gwork->workremain);
  kaapi_assert( gwr >= lwork->workdone );
#endif

  /* update workload information */
  if (KAAPI_ATOMIC_SUB(&gwork->workremain, lwork->workdone) ==0) 
    goto return_label;

  kaapi_assert_debug(KAAPI_ATOMIC_READ(&gwork->workremain) >=0);
  lwork->workdone = 0;

  KAAPI_EVENT_PUSH0(self_thread->stack.proc, 0, KAAPI_EVT_SCHED_IDLE_BEG );
  int retval = kaapic_foreach_globalwork_next_ull( lwork, &first, &last );
  KAAPI_EVENT_PUSH0(self_thread->stack.proc, 0, KAAPI_EVT_SCHED_IDLE_END );
  if (retval)
    goto redo_local_work;
 
return_label:
  KAAPI_EVENT_PUSH0(kproc, 0, KAAPI_EVT_FOREACH_END );

  lwork->workdone = 0;
  lwork->init = 0;
  kaapi_writemem_barrier();

  kaapic_foreach_workend( self_thread, lwork );

#if CONFIG_FOREACH_STATS
  foreach_time += kaapif_get_time_() - time;
#endif
  return 0;
}

