#ifndef KA_LINEAR_WORK_HH_INCLUDED
# define KA_LINEAR_WORK_HH_INCLUDED


#include <new>
#include <cstdlib>
#include "kaapi.h"


namespace ka {
namespace linearWork {


// work interval. the mapping is problem specific
struct range
{
  typedef unsigned long index_type;
  typedef unsigned long size_type;

  index_type _i, _j;

  range() {}

  range(index_type i, index_type j) : _i(i), _j(j) {}

  size_type size() const { return _j - _i; }
  index_type begin() const { return _i; }
  index_type end() const { return _j; }
};


// problem specific results must inherit from baseResult
class baseResult
{
public:
  range _range;
  bool _is_reduced;

  baseResult() : _is_reduced(false) {}

  baseResult(range::index_type i, range::index_type j)
    : _range(i, j), _is_reduced(false) {}
};

class voidResult : public baseResult
{
public:

  voidResult() {}

  template<typename work_type>
  void initialize(const work_type&) {}
};


// problem works have to inherit from that
class baseWork
{
public:
  kaapi_workqueue_t _wq;

  // result_hack: for now, victim result is maintained here
  void* _res;

  // default traits
  static const bool is_reducable = true;
  static const unsigned int seq_grain = 1;
  static const unsigned int par_grain = 1;

  baseWork(range::index_type i, range::index_type j)
  { kaapi_workqueue_init(&_wq, i, j); }

}; // baseWork


// reducer
template<typename work_type, typename result_type>
static void common_reducer
(work_type* vw, result_type* vr, result_type* tr)
{
  // vw the victim work
  // vr the victim result
  // tr the thief result

  range processed((range::index_type)vw->_wq.end, tr->_range.begin());

  // reduce the thief result
  vw->reduce(*vr, *tr, processed);

  // continue the thief work
  kaapi_workqueue_set(&vw->_wq, tr->_range._i, tr->_range._j);
}

template<typename work_type, typename result_type>
static int thief_reducer
(kaapi_taskadaptive_result_t* ktr, void* varg, void* targ)
{
  // called from the thief upon victim preemption request

  work_type* const vw = (work_type*)varg;

  common_reducer<work_type, result_type>
    (vw, (result_type*)vw->_res, (result_type*)ktr->data);

  // inform the victim we did the reduction
  ((result_type*)ktr->data)->_is_reduced = true;

  return 0;
}

template<typename work_type, typename result_type>
static int victim_reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t, void* varg)
{
  // called from the victim to reduce a thief result

  if (((result_type*)tdata)->_is_reduced == false)
  {
    work_type* const vw = (work_type*)varg;
    common_reducer<work_type, result_type>
      (vw, (result_type*)vw->_res, (result_type*)tdata);
  }

  return 0;
}

// linear work splitter

typedef void (*adaptive_body_t)
(void*, kaapi_thread_t*, kaapi_stealcontext_t*);

template<typename work_type, typename result_type>
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);

template<typename work_type, typename result_type>
static int work_splitter
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
{
  adaptive_body_t const entrypoint =
    thief_entrypoint<work_type, result_type>;

  work_type* const vw = (work_type*)args;

  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;

  int nrep = 0;

  kaapi_workqueue_index_t unit_size;

 redo_steal:
  // do not steal if range size <= par_grain
  range_size = kaapi_workqueue_size(&vw->_wq);
  if (range_size <= work_type::par_grain)
    return 0;

  // how much per req
  unit_size = range_size / (nreq + 1);
  if (unit_size == 0)
  {
    nreq = (range_size / work_type::par_grain) - 1;
    unit_size = work_type::par_grain;
  }

  // perform the actual steal. if the range
  // changed size in between, redo the steal
  if (kaapi_workqueue_steal(&vw->_wq, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    // for reduction, a result is needed. take care of initializing it
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(result_type), NULL);

    // initialize the thief work
    work_type* const tw = (work_type*)kaapi_reply_init_adaptive_task
      (sc, req, (kaapi_task_body_t)entrypoint, sizeof(work_type), ktr);
    new (tw) baseWork
      ((range::index_type)(j - unit_size), (range::index_type)j);
    tw->initialize(*vw);

    // result_hack
    tw->_res = ktr->data;

    // initialize ktr task may be preempted before entrypoint
    new (ktr->data) baseResult
      ((range::index_type)(j - unit_size), (range::index_type)j);
    ((result_type*)ktr->data)->initialize(*tw);
    
    // reply head, preempt head
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }

  return nrep;
} // work_splitter


// thief entrypoint

extern "C" void kaapi_synchronize_steal(kaapi_stealcontext_t*);

static int extract_seq
(kaapi_workqueue_t& wq, range& seq_range, unsigned long seq_size)
{
  long i, j;
  if (kaapi_workqueue_pop(&wq, &i, &j, seq_size))
    return -1;
  seq_range = range((range::index_type)i, (range::index_type)j);
  return 0;
}

template<typename work_type, typename result_type>
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  const kaapi_task_splitter_t splitter =
    work_splitter<work_type, result_type>;

  const kaapi_thief_reducer_t reducer =
    thief_reducer<work_type, result_type>;

  // input work
  work_type* const work = (work_type*)args;

  // resulting work
  result_type* const res = (result_type*)kaapi_adaptive_result_data(sc);

  // extracted range
  range seq_range;

  unsigned int is_preempted;

  // set the splitter for this task
  kaapi_steal_setsplitter(sc, splitter, work);

  while (extract_seq(work->_wq, seq_range, work_type::seq_grain) != -1)
  {
    work->execute(*res, seq_range);

    kaapi_steal_setsplitter(sc, NULL, NULL);
    kaapi_synchronize_steal(sc);

    res->_range._i = (range::index_type)work->_wq.beg;
    res->_range._j = (range::index_type)work->_wq.end;

    is_preempted = kaapi_preemptpoint(sc, reducer, NULL, NULL, 0, NULL);
    if (is_preempted) return ;

    kaapi_steal_setsplitter(sc, splitter, work);
  }

  // update our results. use beg == beg
  // to avoid the need of synchronization
  // with potential victim .end update
  res->_range._i = work->_wq.beg;
  res->_range._j = work->_wq.beg;

} // thief_entrypoint


template<typename work_type, typename result_type>
static void execute(work_type& work, result_type& res)
{
  // todo: take into account work traits
  // to use the right flags

  const kaapi_victim_reducer_t reducer =
    victim_reducer<work_type, result_type>;

  const kaapi_task_splitter_t splitter =
    work_splitter<work_type, result_type>;

  // stealcontext flags
  static const unsigned long sc_flags =
    KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION;

  // self thread, task
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;

  range seq_range;

  // result_hack
  work._res = (void*)&res;

  // enter adaptive section
  sc = kaapi_task_begin_adaptive(thread, sc_flags, splitter, &work);

 continue_work:
  while (extract_seq(work._wq, seq_range, work_type::seq_grain) != -1)
    work.execute(res, seq_range);

  // preempt and reduce thieves
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief
      (sc, ktr, (void*)&work, reducer, (void*)&work);
    goto continue_work;
  }

  // wait for thieves
  kaapi_task_end_adaptive(sc);

} // execute


// no result execute version
template<typename work_type>
static void execute(work_type& work)
{ voidResult res; execute(work, res); }


namespace toRemove {
// kaapi runtime constructors
static void initialize(int ac = 0, char** av = 0)
{ kaapi_init(); }

static void finalize()
{ kaapi_finalize(); }

} // ::toRemove


} } // ka::linearWork


#endif // KA_LINEAR_WORK_HH_INCLUDED
