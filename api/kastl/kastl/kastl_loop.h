/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
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

#ifndef _KASTL_LOOP_H_
# define _KASTL_LOOP_H_


#include <sys/types.h>
#include "kaapi.h"
#include "kastl_workqueue.h"
#include "kastl_sequences.h"

#if CONFIG_KASTL_DEBUG
extern "C" unsigned int kaapi_get_current_kid(void);
static volatile unsigned int __attribute__((aligned)) printid = 0;
#endif

/* global lock */
#define CONFIG_KASTL_GLOBAL_LOCK 0
#if CONFIG_KASTL_GLOBAL_LOCK

struct __global_lock
{
  static kastl::rts::atomic_t<32> _lock;

  static void acquire()
  {
    while (!_lock.cas(0, 1))
      ;
  }

  static void release()
  {
    _lock.write(0);
  }
};

kastl::rts::atomic_t<32> __global_lock::_lock = {{0}};

#endif // CONFIG_KASTL_GLOBAL_LOCK


namespace kastl
{
namespace impl
{
  // dummy type
  struct dummy_type {};

  // dummy tag
  struct dummy_tag {};

  // reduce tag
  struct reduce_tag {};
  typedef dummy_tag noreduce_tag;

  // innermost (nano) tag
  struct unrolled_tag {};
  typedef dummy_tag rolled_tag;

  // extractor tag
  struct window_tag {};
  struct linear_tag {};
  struct backoff_tag {};
  struct reverse_tag {};
  struct identity_tag {};
  typedef dummy_tag static_tag;

  // result tag
  struct touchable_tag {};


  // settings allow for static and dynamic parametrisation
  template<typename MacroExtractorTag>
  struct settings
  {
    // which extractor to use for macro loop
    typedef MacroExtractorTag _macro_extractor_tag;

    // nano loop unit size
    const size_t _seq_size;
    // minimum parallel size
    const size_t _par_size;
    // window size
    const size_t _win_size;

    settings(size_t seq_size, size_t par_size)
      : _seq_size(seq_size), _par_size(par_size), _win_size(0)
    {}

    settings(size_t seq_size, size_t par_size, size_t win_size)
      : _seq_size(seq_size), _par_size(par_size), _win_size(win_size)
    {}
  };

  // sugar
  typedef settings<static_tag> static_settings;

  // results
  template<typename Iterator>
  struct touched_algorithm_result
  {
    bool _is_touched;
    Iterator _iter;

    touched_algorithm_result(const Iterator&)
      : _is_touched(false)
    {}

    void set_iter(const Iterator& iter)
    {
      _is_touched = true;
      _iter = iter;
    }
  };

  template<typename Iterator, typename Value>
  struct numeric_result
  {
    Value _value;

    numeric_result(const Value& value)
      : _value(value)
    {}

    numeric_result(const Iterator&)
      : _value(static_cast<Value>(0))
    {}
  };

  // extractors
  template<typename Tag = static_tag>
  struct extractor
  {
    size_t _unit_size;

    template<typename Settings>
    extractor(const Settings& settings)
      : _unit_size(settings._seq_size)
    {}

    template<typename Sequence>
    bool extract(Sequence& seq, typename Sequence::range_type& range)
    {
      return seq.pop(range, _unit_size);
    }
  };

  template<>
  struct extractor<identity_tag>
  {
    template<typename Settings>
    extractor(const Settings& settings)
    {}

    template<typename Sequence>
    bool extract(Sequence& seq, typename Sequence::range_type& range)
    {
      return seq.pop(range, seq.size());
    }
  };

  template<>
  struct extractor<reverse_tag>
  {
    template<typename Sequence, typename Settings>
    extractor(const Settings&)
    {}

    template<typename Sequence>
    bool extract(Sequence& s, typename Sequence::range_type& r)
    {
      return false;
    }
  };

  // kastl typedefs
  typedef int (*kastl_splitter_t)
  (kaapi_stealcontext_t*, int, kaapi_request_t*, void*);

  typedef int (*kastl_reducer_t)
  (kaapi_stealcontext_t*, void*, void*, size_t, void*);

  typedef void (*kastl_entry_t)(void*, kaapi_thread_t*);

  // reducers
  template<typename ReduceTag = noreduce_tag>
  struct reducer
  {
    // no result reducer
    template<typename Result, typename Sequence, typename Body>
    struct thief_context
    {
      typedef thief_context<Result, Sequence, Body> context_type;

      Sequence _seq;
      Result _res;
      bool _is_done;

      thief_context(const Sequence& seq)
	: _seq(seq), _is_done(false)
      {}

      static kaapi_taskadaptive_result_t* allocate
      (kaapi_stealcontext_t* sc, const Sequence& seq, Body& body)
      {
	kaapi_taskadaptive_result_t* const ktr =
	  kaapi_allocate_thief_result(sc, sizeof(context_type), NULL);
	new (ktr->data) context_type(seq);
	return ktr;
      }
    };

    template<typename Result, typename Sequence, typename Body>
    struct victim_context
    {
      Sequence& _seq;

      victim_context(Sequence& seq)
	: _seq(seq)
      {}
    };

    template<typename Result, typename Sequence, typename Body>
    static int reduce_function
    (kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
    {
      typedef typename Sequence::range_type range_type;

      typedef thief_context<Result, Sequence, Body> thief_context_type;
      typedef victim_context<Result, Sequence, Body> victim_context_type;

      victim_context_type* const vc = static_cast<victim_context_type*>(vptr);
      thief_context_type* const tc = static_cast<thief_context_type*>(tptr);

      // join sequences
      kastl::rts::range_t<64> range(tc->_seq._wq._beg, tc->_seq._wq._end);
      vc->_seq._rep = tc->_seq._rep;
      vc->_seq._wq.set(range);

      return 0; // false, continue
    }

    template<typename Result, typename Sequence, typename Body>
    static void reduce
    (kaapi_stealcontext_t* sc,
     bool& has_thief,
     bool&,
     Result&, Sequence& seq, Body&)
    {
      kastl_reducer_t const kastl_reducer =
	reduce_function<Result, Sequence, Body>;

      has_thief = true;

      // no more thief, we are done
      kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
      if (ktr == NULL)
      {
	has_thief = false;
	return ;
      }

      victim_context<Result, Sequence, Body> vc(seq);
      kaapi_preempt_thief(sc, ktr, NULL, kastl_reducer, &vc);
    }

    static bool preempt
    (kaapi_stealcontext_t*, kaapi_taskadaptive_result_t*)
    {
      return false;
    }

  }; // dummy reducer

  template<>
  struct reducer<reduce_tag>
  {
    template<typename Result, typename Sequence, typename Body>
    struct thief_context
    {
      typedef thief_context<Result, Sequence, Body> context_type;

      Sequence _seq;
      Result _res;
      bool _is_done;

      thief_context(const Sequence& seq, Body& body)
	: _seq(seq), _res(seq.begin1()), _is_done(false)
      {}

      static kaapi_taskadaptive_result_t* allocate
      (kaapi_stealcontext_t* sc, const Sequence& seq, Body& body)
      {
	kaapi_taskadaptive_result_t* const ktr =
	  kaapi_allocate_thief_result(sc, sizeof(context_type), NULL);
	new (ktr->data) context_type(seq, body);
	return ktr;
      }
    }; // thief_context

    template<typename Result, typename Sequence, typename Body>
    struct victim_context
    {
      bool& _is_done;
      Result& _res;
      Sequence& _seq;
      Body& _body;

      victim_context(bool& is_done, Result& res, Sequence& seq, Body& body)
	: _is_done(is_done), _res(res), _seq(seq), _body(body)
      {}
    }; // victim_context

    template<typename Result, typename Sequence, typename Body>
    static int reduce_function
    (kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
    {
      typedef typename Sequence::range_type range_type;

      typedef thief_context<Result, Sequence, Body> thief_context_type;
      typedef victim_context<Result, Sequence, Body> victim_context_type;

      victim_context_type* const vc = static_cast<victim_context_type*>(vptr);
      thief_context_type* const tc = static_cast<thief_context_type*>(tptr);

      // if we are done, dont retrieve
      // thief result and sequence, so
      // that there is nothing to steal
      // from us. otherwise, get result
      // and update _is_done accordingly.
      if (vc->_is_done == false)
      {
	// reduce results
	if (vc->_body.reduce(vc->_res, tc->_res) == true)
	{
	  // algorithm completed
	  vc->_is_done = true;
	  return 1;
	}

	// join sequences
	kastl::rts::range_t<64> range(tc->_seq._wq._beg, tc->_seq._wq._end);
	vc->_seq._rep = tc->_seq._rep;
	vc->_seq._wq.set(range);
      }

      return 0;
    }

    template<typename Result, typename Sequence, typename Body>
    static void reduce
    (kaapi_stealcontext_t* sc,
     bool& has_thief,
     bool& is_done,
     Result& res, Sequence& seq, Body& body)
    {
      kastl_reducer_t const kastl_reducer =
	reduce_function<Result, Sequence, Body>;

      has_thief = true;

      // no more thief, we are done
      kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
      if (ktr == NULL)
      {
	has_thief = false;
	return ;
      }

      victim_context<Result, Sequence, Body> vc(is_done, res, seq, body);
      kaapi_preempt_thief(sc, ktr, NULL, kastl_reducer, &vc);
    }

    static bool preempt
    (kaapi_stealcontext_t* sc, kaapi_taskadaptive_result_t* ktr)
    {
      const int is_preempted = kaapi_preemptpoint
	(ktr, sc, NULL, NULL, NULL, 0, NULL);
      return (bool)is_preempted;
    }

  };

  // forward decls
  template<typename ReduceTag, typename UnrollTag> struct outter_loop;

  template<typename Result, typename Sequence, typename Body, typename Settings>
  struct splitter
  {
    typedef splitter<Result, Sequence, Body, Settings> splitter_type;

    struct victim_context
    {
      // adaptive conctext

      Result _res;
      Sequence& _seq;
      Body& _body;
      const Settings& _settings;
    };

    struct task_context
    {
      typedef Result result_type;
      typedef Sequence sequence_type;
      typedef Body body_type;
      typedef Settings settings_type;

      // passed upon thief entry

      Result& _res;
      Sequence& _seq;
      Body _body;
      const Settings& _settings;
      kaapi_stealcontext_t* _master_sc;
      kaapi_taskadaptive_result_t* _ktr;

      task_context
      (Result& res,
       Sequence& seq,
       Body& body,
       const Settings& settings,
       kaapi_stealcontext_t* master_sc = NULL,
       kaapi_taskadaptive_result_t* ktr = NULL)
	: _res(res), _seq(seq), _body(body), _settings(settings),
	  _master_sc(master_sc), _ktr(ktr)
      {}
    };

    template<typename ReduceTag, typename UnrollTag>
    static int split
    (kaapi_stealcontext_t* sc, int request_count,
     kaapi_request_t* request, void* arg)
    {
      typedef typename Sequence::range_type range_type;

      // victim task context
      task_context* const vc = static_cast<task_context*>(arg);

      // compute the balanced unit size.
      const size_t seq_size = vc->_seq.size();
      if (seq_size == 0)
	return 0;

      size_t unit_size = seq_size / (request_count + 1);
      if (unit_size < vc->_settings._par_size)
      {
	request_count = (seq_size / vc->_settings._par_size) - 1;
	if (request_count <= 0)
	  return 0;

	unit_size = vc->_settings._par_size;
      }

      // steal a range smaller or eq to steal_size
      const size_t steal_size = unit_size * (size_t)request_count;

      range_type r;
      if (vc->_seq.steal(r, steal_size) == false)
	return 0;

      // recompute the request count
      if ((size_t)r.size() != steal_size)
      {
	request_count = r.size() / unit_size;

	if ((request_count * (int)unit_size) < r.size())
	  ++request_count;
      }

      typename Sequence::iterator_type pos = r.begin();
      int reply_count = 0;
    
      // balanced workload amongst count thieves
      for (; request_count > 0; ++request)
      {
	if (!kaapi_request_ok(request))
	  continue;

	// pop no more than unit_size
	if (unit_size > (size_t)r.size())
	  unit_size = (size_t)r.size();

	kaapi_thread_t* thief_thread = kaapi_request_getthread(request);
	kaapi_task_t* thief_task = kaapi_thread_toptask(thief_thread);

	// allocate task stack
	task_context* tc = static_cast<task_context*>
	  (kaapi_thread_pushdata_align
	   (thief_thread, sizeof(task_context), 8));

	// allocate task result
	typedef typename reducer<ReduceTag>::template
	  thief_context<Result, Sequence, Body>
	  reducer_thiefcontext_type;

	kaapi_taskadaptive_result_t* const ktr =
	  reducer_thiefcontext_type::allocate
	  (sc, Sequence(pos, unit_size), tc->_body);

	reducer_thiefcontext_type* const rtc =
	  static_cast<reducer_thiefcontext_type*>(ktr->data);

	// initialize task stack
	new (tc) task_context
	  (rtc->_res, rtc->_seq, vc->_body, vc->_settings, sc, ktr);

	kastl_entry_t const thief_entry = outter_loop
	  <ReduceTag, UnrollTag>::template
	  thief_entry<Result, Sequence, Body, Settings>;

	kaapi_task_init(thief_task, thief_entry, tc);
	kaapi_thread_pushtask(thief_thread);
	kaapi_request_reply_head(sc, request, ktr);

	pos += unit_size;

	--request_count;
	++reply_count;
      }

      return reply_count;
    }
  }; // splitter

  // expand iterator, apply body
  template<size_t Count> struct expand_apply_t
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1);
    }
  };

  template<> struct expand_apply_t<2>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2);
    }
  };

  template<> struct expand_apply_t<3>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2, pos.ri3);
    }
  };

  template<> struct expand_apply_t<4>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2, pos.ri3, pos.ri4);
    }
  };

  template<typename Result, typename Iterator, typename Body>
  static bool expand_apply(Result& res, Iterator& pos, Body& body)
  {
    // syntaxic suggar for the above type
    return expand_apply_t<Iterator::iterator_count>::template
      expand_apply(res, pos, body);
  }

  // inner (nano) loop
  template<typename UnrollTag = rolled_tag>
  struct inner_loop
  {
    template<typename Result, typename Range, typename Body>
    static bool run(Result& res, Range& range, Body& body)
    {
      // default, rolled loop
      return body(res, range);
    }
  };

  template<>
  struct inner_loop<unrolled_tag>
  {
    template<typename Result, typename Range, typename Body>
    static bool run(Result& res, Range& range, Body& body)
    {
      // rely upon the first iterator to check for sequence
      // end but iterate with a full sequence iterator

      typename Range::iterator_type pos = range.begin();
      typename Range::iterator1_type end = range.end1();

      for (; pos.ri1 != end; ++pos)
	if (expand_apply(res, pos, body))
	  return true;

      return false;
    }
  };

  // first level, outter (macro) loop
  template<typename ReduceTag, typename UnrollTag>
  struct outter_loop
  {
    // loop entry point
    template<typename Extractor,
	     typename Result,
	     typename Sequence,
	     typename Body,
	     typename Settings>
    static void common_entry
    (kaapi_stealcontext_t* sc,
     kaapi_taskadaptive_result_t* ktr,
     Extractor& xtr,
     Result& res,
     Sequence& seq,
     Body& body,
     const Settings& settings)
    {
      typedef typename reducer<ReduceTag>::template
	thief_context<Result, Sequence, Body>
	thief_context_type;

      thief_context_type* tc = NULL;
      if (ktr != NULL)
	tc = static_cast<thief_context_type*>(ktr->data);

      // is_done is a reference to either ktr or
      bool is_done_storage = false;
      bool& is_done = (tc == NULL) ? is_done_storage : tc->_is_done;

      typedef typename Sequence::range_type range_type;
      range_type subr;

    redo_loop:
      while (xtr.extract(seq, subr))
      {
	is_done = inner_loop<UnrollTag>::template
	  run<Result, range_type, Body>(res, subr, body);

	// check if we have been preempted
	if (ktr != NULL)
	{
	  const bool is_preempted = reducer<ReduceTag>::preempt(sc, ktr);
	  if (is_preempted == true)
	    return ;
	}

	// empty sequence to prevent stealing
	if (is_done == true)
	{
	  seq.empty();
	  break;
	}
      }

    redo_reduce:
      // reduce the remaining thieves or return
      bool has_thief;
      reducer<ReduceTag>::template reduce<Result, Sequence, Body>
	(sc, has_thief, is_done, res, seq, body);
      if (has_thief == true)
      {
	// reduce until no more thief. if done,
	// loop on the redo_reduce label. if
	// not, redo the loop with the sequence
	// got from the thief.

	if (is_done == true)
	  goto redo_reduce;
	goto redo_loop;
      }

      // no more thief
    }

    // task entry points

    template<typename Result, typename Sequence, typename Body, typename Settings>
    static void thief_entry(void* arg, kaapi_thread_t* thread)
    {
      typedef typename splitter<Result, Sequence, Body, Settings>::
	task_context task_context;
      task_context* const tc = static_cast<task_context*>(arg);

      kastl_splitter_t const splitfn =
	splitter<Result, Sequence, Body, Settings>::template
	split<ReduceTag, UnrollTag>;

      kaapi_stealcontext_t* const sc = kaapi_thread_pushstealcontext
	(thread, KAAPI_STEALCONTEXT_DEFAULT, splitfn, arg, tc->_master_sc);

      extractor<static_tag> xtr(tc->_settings);
      outter_loop<ReduceTag, UnrollTag>::common_entry
	(sc, tc->_ktr, xtr, tc->_res, tc->_seq, tc->_body, tc->_settings);

      kaapi_steal_finalize(sc);
    }

    template<typename Context,
	     typename Result,
	     typename Sequence,
	     typename Body,
	     typename Settings>
    static void master_entry(void* arg, kaapi_thread_t* thread)
    {
      typedef splitter<Result, Sequence, Body, Settings> splitter_type;

      Context* const tc = static_cast<Context*>(arg);

      kastl_splitter_t const splitfn = splitter_type::template
	split<ReduceTag, UnrollTag>;

      kaapi_stealcontext_t* const sc = kaapi_thread_pushstealcontext
	(thread, KAAPI_STEALCONTEXT_DEFAULT, splitfn, arg, tc->_master_sc);

      extractor< typename Settings::_macro_extractor_tag > xtr(tc->_settings);
      outter_loop<ReduceTag, UnrollTag>::common_entry
	(sc, NULL, xtr, tc->_res, tc->_seq, tc->_body, tc->_settings);

      kaapi_steal_finalize(sc);
    }

    // exported entry point

    template<typename Result, typename Sequence, typename Body, typename Settings>
    static Result& run
    (Result& res, Sequence& seq, Body& body, const Settings& settings)
    {
      // create a context and call master_entry
      // note that result is inout and returned

      typedef splitter<Result, Sequence, Body, Settings> splitter_type;
      typedef typename splitter_type::task_context context_type;

      kastl_entry_t const entryfn = master_entry
	<context_type, Result, Sequence, Body, Settings>;

      kaapi_thread_t* thread;
      kaapi_task_t* task;
      kaapi_frame_t frame;

      context_type tc(res, seq, body, settings);

      thread = kaapi_self_thread();
      kaapi_thread_save_frame(thread, &frame);
      task = kaapi_thread_toptask(thread);
      kaapi_task_init(task, entryfn, static_cast<void*>(&tc));
      kaapi_thread_pushtask(thread);
      kaapi_sched_sync();
      kaapi_thread_restore_frame(thread, &frame);

      return res;
    }

    template<typename Sequence, typename Body, typename Settings>
    static void run
    (Sequence& seq, Body& body, const Settings& settings)
    {
      dummy_type res;
      run(res, seq, body, settings);
    }

  };

  // sugar
  typedef outter_loop<noreduce_tag, rolled_tag> parallel_loop;
  typedef outter_loop<reduce_tag, rolled_tag> reduce_loop;
  typedef outter_loop<noreduce_tag, unrolled_tag> unrolled_loop;
  typedef outter_loop<reduce_tag, unrolled_tag> reduce_unrolled_loop;


} // impl
} // kastl


#endif // _KASTL_LOOP_H_
