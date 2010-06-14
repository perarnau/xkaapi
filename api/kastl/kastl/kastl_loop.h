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

#ifndef KASTL_LOOP_H_INCLUDED
# define KASTL_LOOP_H_INCLUDED

#include <sys/types.h>
#include "kaapi.h"
#include "kastl_workqueue.h"
#include "kastl_sequences.h"


// missing decls
extern "C" void kaapi_set_workload(kaapi_processor_t*, kaapi_uint32_t);
extern "C" void kaapi_set_self_workload(kaapi_uint32_t);
extern "C" kaapi_processor_t* kaapi_stealcontext_kproc(kaapi_stealcontext_t*);
extern "C" kaapi_processor_t* kaapi_request_kproc(kaapi_request_t*);


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

  // extractor tag
  struct window_tag {};
  struct linear_tag {};
  struct backoff_tag {};
  struct reverse_tag {};
  struct identity_tag {};
  typedef dummy_tag static_tag;

  // result tag
  struct touchable_tag {};

  // operators
  template<typename T>
  struct add
  {
    T operator()(const T& lhs, const T& rhs)
    {
      return lhs + rhs;
    }
  };

  template<typename T>
  struct sub
  {
    T operator()(const T& lhs, const T& rhs)
    {
      return lhs - rhs;
    }
  };

  template<typename T>
  struct mul
  {
    T operator()(const T& lhs, const T& rhs)
    {
      return lhs * rhs;
    }
  };

  // predicates
  template<typename T>
  struct eq
  {
    bool operator()(const T& lhs, const T& rhs)
    {
      return lhs == rhs;
    }
  };

  template<typename Value>
  struct gt
  {
    bool operator()(const Value& lhs, const Value& rhs)
    {
      return lhs > rhs;
    }
  };

  template<typename Value>
  struct lt
  {
    bool operator()(const Value& lhs, const Value& rhs)
    {
      return lhs < rhs;
    }
  };

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
  typedef settings<identity_tag> identity_settings;

  // results
  template<typename Iterator>
  struct void_result
  {
    void_result(const Iterator&)
    {}
  };

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

  template<typename Iterator, bool InitValue>
  struct bool_result
  {
    bool _value;

    bool_result(bool value = InitValue)
      : _value(value)
    {}

    bool_result(const Iterator&)
      : _value(InitValue)
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

  // reduction implmentation.

  // reduce thief context, no termination
  template<typename Result, typename Sequence, typename Body,
	   bool TerminateTag = false,
	   bool ReduceTag = false>
  struct reduce_thief_context
  {
    typedef reduce_thief_context
    <Result, Sequence, Body, TerminateTag, ReduceTag>
    context_type;

    Result _res; // to remove
    Sequence _seq;

    reduce_thief_context(const Sequence& seq)
      : _seq(seq)
    {}
  };

  // reduce thief context, no termination but result
  template<typename Result, typename Sequence, typename Body>
  struct reduce_thief_context<Result, Sequence, Body, false, true>
  {
    typedef reduce_thief_context
    <Result, Sequence, Body, false, true>
    context_type;

    Sequence _seq;
    Result _res;

    reduce_thief_context(const Sequence& seq)
      : _seq(seq), _res(seq.begin1())
    {}
  };

  // reduce thief context, termination, result
  template<typename Result, typename Sequence, typename Body>
  struct reduce_thief_context<Result, Sequence, Body, true, true>
  {
    typedef reduce_thief_context
    <Result, Sequence, Body, true, true>
    context_type;

    Sequence _seq;
    Result _res;
    bool _is_done;

    reduce_thief_context(const Sequence& seq)
      : _seq(seq), _res(seq.begin1()), _is_done(false)
    {}
  };

  // reduce victim context, no termination, no result
  template<typename Result, typename Sequence, typename Body,
	   bool TerminateTag = false,
	   bool ReduceTag = false>
  struct reduce_victim_context
  {
    Sequence& _seq;
    Body& _body;

    reduce_victim_context(Result& res, Sequence& seq, Body& body)
      : _seq(seq), _body(body)
    {}
  };

  // reduce victim context, no termination, result
  template<typename Result, typename Sequence, typename Body>
  struct reduce_victim_context<Result, Sequence, Body, false, true>
  {
    Result& _res;
    Sequence& _seq;
    Body& _body;

    reduce_victim_context(Result& res, Sequence& seq, Body& body)
      : _res(res), _seq(seq), _body(body)
    {}
  };

  // reduce victim context, termination, result
  template<typename Result, typename Sequence, typename Body>
  struct reduce_victim_context
  <Result, Sequence, Body, true, true>
  {
    bool& _is_done;
    Result& _res;
    Sequence& _seq;
    Body& _body;

    reduce_victim_context
    (bool& is_done, Result& res, Sequence& seq, Body& body)
      : _is_done(is_done), _res(res), _seq(seq), _body(body)
    {}
  };

  // allocate a ktr
  template<typename Result, typename Sequence, typename Body,
	   bool TerminateTag,
	   bool ReduceTag>
  static kaapi_taskadaptive_result_t* allocate_ktr
  (kaapi_stealcontext_t* sc, const Sequence& seq, Body& body)
  {
    typedef reduce_thief_context<Result, Sequence, Body, TerminateTag, ReduceTag>
      context_type;

    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(sc, sizeof(context_type), NULL);
    new (ktr->data) context_type(seq);
    return ktr;
  }

  // reducer, no termination, no body defined reduction
  template<typename Result, typename Sequence, typename Body,
	   bool TerminateTag = false,
	   bool ReduceTag = false>
  struct reducer
  {
    static int reduce_function
    (kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
    {
      typedef typename Sequence::range_type range_type;

      typedef reduce_thief_context
	<Result, Sequence, Body, TerminateTag, ReduceTag>
	thief_context_type;

      typedef reduce_victim_context
	<Result, Sequence, Body, TerminateTag, ReduceTag>
	victim_context_type;

      victim_context_type* const vc = static_cast<victim_context_type*>(vptr);
      thief_context_type* const tc = static_cast<thief_context_type*>(tptr);

      // join sequences
      kastl::rts::range_t<64> range(tc->_seq._wq._beg, tc->_seq._wq._end);
      vc->_seq._rep = tc->_seq._rep;
      vc->_seq._wq.set(range);

      return 0; // false, continue
    }

    static void reduce
    (kaapi_stealcontext_t* sc, bool& has_thief,
     Result& res, Sequence& seq, Body& body)
    {
      typedef reduce_victim_context
	<Result, Sequence, Body, TerminateTag, ReduceTag>
	victim_context_type;

      has_thief = true;

      // no more thief, we are done
      kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
      if (ktr == NULL)
      {
	has_thief = false;
	return ;
      }

      victim_context_type vc(res, seq, body);
      kaapi_preempt_thief(sc, ktr, NULL, reduce_function, &vc);
    }
  }; // reducer<false, false>

  // reducer, no termination, body defined reduction
  template<typename Result, typename Sequence, typename Body>
  struct reducer<Result, Sequence, Body, false, true>
  {
    static int reduce_function
    (kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
    {
      typedef typename Sequence::range_type range_type;

      typedef reduce_thief_context
	<Result, Sequence, Body, false, true>
	thief_context_type;

      typedef reduce_victim_context
	<Result, Sequence, Body, false, true>
	victim_context_type;

      victim_context_type* const vc = static_cast<victim_context_type*>(vptr);
      thief_context_type* const tc = static_cast<thief_context_type*>(tptr);

      // reduce results
      vc->_body.reduce(vc->_res, tc->_res);

      // join sequences
      kastl::rts::range_t<64> range(tc->_seq._wq._beg, tc->_seq._wq._end);
      vc->_seq._rep = tc->_seq._rep;
      vc->_seq._wq.set(range);

      return 0;
    }

    static void reduce
    (kaapi_stealcontext_t* sc, bool& has_thief,
     Result& res, Sequence& seq, Body& body)
    {
      typedef reduce_victim_context
	<Result, Sequence, Body, false, true>
	victim_context_type;

      has_thief = true;

      // no more thief, we are done
      kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
      if (ktr == NULL)
      {
	has_thief = false;
	return ;
      }

      victim_context_type vc(res, seq, body);
      kaapi_preempt_thief(sc, ktr, NULL, reduce_function, &vc);
    }
  }; // reducer<false, true>

  // reducer, termination, body defined reduction
  template<typename Result, typename Sequence, typename Body, bool ReduceTag>
  struct reducer<Result, Sequence, Body, true, ReduceTag>
  {
    static int reduce_function
    (kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
    {
      typedef typename Sequence::range_type range_type;

      typedef reduce_thief_context
	<Result, Sequence, Body, true, ReduceTag>
	thief_context_type;

      typedef reduce_victim_context
	<Result, Sequence, Body, true, ReduceTag>
	victim_context_type;

      victim_context_type* const vc = static_cast<victim_context_type*>(vptr);
      thief_context_type* const tc = static_cast<thief_context_type*>(tptr);

      // if we are done, do not retrieve
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

    static void reduce
    (kaapi_stealcontext_t* sc,
     bool& has_thief, bool& is_done,
     Result& res, Sequence& seq, Body& body)
    {
      typedef reduce_victim_context
	<Result, Sequence, Body, true, ReduceTag>
	victim_context_type;

      has_thief = true;

      // no more thief, we are done
      kaapi_taskadaptive_result_t* const ktr = kaapi_get_thief_head(sc);
      if (ktr == NULL)
      {
	has_thief = false;
	return ;
      }

      victim_context_type vc(is_done, res, seq, body);
      kaapi_preempt_thief(sc, ktr, NULL, reduce_function, &vc);
    }
  }; // reducer<true, xxx>

  // preemption
  static bool preempt
  (kaapi_stealcontext_t* sc, kaapi_taskadaptive_result_t* ktr)
  {
    const int is_preempted = kaapi_preemptpoint
      (ktr, sc, NULL, NULL, NULL, 0, NULL);
    return (bool)is_preempted;
  }

  // forward decls
  template<typename Result, typename Sequence, typename Body, typename Settings,
	   bool TerminateTag, bool ReduceTag>
  static void thief_entry(void*, kaapi_thread_t*);

  // split victim context, result
  template
  <typename Result, typename Sequence, typename Body, typename Settings,
   bool ReduceTag = true>
  struct split_victim_context
  {
    Result _res;
    Sequence& _seq;
    Body& _body;
    const Settings& _settings;
  };

  // split victim context, no result
  template
  <typename Result, typename Sequence, typename Body, typename Settings>
  struct split_victim_context<Result, Sequence, Body, Settings, false>
  {
    Sequence& _seq;
    Body& _body;
    const Settings& _settings;
  };

  // task context, result
  template
  <typename Result, typename Sequence, typename Body, typename Settings,
   bool ReduceTag = true>
  struct split_task_context
  {
    typedef Result result_type;
    typedef Sequence sequence_type;
    typedef Body body_type;
    typedef Settings settings_type;

    Result& _res;
    Sequence& _seq;
    Body _body;
    const Settings& _settings;
    kaapi_stealcontext_t* _master_sc;
    kaapi_taskadaptive_result_t* _ktr;

    split_task_context
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

  template
  <typename Result, typename Sequence, typename Body, typename Settings>
  struct split_task_context<Result, Sequence, Body, Settings, false>
  {
    typedef Sequence sequence_type;
    typedef Body body_type;
    typedef Settings settings_type;

    // passed upon thief entry

    Result _res; // to remove
    Sequence& _seq;
    Body _body;
    const Settings& _settings;
    kaapi_stealcontext_t* _master_sc;
    kaapi_taskadaptive_result_t* _ktr;

    split_task_context
    (Result& , // to remove
     Sequence& seq,
     Body& body,
     const Settings& settings,
     kaapi_stealcontext_t* master_sc = NULL,
     kaapi_taskadaptive_result_t* ktr = NULL)
      : _seq(seq), _body(body), _settings(settings),
	_master_sc(master_sc), _ktr(ktr)
    {}
  };

  // splitter function
  template
  <typename Result, typename Sequence, typename Body,
   typename Settings, bool TerminateTag, bool ReduceTag>
  static int split_function
  (kaapi_stealcontext_t* sc, int request_count,
   kaapi_request_t* request, void* arg)
  {
    typedef typename Sequence::range_type range_type;
    typedef split_task_context
      <Result, Sequence, Body, Settings, ReduceTag> context_type;

    // victim task context
    context_type* const vc = static_cast<context_type*>(arg);

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

    size_t vseq_size = vc->_seq.size();
    if (vseq_size < vc->_settings._par_size)
      vseq_size = 0;
    kaapi_set_workload(kaapi_stealcontext_kproc(sc), vseq_size);

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
      context_type* tc = static_cast<context_type*>
	(kaapi_thread_pushdata_align
	 (thief_thread, sizeof(context_type), 8));

      // allocate task result
      typedef reduce_thief_context
	<Result, Sequence, Body, TerminateTag, ReduceTag>
	thief_context_type;

      kaapi_taskadaptive_result_t* const ktr =
	allocate_ktr<Result, Sequence, Body, TerminateTag, ReduceTag>
	(sc, Sequence(pos, unit_size), tc->_body);

      thief_context_type* const rtc =
	static_cast<thief_context_type*>(ktr->data);

      // initialize task stack
      new (tc) context_type
	(rtc->_res, rtc->_seq, vc->_body, vc->_settings, sc, ktr);

      kastl_entry_t const entryfn = thief_entry
	<Result, Sequence, Body, Settings, TerminateTag, ReduceTag>;

      kaapi_task_init(thief_task, entryfn, tc);
      kaapi_thread_pushtask(thief_thread);
      kaapi_request_reply_head(sc, request, ktr);

      kaapi_set_workload(kaapi_request_kproc(request), unit_size);

      pos += unit_size;

      --request_count;
      ++reply_count;
    }

    return reply_count;
  } // split_function

  // expand iterator, apply body
  template<bool TerminateTag = true, unsigned int Count = 1>
  struct expand_apply_t
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1);
    }
  };

  template<>
  struct expand_apply_t<false, 1>
  {
    template<typename Result, typename Iterator, typename Body>
    static void expand_apply(Result& res, Iterator& pos, Body& body)
    {
      body(res, pos.ri1);
    }
  };

  template<> struct expand_apply_t<true, 2>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2);
    }
  };

  template<> struct expand_apply_t<false, 2>
  {
    template<typename Result, typename Iterator, typename Body>
    static void expand_apply(Result& res, Iterator& pos, Body& body)
    {
      body(res, pos.ri1, pos.ri2);
    }
  };

  template<> struct expand_apply_t<true, 3>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2, pos.ri3);
    }
  };

  template<> struct expand_apply_t<false, 3>
  {
    template<typename Result, typename Iterator, typename Body>
    static void expand_apply(Result& res, Iterator& pos, Body& body)
    {
      body(res, pos.ri1, pos.ri2, pos.ri3);
    }
  };

  template<> struct expand_apply_t<true, 4>
  {
    template<typename Result, typename Iterator, typename Body>
    static bool expand_apply(Result& res, Iterator& pos, Body& body)
    {
      return body(res, pos.ri1, pos.ri2, pos.ri3, pos.ri4);
    }
  };

  template<> struct expand_apply_t<false, 4>
  {
    template<typename Result, typename Iterator, typename Body>
    static void expand_apply(Result& res, Iterator& pos, Body& body)
    {
      body(res, pos.ri1, pos.ri2, pos.ri3, pos.ri4);
    }
  };

  // suggar for the above types
  template<typename Result, typename Iterator, typename Body>
  static void expand_apply_noterm(Result& res, Iterator& pos, Body& body)
  {
    expand_apply_t<false, Iterator::iterator_count>::template
      expand_apply(res, pos, body);
  }

  template<typename Result, typename Iterator, typename Body>
  static bool expand_apply(Result& res, Iterator& pos, Body& body)
  {
    return expand_apply_t<true, Iterator::iterator_count>::template
      expand_apply(res, pos, body);
  }

  // inner loop, no termination
  template<typename Result, typename Range, typename Body,
	   bool TerminateTag = false>
  struct inner_loop
  {
    static void run(Result& res, Range& range, Body& body)
    {
      typename Range::iterator_type pos = range.begin();
      typename Range::iterator1_type end = range.end1();

      for (; pos.ri1 != end; ++pos)
	expand_apply_noterm(res, pos, body);
    }
  };

  // inner loop, termination
  template<typename Result, typename Range, typename Body>
  struct inner_loop<Result, Range, Body, true>
  {
    static bool run(Result& res, Range& range, Body& body)
    {
      // rely upon the ri1 to check for sequence end
      // but iterate with a whole sequence iterator

      typename Range::iterator_type pos = range.begin();
      typename Range::iterator1_type end = range.end1();

      for (; pos.ri1 != end; ++pos)
	if (expand_apply(res, pos, body))
	  return true;

      return false;
    }
  };

  // outter loop, no termination.
  template<typename Extractor,
	   typename Result, typename Sequence, typename Body,
	   typename Settings,
	   bool TerminateTag, bool ReduceTag>
  struct outter_loop
  {
    static void run
    (kaapi_stealcontext_t* sc,
     kaapi_taskadaptive_result_t* ktr,
     Extractor& xtr,
     Result& res,
     Sequence& seq,
     Body& body,
     const Settings& settings)
    {
      // more comments in the below version

      typedef reducer<Result, Sequence, Body, TerminateTag, ReduceTag>
	reducer_type;

      typedef reduce_thief_context
	<Result, Sequence, Body, TerminateTag, ReduceTag>
	thief_context_type;

      typedef typename Sequence::range_type range_type;

      typedef inner_loop<Result, range_type, Body, TerminateTag>
	inner_loop_type;

      thief_context_type* tc = NULL;
      if (ktr != NULL)
	tc = static_cast<thief_context_type*>(ktr->data);

      range_type subr;

    redo_loop:
      while (xtr.extract(seq, subr))
      {
	inner_loop_type::run(res, subr, body);

	// check if we have been preempted
	if (ktr != NULL)
	{
	  const bool is_preempted = preempt(sc, ktr);
	  if (is_preempted == true)
	    return ;
	}
      }

      bool has_thief;
      reducer_type::reduce(sc, has_thief, res, seq, body);
      if (has_thief == true)
	goto redo_loop;

      // no more thief
    }
  }; // outter_loop<false>

  // outter loop, termination specialization.
  template<typename Extractor,
	   typename Result, typename Sequence, typename Body,
	   typename Settings,
	   bool ReduceTag>
  struct outter_loop
  <Extractor, Result, Sequence, Body, Settings, true, ReduceTag>
  {
    static void run
    (kaapi_stealcontext_t* sc,
     kaapi_taskadaptive_result_t* ktr,
     Extractor& xtr,
     Result& res,
     Sequence& seq,
     Body& body,
     const Settings& settings)
    {
      typedef reducer<Result, Sequence, Body, true, ReduceTag>
	reducer_type;

      typedef reduce_thief_context
	<Result, Sequence, Body, true, ReduceTag>
	thief_context_type;

      typedef typename Sequence::range_type range_type;

      typedef inner_loop<Result, range_type, Body, true>
	inner_loop_type;

      thief_context_type* tc = NULL;
      if (ktr != NULL)
	tc = static_cast<thief_context_type*>(ktr->data);

      // is_done is a reference to either ktr or
      bool is_done_storage = false;
      bool& is_done = (tc == NULL) ? is_done_storage : tc->_is_done;

      range_type subr;

    redo_loop:
      while (xtr.extract(seq, subr))
      {
	is_done = inner_loop_type::run(res, subr, body);

	// check if we have been preempted
	if (ktr != NULL)
	{
	  const bool is_preempted = preempt(sc, ktr);
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
      reducer_type::reduce(sc, has_thief, is_done, res, seq, body);
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
  }; // outter_loop<true>

  // task entry points
  template<typename Result, typename Sequence, typename Body, typename Settings,
	   bool TerminateTag, bool ReduceTag>
  static void thief_entry(void* arg, kaapi_thread_t* thread)
  {
    typedef split_task_context
      <Result, Sequence, Body, Settings, ReduceTag> context_type;

    typedef extractor<static_tag> xtr_type;

    typedef outter_loop
      <xtr_type, Result, Sequence, Body, Settings, TerminateTag, ReduceTag>
      outter_loop_type;

    context_type* const tc = static_cast<context_type*>(arg);

    kastl_splitter_t const splitfn = split_function
      <Result, Sequence, Body, Settings, TerminateTag, ReduceTag>;

    kaapi_stealcontext_t* const sc = kaapi_thread_pushstealcontext
      (thread, KAAPI_STEALCONTEXT_DEFAULT, splitfn, arg, tc->_master_sc);

    xtr_type xtr(tc->_settings);
    outter_loop_type::run
      (sc, tc->_ktr, xtr, tc->_res, tc->_seq, tc->_body, tc->_settings);

    kaapi_set_self_workload(0);

    kaapi_steal_finalize(sc);
  }

  template
  <typename Result, typename Sequence, typename Body, typename Settings,
   bool TerminateTag, bool ReduceTag>
  static void master_entry(void* arg, kaapi_thread_t* thread)
  {
    typedef split_task_context
      <Result, Sequence, Body, Settings, ReduceTag> context_type;
    typedef extractor<typename Settings::_macro_extractor_tag> xtr_type;

    typedef outter_loop
      <xtr_type, Result, Sequence, Body, Settings, TerminateTag, ReduceTag>
      outter_loop_type;

    context_type* const tc = static_cast<context_type*>(arg);

    kastl_splitter_t const splitfn = split_function
      <Result, Sequence, Body, Settings, TerminateTag, ReduceTag>;

    kaapi_stealcontext_t* const sc = kaapi_thread_pushstealcontext
      (thread, KAAPI_STEALCONTEXT_DEFAULT, splitfn, arg, tc->_master_sc);

    xtr_type xtr(tc->_settings);
    outter_loop_type::run
      (sc, NULL, xtr, tc->_res, tc->_seq, tc->_body, tc->_settings);

    kaapi_steal_finalize(sc);
  }

  // exported entry points
  template
  <typename Result, typename Sequence, typename Body, typename Settings,
   bool TerminateTag, bool ReduceTag>
  static Result& generic_loop
  (Result& res, Sequence& seq, Body& body, const Settings& settings)
  {
    // create a context and call master_entry
    // note that result is inout and returned

    typedef split_task_context
      <Result, Sequence, Body, Settings, ReduceTag> context_type;

    kastl_entry_t const entryfn = master_entry
      <Result, Sequence, Body, Settings, TerminateTag, ReduceTag>;

    kaapi_thread_t* thread;
    kaapi_task_t* task;
    kaapi_frame_t frame;

    kaapi_set_self_workload(seq.size());

    context_type tc(res, seq, body, settings);

    thread = kaapi_self_thread();
    kaapi_thread_save_frame(thread, &frame);
    task = kaapi_thread_toptask(thread);
    kaapi_task_init(task, entryfn, static_cast<void*>(&tc));
    kaapi_thread_pushtask(thread);
    kaapi_sched_sync();
    kaapi_thread_restore_frame(thread, &frame);

    kaapi_set_self_workload(0);

    return res;
  }

  // exported, generic loop selectors
  template<typename Result, typename Sequence, typename Body, typename Settings>
  static inline Result& foreach_loop
  (Result& res, Sequence& seq, Body& body, const Settings& settings)
  {
    return generic_loop<Result, Sequence, Body, Settings, false, false>
      (res, seq, body, settings);
  }

  template<typename Sequence, typename Body, typename Settings>
  static inline void foreach_loop
  (Sequence& seq, Body& body, const Settings& settings)
  {
    dummy_type res;
    foreach_loop(res, seq, body, settings);
  }

  template<typename Result, typename Sequence, typename Body, typename Settings>
  static inline Result& foreach_reduce_loop
  (Result& res, Sequence& seq, Body& body, const Settings& settings)
  {
    return generic_loop<Result, Sequence, Body, Settings, false, true>
      (res, seq, body, settings);
  }

  template<typename Sequence, typename Body, typename Settings>
  static inline void foreach_reduce_loop
  (Sequence& seq, Body& body, const Settings& settings)
  {
    dummy_type res;
    foreach_reduce_loop(res, seq, body, settings);
  }

  template<typename Result, typename Sequence, typename Body, typename Settings>
  static inline Result& while_loop
  (Result& res, Sequence& seq, Body& body, const Settings& settings)
  {
    return generic_loop<Result, Sequence, Body, Settings, true, false>
      (res, seq, body, settings);
  }

  template<typename Sequence, typename Body, typename Settings>
  static inline void while_loop
  (Sequence& seq, Body& body, const Settings& settings)
  {
    dummy_type res;
    while_loop(res, seq, body, settings);
  }

  template<typename Result, typename Sequence, typename Body, typename Settings>
  static inline Result& while_reduce_loop
  (Result& res, Sequence& seq, Body& body, const Settings& settings)
  {
    return generic_loop<Result, Sequence, Body, Settings, true, true>
      (res, seq, body, settings);
  }

  template<typename Sequence, typename Body, typename Settings>
  static inline void while_reduce_loop
  (Sequence& seq, Body& body, const Settings& settings)
  {
    dummy_type res;
    while_reduce_loop(res, seq, body, settings);
  }

} // impl
} // kastl


#endif // KASTL_LOOP_H_INCLUDED
