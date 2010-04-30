#ifndef  KASTL_IMPL_HH_INCLUDED
# define  KASTL_IMPL_HH_INCLUDED



#include <algorithm>
#include <sys/types.h>
#include "kaapi.h"


#if KASTL_DEBUG
extern "C" unsigned int kaapi_get_current_kid(void);
#endif



namespace kastl
{
namespace impl
{
  // predicates

  template<typename T>
  struct isEqualPredicate
  {
    T _ref_value;

    isEqualPredicate(const T& ref_value)
      : _ref_value(ref_value) {}

    inline bool operator()(const T& v)
    {
      return v == _ref_value;
    }
  };


  template<typename Ta, typename Tb>
  static bool isEqualBinaryPredicate(const Ta& a, const Tb& b)
  { return a == b; }


  // sequence types

  template<typename IteratorType>
  struct BasicSequence
  {
    typedef BasicSequence<IteratorType> SequenceType;
    typedef typename std::iterator_traits<IteratorType>::difference_type SizeType;
    typedef IteratorType _IteratorType;

    IteratorType _beg;
    IteratorType _end;

    inline BasicSequence() {}

    inline BasicSequence
    (const IteratorType& beg, const IteratorType& end)
      : _beg(beg), _end(end) {}

    inline IteratorType begin() const
    {
      return _beg;
    }

    inline IteratorType end() const
    {
      return _end;
    }

    inline SizeType size() const
    {
      return std::distance(_beg, _end);
    }

    inline void split(SequenceType& seq, SizeType size)
    {
      seq._end += size;
      _beg += size;
    }

    inline void rsplit(SequenceType& seq, SizeType size)
    {
      seq = SequenceType(_end - size, _end);
      _end -= size;
    }

    inline void empty_seq(SequenceType& seq) const
    {
      seq._beg = _beg;
      seq._end = _beg;
    }

    inline bool is_empty() const
    {
      return _beg == _end;
    }

#if KASTL_DEBUG
    typedef std::pair<SizeType, SizeType> RangeType;
    static RangeType get_range(const SequenceType& a, const SequenceType& b)
    {
      RangeType r(std::distance(a._beg, b._beg), std::distance( a._beg, b._end));
      return r;
    }
#endif

  };


  template<typename Iterator0Type, typename Iterator1Type>
  struct In2EqSizedSequence
  {
    typedef In2EqSizedSequence<Iterator0Type, Iterator1Type>
    SequenceType;

    typedef Iterator0Type _Iterator0Type;
    typedef Iterator1Type _Iterator1Type;

    typedef typename std::iterator_traits<Iterator0Type>::difference_type 
   SizeType;

    BasicSequence<Iterator0Type> _seq0;
    Iterator1Type _beg1;

    inline In2EqSizedSequence() {}

    inline In2EqSizedSequence
    (
     const Iterator0Type& beg0,
     const Iterator0Type& end0,
     const Iterator1Type& beg1
     ) : _seq0(BasicSequence<Iterator0Type>(beg0, end0)), _beg1(beg1) {}

    inline SizeType size() const
    {
      return _seq0.size();
    }

    inline void split(SequenceType& seq, SizeType size)
    {
      _seq0.split(seq._seq0, size);
      _beg1 += size;
    }

    inline void rsplit(SequenceType& seq, SizeType size)
    {
      _seq0.rsplit(seq._seq0, size);
      seq._beg1 = _beg1 + std::distance(_seq0._beg, seq._seq0._beg);
    }

    inline void empty_seq(SequenceType& seq) const
    {
      _seq0.empty_seq(seq._seq0);
      seq._beg1 = _beg1;
    }

    inline bool is_empty() const
    {
      return _seq0.is_empty();
    }

#if KASTL_DEBUG
    typedef typename BasicSequence<Iterator0Type>::RangeType RangeType;
    static RangeType get_range(const SequenceType& a, const SequenceType& b)
    {
      return BasicSequence<Iterator0Type>::get_range(a._seq0, b._seq0);
    }
#endif

  };


  template
  <
    typename InputIterator0Type,
    typename InputIterator1Type,
    typename OutputIteratorType
  >
  struct In2OutSequence
  {
    // 2 input sequences, sizes not eq

    typedef InputIterator0Type _InputIterator0Type;
    typedef InputIterator1Type _InputIterator1Type;
    typedef OutputIteratorType _OutputIteratorType;

    typedef In2OutSequence
    <InputIterator0Type, InputIterator1Type, OutputIteratorType>
    SequenceType;

    typedef typename std::iterator_traits<InputIterator0Type>::difference_type
    SizeType;

    BasicSequence<InputIterator0Type> _iseq0;
    BasicSequence<InputIterator1Type> _iseq1;
    OutputIteratorType _obeg;

    inline In2OutSequence() {}

    inline In2OutSequence
    (
     const InputIterator0Type& beg0,
     const InputIterator0Type& end0,
     const InputIterator1Type& beg1,
     const InputIterator1Type& end1,
     const OutputIteratorType& obeg
     )
      : _iseq0(BasicSequence<InputIterator0Type>(beg0, end0)),
	_iseq1(BasicSequence<InputIterator1Type>(beg1, end1)),
	_obeg(obeg) {}

    inline SizeType size() const
    {
      return std::max(_iseq0.size(), _iseq1.size());
    }

    inline void split(SequenceType& seq, SizeType size)
    {
      SizeType size0 = size;
      SizeType size1 = size;

      if ((size == _iseq0.size()) || (size == _iseq1.size()))
      {
	size0 = _iseq0.size();
	size1 = _iseq1.size();
      }

      _iseq0.split(seq._iseq0, size0);
      _iseq1.split(seq._iseq1, size1);

      _obeg += size0 + size1;
    }

    inline void rsplit(SequenceType& seq, SizeType size)
    {
      const SizeType iseq0_size = _iseq0.size();
      const SizeType iseq1_size = _iseq1.size();

      SizeType split0_size = size;
      SizeType split1_size = size;

      if ((size == iseq0_size) || (size == iseq1_size))
      {
	split0_size = iseq0_size;
	split1_size = iseq1_size;
      }

      const SizeType iseq0_off = iseq0_size - split0_size;
      const SizeType iseq1_off = iseq1_size - split1_size;

      _iseq0.rsplit(seq._iseq0, split0_size);
      _iseq1.rsplit(seq._iseq1, split1_size);

      seq._obeg = _obeg + (iseq0_off + iseq1_off);
    }

    inline void empty_seq(SequenceType& seq) const
    {
      _iseq0.empty_seq(seq._iseq0);
      _iseq1.empty_seq(seq._iseq1);

      seq._obeg = _obeg;
    }

    inline bool is_empty() const
    {
      return _iseq0.is_empty();
    }

#if KASTL_DEBUG
    typedef typename BasicSequence<InputIterator0Type>::RangeType RangeType;
    static RangeType get_range(const SequenceType& a, const SequenceType& b)
    {
      return BasicSequence<InputIterator0Type>::get_range(a._iseq0, b._iseq0);
    }
#endif

  };


  template
  <
    typename InputIterator0Type,
    typename InputIterator1Type,
    typename OutputIteratorType,
    typename ComparatorType
  >
  class OrderedSequence
  {
  public:

    typedef OrderedSequence
    <
    InputIterator0Type,
    InputIterator1Type,
    OutputIteratorType,
    ComparatorType
    >
    SequenceType;

    typedef typename std::iterator_traits
    <InputIterator0Type>::value_type InputValue0Type;

    typedef typename std::iterator_traits
    <InputIterator1Type>::value_type InputValue1Type;

    typedef typename std::iterator_traits
    <OutputIteratorType>::value_type OutputValueType;

    typedef typename std::iterator_traits
    <InputIterator0Type>::difference_type SizeType;

    typedef OutputIteratorType OutputIterator;

    BasicSequence<InputIterator0Type> _iseq0;
    BasicSequence<InputIterator1Type> _iseq1;
    OutputIteratorType _obeg;
    ComparatorType _comp;

    OrderedSequence() {}

    OrderedSequence
    (
     const InputIterator0Type& ipos0,
     const InputIterator0Type& iend0,
     const InputIterator1Type& ipos1,
     const InputIterator1Type& iend1,
     const OutputIteratorType& obeg,
     const ComparatorType& comp
    )
    {
      _iseq0 = BasicSequence<InputIterator0Type>(ipos0, iend0);
      _iseq1 = BasicSequence<InputIterator1Type>(ipos1, iend1);
      _obeg = obeg;
      _comp = comp;
    }

    inline SizeType size() const
    {
      if (_iseq0.size() > _iseq1.size())
	return (SizeType)_iseq0.size();
      return (SizeType)_iseq1.size();
    }

    inline void empty_seq(SequenceType& seq) const
    {
      _iseq0.empty_seq(seq._iseq0);
      _iseq1.empty_seq(seq._iseq1);
    }

    inline bool is_empty() const
    {
      return _iseq0.is_empty() && _iseq1.is_empty();
    }

    static InputIterator1Type find_with_pred
    (
     InputIterator1Type pos,
     InputIterator1Type end,
     const InputValue0Type& value,
     const ComparatorType& comp
    )
    {
      // find greater or eq assuming sorted(pos,end)

      for (; (pos != end) && comp(*pos, value); ++pos)
	;

      return pos;
    }

    template<typename SequenceType>
    inline static void swap_if_greater(SequenceType& a, SequenceType& b)
    {
      if (a.size() > b.size())
	std::swap(a, b);
    }

    void rsplit(SequenceType& oseq, size_t size)
    {
      // size is the size of the largest sequence
      swap_if_greater(_iseq1, _iseq0);

      InputIterator0Type mid0 = _iseq0._beg + size - 1;
      InputIterator1Type mid1 =	find_with_pred
	(_iseq1._beg, _iseq1._end, *mid0, _comp);

      // give [mid, end[
      oseq._iseq0 = BasicSequence<InputIterator0Type>(mid0, _iseq0._end);
      oseq._iseq1 = BasicSequence<InputIterator1Type>(mid1, _iseq1._end);
      oseq._obeg = _obeg + std::distance(_iseq0._beg, mid0) + std::distance(_iseq1._beg, mid1);

      // keep [beg, mid[
      _iseq0._end = mid0;
      _iseq1._end = mid1;
    }

    void split(SequenceType& oseq, size_t size)
    {
      // size is the size of the largest sequence
      swap_if_greater(_iseq1, _iseq0);

      InputIterator0Type mid0 = _iseq0._beg + size;
      InputIterator1Type mid1 =	find_with_pred
	(_iseq1._beg, _iseq1._end, *(mid0 - 1), _comp);

      // give [beg, mid]
      oseq._obeg = _obeg;
      oseq._iseq0 = BasicSequence<InputIterator0Type>(_iseq0._beg, mid0);
      oseq._iseq1 = BasicSequence<InputIterator1Type>(_iseq1._beg, mid1);

      // keep ]mid, end[
      _obeg += std::distance(_iseq0._beg, mid0) + std::distance(_iseq1._beg, mid1);
      _iseq0 = BasicSequence<InputIterator0Type>(mid0, _iseq0._end);
      _iseq1 = BasicSequence<InputIterator1Type>(mid1, _iseq1._end);
    }

#if KASTL_DEBUG
    typedef typename BasicSequence<InputIterator0Type>::RangeType RangeType;
    static RangeType get_range(const SequenceType& a, const SequenceType& b)
    {
      return BasicSequence<InputIterator0Type>::get_range(a._iseq0, b._iseq0);
    }
#endif

  };

  template
  <
    typename InputIteratorType,
    typename OutputIteratorType
  >
  struct InOutSequence
  {
    // _opos is synchronized with _iseq._beg

    typedef InOutSequence<InputIteratorType, OutputIteratorType> SequenceType;

    typedef InputIteratorType _InputIteratorType;
    typedef OutputIteratorType _OutputIteratorType;

    typedef typename std::iterator_traits
    <InputIteratorType>::difference_type SizeType;

    BasicSequence<InputIteratorType> _iseq;
    OutputIteratorType _opos;

    inline InOutSequence() {}

    inline InOutSequence
    (
     const InputIteratorType& l,
     const InputIteratorType& h,
     const OutputIteratorType& opos
    )
    {
      _iseq = BasicSequence<InputIteratorType>(l, h);
      _opos = opos;
    }

    inline SequenceType& operator=(const SequenceType& s)
    {
      _iseq = s._iseq;
      _opos = s._opos;
      return *this;
    }

    inline SizeType size() const
    { return _iseq.size(); }

    static inline void _sync_opos
    (OutputIteratorType& opos, SizeType size)
    { opos += size; }

    static inline void _sync_opos
    (
     OutputIteratorType& opos,
     const InputIteratorType& old_pos,
     const InputIteratorType& new_pos
    )
    { _sync_opos(opos, std::distance(old_pos, new_pos)); }

    inline void split(SequenceType& seq, SizeType size)
    {
      _iseq.split(seq._iseq, size);
      seq._opos = _opos;
      _opos += size;
    }

    inline void rsplit(SequenceType& seq, SizeType size)
    {
      _iseq.rsplit(seq._iseq, size);
      seq._opos = _opos;
      _sync_opos(seq._opos, _iseq._beg, seq._iseq._beg);
    }

    inline void empty_seq(SequenceType& seq) const
    {
      _iseq.empty_seq(seq._iseq);
    }

    inline bool is_empty() const
    {
      return _iseq._beg == _iseq._end;
    }

#if KASTL_DEBUG
    typedef std::pair<SizeType, SizeType> RangeType;
    static RangeType get_range(const SequenceType& a, const SequenceType& b)
    {
      typedef BasicSequence<InputIteratorType> BasicSequenceType;
      return BasicSequenceType::get_range(a._iseq, b._iseq);
    }
#endif

  };


  // extractor types

  // . macro extracts an instruction sequence
  // that can be processed in parallel
  // . nano extracts an instruction sequence
  // that can only be processed sequentially,
  // ie. there is no possible stealing

  template<typename SequenceType>
  struct BaseExtractor
  {
    // extractors may inherit from this one
    inline void prepare(SequenceType&, const SequenceType&) {}
    inline void extract(SequenceType&, SequenceType&) {}
  };


  template<typename SequenceType>
  struct IdentityExtractor : public BaseExtractor<SequenceType>
  {
    // the identity macro extractor returns
    // the whole sequence when called, ie.
    // for algorithms that dont need macro
    // steps (typically the ones processing
    // the whole sequence [but not prefixes])

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      if (src_seq.is_empty())
	return false;

      dst_seq = src_seq;
      dst_seq.empty_seq(src_seq);

      return true;
    }
  };


  template<typename SequenceType, size_t MinSize, size_t MaxSize>
  struct BackoffExtractor : public BaseExtractor<SequenceType>
  {
    typedef typename SequenceType::SizeType SizeType;

    SizeType _backoff_size;

    BackoffExtractor()
      : _backoff_size((SizeType)MinSize) {}

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      if (src_seq.is_empty())
	return false;

      const SizeType size = std::min(src_seq.size(), _backoff_size);
      src_seq.split(dst_seq, size);

      if (_backoff_size < (SizeType)MaxSize)
      {
	_backoff_size *= 2;
	if (_backoff_size > (SizeType)MaxSize)
	  _backoff_size = (SizeType)MaxSize;
      }

      return true;
    }
  };


  template<typename SequenceType, size_t MinSize, size_t MaxSize, size_t StepSize>
  struct LinearExtractor : public BaseExtractor<SequenceType>
  {
    typedef typename SequenceType::SizeType SizeType;

    SizeType _macro_size;

    LinearExtractor()
      : _macro_size((SizeType)MinSize) {}

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      if (src_seq.is_empty())
	return false;

      const SizeType size = std::min(src_seq.size(), _macro_size);
      src_seq.split(dst_seq, size);

      if (_macro_size == (SizeType)MaxSize)
	return true;

      _macro_size += StepSize;
      if (_macro_size > (SizeType)MaxSize)
	_macro_size = (SizeType)MaxSize;

      return true;
    }
  };


  template<typename SequenceType, unsigned int UnitSize>
  struct StaticExtractor : public BaseExtractor<SequenceType>
  {
    typedef typename SequenceType::SizeType SizeType;

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      if (src_seq.is_empty())
	return false;

      const SizeType size = std::min((SizeType)src_seq.size(), (SizeType)UnitSize);

      src_seq.split(dst_seq, size);

      return true;
    }
  };

  template<typename SequenceType, unsigned int UnitSize>
  struct MergeExtractor : public BaseExtractor<SequenceType>
  {
    typedef typename SequenceType::SizeType SizeType;

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      const SizeType split0_size =
	std::min((SizeType)UnitSize - dst_seq._iseq0.size(),
		 src_seq._iseq0.size());

      const SizeType split1_size =
	std::min((SizeType)UnitSize - dst_seq._iseq1.size(),
		 src_seq._iseq1.size());

      src_seq._iseq0.split(dst_seq._iseq0, split0_size);
      src_seq._iseq1.split(dst_seq._iseq1, split1_size);

      src_seq._obeg += split0_size + split1_size;

      return dst_seq._iseq0.size() || dst_seq._iseq1.size();
    }
  };

  template<typename SequenceType>
  struct StaticReverseExtractor : public BaseExtractor<SequenceType>
  {
    typedef typename SequenceType::SizeType SizeType;

    SizeType _unit_size;

    StaticReverseExtractor(const SizeType& unit_size)
      : _unit_size(unit_size) {}

    inline bool extract(SequenceType& dst_seq, SequenceType& src_seq)
    {
      if (src_seq.is_empty())
	return false;

      const SizeType size =
	std::min((SizeType)src_seq.size(), (SizeType)_unit_size);

      src_seq.rsplit(dst_seq, size);

      return true;
    }
  };


  // invalid types

  struct InvalidType
  {
    inline InvalidType() {}
    inline InvalidType(int) {}
  } InvalidType;

  typedef struct InvalidType InvalidResultType;
  typedef struct InvalidType InvalidConstantType;


  // reduce xkaapi thief result

  template<typename WorkType>
  int reduce_thief
  (
   kaapi_stack_t* stack,
   kaapi_task_t* task,
   void* thief_data,
   void* victim_data,
   void* seq_voidptr
  )
  {
    typedef typename WorkType::SequenceType SequenceType;

    WorkType* const thief_work =
      static_cast<WorkType*>(thief_data);

    WorkType* const victim_work =
      static_cast<WorkType*>(victim_data);

    SequenceType* const victim_seq =
      static_cast<SequenceType*>(seq_voidptr);

#if KASTL_DEBUG
      const typename SequenceType::RangeType vr =
	SequenceType::get_range(victim_work->_ori_seq, victim_work->_seq);
      const typename SequenceType::RangeType tr =
	SequenceType::get_range(thief_work->_ori_seq, thief_work->_seq);
      printf("r: %c#%u [%lu - %lu] <- %c#%u [%lu - %lu]\n",
	     victim_work->_is_master ? 'm' : 's',
	     (unsigned int)victim_work->_kid,
	     vr.first, vr.second,
	     thief_work->_is_master ? 'm' : 's',
	     (unsigned int)thief_work->_kid,
	     tr.first, tr.second);
#endif

    victim_work->reduce(*thief_work);

    *victim_seq = thief_work->_seq;

    // always return 1 so that the
    // victim knows about preemption

    return 1;
  }


  // xkaapi splitter types

  template<typename WorkType>
  void child_entry(kaapi_task_t*, kaapi_stack_t*);

  struct IdentitySplitter
  {
    template<typename WorkType>
    static int handle_requests
    (
     kaapi_stack_t*,
     kaapi_task_t*,
     int,
     kaapi_request_t*,
     void*
    )
    { return 0; }
  };

  template<size_t ParSize>
  struct StaticSplitter
  {
    template<typename WorkType>
    static int handle_requests
    (
     kaapi_stack_t* victim_stack,
     kaapi_task_t* victim_task,
     int request_count,
     kaapi_request_t* request,
     void* seq_voidptr
    )
    {
      typedef typename WorkType::SequenceType SequenceType;

      WorkType* const victim_work =
	kaapi_task_getargst(victim_task, WorkType);

      SequenceType* const victim_seq =
	static_cast<SequenceType*>(seq_voidptr);

      const int saved_count = request_count;
      int replied_count = 0;

      const size_t work_size = victim_seq->size();
      size_t par_size = work_size / (1 + (size_t)request_count);
      if (par_size < ParSize)
      {
	request_count = work_size / ParSize - 1;
	par_size = ParSize;
      }

      StaticReverseExtractor<SequenceType> extractor(par_size);

      // request_count can be <= 0
      for (; request_count > 0; ++request)
      {
	if (!kaapi_request_ok(request))
	  continue ;

	// allocate on the thief stack
	kaapi_stack_t* const thief_stack = request->stack;
	kaapi_task_t* const thief_task = kaapi_stack_toptask(thief_stack);

	kaapi_task_initadaptive
	(
	 thief_stack, thief_task,
	 child_entry<WorkType>, NULL,
	 KAAPI_TASK_ADAPT_DEFAULT
	);
                
	WorkType* const thief_work =
	  static_cast<WorkType*>
	  (kaapi_stack_pushdata(thief_stack, sizeof(WorkType)));

	thief_work->_is_done = false;

	extractor.extract(thief_work->_seq, *victim_seq);

#if KASTL_DEBUG
	const typename SequenceType::RangeType vr =
	  SequenceType::get_range(victim_work->_ori_seq, *victim_seq);
	const typename SequenceType::RangeType tr =
	  SequenceType::get_range(victim_work->_ori_seq, thief_work->_seq);
	printf("s: %c#%x [%lu - %lu] -> %c#%x [%lu - %lu]\n",
	       victim_work->_is_master ? 'm' : 's',
	       (unsigned int)victim_work->_kid,
	       vr.first, vr.second,
	       thief_work->_is_master ? 'm' : 's',
	       (unsigned int)thief_work->_kid,
	       tr.first, tr.second);
#endif

	thief_work->_const = victim_work->_const;
	thief_work->prepare();

#if KASTL_DEBUG
	thief_work->_is_master = false;
	thief_work->_ori_seq = victim_work->_ori_seq;
#endif
        
	kaapi_task_setargs(thief_task, thief_work);
	kaapi_stack_pushtask(thief_stack);
        
	kaapi_request_reply_head
	(
	 victim_stack, victim_task,
	 request, thief_stack,
	 sizeof(WorkType), 1
	);

	--request_count;
	++replied_count;
      }

      return saved_count - replied_count;
    }
  };


  // base work type

  template
  <
    typename _SequenceType,
    typename _ConstantType,
    typename _ResultType,
    typename _MacroType,
    typename _NanoType,
    typename _SplitterType
  >
  struct BaseWork
  {
    typedef _SequenceType SequenceType;
    typedef _ConstantType ConstantType;
    typedef _ResultType ResultType;
    typedef _MacroType MacroType;
    typedef _NanoType NanoType;
    typedef _SplitterType SplitterType;

    typedef BaseWork
    <SequenceType, ConstantType, ResultType, MacroType, NanoType, SplitterType> BaseType;

    SequenceType _seq;
    const ConstantType* _const;
    ResultType _res;

    bool _is_done;

#if KASTL_DEBUG
    SequenceType _ori_seq;
    bool _is_master;
    typedef unsigned int kaapi_processor_id_t;
    kaapi_processor_id_t _kid;
#endif

    BaseWork() : _is_done(false) {}

    BaseWork
    (
     const SequenceType& s,
     const ConstantType* c,
     const ResultType& r
    ) : _seq(s), _const(c), _res(r), _is_done(false) {}

    inline void prepare() {}

    inline void reduce(const BaseWork&) {}

    inline bool is_empty() const
    {
      return _seq.is_empty();
    }

  };


  // user tuning params

  enum TuningTag
  {
    // tags are exposed to the user to
    // modifiy parts of the algorithm
    // behavior, such as macro extractor
    // and splitter

    TAG_IDENTITY = 0,
    TAG_STATIC,
    TAG_BACKOFF,
    TAG_LINEAR,
    TAG_MERGE
  };

  struct BaseTuningParams
  {
    // this is the documented structure
    // the user can use to tune the
    // algorithm behavior. every field
    // must be documented.
    // A default implementation is provided
    // for every algorithm, with fit-all
    // values. It can be derived to provide
    // better values whenever apropriated.

    static const enum TuningTag splitter_tag = TAG_STATIC;
    static const size_t par_size = 128;

    static const enum TuningTag macro_tag = TAG_IDENTITY;
    static const size_t macro_min_size = 1024;
    static const size_t macro_max_size = 32768;

    static const enum TuningTag nano_tag = TAG_STATIC;
    static const size_t nano_size = 512;
  };

  // default per algo class tuning

  struct Daouda0TuningParams : BaseTuningParams
  {
    // no macroloop
  };


  struct Daouda1TuningParams : BaseTuningParams
  {
    // add a macroloop to prevent extraction
    // from the very end in case of anticipated
    // terminating algo such as search, find_if

    static const enum TuningTag macro_tag = TAG_BACKOFF;
  };


  // macro type factory

  template<enum TuningTag Tag, typename Params, typename SequenceType>
  struct make_macro_type
  {
    // dont provide any default implm
  };

  template<typename Params, typename SequenceType>
  struct make_macro_type<TAG_IDENTITY, Params, SequenceType>
  {
    typedef IdentityExtractor
    <SequenceType> Type;
  };

  template<typename Params, typename SequenceType>
  struct make_macro_type<TAG_STATIC, Params, SequenceType>
  {
    typedef StaticExtractor
    <SequenceType, Params::macro_max_size> Type;
  };

  template<typename Params, typename SequenceType>
  struct make_macro_type<TAG_BACKOFF, Params, SequenceType>
  {
    typedef BackoffExtractor
    <SequenceType, Params::macro_min_size, Params::macro_max_size> Type;
  };

  template<typename Params, typename SequenceType>
  struct make_macro_type<TAG_LINEAR, Params, SequenceType>
  {
    typedef LinearExtractor
    <
      SequenceType,
      Params::macro_min_size,
      Params::macro_max_size,
      Params::macro_step_size
    > Type;
  };

  // nano type factory

  template<enum TuningTag Tag, typename Params, typename SequenceType>
  struct make_nano_type
  {
    // dont provide any default implm
  };

  template<typename Params, typename SequenceType>
  struct make_nano_type<TAG_IDENTITY, Params, SequenceType>
  {
    typedef IdentityExtractor
    <SequenceType> Type;
  };

  template<typename Params, typename SequenceType>
  struct make_nano_type<TAG_STATIC, Params, SequenceType>
  {
    typedef StaticExtractor
    <SequenceType, Params::nano_size> Type;
  };

  template<typename Params, typename SequenceType>
  struct make_nano_type<TAG_MERGE, Params, SequenceType>
  {
    typedef MergeExtractor
    <SequenceType, Params::nano_size> Type;
  };

  // splitter type factory

  template<enum TuningTag Tag, typename Params>
  struct make_splitter_type
  {
    // dont provide any default implm
  };

  template<typename Params>
  struct make_splitter_type<TAG_IDENTITY, Params>
  {
    typedef IdentitySplitter Type;
  };

  template<typename Params>
  struct make_splitter_type<TAG_STATIC, Params>
  {
    typedef StaticSplitter<Params::par_size> Type;
  };


  // xkaapi entrypoint routines

  typedef int (*kastl_splitter_t)
  (kaapi_stack_t*, kaapi_task_t*, int, kaapi_request_t*, void*);

  typedef int (*kastl_reducer_t)
  (kaapi_stack_t*, kaapi_task_t*, void*, void*, void*);

#define KASTL_MASTER_SLAVE 1
#if KASTL_MASTER_SLAVE

  template<typename WorkType>
  void process_sequence
  (
   kaapi_task_t* task,
   kaapi_stack_t* stack,
   WorkType* work,
   typename WorkType::SequenceType* macro_seq
  )
  {
    typedef typename WorkType::NanoType NanoType;
    typedef typename WorkType::SequenceType SequenceType;
    typedef typename WorkType::SplitterType SplitterType;

    kastl_splitter_t const splitter =
      SplitterType::template handle_requests<WorkType>;

    kastl_reducer_t const reducer = reduce_thief<WorkType>;

    NanoType nano_extractor;
    SequenceType nano_seq;

#if KASTL_DEBUG
    work->_kid = kaapi_get_current_kid();
#endif

    macro_seq->empty_seq(nano_seq);

  advance_work:
    
    while (nano_extractor.extract(nano_seq, *macro_seq))
    {
      // cooperative stealing
      kaapi_stealpoint(stack, task, splitter, macro_seq);

      // conccurent workstealing
      kaapi_stealbegin(stack, task, splitter, macro_seq);

#if KASTL_DEBUG
      const typename SequenceType::RangeType r =
	SequenceType::get_range(work->_ori_seq, nano_seq);
      printf("c: %c#%x [%lu - %lu] \n",
	     work->_is_master ? 'm' : 's',
	     (unsigned int)work->_kid,
	     r.first, r.second);
#endif

      work->compute(nano_seq);
      kaapi_stealend(stack, task);

      if (work->_is_done)
	return ;
    }

    if (kaapi_preemptpoint(stack, task, NULL, work, sizeof(WorkType)))
      return ;

    if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, work, macro_seq))
      return ;

    if (work->_is_done)
      return ;

    macro_seq->empty_seq(nano_seq);

    goto advance_work;
  }

  template<typename WorkType>
  static void reduce_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    // notes: in the daouda thesis, the
    // splitter is binary. we use nano.

    WorkType* const work = kaapi_task_getargst(task, WorkType);

    SequenceType nseq;

    work->_seq.empty_seq(nseq);

    while (true)
    {
      if (nanoloop.next(nseq) == false)
	break ;

      stealpoint(reduce_task_splitter, nseq);

      work->reduce(nseq);
    }
  }

  // daouda thesis Ps process

  template<typename WorkType>
  static void seq_task_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    typedef typename WorkType::NanoType NanoType;
    typedef typename WorkType::SequenceType SequenceType;
    typedef typename WorkType::SplitterType SplitterType;

    kastl_splitter_t const splitter =
      SplitterType::template handle_requests<WorkType>;

    kastl_reducer_t const reducer = reduce_thief<WorkType>;

    WorkType* const work = kaapi_task_getargst(task, WorkType);
    SequenceType work_seq = work->_seq;

    SequenceType macro_seq;
    MacroType macro_extractor;

    SequenceType nano_seq;
    NanoType nano_extractor;

    work_seq.empty_seq(macro_seq);

  feed_macro:
    // consume all the work_seq
    if (macro_extractor.extract(macro_seq, work_seq) == false)
      return ;

    macro_seq.empty_seq(nano_seq);

  feed_nano:
    // consume all the macro
    if (nano_extractor.extract(nano_seq, macro_seq) == true)
    {
      kaapi_stealpoint(stack, task, splitter, macro_seq);

      work->compute(nseq);

      if (work->_is_done)
	return ;

      goto feed_nano;
    }

    // no nano, no thief, feed macro
    if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, work, macro_seq))
      goto feed_macro ;

    // thief terminated the work
    if (work->_is_done)
      return ;

    // preempting means nano items
    goto feed_nano;
  }

  // daouda thesis Pv process

  template<typename WorkType>
  static void steal_task_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    typedef typename WorkType::NanoType NanoType;
    typedef typename WorkType::SequenceType SequenceType;
    typedef typename WorkType::SplitterType SplitterType;

    kastl_splitter_t const splitter =
      SplitterType::template handle_requests<WorkType>;

    kastl_reducer_t const reducer = reduce_thief<WorkType>;

    NanoType nano_extractor;
    SequenceType nano_seq;

#if KASTL_DEBUG
    work->_kid = kaapi_get_current_kid();
#endif

    macro_seq->empty_seq(nano_seq);

  advance_work:
    
    while (nano_extractor.extract(nano_seq, *macro_seq))
    {
      // cooperative stealing
      kaapi_stealpoint(stack, task, splitter, macro_seq);

      // conccurent workstealing
      kaapi_stealbegin(stack, task, splitter, macro_seq);

#if KASTL_DEBUG
      const typename SequenceType::RangeType r =
	SequenceType::get_range(work->_ori_seq, nano_seq);
      printf("c: %c#%x [%lu - %lu] \n",
	     work->_is_master ? 'm' : 's',
	     (unsigned int)work->_kid,
	     r.first, r.second);
#endif

      work->compute(nano_seq);
      kaapi_stealend(stack, task);

      if (work->_is_done)
	return ;
    }

    if (kaapi_preemptpoint(stack, task, NULL, work, sizeof(WorkType)))
      return ;

    if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, work, macro_seq))
      return ;

    if (work->_is_done)
      return ;

    macro_seq->empty_seq(nano_seq);

    goto advance_work;
  }

  template<typename WorkType>
  void child_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    WorkType* const work = kaapi_task_getargst(task, WorkType);
    process_sequence<WorkType>(task, stack, work, &work->_seq);
    kaapi_return_steal(stack, task, work, sizeof(WorkType));
  }

  template<typename WorkType>
  void root_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    typedef typename WorkType::SequenceType SequenceType;
    typedef typename WorkType::MacroType MacroType;

    WorkType* const work = kaapi_task_getargst(task, WorkType);

    MacroType macro_extractor;
    SequenceType macro_seq;

    work->_seq.empty_seq(macro_seq);

    while (macro_extractor.extract(macro_seq, work->_seq))
    {
      process_sequence(task, stack, work, &macro_seq);

      if (work->_is_done == true)
	break;
    }
  }

#else // ! KASTL_MASTER_SLAVE

  template<typename WorkType>
  void execute_work
  (
   kaapi_task_t* task,
   kaapi_stack_t* stack
  )
  {
    typedef typename WorkType::SequenceType SequenceType;
    typedef typename WorkType::MacroType MacroType;
    typedef typename WorkType::NanoType NanoType;
    typedef typename WorkType::SplitterType SplitterType;

    kastl_splitter_t const splitter =
      SplitterType::template handle_requests<WorkType>;

    kastl_reducer_t const reducer = reduce_thief<WorkType>;

    WorkType* const work = kaapi_task_getargst(task, WorkType);

#if KASTL_DEBUG
    work->_kid = kaapi_get_current_kid();
#endif

    MacroType macro_extractor;
    NanoType nano_extractor;

    SequenceType macro_seq;
    SequenceType nano_seq;

    // .
    // preempt thieves to complete the current
    // macro loop. if there is no theif, the
    // macro is done and we extract another one
  advance_work:

    // .
    // here the local macro is empty, and thieves
    // may own parts of it they stole. we preempt
    // them to really finish the macro.
    if (!kaapi_preempt_nextthief(stack, task, NULL, reducer, work, &macro_seq))
    {
      // .
      // macro sequence really empty (ie. no
      // thieves). extract new macro from seq
      // or return if no more work to be done
      if (!macro_extractor.extract(macro_seq, work->_seq))
	return ;
    }

    // .
    // work is done, terminate
    if (work->_is_done)
      return ;

    // .
    // process macro work by extracting
    // unstealable nano steps
    while (nano_extractor.extract(nano_seq, macro_seq))
    {
      // cooperative stealing
      kaapi_stealpoint(stack, task, splitter, &macro_seq);

      // conccurent workstealing
      kaapi_stealbegin(stack, task, splitter, &macro_seq);

#if KASTL_DEBUG
      const typename SequenceType::RangeType r =
	SequenceType::get_range(work->_ori_seq, nano_seq);
      printf("c: %c#%x [%lu - %lu] \n",
	     work->_is_master ? 'm' : 's',
	     (unsigned int)work->_kid,
	     r.first, r.second);
#endif

      work->compute(nano_seq);
      kaapi_stealend(stack, task);

      // .
      // work is done, return.
      if (work->_is_done)
	return ;
    }

    // .
    // current macro done. check for preemption
    if (kaapi_preemptpoint(stack, task, NULL, work, sizeof(WorkType)))
      return ;

    // .
    // continue processing, goto 1.
    goto advance_work;
  }

  template<typename WorkType>
  void child_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    execute_work<WorkType>(task, stack);
    WorkType* const work = kaapi_task_getargst(task, WorkType);
    kaapi_return_steal(stack, task, work, sizeof(WorkType));
  }

  template<typename WorkType>
  void root_entry(kaapi_task_t* task, kaapi_stack_t* stack)
  {
    execute_work<WorkType>(task, stack);
  }

#endif // KASTL_MASTER_SLAVE

  // bootstrap the computation

#if 1 // synchronous only

  template<typename WorkType> void compute(WorkType& work)
  {
    // create an adpative task and sync

    kaapi_stack_t* const stack = kaapi_self_stack();
    kaapi_task_t* const task = kaapi_stack_toptask(stack);
    kaapi_frame_t frame;

#if KASTL_DEBUG
    work._is_master = true;
    work._ori_seq = work._seq;
#endif

    kaapi_stack_save_frame(stack, &frame);

    kaapi_task_initadaptive
    (
     stack, task,
     root_entry<WorkType>,
     static_cast<void*>(&work),
     KAAPI_TASK_ADAPT_DEFAULT
    );

    kaapi_stack_pushtask(stack);
    kaapi_finalize_steal(stack, task);
    kaapi_sched_sync(stack);

    kaapi_stack_restore_frame(stack, &frame);
  }

#else

  // compute, async version
  template<typename WorkType>
  void compute_async
  (
   typename const WorkType::SequenceType& s, 
   typename const WorkType::ConstantType& c, 
   typename WorkType::ResultType& r
  )
  {
    kaapi_stack_t* const stack = kaapi_self_stack();
    kaapi_task_t* const task = kaapi_stack_toptask(stack);
    WorkType* const work = kaapi_stack_pushdata(stack, sizeof(WorkType)),

    kaapi_task_initadaptive
    (
     stack, task,
     root_entry<WorkType>,
     static_cast<void*>(work),
     KAAPI_TASK_ADAPT_DEFAULT
    );

    // asynchronous, do copy
    kaapi_stack_pushdata(stack, sizeof(WorkType))
    WorkType* const work = static_cast<WorkType*>(kaapi_task_getargs(thief_tas));
    work->_seq   = s;
    work->_const = c;
    work->_res   = res;

    kaapi_stack_pushtask(stack);
  }

  // compute, sync version 
  template<typename WorkType>
  void compute
  (
   typename const WorkType::SequenceType& s,
   typename const WorkType::ConstantType& c,
   typename WorkType::ResultType* r
  )
  {
    kaapi_stack_t* const stack = kaapi_self_stack();
    kaapi_frame_t frame;

    kaapi_stack_save_frame(stack, &frame);
    compute_async(s, c, r, p);
    kaapi_sched_sync(stack);
    kaapi_stack_restore_frame(stack, &frame);
  }
#endif

}
} // kastl::impl


#endif // ! KASTL_IMPL_HH_INCLUDED
