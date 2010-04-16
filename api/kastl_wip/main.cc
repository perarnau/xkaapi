#include <stddef.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include "kastl.hh"
#include "pinned_array.hh"


#define DEFAULT_INPUT_SIZE 100000
#define DEFAULT_ITER_SIZE 10

typedef uint32_t ValueType;
//typedef std::vector<ValueType> SequenceType;
typedef pinned_array<ValueType> SequenceType;
typedef std::pair<SequenceType, SequenceType> InputType;
typedef SequenceType OutputType;


// error strings
static std::string index_error_string(ptrdiff_t s, ptrdiff_t k)
{ 
  std::ostringstream oss;
  oss << "index differ at s: " << s << ", k: " << k;
  return oss.str();
}

static std::string size_error_string(size_t s, size_t k)
{
  std::ostringstream oss;
  oss << "sizes differ s: " << s << ", k: " << k;
  return oss.str();
}

template<typename T>
static std::string value_error_string(const T& s, const T& k)
{
  std::ostringstream oss;
  oss << "values differ s: " << s << ", k: " << k;
  return oss.str();
}


// sequences

template<typename T>
inline static T gen_rand(const T& max)
{
  static bool is_seed = false;

  if (is_seed == false)
    {
      ::srand(::getpid() * ::time(NULL));
      is_seed = true;
    }

  return static_cast<T>(rand() % (max - 1) + 1);
}


static void gen_rand_seq(SequenceType& s, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    s[i] = gen_rand<ValueType>(1000);
}


static void gen_rand_asc_seq(SequenceType& s, size_t n)
{
  s[0] = gen_rand<ValueType>(10);

  for (size_t i = 1; i < n; ++i)
    s[i] = s[i - 1] + gen_rand<ValueType>(10);
}


static void gen_asc_seq(SequenceType& s, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    s[i] = i;
}


static void gen_desc_seq(SequenceType& s, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    s[i] = n - i;
}


static void gen_one_seq(SequenceType& s, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    s[i] = 1;
}


enum seq_order
  {
    SEQ_ORDER_RAND = 0,
    SEQ_ORDER_RAND_ASC,
    SEQ_ORDER_ASC,
    SEQ_ORDER_DESC,
    SEQ_ORDER_ONE,
    SEQ_ORDER_INVALID
  };


static void gen_seq(SequenceType& s, size_t n, enum seq_order o)
{
  static void (*gen_seq[])(SequenceType&, size_t) =
    {
      gen_rand_seq,
      gen_rand_asc_seq,
      gen_asc_seq,
      gen_desc_seq,
      gen_one_seq
    };

  return gen_seq[(size_t)o](s, n);
}


static bool cmp_sequence
(
 SequenceType::iterator& ipos,
 SequenceType::iterator& jpos,
 SequenceType::iterator& jend
)
{
  // assume same size
  for (; jpos != jend; ++ipos, ++jpos)
    if (*ipos != *jpos)
      return false;
  return true;
}


static void __attribute__((unused)) print_sequences
(
 SequenceType::iterator ipos,
 SequenceType::iterator jpos,
 SequenceType::iterator jend
)
{
  for (; jpos != jend; ++ipos, ++jpos)
    ::printf("%u %u\n", *ipos, *jpos);
}


static void __attribute__((unused)) print_sequence
(
 SequenceType::iterator pos,
 SequenceType::iterator end
)
{
  for (; pos != end; ++pos)
    ::printf("%u\n", *pos);
}


// timing

#if 1 // gettimeofday

namespace timing
{
  typedef struct timeval TimerType;

  static void initialize()
  { }

  static void get_now(TimerType& now)
  { gettimeofday(&now, NULL); }

  static void sub_timers(const TimerType& a, const TimerType& b, TimerType& d)
  { timersub(&a, &b, &d); }

  static unsigned long timer_to_usec(const TimerType& t)
  { return t.tv_sec * 1000000 + t.tv_usec; }

};

#else // ia32 realtime clock

#include "timing.h"

namespace timing
{
  typedef tick_t TimerType;

  static void initialize()
  { timing_init(); }

  static void get_now(TimerType& now)
  { GET_TICK(now); }

  static void sub_timers(const TimerType& a, const TimerType& b, TimerType& d)
  { d.tick = TICK_DIFF(b, a); }

  static unsigned long timer_to_usec(const TimerType& t)
  { return (unsigned long)tick2usec(t.tick); }

};

#endif



// run factory

class RunInterface
{
public:

  ~RunInterface() {}

  virtual void run_stl(InputType&, OutputType&) = 0;
  virtual void run_kastl(InputType&, OutputType&) = 0;

  virtual bool check(OutputType&, OutputType&, std::string&) const = 0;

  virtual void prepare(InputType&) {}

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_RAND;
    are_equal = true;
  }

  static RunInterface* create(const std::string&);
};


class InvalidRun : public RunInterface
{
public:
  InvalidRun() {}

  void run_stl(InputType&, OutputType&) {}
  void run_kastl(InputType&, OutputType&) {}

  bool check(OutputType&, OutputType&, std::string&) const { return false; }
};


class CountRun : public RunInterface
{
  ptrdiff_t _kastl_res;
  ptrdiff_t _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_RAND;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::count
      (i.first.begin(), i.first.end(), 42);
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::count
      (i.first.begin(), i.first.end(), 42);
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = value_error_string(_stl_res, _kastl_res);

    return false;
  }

};


class ForEachRun : public RunInterface
{
  // todo hack hack hack
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

  static void inc(unsigned int& n) { ++n; }

public:

  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::for_each(i.first.begin(), i.first.end(), inc);
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::for_each(i.second.begin(), i.second.end(), inc);
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    SequenceType::iterator spos = _spos;
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};


class TransformRun : public RunInterface
{
  struct inc
  {
    inc() {}

    ValueType operator()(const ValueType& v)
    { return v + 1; }
  };

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ASC;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    kastl::transform
      (
       i.first.begin(), i.first.end(),
       o.begin(), inc()
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::transform
      (
       i.first.begin(), i.first.end(),
       o.begin(), inc()
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.end();

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }

    return true;
  }

};

class SearchRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

  ptrdiff_t _kastl_index;
  ptrdiff_t _stl_index;

  SequenceType _ref_seq;

  static void gen_ref_seq(SequenceType& seq, unsigned int n)
  {
    static const size_t size = 10;

    // generate reference sequence
    seq.resize(size);
    for (size_t count = 0; count < size;  ++count, ++n)
      seq[count] = n;
  }

public:

  virtual void prepare(InputType& i)
  {
    // only i.frist.begin

    gen_ref_seq(_ref_seq, 10);

    std::copy
    (
       _ref_seq.begin(), _ref_seq.end(),
       i.first.begin() + (i.first.size() / 2)
    );
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::search
      (
       i.first.begin(),
       i.first.end(),
       _ref_seq.begin(),
       _ref_seq.end()
      );

    _kastl_index = std::distance(i.first.begin(), _kastl_res);
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::search
      (
       i.first.begin(),
       i.first.end(),
       _ref_seq.begin(),
       _ref_seq.end()
      );

    _stl_index = std::distance(i.first.begin(), _stl_res);
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = index_error_string(_stl_index, _kastl_index);

    return false;
  }

};

#if 0 // speed compile time up

class AccumulateRun : public RunInterface
{
  ValueType _kastl_res;
  ValueType _stl_res;

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kastl_res = kastl::accumulate
      (i.first.begin(), i.first.end(), ValueType(0));
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::accumulate
      (i.first.begin(), i.first.end(), ValueType(0));
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    return _kastl_res == _stl_res;
  }

};


class EqualRun : public RunInterface
{
  bool _kastl_res;
  bool _stl_res;

public:

  virtual void prepare(InputType& i)
  {
    i.first[i.first.size() / 2] = 42;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::equal
      (i.first.begin(), i.first.end(), i.second.begin());
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::equal
      (i.first.begin(), i.first.end(), i.second.begin());
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    return _kastl_res == _stl_res;
  }

};


#if 0

class CountIfRun : public RunInterface
{
  ptrdiff_t _kastl_res;
  ptrdiff_t _stl_res;

  static bool is_zero(unsigned int v)
  {
    return v == 0;
  }

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::count_if
      (i.first.begin(), i.first.end(), is_zero);
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::count_if
      (i.first.begin(), i.first.end(), is_zero);
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = value_error_string(_stl_res, _kastl_res);

    return false;
  }

};

#endif


class FindIfRun : public RunInterface
{
  SequenceType::iterator _kastl_res[2];
  SequenceType::iterator _stl_res[2];

  ptrdiff_t _kastl_index[2];
  ptrdiff_t _stl_index[2];

  static bool is_zero(unsigned int v)
  {
    return v == 0;
  }

  static bool is_magic(unsigned int v)
  {
    return v == 42;
  }

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res[0] = kastl::find_if
      (i.first.begin(), i.first.end(), is_zero);
    _kastl_index[0] = _kastl_res[0] - i.first.begin();

    _kastl_res[1] = kastl::find_if
      (i.first.begin(), i.first.end(), is_magic);
    _kastl_index[1] = _kastl_res[1] - i.first.begin();
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res[0] = std::find_if
      (i.first.begin(), i.first.end(), is_zero);
    _stl_index[0] = _stl_res[0] - i.first.begin();

    _stl_res[1] = std::find_if
      (i.first.begin(), i.first.end(), is_magic);
    _stl_index[1] = _stl_res[1] - i.first.begin();
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res[0] == _stl_res[0])
      if (_kastl_res[1] == _stl_res[1])
	return true;

    if (_kastl_res[1] != _stl_res[1])
      error_string = index_error_string(_stl_index[1], _kastl_index[1]);

    if (_kastl_res[0] != _stl_res[0])
      error_string = index_error_string(_stl_index[0], _kastl_index[0]);

    return false;
  }

};


class ReverseRun : public RunInterface
{
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::reverse(i.first.begin(), i.first.end());
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::reverse(i.second.begin(), i.second.end());
  }

  virtual bool check
  (
   OutputType&,
   OutputType&,
   std::string& es
  ) const
  {
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator spos = _spos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};


class FindRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

  ptrdiff_t _kastl_index;
  ptrdiff_t _stl_index;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
#define FIND_VALUE 42
    _kastl_res = kastl::find
      (i.first.begin(), i.first.end(), FIND_VALUE);

    _kastl_index = std::distance(i.first.begin(), _kastl_res);
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::find
      (i.first.begin(), i.first.end(), FIND_VALUE);

    _stl_index = std::distance(i.first.begin(), _stl_res);
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = index_error_string(_stl_index, _kastl_index);

    return false;
  }

};


class FindFirstOfRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

  ptrdiff_t _kastl_index;
  ptrdiff_t _stl_index;

public:

  virtual void prepare(InputType& i)
  {
    i.second.resize(4);

    i.second[0] = 6;
    i.second[1] = 12;
    i.second[2] = 24;
    i.second[3] = 42;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::find_first_of
      (i.first.begin(), i.first.end(),
       i.second.begin(), i.second.end());

    _kastl_index = _kastl_res - i.first.begin();
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::find_first_of
      (i.first.begin(), i.first.end(),
       i.second.begin(), i.second.end());

    _stl_index = _stl_res - i.first.begin();
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = index_error_string(_stl_index, _kastl_index);

    return false;
  }

};


class PartialSumRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ONE;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::partial_sum
    (
     i.first.begin(),
     i.first.end(),
     o.begin()
    );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::partial_sum
    (
     i.first.begin(),
     i.first.end(),
     o.begin()
    );
  }

  virtual bool check
  (OutputType& ko, OutputType& so, std::string& error_string) const
  {
    const ptrdiff_t kastl_index = std::distance(ko.begin(), _kastl_res);
    const ptrdiff_t stl_index = std::distance(so.begin(), _stl_res);

    if (kastl_index != stl_index)
    {
      error_string = index_error_string(stl_index, kastl_index);
      return false;
    }

    SequenceType::iterator kpos = ko.begin();
    SequenceType::iterator spos = so.begin();
    SequenceType::iterator send = so.end();

    if (cmp_sequence(kpos, spos, send) == true)
      return true;

#if 0
    error_string = index_error_string
    (
     std::distance(so.begin(), spos),
     std::distance(ko.begin(), kpos)
    );
#else
    error_string = index_error_string
    (  
     std::distance(so.begin(), spos),
     std::distance(ko.begin(), kpos)
    );
    error_string.append(std::string("\n"));
    error_string.append(value_error_string(*spos, *kpos));
#endif

    return false;
  }

};


class MismatchRun : public RunInterface
{
  std::pair<SequenceType::iterator, SequenceType::iterator> _kastl_res;
  std::pair<SequenceType::iterator, SequenceType::iterator> _stl_res;

public:

  virtual void prepare(InputType& i)
  {
    if (!(::rand() % 10))
      return ;

    i.first[::rand() % i.first.size()] = 42;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::mismatch
      (
       i.first.begin(),
       i.first.end(),
       i.second.begin()
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::mismatch
      (
       i.first.begin(),
       i.first.end(),
       i.second.begin()
      );
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;
    return false;
  }

};


#if 0

class SearchNRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

  ptrdiff_t _kastl_index;
  ptrdiff_t _stl_index;

  static void gen_sub_seq(SequenceType& seq, size_t i)
  {
    static const size_t size = 10;

    // generate reference sequence

    for (size_t count = 0; count < size;  ++count, ++i)
      seq[i] = 42;
  }

public:

#if 1
  virtual void prepare(InputType& i)
  {
    // prepare only the first input
    // sequence, second not used

    const size_t off = i.first.size() / 2 - (::rand() % 5);
    gen_sub_seq(i.first, off);
  }
#else
  virtual void prepare(InputType& i)
  {
    i.first[INPUT_SIZE - 3] = 513;
    i.first[INPUT_SIZE - 2] = 513;
    i.first[INPUT_SIZE - 1] = 513;

    i.first.resize(INPUT_SIZE - 1);
  }
#endif

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    SequenceType ref_seq;

    _kastl_res = kastl::search_n
      (
       i.first.begin(),
       i.first.end(),
       2, 513
      );

    _kastl_index = _kastl_res - i.first.begin();
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    SequenceType ref_seq;

    _stl_res = std::search_n
      (
       i.first.begin(),
       i.first.end(),
       2, 513
      );

    _stl_index = _stl_res - i.first.begin();
  }

  virtual bool check(OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = index_error_string(_stl_index, _kastl_index);

    return false;
  }

};

#endif


class SortRun : public RunInterface
{
  // todo hack hack hack
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::sort(i.first.begin(), i.first.end());
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::sort(i.second.begin(), i.second.end());
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    SequenceType::iterator spos = _spos;
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};


class PartitionRun : public RunInterface
{
  // todo hack hack hack

  SequenceType::iterator _spos;
  SequenceType::iterator _send;
  SequenceType::iterator _sres;

  SequenceType::iterator _kpos;
  SequenceType::iterator _kend;
  SequenceType::iterator _kres;

  static bool predicate(const ValueType& v)
  {
    return v <= 42;
  }

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();
    _kend = i.first.end();

    _kres = kastl::partition(i.first.begin(), i.first.end(), predicate);
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    _sres = std::partition(i.second.begin(), i.second.end(), predicate);
  }

  virtual bool check(OutputType&, OutputType&, std::string& es) const
  {
    const ptrdiff_t si = std::distance(_spos, _sres);
    const ptrdiff_t ki = std::distance(_kpos, _kres);

    if (si != ki)
    {
      es = index_error_string(si, ki);
      return false;
    }

    // weak ordering, cmp_seq not possible
    SequenceType::iterator pos = _kpos;
    for (; (pos != _kend) && predicate(*pos); ++pos)
      ;

    if (pos != _kres)
    {
      es = std::string("pred not true at");
      es += index_error_string(0, std::distance(_kpos, pos));
      return false;
    }

    return true;
  }

};


class FillRun : public RunInterface
{
  // todo hack hack hack
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::fill(i.first.begin(), i.first.end(), 42);
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::fill(i.second.begin(), i.second.end(), 42);
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    SequenceType::iterator spos = _spos;
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};


#if 0

class ReplaceIfRun : public RunInterface
{
  // todo hack hack hack
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

  static bool is_magic(unsigned int v)
  {
    return v == 42;
  }

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::replace_if(i.first.begin(), i.first.end(), is_magic, 24);
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::replace_if(i.second.begin(), i.second.end(), is_magic, 24);
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    SequenceType::iterator spos = _spos;
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};

#endif


class ReplaceRun : public RunInterface
{
  // todo hack hack hack
  SequenceType::iterator _spos;
  SequenceType::iterator _kpos;
  SequenceType::iterator _send;

public:
  virtual void run_kastl(InputType& i, OutputType&)
  {
    _kpos = i.first.begin();

    kastl::replace(i.first.begin(), i.first.end(), 42, 24);
  }

  virtual void run_stl(InputType& i, OutputType&)
  {
    _spos = i.second.begin();
    _send = i.second.end();

    std::replace(i.second.begin(), i.second.end(), 42, 24);
  }

  virtual bool check(OutputType&, OutputType&, std::string&) const
  {
    SequenceType::iterator spos = _spos;
    SequenceType::iterator kpos = _kpos;
    SequenceType::iterator send = _send;

    return cmp_sequence(kpos, spos, send);
  }

};


class GenerateRun : public RunInterface
{
  struct gen_one
  {
    gen_one() {}

    unsigned int operator()() { return 1; }
  };

public:
  virtual void run_kastl(InputType&, OutputType& o)
  {
    kastl::generate(o.begin(), o.end(), gen_one());
  }

  virtual void run_stl(InputType&, OutputType& o)
  {
    std::generate(o.begin(), o.end(), gen_one());
  }

  virtual bool check
  (
   OutputType& os,
   OutputType& ok,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = ok.begin();
    SequenceType::iterator spos = os.begin();
    SequenceType::iterator send = os.end();

    return cmp_sequence(kpos, spos, send);
  }

};


#if 0

class GenerateNRun : public RunInterface
{
  struct gen_one
  {
    gen_one() {}

    unsigned int operator()() { return 1; }
  };

public:
  virtual void run_kastl(InputType&, OutputType& o)
  {
    kastl::generate_n(o.begin(), o.size(), gen_one());
  }

  virtual void run_stl(InputType&, OutputType& o)
  {
    std::generate_n(o.begin(), o.size(), gen_one());
  }

  virtual bool check
  (
   OutputType& os,
   OutputType& ok,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = ok.begin();
    SequenceType::iterator spos = os.begin();
    SequenceType::iterator send = os.end();

    return cmp_sequence(kpos, spos, send);
  }

};

#endif

class InnerProductRun : public RunInterface
{
  ValueType _kastl_res;
  ValueType _stl_res;

public:
  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ONE;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::inner_product
      (i.first.begin(), i.first.end(), i.second.begin(), ValueType(0));
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::inner_product
      (i.first.begin(), i.first.end(), i.second.begin(), ValueType(0));
  }

  virtual bool check
  (OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    error_string = value_error_string(_stl_res, _kastl_res);

    return false;
  }

};


#if 0

class AdjacentFindRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::adjacent_find(i.first.begin(), i.first.end());
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::adjacent_find(i.first.begin(), i.first.end());
  }

  virtual bool check
  (OutputType&, OutputType&, std::string& error_string) const
  {
    if (_kastl_res == _stl_res)
      return true;

    return false;
  }

};

#endif

class MinElementRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::min_element
      (i.first.begin(), i.first.end());
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::min_element
      (i.first.begin(), i.first.end());
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    if (*_kastl_res == *_stl_res)
      return true;

    error_string = value_error_string<ValueType>
      (*_stl_res, *_kastl_res);

    return false;
  }

};


class MaxElementRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::max_element
      (i.first.begin(), i.first.end());
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::max_element
      (i.first.begin(), i.first.end());
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    if (*_kastl_res == *_stl_res)
      return true;

    error_string = value_error_string<ValueType>
      (*_stl_res, *_kastl_res);

    return false;
  }

};


class MergeRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ASC;
    are_equal = false;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    // const ptrdiff_t size = i.second.size() / 2;
    const ptrdiff_t size = i.second.size();

    _kastl_res = kastl::merge
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.begin() + size,
     o.begin()
    );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    // const ptrdiff_t size = i.second.size() / 2;
    const ptrdiff_t size = i.second.size();

    _stl_res = std::merge
      (
       i.first.begin(), i.first.end(),
       i.second.begin(), i.second.begin() + size,
       o.begin()
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    const ptrdiff_t kastl_size = std::distance(kastl_output.begin(), _kastl_res);
    const ptrdiff_t stl_size = std::distance(stl_output.begin(), _stl_res);

    if (kastl_size != stl_size)
    {
      error_string = size_error_string(stl_size, kastl_size);
      return false;
    }

    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = _stl_res;

    if (cmp_sequence(kpos, spos, send) == false)
    {
      error_string = index_error_string
      (
       std::distance(stl_output.begin(), spos),
       std::distance(kastl_output.begin(), kpos)
      );

#if 0
      print_sequences(stl_output.begin(), kastl_output.begin(), kastl_output.end());
#endif

      return false;
    }

    return true;
  }

};


class SetUnionRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_RAND_ASC;
    are_equal = false;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::set_union
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::set_union
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    const ptrdiff_t kastl_size = std::distance(kastl_output.begin(), _kastl_res);
    const ptrdiff_t stl_size = std::distance(stl_output.begin(), _stl_res);

    if (kastl_size != stl_size)
    {
      error_string = size_error_string(stl_size, kastl_size);
      return false;
    }

    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = _stl_res;

    if (cmp_sequence(kpos, spos, send) == false)
    {
      error_string = index_error_string
      (
       std::distance(stl_output.begin(), spos),
       std::distance(kastl_output.begin(), kpos)
      );

#if 0
      print_sequences(stl_output.begin(), kastl_output.begin(), kastl_output.end());
#endif

      return false;
    }

    return true;
  }

};


class SetIntersectionRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ASC;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::set_intersection
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::set_intersection
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    const ptrdiff_t kastl_size = std::distance(kastl_output.begin(), _kastl_res);
    const ptrdiff_t stl_size = std::distance(stl_output.begin(), _stl_res);

    if (kastl_size != stl_size)
    {
      error_string = size_error_string(stl_size, kastl_size);
      print_sequences(stl_output.begin(), kastl_output.begin(), _kastl_res);
      return false;
    }

    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = _stl_res;

    if (cmp_sequence(kpos, spos, send) == false)
    {
      error_string = index_error_string
      (
       std::distance(stl_output.begin(), spos),
       std::distance(kastl_output.begin(), kpos)
      );

      print_sequences(stl_output.begin(), kastl_output.begin(), _kastl_res);

      return false;
    }

    return true;
  }

};


class SetDifferenceRun : public RunInterface
{
  SequenceType::iterator _kastl_res;
  SequenceType::iterator _stl_res;

public:

  virtual void get_seq_constraints
  (
   enum seq_order& seq_order,
   bool& are_equal
  ) const
  {
    seq_order = SEQ_ORDER_ASC;
    are_equal = true;
  }

  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _kastl_res = kastl::set_difference
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    _stl_res = std::set_difference
    (
     i.first.begin(), i.first.end(),
     i.second.begin(), i.second.end(),
     o.begin()
    );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    const ptrdiff_t kastl_size = std::distance(kastl_output.begin(), _kastl_res);
    const ptrdiff_t stl_size = std::distance(stl_output.begin(), _stl_res);

    if (kastl_size != stl_size)
    {
      error_string = size_error_string(stl_size, kastl_size);
      print_sequences(stl_output.begin(), kastl_output.begin(), _kastl_res);
      return false;
    }

    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = _stl_res;

    if (cmp_sequence(kpos, spos, send) == false)
    {
      error_string = index_error_string
      (
       std::distance(stl_output.begin(), spos),
       std::distance(kastl_output.begin(), kpos)
      );

      print_sequences(stl_output.begin(), kastl_output.begin(), _kastl_res);

      return false;
    }

    return true;
  }

};

#if 0

class AdjacentDifferenceRun : public RunInterface
{
public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    kastl::adjacent_difference
      (
       i.first.begin(),
       i.first.end(),
       o.begin()
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::adjacent_difference
      (
       i.first.begin(),
       i.first.end(),
       o.begin()
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.end();

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }

    return true;
  }

};

#endif


class CopyRun : public RunInterface
{
public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    kastl::copy
      (
       i.first.begin(),
       i.first.end(),
       o.begin()
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::copy
      (
       i.first.begin(),
       i.first.end(),
       o.begin()
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.end();

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }

    return true;
  }

};


#if 0

class ReplaceCopyIfRun : public RunInterface
{
  struct pred
  {
    pred() {}
    bool operator() (ValueType& n){ return n == 42; }
  };

  size_t _size;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _size = i.first.end() - i.first.begin();

    kastl::replace_copy_if
      (
       i.first.begin(),
       i.first.end(),
       o.begin(),
       pred(),
       24
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::replace_copy_if
      (
       i.first.begin(),
       i.first.end(),
       o.begin(),
       pred(),
       24
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.begin() + _size;

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }

    return true;
  }

};


class ReplaceCopyRun : public RunInterface
{
  size_t _size;

public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    _size = i.first.end() - i.first.begin();

    kastl::replace_copy
      (
       i.first.begin(),
       i.first.end(),
       o.begin(),
       42,
       24
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::replace_copy
      (
       i.first.begin(),
       i.first.end(),
       o.begin(),
       42,
       24
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.begin() + _size;

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }

    return true;
  }

};

#endif


class SwapRangesRun : public RunInterface
{
public:
  virtual void run_kastl(InputType& i, OutputType& o)
  {
    kastl::swap_ranges
      (
       i.first.begin(),
       i.first.end(),
       i.second.begin()
      );
  }

  virtual void run_stl(InputType& i, OutputType& o)
  {
    std::swap_ranges
      (
       i.first.begin(),
       i.first.end(),
       i.second.begin()
      );
  }

  virtual bool check
  (
   OutputType& kastl_output,
   OutputType& stl_output,
   std::string& error_string
  ) const
  {
#if 0
    SequenceType::iterator kpos = kastl_output.begin();
    SequenceType::iterator spos = stl_output.begin();
    SequenceType::iterator send = stl_output.end();

    if (cmp_sequence(kpos, spos, send) == false)
      {
	error_string = index_error_string
	  (
	   spos - stl_output.begin(),
	   kpos - kastl_output.begin()
	  );

	return false;
      }
#endif

    return true;
  }

};

#endif // speed compile time up


RunInterface* RunInterface::create(const std::string& name)
{
#define MATCH_AND_CREATE(NAME)	\
  if (name == #NAME)		\
    return new NAME ## Run();

  MATCH_AND_CREATE( Count );
  MATCH_AND_CREATE( ForEach );
  MATCH_AND_CREATE( Transform );
  MATCH_AND_CREATE( Search );

#if 0 // speed compile time up
  MATCH_AND_CREATE( Merge );
  MATCH_AND_CREATE( Sort );
  MATCH_AND_CREATE( PartialSum );
  MATCH_AND_CREATE( Accumulate );
  MATCH_AND_CREATE( MinElement );
  MATCH_AND_CREATE( MaxElement );
  MATCH_AND_CREATE( Find );
  MATCH_AND_CREATE( InnerProduct );
  MATCH_AND_CREATE( Copy );
  MATCH_AND_CREATE( Fill );
  MATCH_AND_CREATE( Replace );
  MATCH_AND_CREATE( Generate );
  MATCH_AND_CREATE( Equal );
  MATCH_AND_CREATE( Mismatch );
  MATCH_AND_CREATE( FindIf );
  MATCH_AND_CREATE( FindFirstOf );
  MATCH_AND_CREATE( SwapRanges );
  MATCH_AND_CREATE( Reverse );
  MATCH_AND_CREATE( Partition );
  MATCH_AND_CREATE( SetUnion );
  MATCH_AND_CREATE( SetIntersection );
  MATCH_AND_CREATE( SetDifference );
#endif // speed compile time up

#if 0
  MATCH_AND_CREATE( CountIf );
  MATCH_AND_CREATE( GenerateN );
  MATCH_AND_CREATE( ReplaceCopyIf );
  MATCH_AND_CREATE( ReplaceCopy );
  MATCH_AND_CREATE( ReplaceIf );
  MATCH_AND_CREATE( AdjacentDifference );
  MATCH_AND_CREATE( AdjacentFind );
  MATCH_AND_CREATE( SearchN );
#endif

  return NULL;
}


// runner

class RunLauncher
{
  timing::TimerType _kastl_tm;
  timing::TimerType _stl_tm;

public:
  void launch
  (
   RunInterface* run,
   InputType& input,
   OutputType& stl_output,
   OutputType& kastl_output
  )
  {
    // count the run count

    timing::TimerType start_tm;
    timing::TimerType now_tm;

    timing::get_now(start_tm);
    run->run_stl(input, stl_output);
    timing::get_now(now_tm);
    timing::sub_timers(now_tm, start_tm, _stl_tm);

    timing::get_now(start_tm);
    run->run_kastl(input, kastl_output);
    timing::get_now(now_tm);
    timing::sub_timers(now_tm, start_tm, _kastl_tm);
  }

  inline unsigned long kastl_usec() const
  { return timing::timer_to_usec(_kastl_tm); }

  inline unsigned long stl_usec() const
  { return timing::timer_to_usec(_stl_tm); }

};


// checker

class RunChecker
{
public:

  bool check
  (
   RunInterface* run,
   OutputType& stl_output,
   OutputType& kastl_output,
   std::string& error_string
  )
  {
    return run->check
      (
       kastl_output,
       stl_output,
       error_string
      );
  }

};


// main

int main(int ac, char** av)
{
  const size_t input_size =
    (ac > 2) ? atoi(av[2]) : DEFAULT_INPUT_SIZE;

  const size_t iter_count =
    (ac > 3) ? atoi(av[3]) : DEFAULT_ITER_SIZE;

  const char* const algo_name = av[1];

  timing::initialize();

  // run
  RunInterface* const run =
    RunInterface::create(std::string(algo_name));

  if (run == NULL)
  {
    printf("cannot create run for %s\n", algo_name);
    return -1 ;
  }

  // create sequences
  InputType input;
  input.first.resize(input_size);
  input.second.resize(input.first.size());
  enum seq_order seq_order;
  bool are_equal;
  run->get_seq_constraints(seq_order, are_equal);
  gen_seq(input.first, input.first.size(), seq_order);
  if (are_equal == true)
    std::copy(input.first.begin(), input.first.end(), input.second.begin());
  else
    gen_seq(input.second, input.second.size(), seq_order);

  // output
  OutputType stl_output;
  stl_output.resize(input.first.size() + input.second.size());
  OutputType kastl_output;
  kastl_output.resize(input.first.size() + input.second.size());

  run->prepare(input);

  for (size_t i = 0; i < iter_count; ++i)
  {
    RunLauncher l;

    l.launch(run, input, stl_output, kastl_output);
    std::string error_string;
    const bool is_success =
      RunChecker().check(run, stl_output, kastl_output, error_string);

    printf("%s %lu %lu [%c] (%s)\n",
	   algo_name,
	   l.kastl_usec(),
	   l.stl_usec(),
	   is_success ? 'x' : '!',
	   error_string.c_str());

    if (is_success == false)
      return -1;
  }

  delete run;

  return 0;
}
