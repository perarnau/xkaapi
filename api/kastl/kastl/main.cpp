#include <stdio.h>
#include <sys/time.h>
#include "kastl_loop.h"
#include "kastl_sequences.h"



// find
template<typename Iterator, typename Value>
struct find_body
{
  typedef Value result_type;

  const Value& _value;

  find_body(const Value& value) : _value(value) {}

  bool operator()(result_type& result, const Iterator& pos)
  {
    // return true if result is found
    if (*pos != _value)
      return false;

    result = pos;
    return true;
  }

  bool reduce(Value&, const Value&)
  {
    return false;
  }

  void init_result(result_type&)
  {
  }

};

template<typename Iterator, typename Value>
static Iterator find(Iterator first, Iterator last, const Value& value)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  kastl::impl::static_settings settings(512, 512);
  find_body<Iterator, Value> body(value);
  Iterator res = last;
  return kastl::impl::reduce_loop::run(res, seq, body, settings);
}


// accumulate
template<typename Iterator, typename Value>
struct accumulate_body
{
  // typedef Sequence<Iterator> sequence_type;

  bool operator()(Value& result, Iterator& pos)
  {
    result += *pos;
    *pos += 1;
    return false;
  }

  bool reduce(Value& lhs, const Value& rhs)
  {
    lhs += rhs;
    return false;
  }

  void init_result(Value& result)
  {
    result = static_cast<Value>(0);
  }

};

template<typename Iterator, typename Value>
Value accumulate(Iterator first, Iterator last, const Value& value)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  kastl::impl::static_settings settings(1024, 1024);
  accumulate_body<Iterator, Value> body;
  Value result = value;
  return kastl::impl::reduce_unrolled_loop::run
    (result, seq, body, settings);
}


// for_each
template<typename Iterator, typename Operation>
struct for_each_body
{
  typedef kastl::rts::Sequence<Iterator> sequence_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::dummy_type result_type;

  Operation _op;

#if CONFIG_KASTL_DEBUG
  const Iterator& _first;
#endif

  for_each_body(const Operation& op, const Iterator& first)
    : _op(op)
#if CONFIG_KASTL_DEBUG
    , _first(first)
#endif
  {}

  bool operator()(result_type&, range_type& range)
  {
    typedef typename range_type::iterator1_type iterator_type;

    iterator_type end = range.end();

#if CONFIG_KASTL_DEBUG
    printf(">>(%ld - %ld)\n", range.begin() - _first, range.end() - _first);
    fflush(stdout);
#endif

    for (iterator_type pos = range.begin(); pos != end; ++pos)
      _op(*pos);

#if CONFIG_KASTL_DEBUG
    printf("<<\n"); fflush(stdout);
#endif

    return false;
  }
};

template<typename Iterator, typename Operation, typename Settings>
void for_each(Iterator first, Iterator last, Operation op, const Settings& settings)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  for_each_body<Iterator, Operation> body(op, first);
  kastl::impl::parallel_loop::run(seq, body, settings);
}

template<typename Iterator, typename Operation>
void for_each(Iterator first, Iterator last, Operation op)
{
  kastl::impl::static_settings settings(1024, 1024);
  for_each(first, last, op, settings);
}

template<typename T>
struct op
{
#define FOO_VALUE 1

  const T _value;

  op(const T& value) : _value(value)
  {}

  void operator()(T& value)
  {
    value = _value;
  }
};


template<typename Iterator>
static void init_seq(Iterator seq, size_t count)
{
  for (; count; --count, ++seq)
    *seq = 1;
}

template<typename Iterator>
static void check_seq(Iterator seq, size_t count)
{
  unsigned int i = 0;

  for (; count; --count, ++seq, ++i)
    if (*seq != FOO_VALUE)
    {
      printf("foo @%u\n", i);
      break;
    }
}

template<typename Iterator>
static void __attribute__((unused)) check_seq_2(Iterator seq, size_t count)
{
  unsigned int i = 0;
  for (; count; --count, ++seq, ++i)
    if (*seq != 2)
    {
      printf("invalid@%u(%lf)\n", i, *seq);
      break;
    }
}

int main()
{
  typedef double value_type;

  struct timeval tm_start;
  struct timeval tm_end;
  struct timeval tm_diff;

  gettimeofday(&tm_start, NULL);

#define ITEM_COUNT 30000
  static value_type foo[ITEM_COUNT] = {};
  init_seq(foo, ITEM_COUNT);

#if 0 // foreach
  for_each(foo, foo + ITEM_COUNT, op<value_type>(FOO_VALUE));
  gettimeofday(&tm_end, NULL);
  check_seq(foo, ITEM_COUNT);
#else // accum
  const double accum = accumulate(foo, foo + ITEM_COUNT, value_type(0));
  gettimeofday(&tm_end, NULL);
  printf("res: %lf\n", accum);
  check_seq_2(foo, ITEM_COUNT);
#endif

  timersub(&tm_end, &tm_start, &tm_diff);
  printf("%lu.%lu\n", tm_diff.tv_sec, tm_diff.tv_usec);

  return 0;
}
