// debug
double* __first;
extern "C" unsigned int kaapi_get_current_kid(void);


#include <stdio.h>
#include <sys/time.h>
#include <iterator>
#include "kastl_loop.h"
#include "kastl_sequences.h"


// find

template<typename Iterator>
struct find_body
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef kastl::impl::algorithm_result<Iterator> result_type;

  const value_type& _value;

  find_body(const value_type& value) : _value(value) {}

  bool operator()(result_type& res, const Iterator& pos)
  {
    // return true if result is found
    if (*pos != _value)
      return false;

    res.set_iter(pos);
    return true;
  }

  bool reduce(result_type& lhs, const result_type& rhs)
  {
    if (rhs._is_touched == false)
      return false;
    lhs.set_iter(rhs._iter);
    return true;
  }

};

template<typename Iterator, typename Value>
static Iterator find(Iterator first, Iterator last, const Value& value)
{
  // debug
  __first = first;

  kastl::rts::Sequence<Iterator> seq(first, last - first);
  kastl::impl::static_settings settings(128, 128);
  find_body<Iterator> body(value);
  kastl::impl::algorithm_result<Iterator> res(last);
  kastl::impl::reduce_unrolled_loop::run(res, seq, body, settings);
  return res._iter;
}


// accumulate
template<typename Iterator, typename Value>
struct accumulate_body
{
  // typedef Sequence<Iterator> sequence_type;

  typedef kastl::impl::numeric_result<Value> result_type;

  bool operator()(result_type& result, const Iterator& pos)
  {
    result._value += *pos;
    return false;
  }

  bool reduce(result_type& lhs, const result_type& rhs)
  {
    lhs._value += rhs._value;
    return false;
  }
};

template<typename Iterator, typename Value>
Value accumulate(Iterator first, Iterator last, const Value& value)
{
  kastl::rts::Sequence<Iterator> seq(first, last - first);
  kastl::impl::static_settings settings(128, 128);
  accumulate_body<Iterator, Value> body;
  kastl::impl::numeric_result<Value> result(value);
  kastl::impl::reduce_unrolled_loop::run
    (result, seq, body, settings);
  return result._value;
}


// for_each
template<typename Iterator, typename Operation>
struct for_each_body
{
  typedef kastl::rts::Sequence<Iterator> sequence_type;
  typedef typename sequence_type::range_type range_type;
  typedef kastl::impl::dummy_type result_type;

  Operation _op;

  for_each_body(const Operation& op, const Iterator& first)
    : _op(op)
  {}

  bool operator()(result_type&, range_type& range)
  {
    typedef typename range_type::iterator1_type iterator_type;

    iterator_type end = range.end();
    for (iterator_type pos = range.begin(); pos != end; ++pos)
      _op(*pos);

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
  kastl::impl::static_settings settings(128, 128);
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
    ++value; //  = _value
  }
};


template<typename Iterator>
static void init_seq(Iterator seq, size_t count)
{
  for (; count; --count, ++seq)
    *seq = 1;
}

template<typename Iterator>
static void __attribute__((unused)) check_seq(Iterator seq, size_t count)
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

#define ITEM_COUNT 1000000
  static value_type foo[ITEM_COUNT] = {};
  init_seq(foo, ITEM_COUNT);

#if 0 // foreach
  for_each(foo, foo + ITEM_COUNT, op<value_type>(FOO_VALUE));
  gettimeofday(&tm_end, NULL);
  check_seq_2(foo, ITEM_COUNT);
#elif 1 // accum
  const double accum = accumulate(foo, foo + ITEM_COUNT, value_type(0));
  gettimeofday(&tm_end, NULL);
  printf("res: %lf\n", accum);
  // check_seq_2(foo, ITEM_COUNT);
#elif 0 // find
  const size_t item_index = ITEM_COUNT - (ITEM_COUNT / 4);
  // const size_t item_index = ITEM_COUNT / 4;
  foo[item_index] = 42;
  value_type* const res = find(foo, foo + ITEM_COUNT, value_type(42));
  gettimeofday(&tm_end, NULL);
  if (res != (foo + item_index))
    printf("invalidResult(%lu != %lu)\n", res - foo, item_index);
#else
# error select an algorithm
#endif

  timersub(&tm_end, &tm_start, &tm_diff);
  printf("%lu.%lu\n", tm_diff.tv_sec, tm_diff.tv_usec);

  return 0;
}
