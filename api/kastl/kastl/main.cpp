#include <stdio.h>
#include <sys/time.h>
#include <iterator>
#include "kastl/numeric"
#include "kastl/algorithm"


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
  kastl::for_each(foo, foo + ITEM_COUNT, op<value_type>(FOO_VALUE));
  gettimeofday(&tm_end, NULL);
  check_seq_2(foo, ITEM_COUNT);
#elif 1 // accum
  const double accum = kastl::accumulate(foo, foo + ITEM_COUNT, value_type(0));
  gettimeofday(&tm_end, NULL);
  printf("res: %lf\n", accum);
  // check_seq_2(foo, ITEM_COUNT);
#elif 0 // find
  const size_t item_index = ITEM_COUNT - (ITEM_COUNT / 4);
  // const size_t item_index = ITEM_COUNT / 4;
  foo[item_index] = 42;
  value_type* const res = kastl::find(foo, foo + ITEM_COUNT, value_type(42));
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
