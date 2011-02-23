#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "modp.h"
#include "kaLinearWork.hh"


// index to degree mapping functions
static inline unsigned long to_index
(unsigned long n, unsigned long polynom_degree)
{ return polynom_degree - n; }

static inline unsigned long to_degree
(unsigned long i, unsigned long polynom_degree)
{ return polynom_degree - i; }


// fwd decl
class hornerWork;

class hornerResult : public ka::linearWork::baseResult
{
  // every result must inherit from baseResult
  // must implement a constructor with the sig
  // result_type(const work_type&, ka::linearWork::splitTag)

public:
  unsigned long _res;

  hornerResult(const unsigned long* a, unsigned long n)
  { _res = a[to_index(n, n)]; }

  hornerResult(unsigned long res) : _res(res) {}

  void initialize(const hornerWork&) { _res = 0; }
};

class hornerWork : public ka::linearWork::baseWork
{
public:

  typedef ka::linearWork::range range_type;

  // problem specific data

  unsigned long _x;
  const unsigned long* _a;
  unsigned long _n;

  hornerWork
  (unsigned long x, const unsigned long* a, unsigned long n)
    : baseWork(0, n), _x(x), _a(a), _n(n) {}

  // implements splittableWork interface
  // the interface is composed of:
  // overridable traits
  // void initialize(const work_type&);
  // void execute(result_type&, const range&);
  // void reduce(result_type&, const result_type&);

  static const bool is_reducable = true;
  static const unsigned int seq_grain = 256;
  static const unsigned int par_grain = 256;

  void initialize(const hornerWork& w)
  {
    // initialize the splitted work
    _x = w._x;
    _a = w._a;
    _n = w._n;
  }

  void execute(hornerResult& res, const range_type& r)
  {
    // map range to actual work and process

    unsigned long hi = to_degree(r.begin(), _n);
    const unsigned long lo = to_degree(r.end(), _n);

    unsigned long local_res = res._res;

    for (unsigned long j = to_index(hi - 1, _n); hi > lo; --hi, ++j)
      local_res = axb_modp(local_res, _x, _a[j]);

    res._res = local_res;
  }

  void reduce
  (hornerResult& lhs, const hornerResult& rhs, const range_type& processed)
  {
    // lhs += rhs with rhs the preempted work
    lhs._res = axnb_modp(lhs._res, _x, processed.size(), rhs._res);
  }

};


// main

static unsigned long* make_rand_polynom(unsigned long n)
{
  unsigned long* const a = (unsigned long*)malloc
    ((n + 1) * sizeof(unsigned long));

  for (unsigned long i = 0; i <= n; ++i)
    a[i] = modp(rand());

  return a;
}

int main(int ac, char** av)
{
  static const unsigned long n = 1024 * 1024;
  unsigned long* const a = make_rand_polynom(n);
  static const unsigned long x = 2;

  ka::linearWork::toRemove::initialize();

  volatile unsigned long sum_par = 0;

  uint64_t start = kaapi_get_elapsedns();

  for (unsigned int iter = 0; iter < 100; ++iter)
  {
    hornerWork work(x, a, n);
    hornerResult res(a, n);
    ka::linearWork::execute(work, res);
    sum_par += res._res;
  }

  uint64_t stop = kaapi_get_elapsedns();
  double par_time = (double)(stop - start) / (100 * 1E6);
  printf("%u %lf %lu\n", kaapi_getconcurrency(), par_time, sum_par);

  ka::linearWork::toRemove::finalize();

  free(a);

  return 0;
}
