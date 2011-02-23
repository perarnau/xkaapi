#include <stdio.h>
#include <sys/types.h>


// just a small for now. will change as soon as the
// linear work adaptive interface gets integrated

#include "kaLinearWork.hh"

class varWork;

class varResult : public ka::linearWork::baseResult
{
public:
  double _sum_x;
  double _sum_xx;

  varResult() : _sum_x(0.), _sum_xx(0.) { }

  void initialize(const varWork&)
  {
    _sum_x = 0.;
    _sum_xx = 0.;
  }
};


class varWork : public ka::linearWork::baseWork
{
  // compute the sequence variance

public:

  typedef ka::linearWork::range range_type;

  static const bool is_reducable = true;
  static const unsigned int seq_grain = 256;
  static const unsigned int par_grain = 256;

  const double* _x;

  varWork(const double* x, size_t n) :
    ka::linearWork::baseWork(0, (range_type::index_type)n), _x(x)
  {}

  void initialize(const varWork& w)
  {
    // initialize the splitted work
    _x = w._x;
  }

  void execute(varResult& res, const range_type& r)
  {
    double sum_x = 0.;
    double sum_xx = 0.;

    for (range_type::index_type i = r.begin(); i < r.end(); ++i)
    {
      const double x = _x[i];
      sum_x += x;
      sum_xx += x * x;
    }

    res._sum_x += sum_x;
    res._sum_xx += sum_xx;
  }

  void reduce
  (varResult& lhs, const varResult& rhs, const range_type&)
  {
    lhs._sum_x += rhs._sum_x;
    lhs._sum_xx += rhs._sum_xx;
  }

};


static double var
(const double* x, size_t _n, bool)
{
  const double n = (double)_n;

  varWork work(x, _n);
  varResult res;
  ka::linearWork::execute(work, res);

  const double ave = res._sum_x / n;
  return (res._sum_xx + ave * (n * ave - 2 * res._sum_x)) / n;
}


// sequential implem

static double mean(const double* x, size_t n)
{
  double sum = 0.;
  for (size_t i = 0; i < n; ++i) sum += x[i];
  return sum / (double)n;
}

__attribute__((unused))
static double var(const double* x, size_t n)
{
  // sum( (xi - ave)^2 )

  const double ave = mean(x, n);

  double sum = 0.;
  for (size_t i = 0; i < n; ++i)
  {
    const double dif = x[i] - ave;
    sum += dif * dif;
  }

  return sum / (double)n;
}


int main(int ac, char** av)
{
  ka::linearWork::toRemove::initialize();

  // generate a random vector
  const size_t n = 1024 * 1024;
  double* const x = (double*)malloc(n * sizeof(double));
  for (size_t i = 0; i < n; ++i) x[i] = rand() % 100;

  uint64_t start = kaapi_get_elapsedns();

  volatile double dont_optimize;
  for (unsigned int iter = 0; iter < 1000; ++iter)
    dont_optimize = var(x, n, true);

  uint64_t stop = kaapi_get_elapsedns();
  double par_time = (double)(stop - start) / (1000 * 1E6);

  printf("%lf\n", par_time);

  free(x);

  ka::linearWork::toRemove::finalize();

  return 0;
}
