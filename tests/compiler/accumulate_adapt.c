#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


typedef double real_type;


/* reduction */

static void reduce_sum(real_type* lhs, const real_type* rhs)
{
  /* printf("reduce_sum: %lx(%lf), %lx(%lf)\n", (uintptr_t)lhs, *lhs, (uintptr_t)rhs, *rhs); */
  *lhs += *rhs;
}

#pragma kaapi declare reduction(reduce_sum_id: reduce_sum)


static real_type accumulate(const real_type* v, unsigned int n)
{
  unsigned int i;
  real_type sum = 0;
#pragma kaapi loop reduction(reduce_sum_id:sum)
  for (i = 0; i < n; ++i) sum += v[i];
  return sum;
}


/* temporary */

#ifndef __USE_BSD
# define __USE_BSD
#endif

#include <time.h>
#include <sys/time.h>

struct timeval tms[3];

static void pragma_kaapi_start_time(void)
{
  gettimeofday(&tms[0], NULL);
}

static void pragma_kaapi_stop_time(void)
{
  double msecs;
  gettimeofday(&tms[1], NULL);
  timersub(&tms[1], &tms[0], &tms[2]);
  msecs = (double)tms[2].tv_sec * 1E3 + (double)tms[2].tv_usec / 1E3;
  printf("%lf\n", msecs);
}


/* main */

int main(int ac, char** av)
{
  static const unsigned int n = 4 * 1024 * 1024;
  real_type* const v = malloc(n * sizeof(real_type));
  real_type sum;
  unsigned int j;
  unsigned int i;
  for (i = 0; i < n; ++i) v[i] = 1;

#pragma kaapi parallel
  {
    pragma_kaapi_start_time();

    for (j = 0; j < 100; ++j)
    {
      sum = accumulate(v, n);
      if (sum != n) printf("invalid: %lf\n", sum);
    }

    pragma_kaapi_stop_time();
  }

  free(v);
  return 0;
}
