#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


typedef double real_type;


/* reduction */

static void reduce_sum(real_type* lhs, const real_type* rhs)
{
  printf("reduce_sum: %lx(%lf), %lx(%lf)\n", (uintptr_t)lhs, *lhs, (uintptr_t)rhs, *rhs);
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

/* main */

int main(int ac, char** av)
{
  static const unsigned int n = 1024 * 1024;
  real_type* const v = malloc(n * sizeof(real_type));
  real_type sum;
  unsigned int i;
  for (i = 0; i < n; ++i) v[i] = 1;

#pragma kaapi parallel
  sum = accumulate(v, n);

  printf("%lf\n", sum);
  free(v);
  return 0;
}
