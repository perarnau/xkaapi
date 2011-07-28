#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


typedef double real_type;


/* reduction */

static void reduce_sum(real_type* lhs, const real_type* rhs)
{
  *lhs += *rhs;
}

#pragma kaapi declare reduction(reduce_sum_id: reduce_sum)


static void accum3
(
 const real_type* u, real_type* usum,
 const real_type* v, real_type* vsum,
 const real_type* w, real_type* wsum,
 unsigned int n
)
{
  unsigned int i;

  real_type local_usum = 0;
  real_type local_vsum = 0;
  real_type local_wsum = 0;


#pragma kaapi loop \
reduction(reduce_sum_id:local_usum, reduce_sum_id:local_vsum, reduce_sum_id:local_wsum)
  for (i = 0; i < n; ++i)
  {
    local_usum += u[i];
    local_vsum += v[i];
    local_wsum += w[i];
  }

  *usum = local_usum;
  *vsum = local_vsum;
  *wsum = local_wsum;
}

/* main */

int main(int ac, char** av)
{
  static const unsigned int n = 8 * 1024 * 1024;
  real_type* const u = malloc(n * sizeof(real_type));
  real_type* const v = malloc(n * sizeof(real_type));
  real_type* const w = malloc(n * sizeof(real_type));
  real_type usum, vsum, wsum;
  unsigned int i;

  for (i = 0; i < n; ++i)
  {
    u[i] = 1;
    v[i] = 2;
    w[i] = 3;
  }

  for (i = 0; i < 10; ++i)
  {
#pragma kaapi parallel
    accum3(u, &usum, v, &vsum, w, &wsum, n);

    if (usum != (real_type)n)
      printf("invalid: %lf\n", usum);
    else if (vsum != (real_type)(2 * n))
      printf("invalid: %lf\n", vsum);
    else if (wsum != (real_type)(3 * n))
      printf("invalid: %lf\n", wsum);
  }

  free(u);
  free(v);
  free(w);

  return 0;
}
