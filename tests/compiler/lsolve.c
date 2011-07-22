#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef double real_type;


/* reduction */

static void reduce_sum(real_type* lhs, const real_type* rhs)
{
  *lhs += *rhs;
}

#pragma kaapi declare reduction(reduce_sum_id: reduce_sum)


/* solve ax = b */

static void lsolve
(
 real_type* a,
 unsigned int lda,
 unsigned int m,
 real_type* b
)
{
  /* solve a * x = b where:
     a a m,m lower triangular matrix
     b a m vector
     x is written on b
  */

  real_type* const x = b;

  unsigned int i;
  unsigned int j;
  real_type sum;

  for (i = 0; i < m; ++i)
  {
    sum = 0;
#pragma kaapi loop reduction(reduce_sum_id:sum)
    for (j = 0; j < i; ++j)
      sum += a[i * lda + j] * x[j];
    x[i] = (b[i] - sum) / a[i * lda + i];
  }
}


/* unit testing */

static int check
(
 const real_type* a,
 unsigned int lda,
 unsigned int m,
 const real_type* x,
 const real_type* b
)
{
  /* a[m, m] of leading dimension lda */

  unsigned int i;
  unsigned int j;
  real_type sum;

  for (i = 0; i < m; ++i)
  {
    sum = 0;
    for (j = 0; j <= i; ++j)
      sum += a[i * lda + j] * x[j];

    /* above threshold error */
    if (fabs(sum - b[i]) > 0.001) return -1;
  }

  return 0;
}

static void get_ab
(
 real_type* a,
 unsigned int lda,
 unsigned int m,
 real_type* b
)
{
  unsigned int i;
  unsigned int j;

  for (i = 0; i < m; ++i)
  {
    b[i] = 1;
    for (j = 0; j < m; ++j)
      a[i * lda + j] = 1;
  }
}


/* main */

int main(int ac, char** av)
{
  static const unsigned int m = 4;
  static const unsigned int lda = 4;

  real_type* const a = malloc(m * m * sizeof(real_type));
  real_type* const b = malloc(m * sizeof(real_type));
  real_type* const x = b;
  real_type* const saved_b = malloc(m * sizeof(real_type));

  get_ab(a, lda, m, b);
  memcpy(saved_b, b, m * sizeof(real_type));

#pragma kaapi parallel
  lsolve(a, lda, m, b);

  if (check(a, lda, m, x, saved_b))
    printf("invalid\n");

  free(a);
  free(b);
  free(saved_b);

  return 0;
}
