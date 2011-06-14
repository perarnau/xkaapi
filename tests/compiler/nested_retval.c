#include <stdio.h>


#pragma kaapi task value(n)
static double fu(double n)
{
  /* printf("%s(%lf)\n", __FUNCTION__, n); */
  return n + 1;
}

static double bar(double m, double n)
{
  return m + n;
}

int main(int ac, char** av)
{
  double res;

#pragma kaapi start
  {
    /* bar(fu(42), fu(24)); */
    res = bar(fu(40), 1);
  }
#pragma kaapi barrier
#pragma kaapi finish

  printf("res == %lf\n", res);

  return 0;
}
