#include <stdio.h>


#pragma kaapi task value(n)
static double fu(double n)
{
  /* printf("%s(%lf)\n", __FUNCTION__, n); */
  return n + 1;
}

static double bar(double m, double n)
{
  return fu(m) + n;
}

int main(int ac, char** av)
{
  double res;

#pragma kaapi start
  {
    res = bar(41, 1);
  }
#pragma kaapi barrier
#pragma kaapi finish

  printf("%lf\n", res);

  return 0;
}
