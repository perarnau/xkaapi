#include <stdio.h>


#pragma kaapi task value(n)
static double fu(double n)
{
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
    res = bar(fu(10) + fu(0), fu(29));
  }
#pragma kaapi barrier
#pragma kaapi finish

  printf("res == %lf\n", res);

  return 0;
}
