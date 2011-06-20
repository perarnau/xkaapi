#include <stdio.h>


#pragma kaapi task value(n)
static double fu(double n)
{
  printf("%s(%lf)\n", __FUNCTION__, n);
  return n + 1;
}

int main(int ac, char** av)
{
  double res = 0.;

#pragma kaapi start
  {
    res = fu(0);
#pragma kaapi barrier
    res = fu(res);
#pragma kaapi barrier
  }
#pragma kaapi finish


  return 0;
}
