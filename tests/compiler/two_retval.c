#include <stdio.h>

#pragma kaapi task value(n)
static double fu(double n)
{
  printf("%s(%lf)\n", __FUNCTION__, n);
  return n + 1;
}

static void bar(double a, double b) {}

int main(int ac, char** av)
{
#pragma kaapi start
  {
    bar(fu(0), fu(1));
  }
#pragma kaapi finish


  return 0;
}
