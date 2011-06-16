#pragma kaapi task write(m) read(n)
static void fu(double* m, double* n)
{
  *n = 42;
}

int main(int ac, char** av)
{
  double bar = 0;

#pragma kaapi start
  {
    fu(&bar, &bar);
  }
#pragma kaapi barrier
#pragma kaapi finish

  return 0;
}
