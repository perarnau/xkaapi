static void baz(double* n)
{
  *n = 0;
}

#pragma kaapi task read(m) write(n)
static void fu(double* m, double* n)
{
  *n = 0;
  *m = 42;
  baz(n);
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
