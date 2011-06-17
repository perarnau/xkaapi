static void baz(unsigned int p, double* q)
{
  *q = 0;
}

#pragma kaapi task read(m) write(n)
static void fu(double* m, double* n)
{
  *n = 0;
  baz(42, m + 2);
  baz(42, m);
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
