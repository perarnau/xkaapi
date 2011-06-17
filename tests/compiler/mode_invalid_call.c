static void fubar(double* r, double* w)
{
  *w = 0;
}

static void baz(double* p, double* q)
{
  fubar(p, q);
}

#pragma kaapi task read(m) write(n)
static void fu(double* m, double* n)
{
  baz(n, m + 2);
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
