/* static void baz(double* n) */
/* { */
/*   *n = 0; */
/* } */

#pragma kaapi task read(n)
static void fu(double* n)
{
  *n = 0;
  /* baz(n); */
}

int main(int ac, char** av)
{
  double bar = 0;

#pragma kaapi start
  {
    fu(&bar);
  }
#pragma kaapi barrier
#pragma kaapi finish

  return 0;
}
