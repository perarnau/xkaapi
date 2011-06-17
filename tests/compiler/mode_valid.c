#pragma kaapi task write(n) read(m)
static void fu(unsigned int* m, unsigned int* n)
{
  m += 2;
  if (*m < 0) ;
  *n = *m;
  *n = 2 + *m;
  m = 0;
  n[(unsigned int)m + 5] = 0;
  n[*m] = 0;
  n[*m + 5] = 0;
}

int main(int ac, char** av)
{
  unsigned int bar = 0;

#pragma kaapi start
  {
    fu(&bar, &bar);
  }
#pragma kaapi barrier
#pragma kaapi finish

  return 0;
}
