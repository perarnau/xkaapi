extern void baz(unsigned int*);

#pragma kaapi task write(n) read(m)
static void fu(unsigned int* m, unsigned int* n)
{
  baz(m);
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
