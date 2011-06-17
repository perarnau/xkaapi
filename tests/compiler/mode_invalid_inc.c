#pragma kaapi task read(m)
static void fu(unsigned int* m)
{
  ++*m;
  (*m)++;
}

int main(int ac, char** av)
{
  unsigned int bar = 0;

#pragma kaapi start
  {
    fu(&bar);
  }
#pragma kaapi barrier
#pragma kaapi finish

  return 0;
}
