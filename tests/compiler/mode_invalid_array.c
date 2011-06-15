#pragma kaapi task read(arr) value(size)
static void fu(double* arr, unsigned int size)
{
  unsigned int i;
  for (i = 0; i < size; ++i)
    arr[i] = 0;
}

int main(int ac, char** av)
{
  double bar[42];

#pragma kaapi start
  {
    fu(bar, 42);
  }
#pragma kaapi barrier
#pragma kaapi finish

  return 0;
}
