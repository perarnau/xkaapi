#pragma kaapi task read(arr) value(size)
static void fu(double* arr, unsigned int size)
{
  /* this triggers a bug in the kaapi mode analysis
     due to SageInterface::collectReadWriteVariables
     not handling correclty the name associated with
     *(arr + i).
     to fix it, we should run the ref version of the
     analysis and process the results ourselves taking
     care of not doing the same mistk when handling
     names.
     ie. file: sageInterface.C, line: 13176
   */

  unsigned int i;
  double r;
  for (i = 0; i < size; ++i)
    *(arr + i) = 0;
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
