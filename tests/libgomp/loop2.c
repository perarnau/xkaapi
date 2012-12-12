#include <stdio.h>

int
main (int argc, char **argv)
{
  int tab[123] = { 0 };
  int res = 0;
  int i,j;
  
  for (j = 0; j < 1000; ++j)
  {
#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < 123; ++i)
      tab[i]++;
  }
  
  for (i = 0; i < 123; ++i)
    res += tab[i];
  
  return !(res == 123000);
}
