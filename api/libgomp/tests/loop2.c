#include <stdio.h>

int
main (int argc, char **argv)
{
  int tab[123] = { 0 };
  int res = 0;
  
  for (int j = 0; j < 1000; ++j)
  {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 123; ++i)
      tab[i]++;
  }
  
  for (int i = 0; i < 123; ++i)
    res += tab[i];
  
  return !(res == 123000);
}
