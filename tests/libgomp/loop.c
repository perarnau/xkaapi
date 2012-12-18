#include <stdio.h>

int
main (int argc, char **argv)
{
  int tab[123] = { 0 };
  int i;
  
#pragma omp parallel for schedule(dynamic) shared (tab)
  for (i = 0; i < 123; ++i)
    tab[i] = 1;

  int res = 0;
  
  for (i = 0; i < 123; ++i)
    res += tab[i];
    
  return !(res == 123);
}
