#include <stdio.h>

int
main (int argc, char **argv)
{
  int i,j;
  
  for (j=0; j<1000; ++j)
  {
#pragma omp parallel for schedule(dynamic) private(i)
    for (i=0; i<123; ++i)
      printf ("Hello world! %i\n", i);
  }
  return 0;
}
