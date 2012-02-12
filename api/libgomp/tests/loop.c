#include <stdio.h>

int
main (int argc, char **argv)
{
  int i;
  
#pragma omp parallel for schedule(dynamic)
  for (i=0; i<123; ++i)
  {
    printf ("Hello world! %i\n", i);
  }

  return 0;
}
