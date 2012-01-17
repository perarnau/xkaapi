#include <stdio.h>

int
main (int argc, char **argv)
{
  int cpt = 123321;

#pragma omp parallel shared (cpt)
  {
    printf ("par-shared: cpt = %i\n", cpt);
  }

  return 0;
}
