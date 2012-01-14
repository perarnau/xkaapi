#include <stdio.h>

int
main (int argc, char **argv)
{
  int cpt = 0;

#pragma omp parallel shared (cpt)
  {
#pragma omp critical
    cpt++;
  }

  printf ("par-shared: cpt = %i\n\n", cpt);

  return 0;
}
