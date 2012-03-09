#include <stdio.h>

int
main (int argc, char **argv)
{
  int cpt = 123321;

#pragma omp parallel shared (cpt)
  {
    printf ("%i::par-shared: cpt = %i\n", omp_get_thread_num(), cpt);
  }

  return 0;
}
