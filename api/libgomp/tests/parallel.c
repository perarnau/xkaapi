#include <stdio.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  int cpt = 0;
#pragma omp parallel num_threads(10) shared (cpt)
  {
#pragma omp critical
    cpt++;
  }

  return !(cpt == 10);
}
