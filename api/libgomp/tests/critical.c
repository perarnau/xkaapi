#include <stdio.h>
#include <omp.h>

#include "test-toolbox.h"

int
main (int argc, char **argv)
{
  int cpt = 0;
  int nthreads = -1;

#pragma omp parallel shared (cpt)
  {
    nthreads = omp_get_num_threads ();

#pragma omp critical
    cpt++;
  }

  test_check ("critical", (cpt == nthreads));

  return 0;
}
