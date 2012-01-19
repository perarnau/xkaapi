#include <stdio.h>

#include "test-toolbox.h"

int
main (int argc, char **argv)
{
  volatile int cpt = 0;
  
#pragma omp parallel shared (cpt)
  {
#pragma omp single
    cpt++;
  }

  test_check ("single", (cpt == 1));
  
  return 0;
}
