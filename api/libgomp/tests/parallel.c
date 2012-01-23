#include <stdio.h>

int
main (int argc, char **argv)
{

#pragma omp parallel num_threads(10)
  {
    printf ("Hello world! %i\n", omp_get_thread_num());
  }

  return 0;
}
