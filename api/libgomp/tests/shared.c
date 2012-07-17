#include <stdio.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  int cpt = 123321;
  int res[4];

#pragma omp parallel shared (cpt, res) num_threads (4)
  res[omp_get_thread_num ()] = cpt;
  
  return !(res[0] == res[1] && res[0] == res[2] && res[0] == res[3]);
}
