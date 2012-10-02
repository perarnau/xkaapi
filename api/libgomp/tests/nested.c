#include <stdio.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  omp_set_nested (1);
  int cpt = 0;
  
#pragma omp parallel num_threads (2) shared (cpt)
  {
#pragma omp parallel num_threads (2) shared (cpt)
    {
#pragma omp atomic
      cpt++;
    }
  }
  
  return !(cpt == 4);
}
