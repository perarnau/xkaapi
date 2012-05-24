#include <stdio.h>
#include <omp_ext.h>

int
main (int argc, char **argv)
{
  int i;
  
  omp_set_datadistribution_bloccyclic( 32, 64 );
#pragma omp parallel for schedule(dynamic)
  for (i=0; i<123; ++i)
  {
  }

  return 0;
}
