#include <stdio.h>
#include <omp_ext.h>

int
main (int argc, char **argv)
{
  int i;
  
  omp_set_datadistribution_bloccyclic( 8, 4 );
#pragma omp parallel for schedule(dynamic)
  for (i=0; i<32; ++i)
  {
    printf("TAB[%i] !\n",i);
  }

  return 0;
}
