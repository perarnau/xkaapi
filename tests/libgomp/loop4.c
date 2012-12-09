#include <stdio.h>
#include <omp.h>
#include <omp_ext.h>
#include <unistd.h>

int
main (int argc, char **argv)
{
  int i;

  int TAB[32];
  
  omp_set_datadistribution_bloccyclic( 8, 4 );
#pragma omp parallel 
  {
    int r = omp_get_thread_num()*20;
#pragma omp for schedule(runtime)
    for (i=0; i<32; ++i)
    {
      TAB[i] = omp_get_thread_num();
      usleep(r);
    }
  }

  for (i=0; i<32; ++i)
    printf("TAB[%i] =%i\n",i,TAB[i]);

  return 0;
}
