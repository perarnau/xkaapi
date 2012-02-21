#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <strings.h>
#include <omp.h>

/**
*/
double get_elapsedtime()
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}


void fu(int i)
{
  int l = (rand_r(&i) % 100);
  for ( ; l>0; --l)
    sin((double)(unsigned long)&i);
}

int
main (int argc, char **argv)
{
  int i,j;
  int iter = 1000;
  double t0, t1;  
  double alldelay[48];
  bzero(alldelay, sizeof(double)*48 );

  t0 = get_elapsedtime();
  for (j=0; j<iter; ++j)
  {
#pragma omp parallel 
   {
      double t0, t1;  
      t0 = get_elapsedtime();
#pragma omp for schedule(runtime) private(i)
      for (i=0; i<10000; ++i)
        fu(i);
      t1 = get_elapsedtime();
      alldelay[omp_get_thread_num()] += t1-t0;
    }
    fputc('.',stdout);
    if (j % 50 ==0) fflush(stdout);
  }
  t1 = get_elapsedtime();

  printf("Time: %f (s)\n", (t1-t0)/iter );
  return 0;
}
