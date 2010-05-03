#include <iostream>
#include <sys/time.h>
#include <stdlib.h>

/**
*/
double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

/*
*/
long fibonacci(long n)
{
   if (n < 2) return n;
   long x, y;
#pragma omp task shared(x)
   x = fibonacci(n - 1);
#pragma omp task shared(y)
   y = fibonacci(n - 2);
#pragma omp taskwait
   return x+y;
}

int main(int argc, char* argv[])
{
  unsigned int n = 30;
  if (argc > 1) n = atoi(argv[1]);
  unsigned int iter = 1;
  if (argc > 2) iter = atoi(argv[2]);

  double start_time, stop_time;
  long res = 0;

  start_time = get_elapsedtime();
  for (int i=0; i<iter; ++i)
#pragma omp parallel
  {
#pragma omp single
    {
       res = fibonacci(n);
    }
  } /* end omp parallel */
  stop_time = get_elapsedtime();
  std::cout << ": Res = " << res << std::endl;
  std::cout << ": Time(s): " << (stop_time-start_time)/iter << std::endl;
  return 0;
}

