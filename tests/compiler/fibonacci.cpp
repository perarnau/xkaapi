//#include <iostream>
//#include <vector>
//#include <sys/time.h>

/**
*/
double get_elapsedtime()
{
  return 0;
#if 0
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
#endif
}


#pragma kaapi task write (result) read(r1,r2) 
template<class T>
void sum( T* result, T* r1, T* r2)
{
  *result = *r1 + *r2;
}

/* specialization is not a task */
void sum( double* result, double* r1, double* r2)
{
  *result = *r1 + *r2;
}

// --------------------------------------------------------------------
/* Sequential fibo function
 */
#pragma kaapi task write(result) value(n)
void fibonacci(long* result, const long n)
{
  if(n<2)
  {
    *result = n;
  } 
  else 
  {
#pragma kaapi data alloca(r1,r2)
    long r1,r2;
    fibonacci( &r1, n-1 );
#pragma kaapi notask
    fibonacci( &r2, n-2 );
    sum( result, &r1, &r2 );
  }
}


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
#pragma kaapi init
  int n;
  double t0, t1;
  int niter;
  int i;

  if (argc >1)
    n = 1; //atoi(argv[1]);
  else 
    n = 30;
  if (argc >2)
    niter =  10; //atoi(argv[2]);
  else 
    niter = 1;

  long result;
  t0 = get_elapsedtime();
  for ( i=-1; i<niter; ++i)
  {
    if (i ==0) t0 = get_elapsedtime();
    fibonacci(&result, n);
  }
#pragma kaapi barrier
  t1 = get_elapsedtime();

  double s;
  sum( &s, &t0, &t1);

#pragma kaapi waiton(result);
#if 0
  std::cout << "After sync: Fibo(" << n << ")=" << result << std::endl;
  std::cout << "Time Fibo(" << n << "): " << (t1-t0)/niter << std::endl;
#endif
#pragma kaapi finish
  return 0;
}

