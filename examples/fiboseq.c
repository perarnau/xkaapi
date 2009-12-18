#include <stdio.h>
#include <stdlib.h>

/** The macro FIBOCODE and MAIN CODE are used to allows separate compilation
    of the fibo code and the main code.
*/
extern double kaapi_get_elapsedtime();

#if defined(FIBOCODE)
/* Sequential fibo function
*/
void fiboseq(int n, int* r)
{ 
  if (n <2) {
    *r = n;
  }
  else {
    int r1;
    int r2;
    fiboseq(n-1, &r1);
    fiboseq(n-2, &r2);
    *r = r1+r2;
  }
}
#else
extern void fiboseq(int n, int* r);
#endif

#if defined(MAINCODE)
int main(int argc, char** argv)
{
  int i;
  int n;
  int niter;
  int result = 0;
  double t0, t1;

  if (argc >1)
    n = atoi(argv[1]);
  else 
    n = 20;
  if (argc >2)
    niter =  atoi(argv[2]);
  else 
    niter = 1;
    
  t0 = kaapi_get_elapsedtime();
  {
    for (i=0; i<niter; ++i)
    {
        fiboseq(n, &result);
    }
  }
  t1 = kaapi_get_elapsedtime();
/*  printf("Fibo(%i) = %i *** Time: t1=%e(s), t0=%e(s)\n", n, result, t1,t0 );*/
  printf("Fibo(%i) = %i *** Time(s): %e\n", n, result, (t1-t0)/(double)niter );
  return 0;
}
#endif
