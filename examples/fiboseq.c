#include <stdio.h>
#include "kaapi_time.h"

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

int main(int argc, char** argv)
{
  int n = atoi(argv[1]);
  int result;
  double t0 = kaapi_get_elapsedtime();
  fiboseq(n, &result);
  double t1 = kaapi_get_elapsedtime();
  printf("Fibo(%i) = %i *** Time: %e(s)\n", n, result, t1-t0 );
  return 0;
}

