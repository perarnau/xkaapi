/*
** xkaapi
** 
** Copyright 2011 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
double kaapi_get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

#pragma kaapi task write(result) read(r1,r2)
void sum( long* result, const long* r1, const long* r2)
{
  *result = *r1 + *r2;
}

#pragma kaapi task write(result) value(n)
void fibonacci(long* result, const long n)
{
  if (n<2)
    *result = n;
  else 
  {
#pragma kaapi data alloca(r1,r2)
    long r1,r2;
    fibonacci( &r1, n-1 );
#pragma kaapi notask    
    fibonacci( &r2, n-2 );
    sum( result, &r1, &r2);
  }
}

void print_result( double delay, long n, const long* result )
{
  printf("Fibonacci(%li)=%li\n", n, *result);
  printf("Time (s)=%lf\n", delay);
}

int main( int argc, char** argv)
{
  long result;
  double t0, t1;
  int n;
  if (argc >1) n = atoi(argv[1]);
  else n = 30;
  t0 = kaapi_get_elapsedtime();
#pragma kaapi parallel
  {
    fibonacci(&result, n);
  }
  t1 = kaapi_get_elapsedtime();
  print_result(t1-t0, n, &result);
  return 0;
}
