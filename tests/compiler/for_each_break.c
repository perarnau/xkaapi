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
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/**
*/
double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

#pragma kaapi task value(size) write(array[size]) value (op)
void for_each( double* array, size_t size, void (*op)(double*) )
{
  for (size_t i=0; i<size; ++i)
    op(&array[i]);
}


/**
 */
static void apply_cos( double* v )
{
  *v += cos(*v);
}

/**
 */
int main(int argc, char** argv)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;
  size_t niter;
  
  niter = atoi(argv[1]);
  
#define ITEM_COUNT 100000
  static double array[ITEM_COUNT];

  /* initialize, apply, check */
  for (i = 0; i < ITEM_COUNT; ++i)
    array[i] = 0.f;
  
#pragma kaapi parallel 
{
  /* initialize the runtime */
  t0 = get_elapsedtime();
  for (iter = 0; iter < niter; ++iter)
  {
    for_each( array, ITEM_COUNT, apply_cos );
    if (iter == 2) goto end;
    if (iter == 3) break;
    if (iter == 4) return 0;
    ++iter;
  }
  t1 = get_elapsedtime();
end:
  sum += (t1-t0)*1000/niter; /* ms */
}

  printf("done: %lf (ms)\n", sum / 100);

  return 0;
}
