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
#include <iostream>

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
template<class T, class OP>
void for_each( T* array, size_t size, OP op )
{
  if (size <2)
  {
    for (size_t i=0; i<size; ++i)
      op(&array[i]);
  }
  else {
    /* simple recursive for_each */
    size_t med = size/2;
    for_each( array, med, op);
    for_each( array+med, size-med, op);
  }
}


/**
 */
static void apply_cos( double* v )
{
  *v += cos(*v);
}

/**
 */
int main(int ac, char** av)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;
  size_t size;
  
  if (ac >1)
    size = atoi(av[1]); 

  double* array = new double[size];
  
  /* initialize the runtime */
#pragma kaapi init  
  for (iter = 0; iter < 100; ++iter)
  {
    /* initialize, apply, check */
    for (i = 0; i < size; ++i)
      array[i] = 0.f;

#pragma kaapi barrier
    t0 = get_elapsedtime();
    for_each( array, size, apply_cos );
#pragma kaapi barrier
    t1 = get_elapsedtime();
    sum += (t1-t0)*1000; /* ms */

    for (i = 0; i < size; ++i)
      if (array[i] != 1.f)
      {
        printf("invalid @%lu == %lf\n", i, array[i]);
        break ;
      }
  }

  printf("done: %lf (ms)\n", sum / 100);

  /* finalize the runtime */
#pragma kaapi finish
  
  return 0;
}
