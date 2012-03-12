/* { dg-do run } */
/* { dg-options "-O2" } */
/* { dg-require-effective-target tls_runtime } */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int thr = 32;
#pragma omp threadprivate (thr)

int
main (void)
{
  int l = 0;

  omp_set_dynamic (0);
  omp_set_num_threads (6);

#pragma omp parallel copyin (thr) reduction (||:l)
  {
    l = thr != 32;
printf("%li:: << tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
    thr = omp_get_thread_num () + 11;
printf("%li:: >> tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
  }

printf("%li:: == tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
  if (l || thr != 11)
    abort ();

#pragma omp parallel reduction (||:l)
{
printf("%li:: << tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
  l = (thr != omp_get_thread_num () + 11);
printf("%li:: >> tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
}

printf("%li:: == tid:%i #num:%i -> thr = %i, l=%i\n", 
  pthread_self(), omp_get_thread_num(), omp_get_num_threads(), thr, l );
  if (l)
    abort ();
  return 0;
}
