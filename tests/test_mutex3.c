/*
 * test_mutex3.c
 * ckaapi
 *
 * Output :
 *
 * Creating (n/2) SYSTEM threads and (n/2) PROCESS threads, and NB_MUTEX_ITERATIONS iterations on mutex for each thread
 * Thread 1049632 has the mutex (1 times)
 * Thread 1049632 has the mutex (2 times)
 * ...
 * Return from the thread 1049392 = 'Ended'
 * Return from the thread 1049632 = 'Ended'
 * ...
 *
 * Created by CL and TG on 06/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NB_MUTEX_ITERATIONS 200

kaapi_mutex_t mutex;

void* startup_routine(void* argv)
{
  int i, rc;
  for (i = 1; i <= NB_MUTEX_ITERATIONS; ++i)
  {
    rc = kaapi_mutex_lock(&mutex);
    if (rc != 0 ) exit (rc);
    
    printf("Thread %li has the mutex (%i times)\n", (long) kaapi_self(), i);
    
    rc = kaapi_mutex_unlock(&mutex);
    if (rc != 0 ) exit (rc);
    
    kaapi_yield();
  }
  return "Ended";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_threads();
  
  int rc;
  long i;
  int nthreads = atoi (argv[1]);
  
  kaapi_attr_t attr;
  kaapi_t thread[nthreads];
  
  rc = kaapi_mutex_init(&mutex, 0);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_attr_init(&attr);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_setconcurrency( 1 );
  if (rc != 0 ) exit (rc);
  
  printf("Creating %i SYSTEM threads and %i PROCESS threads, and %i iterations on mutex for each thread\n", (int) (((double)nthreads/2)+0.5), nthreads/2, NB_MUTEX_ITERATIONS );
  for (i=0; i<nthreads; ++i)
  {
    if (i%2 == 0)
    {
      rc = kaapi_create( &thread[i], &attr, startup_routine, (void*)i );
      if (rc != 0 ) exit (rc);
    }
    else
    { 
      rc = kaapi_create( &thread[i], 0, startup_routine, (void*)i );
      if (rc != 0 ) exit (rc);
    }
  }

  
  for (i=0; i<nthreads; ++i)
  {
    void *result;
    
    rc = kaapi_join( thread[i], &result );
    if (rc != 0 ) exit (rc);
    
    printf("Return from the thread %li = '%s'\n", (long)thread[i], (char*)result );
  }
  
  rc = kaapi_mutex_destroy(&mutex);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_attr_destroy(&attr);
  if (rc != 0 ) exit (rc);
  
  return 0;
}
