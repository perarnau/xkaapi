/*
 * test_mutex1.c
 * xkaapi
 *
 * Output :
 *
 * Creating n threads (with SYSTEM SCOPE) and NB_MUTEX_ITERATIONS iterations on mutex for each thread
 * Thread 1048928 has the mutex (1 times)
 * Thread 1049168 has the mutex (1 times)
 * Thread 1048928 has the mutex (2 times)
 * Thread 1049168 has the mutex (2 times)
 * ...
 * Return from the thread 1048928 = 'Ended'
 * Return from the thread 1049168 = 'Ended' 
 * ...
 *
 * Created by CL and TG on 04/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
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

  kaapi_t thread[nthreads];

  rc = kaapi_mutex_init(&mutex, 0);
  if (rc != 0 ) exit (rc);
  
  printf("Creating %i threads (with SYSTEM SCOPE) and %i iterations on mutex for each thread\n", nthreads, NB_MUTEX_ITERATIONS );
  for (i=0; i<nthreads; ++i)
  {
    rc = kaapi_create( &thread[i], 0, startup_routine, (void*)i );
    if (rc != 0 ) exit (rc);
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
  
  return 0;
}
