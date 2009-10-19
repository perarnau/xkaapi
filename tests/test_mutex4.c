/*
 *  test_mutex4.c
 *  xkaapi
 *
 *  Created by CL and TG on 13/03/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

#define NB_MUTEX_ITERATIONS 300
#define NB_THREADS 50

kaapi_mutex_t mutex;

void* startup_routine(void* argv)
{
  int i, rc;
  
  for (i = 1; i <= NB_MUTEX_ITERATIONS; ++i)
  {
    printf("Thread %li hope to take the recursive mutex (%i times)\n", (long) kaapi_self(), i);
    rc = kaapi_mutex_lock(&mutex);
    if (rc != 0 ) exit (rc);
    
    printf("Thread %li has the recursive mutex (%i times)\n", (long) kaapi_self(), i);
  }
  
  for (i = 1; i <= NB_MUTEX_ITERATIONS; ++i)
  {
    printf("Thread %li release the recursive mutex (%i times)\n", (long) kaapi_self(), i);
    
    rc = kaapi_mutex_unlock(&mutex);
    if (rc != 0 ) exit (rc);
    
    kaapi_yield();
  }
  
  return "Ended";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_scope();
  
  int rc;
  long i;
  
  kaapi_t thread[NB_THREADS];
  kaapi_attr_t attr;
  kaapi_mutexattr_t mutex_attr;
  
  rc = kaapi_attr_init(&attr);
  if (rc != 0 ) exit (rc);
  rc = kaapi_mutexattr_init(&mutex_attr);
  if (rc != 0 ) exit (rc);
  rc = kaapi_mutexattr_settype(&mutex_attr, KAAPI_MUTEX_RECURSIVE);
  if (rc != 0 ) exit (rc);
  rc = kaapi_mutex_init(&mutex, &mutex_attr);
  if (rc != 0 ) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0 ) exit (rc);
    printf("Creating %i threads (with SYSTEM SCOPE) and %i iterations on recursive mutex for each thread\n", NB_THREADS, NB_MUTEX_ITERATIONS );
  }
  
  if (atoi (argv[1]) == 2)
  {
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0 ) exit (rc);
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0 ) exit (rc);
    printf("Creating %i threads (with PROCESS SCOPE) and %i iterations on recursive mutex for each thread\n", NB_THREADS, NB_MUTEX_ITERATIONS );
  }

  for (i=0; i<NB_THREADS; ++i)
  {
    rc = kaapi_create( &thread[i], 0, startup_routine, (void*)i );
    if (rc != 0 ) exit (rc);
  }
  
  for (i=0; i<NB_THREADS; ++i)
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
