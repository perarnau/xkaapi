/*
 * test_mutex_trylock.c
 * ckaapi
 *
 * Output :
 *
 * Creating NB_THREADS threads (with [PROCESS|SYSTEM] SCOPE)
 * I got the mutex (thread 1049168)
 * I didn't got the mutex (thread 1049408)
 * I didn't got the mutex (thread 1049648)
 * ...
 * Return from the thread 1048928 = 'Ended'
 * Return from the thread 1049168 = 'Ended'
 * ...
 *
 * Created by CL and TG on 12/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

#define NB_THREADS 50

kaapi_mutex_t mutex;

void* startup_routine(void* argv)
{
  int rc;
  rc = kaapi_mutex_trylock(&mutex);
  if (rc == 0)
    printf ("I got the mutex (thread %li)\n", (long)kaapi_self());
  else
    printf("I didn't got the mutex (thread %li)\n", (long)kaapi_self());

  return "Ended";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_scope();
  
  int rc;
  long i;
  
  kaapi_attr_t attr;
  kaapi_t thread[NB_THREADS];
  
  rc = kaapi_mutex_init(&mutex, 0);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_attr_init(&attr);
  if (rc != 0 ) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0 ) exit (rc);
    
    printf("Creating %i threads (with SYSTEM SCOPE)\n", NB_THREADS);
  }
  
  if (atoi (argv[1]) == 2)
  {
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0 ) exit (rc);
    
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0 ) exit (rc);
    
    printf("Creating %i threads (with PROCESS SCOPE)\n", NB_THREADS);
  }
  
  for (i = 0; i < NB_THREADS; ++i)
  {
    rc = kaapi_create( &thread[i], &attr, startup_routine, (void*)i );
    if (rc != 0 ) exit (rc);
  }
  
  for (i=0; i < NB_THREADS; ++i)
  {
    void *result;
    
    rc = kaapi_join( thread[i], &result );
    if (rc != 0 ) exit (rc);
    
    printf("Return from the thread %li = '%s'\n", (long) thread[i], (char*)result );
  }
  
  rc = kaapi_mutex_unlock (&mutex);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_mutex_destroy(&mutex);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_attr_destroy(&attr);
  if (rc != 0 ) exit (rc);
  
  return 0;
}
