/*
 * test_key.c
 * xkaapi
 *
 * Output :
 *
 * Creating NB_THREADS threads (with [PROCESS|SYSTEM] SCOPE) and NB_KEYS keys for each thread
 * Thread 1049168 - 	keys[0] : 4
 * Thread 1049168 - 	keys[1] : 6
 * ...
 * Destructor called for value 4 by thread 1049168
 * Destructor called for value 6 by thread 1049168
 * ...
 * Thread 1048928 - 	keys[0] : 2
 * Thread 1048928 - 	keys[1] : 4
 * ...
 * Destructor called for value 2 by thread 1048928
 * Destructor called for value 4 by thread 1048928
 * ...
 * Return from the thread 1048928 = 'Ended'
 * Return from the thread 1049168 = 'Ended'
 * ...
 *
 * Created by CL and TG on 16/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

#define NB_THREADS 50
#define NB_KEYS 20

static kaapi_key_t keys[NB_KEYS];

void destruction (void *arg)
{
  printf ("Destructor called for value %li by thread %li\n", (long) arg, (long)kaapi_self());
}

void* startup_routine(void* argv)
{
  int i, rc;
  long th = (long) argv;
  
  for (i = 0; i < NB_KEYS; i++)
  { 
    rc = kaapi_setspecific(keys[i], (void*) ((th + i)* 2));
    if (rc != 0 ) exit (rc);
  }
  
  for (i = 0; i < NB_KEYS; i++)
    printf ("Thread %li - \tkeys[%i] : %li\n", (long) kaapi_self(), i, (long)kaapi_getspecific(keys[i]));
  
  return "Ended";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_scope();
  
  int rc;
  long i;
  void *result;
  kaapi_t thread[NB_THREADS];
  kaapi_attr_t attr;
  
  for (i = 0; i < NB_KEYS; i++)
  { 
    rc = kaapi_key_create(&keys[i], destruction);
    if (rc != 0 ) exit (rc);
  }
  
  rc = kaapi_attr_init(&attr);
  if (rc != 0 ) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0 ) exit (rc);
    printf("Creating %i threads (with SYSTEM SCOPE) and %i keys for each thread\n", NB_THREADS, NB_KEYS );
  }
  
  if (atoi (argv[1]) == 2)
  {
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0 ) exit (rc);
    
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0 ) exit (rc);
    
    printf("Creating %i threads (with PROCESS SCOPE) and %i keys for each thread\n", NB_THREADS, NB_KEYS );
  }
  
  for (i = 0; i < NB_THREADS; i++)
  {
    rc = kaapi_create( &thread[i], &attr, startup_routine, (void*)(i+1) );
    if (rc != 0 ) exit (rc);
  }
  
  for (i = 0; i < NB_THREADS; i++)
  {
    rc = kaapi_join( thread[i], &result );
    if (rc != 0 ) exit (rc);
    printf("Return from the thread %li = '%s'\n", (long)thread[i], (char*)result );
  }
  
  rc = kaapi_attr_destroy(&attr);
  if (rc != 0 ) exit (rc);
  
  return 0;
}
