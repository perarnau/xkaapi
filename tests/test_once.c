/*
 * test_once.c
 * ckaapi
 *
 * Output :
 *
 * Creating NB_THREADS threads with [PROCESS|SYSTEM] SCOPE
 * I'm the first thread 1049392! (first output)
 * Hello world from thread 1049392! (x NB_THREADS)
 * Return from the thread 0 = 'Ended' (x NB_THREADS)
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

static kaapi_once_t global_once = KAAPI_ONCE_INIT;


void hello ()
{
  printf("Hello world from thread %li!\n", (long)kaapi_self());
}

void global ()
{
  printf("I'm the first thread %li!\n", (long)kaapi_self());
}

void* startup_routine(void* argv)
{
  int rc;
  kaapi_once_t once_control = KAAPI_ONCE_INIT;

  rc = kaapi_once (&global_once,  global);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_once (&once_control, hello);
  if (rc != 0 ) exit (rc);
  
  rc = kaapi_once (&once_control, hello);
  if (rc != 0 ) exit (rc);
  
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
  
  rc = kaapi_attr_init(&attr);
  if (rc != 0 ) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0 ) exit (rc);
    printf("Creating %i threads with SYSTEM SCOPE\n", NB_THREADS );
  }
  
  if (atoi (argv[1]) == 2)
  {
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0 ) exit (rc);
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0 ) exit (rc);
    printf("Creating %i threads with PROCESS SCOPE\n", NB_THREADS );
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
    printf("Return from the thread %li = '%s'\n", i, (char*)result );
  }
  
  rc = kaapi_attr_destroy(&attr);
  if (rc != 0 ) exit (rc);
  
  return 0;
}

