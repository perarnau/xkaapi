/*
 * test_create1.c
 * xkaapi
 *
 * Output :
 * n lines of thread creation
 * n lines of thread termination
 *
 * Created by CL and TG on 22/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

void* startup_routine(void* argv)
{
  printf("Hello World, I'm thread %li\n", (long) argv);
  return (long*) argv;
}

int main(int argc, char** argv)
{
  int nb_thread, rc;
  long i;
  void* result;
  
  if (argc != 2) usage_threads();

  nb_thread = atoi (argv[1]);
  kaapi_t thread[nb_thread];
  
  printf("Creating %i threads with SYSTEM SCOPE\n", nb_thread );
  for (i = 1; i <= nb_thread; i++)
  {
    rc = kaapi_create( &thread[i], 0, startup_routine, (void *)i );
    if (rc != 0 ) exit (rc);
  }
    

  for (i = 1; i <= nb_thread; i++)
  {
    rc = kaapi_join( thread[i], &result );
    if (rc != 0 ) exit (rc);
    printf("Thread %li Ended\n", (long)result );
  }
  
  return 0;
}
