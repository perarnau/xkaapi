/*
 * test_cond2.c
 * ckaapi
 *
 * Output :
 *
 * Creating NB_THREADS threads with [PROCESS|SYSTEM] SCOPE
 * Thread 0 Ended (return value : Master ended)
 * Thread 1 Ended (return value : Slave ended)
 * Thread 2 Ended (return value : Slave ended)
 * ...
 *
 * Created by CL and TG on 05/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

#define NB_THREADS 2

kaapi_cond_t cond_master, cond_slave;
kaapi_mutex_t mutex_master, mutex_slave;
volatile int nb_sleep = 0;

void* master (void* argv)
{
  int rc, i;
  i = (int)argv;
  printf("Master %i\n", i);
  rc = kaapi_mutex_lock(&mutex_slave);
  if (rc != 0) exit (rc);
  while (nb_sleep != NB_THREADS)
  {    
    printf("Master begin sleep %i\n", i);
    rc = kaapi_cond_wait (&cond_master, &mutex_slave);
    printf("Master end sleep %i\n", i);
    if (rc != 0) exit (rc);    
  }

  rc = kaapi_mutex_unlock(&mutex_slave);
  if (rc != 0) exit (rc);
  
  printf("Master begin signal all slave %i\n", i);
  for (i = 0; i < NB_THREADS; i++)
  { 
    rc = kaapi_cond_broadcast(&cond_slave);
    if (rc != 0) exit (rc);    
  }
  printf("Master end signal all slave %i\n", i);

  return "Master ended";
}

void* slave (void* argv)
{
  int rc, i;
  i = (int)argv;

  printf("Slave %i\n", i);
  rc = kaapi_mutex_lock (&mutex_slave);
  if (rc != 0) exit (rc);
  printf("Slave have lock %i\n", i);
  
  nb_sleep++;
  
  if (nb_sleep == NB_THREADS)
  {
    printf("Slave do signal %i\n", i);
    rc = kaapi_cond_signal(&cond_master);
    if (rc != 0) exit (rc);
  }
  
  printf("Slave begin wait %i\n", i);
  rc = kaapi_cond_wait (&cond_slave, &mutex_slave);
  if (rc != 0) exit (rc);
  printf("Slave end wait %i\n", i);
  
  rc = kaapi_mutex_unlock (&mutex_slave);
  if (rc != 0) exit (rc);
  printf("Slave end %i\n", i);
  
  return "Slave ended";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_scope ();
  
  void *result;
  int rc;
  long i;

  kaapi_t thread[NB_THREADS + 1];
  kaapi_attr_t attr;
  
  rc = kaapi_mutex_init (&mutex_master, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_mutex_init (&mutex_slave, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_cond_init(&cond_master, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_cond_init(&cond_slave, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_attr_init(&attr);
  if (rc != 0) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0) exit (rc);
    printf("Creating %i threads with SYSTEM SCOPE\n", NB_THREADS );
  }
  
  if (atoi (argv[1]) == 2)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0) exit (rc);
    printf("Creating %i threads with PROCESS SCOPE\n", NB_THREADS );
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0) exit (rc);
  }
  
  rc = kaapi_create( &thread[0], &attr, master, (void *)0 );
  if (rc != 0 ) exit (rc);

  for (i = 1; i <= NB_THREADS ; i++)
  {
    rc = kaapi_create( &thread[i], &attr, slave, (void *)i );
    if (rc != 0 ) exit (rc);
  }

  for (i = 0; i <= NB_THREADS; i++)
  {
    rc = kaapi_join( thread[i], &result );
    if (rc != 0 ) exit (rc);
    printf("Thread %li Ended (return value : %s)\n", i, (char*)result );
  }
  
  return 0;
}
