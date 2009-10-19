/*
 * prod-cons.c
 * xkaapi
 *
 * Output :
 *
 * Running producer/consumer test with [PROCESS|SYSTEM] SCOPE threads (buffer size : SIZE_BUF , production size : LOOP_LIMIT)
 * Prod 	: 0
 * Read 	: 0
 * Prod 	: 1
 * Read 	: 1
 * ...
 * Return from the thread 1 = 'Producer ended!'
 * Return from the thread 2 = 'Consumer ended!'
 *
 * Created by CL and TG on 05/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE_BUF 25
#define LOOP_LIMIT 100

kaapi_cond_t cond;
kaapi_mutex_t mutex;
int buff[SIZE_BUF];
int next_read = 0;
int next_write = 0;
int buff_elements = 0;

void* producteur(void* argv)
{
  int i, rc;
  
  for (i = 0; i < LOOP_LIMIT; i++)
  {
    rc = kaapi_mutex_lock (&mutex);
    if (rc != 0) exit (rc);
    
    while (buff_elements == SIZE_BUF)
    {
      rc = kaapi_cond_wait (&cond, &mutex);
      if (rc != 0) exit (rc);
    }
    
    if (buff_elements == 0)
    {
      rc = kaapi_cond_signal (&cond);
      if (rc != 0) exit (rc);      
    }
    
    buff[next_write] = i;
    printf ("Prod \t: %i\n", buff[next_write]);
    next_write = (next_write + 1) % SIZE_BUF;
    buff_elements++;
    
    rc = kaapi_mutex_unlock (&mutex);
    if (rc != 0) exit (rc);
    
  }
  return "Producer ended!";
}

void* consomateur(void* argv)
{
  int i, rc;
  
  for (i = 0; i < LOOP_LIMIT; i++)
  {
    rc = kaapi_mutex_lock (&mutex);
    if (rc != 0) exit (rc);
    
    while (buff_elements == 0)
    {
      rc = kaapi_cond_wait (&cond, &mutex);
      if (rc != 0) exit (rc);
    }
    
    if (buff_elements == SIZE_BUF)
    {
      rc = kaapi_cond_signal (&cond);
      if (rc != 0) exit (rc);
    }
      
    printf ("Read \t: %i\n", buff[next_read]);
    next_read = (next_read + 1) % SIZE_BUF;
    buff_elements--;
    
    rc = kaapi_mutex_unlock (&mutex);
    if (rc != 0) exit (rc);     
  }
  return "Consumer ended!";
}

int main(int argc, char** argv)
{
  if (argc != 2) usage_scope ();
  
  int i, rc;
  void *result1, *result2;
  kaapi_t thread1, thread2;
  kaapi_attr_t attr;
  
  for (i = 0; i < SIZE_BUF; i++)
    buff[i] = 0;
  
  rc = kaapi_mutex_init (&mutex, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_cond_init(&cond, 0);
  if (rc != 0) exit (rc);
  rc = kaapi_attr_init(&attr);
  if (rc != 0) exit (rc);
  
  if (atoi (argv[1]) == 1)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    if (rc != 0) exit (rc);
    printf("Running producer/consumer test with SYSTEM SCOPE threads (buffer size : %i , production size : %i)\n", SIZE_BUF, LOOP_LIMIT);
  }
  else if (atoi (argv[1]) == 2)
  {
    rc = kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);
    if (rc != 0) exit (rc);
    printf("Running producer/consumer test with PROCESS SCOPE threads (buffer size : %i , production size : %i)\n", SIZE_BUF, LOOP_LIMIT);
    rc = kaapi_setconcurrency( 1 );
    if (rc != 0) exit (rc);
  }
  else {
    printf("Running producer/consumer : bad value for scope of threads\n");
    exit(1);
  }
  
  rc = kaapi_create( &thread2, &attr, &consomateur, (void*)321 );
  if (rc != 0) exit (rc);
  rc = kaapi_create( &thread1, &attr, &producteur, (void*)123 );
  if (rc != 0) exit (rc);
  
  rc = kaapi_join( thread1, &result1 );
  if (rc != 0) exit (rc);
  printf("Return from the thread 1 = '%s'\n", (char *)result1 );
  
  rc = kaapi_join( thread2, &result2 );
  if (rc != 0) exit (rc);
  printf("Return from the thread 2 = '%s'\n", (char *)result2 );
  
  return 0;
}
