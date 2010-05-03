/*
 * test_create2.c
 *
 * Output :
 * n lines of thread creation
 * n lines of thread termination
 *
 * Created by CL and TG on 22/02/2009
 * Copyright 2009 INRIA. All rights reserved.
 *
 */
#include <signal.h>
#include <ucontext.h>
#include <pthread.h>
#include "usage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void* startup_routine(void* argv)
{
  unsigned long l;
  while (1)
  {
    l += 1;
  }
  return (long*) argv;
}

inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}


double t_send;

void  mysa_sigaction(int sig, struct __siginfo* info, ucontext_t *uap)
{
  double t_recv = gettime();
  printf("Action called thread:%ui, Delay=%lfs\n", pthread_self(), t_recv-t_send);
}



int main(int argc, char** argv)
{
  pthread_t thread[16];
  
  
  int err, nb_thread, rc;
  long i;
  void* result;
  
  if (argc != 2) usage_threads ();
  nb_thread = atoi (argv[1]);


  struct sigaction sa;
  struct sigaction sigactold;
  sa.sa_flags = SA_SIGINFO|SA_RESTART;
  sigemptyset(&sa.sa_mask);
  sa.__sigaction_u.__sa_sigaction = &mysa_sigaction;
  
  if (sigaction(SIGUSR1, &sa, NULL) == -1)
  {
    printf("Cannot set signal action\n");
  }
  
    
  for (i = 0; i < nb_thread; i++)
  {
    rc = pthread_create( &thread[i], 0, startup_routine, (void *)i );
    printf("Create thread %i (%ui) with SYSTEM SCOPE\n", i, thread[i] );
    if (rc != 0 ) exit (rc);
  }

  while (1)
  {
    int c = getchar();
    if (isdigit(c)) 
    {
      int t = c - '0';
      printf("Send signal SIGSTOP to thread %i (%ui)\n", t, thread[t] );
      t_send = gettime();
      err = pthread_kill( thread[t], SIGUSR1 ); 
      if (err !=0)
        printf("Error: %i, msg='%s'\n", err, strerror(err) );
    }
  }  
  
  return 0;
}



