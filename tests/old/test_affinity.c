#define _GNU_SOURCE 1

#include <sched.h>
#include "kaapi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void* startup_routine(void* argv)
{
  double d;
  int i;
  printf("Hello world!\n");
  while (1)
    for (i=0; i<10000000; ++i)
      d = sin(i);

  return "As-tu le resultat ?";
}

int main(int argc, char** argv)
{
  int i, ncpu = 1, sysscope = 1;
  if (argc >1) ncpu = atoi( argv[1] );
  if (argc >2) sysscope = atoi( argv[2] ) == 1;
  
  cpu_set_t cpuset;
  if (!sysscope) 
  {
    kaapi_setconcurrency(2);
    printf("Creating %i threads with PROCESS SCOPE\n", ncpu );
  } else {
    printf("Creating %i threads with SYSTEM SCOPE\n", ncpu );
  }    
  
  kaapi_t thread[16];
  for (i =0; i<ncpu; ++i)
  {
    kaapi_attr_t attr;
    kaapi_attr_init(&attr);
    CPU_ZERO( &cpuset );
    CPU_SET (i, &cpuset);
    kaapi_attr_setaffinity( &attr, sizeof(cpuset), &cpuset );
    if (sysscope)
      kaapi_attr_setscope(&attr, KAAPI_SYSTEM_SCOPE);
    else
      kaapi_attr_setscope(&attr, KAAPI_PROCESS_SCOPE);

    kaapi_create( &thread[i], &attr, startup_routine, (void *)(long)i );
  }
  for (i =0; i<ncpu; ++i)
  {
   kaapi_join( thread[0], 0 );
   kaapi_join( thread[1], 0 );
  }
  while (1) sleep(60);
  return 0;
}
