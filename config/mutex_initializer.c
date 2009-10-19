#include <pthread.h>
#define KAAPI_KEYS_MAX PTHREAD_KEYS_MAX
#include "kaapi_private_structure.h"
#define _KAAPI_IMPL_H 1
#define KAAPI_MUTEX_NORMAL 0
#include <errno.h>
#include "kaapi_atomic.h"
#include "kaapi_datastructure.h"
#include "kaapi_mutex_init.c"
#include <stdio.h>


int main (int argc, char** argv)
{
  int i;
  kaapi_mutex_t mutex;

  kaapi_mutex_init (&mutex, NULL);

  char* toto = &mutex;

  printf("{{ %d", toto[0]);

  for (i = 1; i < sizeof (mutex); i++)
  {
    printf (", %d", toto[i]); 
  }

  printf("}}\n");
  
  return 1;
}
