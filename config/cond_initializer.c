#include <pthread.h>
#define KAAPI_KEYS_MAX PTHREAD_KEYS_MAX

#if defined(NDEBUG)
#  define ckaapi_assert( cond ) if (!(cond)) abort();
#  define ckaapi_assert_debug( cond )
#else
#  include <stdio.h>
#  include <stdlib.h>
#  define ckaapi_assert( cond ) if (!(cond)) { printf("Bad assertion, line:%i, file:'%s'\n", __LINE__, __FILE__ ); abort(); }
#  define ckaapi_assert_debug( cond ) if (!(cond)) { printf("Bad assertion, line:%i, file:'%s'\n", __LINE__, __FILE__ ); abort(); }
#endif

#include "kaapi_private_structure.h"
#define _KAAPI_IMPL_H 1
#include <errno.h>
#include "kaapi_atomic.h"
#include "kaapi_datastructure.h"
#include "kaapi_cond_init.c"
#include <stdio.h>


int main (int argc, char** argv)
{
  int i;
  kaapi_cond_t cond;

  kaapi_cond_init (&cond, NULL);

  char* toto = &cond;

  printf("{{ %d", toto[0]);

  for (i = 1; i < sizeof (cond); i++)
  {
    printf (", %d", toto[i]); 
  }

  printf("}}\n");
  
  return 1;
}
