#include "kaapi.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


int main( int argc, char** argv )
{
  printf(" size(pthread_mutex_t) = %lu\n", sizeof(pthread_mutex_t) );
//  printf(" size(kaapi_thief_request_t) = %lu\n", sizeof(kaapi_thief_request_t) );
//  printf(" size(kaapi_steal_reply_t) = %lu\n", sizeof(kaapi_steal_reply_t) );
  printf(" size(kaapi_mutex_t) = %lu\n", sizeof(kaapi_mutex_t) );
  printf(" size(kaapi_mutexattr_t) = %lu\n", sizeof(kaapi_mutexattr_t) );
  printf(" size(kaapi_cond_t) = %lu\n", sizeof(kaapi_cond_t) );
  printf(" size(kaapi_condattr_t) = %lu\n", sizeof(kaapi_condattr_t) );
//  printf(" size(kaapi_timed_test_and_lock__t) = %lu\n", sizeof (kaapi_timed_test_and_lock__t));
  printf(" size(kaapi_attr_t) = %lu\n", sizeof(kaapi_attr_t) );
//  printf(" size(kaapi_thread_descr_t) = %lu\n", sizeof(kaapi_thread_descr_t) );
//  printf(" size(kaapi_processor_t) = %lu\n", sizeof(kaapi_processor_t) );
  return 0;
}
