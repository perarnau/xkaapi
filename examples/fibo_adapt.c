#include "kaapi.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

/** C Adaptive version of fibo
*/
typedef struct fibo_arg_t {
    int n;
    int* res;
    kaapi_stack_t* stack;
} fibo_arg_t;

void fibo( kaapi_stack_t* stack, int n, int* result )
{
  int result1;
  int result2;

  printf("[%u] fibo(%d)\n", (unsigned int)pthread_self(), n);

  int splitter( kaapi_stack_t* victim_stack, kaapi_task_t* task, int count, kaapi_request_t* request )
  { 
    void entrypoint( kaapi_task_t* task, kaapi_stack_t* stack ) 
    {
      fibo_arg_t* arg = kaapi_task_getargst( task, fibo_arg_t);
      fibo( stack, arg->n, arg->res );
    }
    int i;
    for (i=0; ; ++i)
    {
      if ( kaapi_request_ok(&request[i]) )
      {
        fibo_arg_t* arg;
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_init( thief_stack, thief_task, &entrypoint, kaapi_stack_pushdata(thief_stack, sizeof(fibo_arg_t)), KAAPI_TASK_ADAPTIVE);
        arg = kaapi_task_getargst(thief_task, fibo_arg_t);                
        arg->n   = n-1;
        arg->res = &result2;
        kaapi_stack_pushtask( thief_stack );
        kaapi_request_reply( victim_stack, task, &request[i], thief_stack, 0, 1 );

	/* remove the splitter so that wont be stolen twice */
	task->splitter = NULL;

        return 1;
      }
    }
    return 0;
  }
    
  if (n < 2)
  {
    *result = n;
  }
  else {
    void entrypoint( kaapi_task_t* task, kaapi_stack_t* stack ) 
    {
      fibo_arg_t* arg = kaapi_task_getargst( task, fibo_arg_t);
      fibo( stack, arg->n, arg->res );
    }
    /* set an action to reply to a steal request*/
    fibo_arg_t* arg;
    kaapi_task_t* task = kaapi_stack_toptask(stack);
    kaapi_task_init( stack, task, &entrypoint, kaapi_stack_pushdata(stack, sizeof(fibo_arg_t)), KAAPI_TASK_ADAPTIVE);
    arg = kaapi_task_getargst(task, fibo_arg_t);                
    arg->n   = n-1;
    arg->res = &result1;
    kaapi_stack_pushtask( stack );

    /* let a chance to be stolen */
    kaapi_stealpoint(stack, task, splitter);

    /* recursive sequential call */
    task->splitter = &splitter;
    fibo( stack, n-2, &result2 );
    task->splitter = 0;

    /* preempt thief, if any */
    int reducer(kaapi_stack_t* stack, kaapi_task_t* task, const int* result1, int* result, const int* result2)
    {
      if (result1 == NULL)
	return 0;

      *result = *result1 + *result2;

      printf("[%u] reducing, %d == %d + %d\n", (unsigned int)pthread_self(), *result, *result1, *result2);

      return 1;
    }

    if (!kaapi_preempt_nextthief(stack, task, 0, &reducer, &result, &result2))
    {
      /* no thief stole the n-1, compute it */
      fibo(stack, n - 1, &result1);

      /* sum fibo(n-1), fibo(n-2) */
      *result = result1 + result2;

      printf("[%u] !kaapi_preempt(%d): %d = %d + %d\n", (unsigned int)pthread_self(), n, *result, result1, result2);
    }
    /* else reducer did the sum */
  }

  printf("[%u] out == %d\n", (unsigned int)pthread_self(), *result);

  kaapi_finalize_steal(stack, kaapi_stack_toptask(stack), result, sizeof(*result));
}


int main(int argc, char** argv)
{
  int err = 0;
  int result = 0;
  double t0, t1;
  
  printf("sizeof(kaapi_task_t)=%lu\n", sizeof(kaapi_task_t));
  printf("sizeof(kaapi_stack_t)=%lu\n", sizeof(kaapi_stack_t));

  t0 = kaapi_get_elapsedtime();
  fibo( kaapi_self_stack(), atoi(argv[1]), &result );
  t1 = kaapi_get_elapsedtime();

  if ((err != 0) && (err != ENOEXEC)) printf("error in executing task: %i, '%s'\n", err, strerror(err) );
  printf("Fibo(%i) = %i *** Time: %e(s)\n", atoi(argv[1]), result, t1-t0 );
  
  return 0;
}
