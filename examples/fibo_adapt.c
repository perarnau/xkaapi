#include "kaapi_task.h"
#include "kaapi_time.h"
#include "kaapi_error.h"

#define KAAPI_EVENT_STEAL 1
#define KAAPI_EVENT_EXEC 0

void fibo_body( kaapi_stack_t* stack, int n, int* result )
{
    int result1;
    int result2;

    void splitter( int event, kaapi_stack_t* thief_stack )
    { 
      typedef struct arg_fibo {
        int n;
        int* res;
      } arg_fibo;
      void entrypoint( kaapi_task_t* task, kaapi_stack_t* stack ) 
      {
        arg_fibo* a = (arg_fibo*)task->sp;
        fibo_body( stack, a->n, a->res );
      }
      kaapi_task_t* task = kaapi_stack_top(thief_stack);
      arg_fibo* a = (arg_fibo*)kaapi_stack_pushdata( thief_stack, sizeof(arg_fibo));
      task->sp = a;
      a->n   = n-1;
      a->res = &result2;
      task->body[KAAPI_EVENT_EXEC] = (kaapi_task_body_t)&entrypoint;
      kaapi_stack_push( thief_stack );
    }
    
  if (n < 2)
  {
    *result = n;
  }
  else {
    /* set an action to reply to a steal request*/
    kaapi_task_t* task = kaapi_stack_top(stack);
    task->event = KAAPI_TASK_F_ADAPTIVE|KAAPI_EVENT_STEAL;
    task->body[KAAPI_EVENT_STEAL] = &splitter;
    kaapi_stack_push( stack );
        
    /* recursive call */
    fibo_body( stack, n-2, &result2 );
    
    if (kaapi_finalize_steal(stack, task) == 0) 
    {
      /* no theft task, do it in sequential on n-1 */
      fibo_body(stack, n-1, &result1 );
    } /* else wait */

    /* return with sum */
    *result = result1 + result2;
 }
}


int main(int argc, char** argv)
{
  int err = 0;
  kaapi_stack_t stack;
  kaapi_stack_alloc(&stack, 1024, 1024);

  printf("sizeof(kaapi_task_t)=%i\n", sizeof(kaapi_task_t));
  int result = 0;
  double t0 = kaapi_get_elapsedtime();
  fibo_body( &stack, atoi(argv[1]), &result );

#if 1
  do {
//    fibo_body( maintask, &stack ); err = 1;
    kaapi_stack_taskexecall(&stack);
  } while (!kaapi_stack_isempty(&stack));
#endif
  double t1 = kaapi_get_elapsedtime();
  if ((err != 0) && (err != ENOEXEC)) printf("error in executing task: %i, '%s'\n", err, strerror(err) );
  printf("Fibo(%i) = %i *** Time: %e(s)\n", atoi(argv[1]), result, t1-t0 );
  
  kaapi_stack_free(&stack);
  return 0;
}
