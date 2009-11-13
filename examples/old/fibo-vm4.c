#include "kaapi_task.h"
#include "kaapi_time.h"
#include "kaapi_error.h"


void sum_body( int* res, int r1, int r2 )
{
  *res = r1 + r2;
}

typedef struct arg_fibo {
  int n;
  int* res;
} arg_fibo;

void fibo_body( int n, int* res, kaapi_stack_t* stack )
{
  if (n < 2)
  {
    *res = n;
  }
  else {
    kaapi_frame_t frame;
    kaapi_stack_save_frame( stack, &frame );

    int result1;
    int result2;

    void create_task(kaapi_stack_t* thief)
    {
      printf("N=%i\n", n);
#if 0
        kaapi_task_t* task1 = kaapi_stack_top(stack);
        arg_fibo* arg = kaapi_stack_pushdata(stack, sizeof(arg_fibo));
        arg->n = arg->n-1; 
        arg->res = &result1;
        task1->sp_data = (char*)arg;
        kaapi_stack_push(stack);
#endif
    }

    /* represent task that may be stolen */
    kaapi_task_t* task1 = kaapi_stack_top(stack);
    task1->body  = (kaapi_task_body_t)&create_task;
    kaapi_stack_push(stack);

    fibo_body(n-2, &result2, stack );
    if (task1->body !=0) { task1->body = 0, fibo_body( n-1, &result1, stack ); }
    kaapi_stack_restore_frame( stack, &frame );
    sum_body( res, result1, result2 );
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
  fibo_body( atoi(argv[1]), &result, &stack );

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
