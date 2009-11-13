#include "kaapi_task.h"
#include "kaapi_time.h"
#include "kaapi_error.h"


typedef struct arg_sum {
  int* res;
  int* r1;
  int* r2;
} arg_sum;

void sum_body( void* sp, kaapi_stack_t* stack )
{
  arg_sum* arg = (arg_sum*)sp;
  *arg->res = *arg->r1 + *arg->r2;
}

typedef struct arg_fibo {
  int n;
  int* res;
} arg_fibo;

void fibo_body( void* sp, kaapi_stack_t* stack )
{
  arg_fibo* arg = (arg_fibo*)sp;
  if (arg->n < 2)
  {
    *arg->res = arg->n;
  }
  else {
    kaapi_frame_t frame;
    kaapi_stack_save_frame( stack, &frame );

    int result1;
    int result2;

    kaapi_task_t* task1 = kaapi_stack_top(stack);
    task1->body  = &fibo_body;
    arg_fibo* a1 = kaapi_stack_pushdata(stack, sizeof(arg_fibo));
    a1->n = arg->n-1;
    a1->res = &result1;
    task1->sp_data = (char*)a1;
    kaapi_stack_push(stack);

    kaapi_task_t* task2 = kaapi_stack_top(stack);
    task2->body  = &fibo_body;
    arg_fibo* a2 = kaapi_stack_pushdata(stack, sizeof(arg_fibo));
    a2->n = arg->n-2;
    a2->res = &result2;
    task2->sp_data = (char*)a2;
    kaapi_stack_push(stack);

    kaapi_task_t* task_sum = kaapi_stack_top(stack);
    task_sum->body  = &sum_body;
    arg_sum* args = kaapi_stack_pushdata(stack, sizeof(arg_sum));;
    args->res = arg->res;
    args->r1 = a1->res;
    args->r2 = a2->res;
    task_sum->sp_data = (char*)args;
    kaapi_stack_push(stack);    


    fibo_body( task1->sp_data, stack);
    task1->body = 0;
    fibo_body( task2->sp_data, stack);
    task2->body = 0;
    sum_body( task_sum->sp_data, stack);
    task_sum->body = 0;

    kaapi_stack_restore_frame( stack, &frame );
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
  kaapi_task_t* maintask = kaapi_stack_top(&stack);
  maintask->body  = &fibo_body;
  arg_fibo mainarg;
  mainarg.n = atoi(argv[1]);
  mainarg.res = &result;
  maintask->sp_data = (char*)&mainarg;  
  kaapi_stack_push(&stack);
//  kaapi_stack_taskexecall( &stack );

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
