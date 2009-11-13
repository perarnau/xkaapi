#include "kaapi_task.h"
#include "kaapi_time.h"
#include "kaapi_error.h"


#define VALUE_AT_OFFSET( arg, offset, type) \
  ((type*)((arg) + (offset)))

typedef struct arg_sum {
  int* res;
  int* v1;
  int* v2;
} arg_sum;

void sum_body( void* a, kaapi_stack_t* stack )
{
  arg_sum* arg = (arg_sum*)a;
  *arg->res = *arg->v1 + *arg->v2;
}

typedef struct arg_fibo {
  int n;
  int* res;
} arg_fibo;

void fibo_body( void* sp, kaapi_stack_t* stack )
{
//  printf("fibo(%i)\n", task->idata[0]);
  int  n = ((arg_fibo*)sp)->n;
  int* res = ((arg_fibo*)sp)->res;
  if (n < 2)
  {
    *res = n;
  }
  else {
    char* top = kaapi_stack_topdata2(stack);
#ifdef REC 
    kaapi_frame_t frame;
    kaapi_stack_save_frame( stack, &frame );
#endif
    int* result1 = (int*)&top[0];
    int* result2 = (int*)&top[sizeof(int)];

    kaapi_task_t* task1 = kaapi_stack_top(stack);
    task1->body  = &fibo_body;
    arg_fibo* arg = (arg_fibo*)&top[2*sizeof(int)];
    arg->n = n-1;
    arg->res = result1;
    task1->sp_data =(char*)arg;
    kaapi_stack_push(stack);

    kaapi_task_t* task2 = kaapi_stack_top(stack);
    task2->body  = &fibo_body;
    arg = (arg_fibo*)&top[2*sizeof(int)+sizeof(arg_fibo)];
    arg->n = n-2;
    arg->res = result2;
    task2->sp_data = (char*)arg;
    kaapi_stack_push(stack);

    kaapi_task_t* task_sum = kaapi_stack_top(stack);
    task_sum->body  = &sum_body;
    arg_sum* args = (arg_sum*)&top[2*sizeof(int)+2*sizeof(arg_fibo)];
    args->res = res;
    args->v1  = result1;
    args->v2  = result2;
    task_sum->sp_data = (char*)args;
    kaapi_stack_push(stack);
    kaapi_stack_pushdata2( stack, 2*sizeof(int)+2*sizeof(arg_fibo)+sizeof(arg_sum));

#ifdef REC 
#  if 1
    fibo_body( task1->sp_data, stack);
    task1->body = 0;
    fibo_body( task2->sp_data, stack);
    task2->body = 0;
    sum_body( task_sum->sp_data, stack);
    task_sum->body = 0;
#  else
    (*task1->body)( task1, stack);
    (*task1->body)( task2, stack);
    (*task_sum->body)( task_sum, stack);
#  endif
 
    kaapi_stack_restore_frame( stack, &frame );
#endif
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
/*  maintask->state = KAAPI_TASK_INIT;*/
  arg_fibo* arg = (arg_fibo*)(maintask->sp_data = kaapi_stack_pushdata(&stack, sizeof(arg_fibo)));
  arg->n = atoi(argv[1]);
  arg->res = &result;
  
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
