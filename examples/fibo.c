#include "kaapi.h"
#include "kaapi_task.h"
#include "kaapi_time.h"



void sum_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
  *(int*)task->pdata[0] = *(int*)task->pdata[1] + *(int*)task->pdata[2];
}

void fibo_body( kaapi_task_t* task, kaapi_stack_t* stack )
{
//  printf("fibo(%i)\n", task->idata[0]);
  if (task->idata[0] < 2)
  {
    *(int*)task->pdata[1] = task->idata[0];
  }
  else {
#ifdef REC 
    kaapi_frame_t frame;
    kaapi_stack_save_frame( stack, &frame );

    int result1;
    int result2;
#else
    int* result1 = (int*)kaapi_stack_pushdata(stack, sizeof(int));
    int* result2 = (int*)kaapi_stack_pushdata(stack, sizeof(int));
#endif

    kaapi_task_t* task1 = kaapi_stack_top(stack);
    task1->state = KAAPI_TASK_INIT;
    task1->body  = &fibo_body;
    task1->idata[0] = task->idata[0]-1;
#ifdef REC
    task1->pdata[1] = &result1;
#else
    task1->pdata[1] = result1;
#endif
    kaapi_stack_push(stack);

    kaapi_task_t* task2 = kaapi_stack_top(stack);
    task2->state = KAAPI_TASK_INIT;
    task2->body  = &fibo_body;
    task2->idata[0] = task->idata[0]-2;
#ifdef REC
    task2->pdata[1] = &result2;
#else
    task2->pdata[1] = result2;
#endif
    kaapi_stack_push(stack);

    kaapi_task_t* task_sum = kaapi_stack_top(stack);
    task_sum->state = KAAPI_TASK_INIT;
    task_sum->body  = &sum_body;
    task_sum->pdata[0] = task->pdata[1];
#ifdef REC
    task_sum->pdata[1] = &result1;
    task_sum->pdata[2] = &result2;
#else
    task_sum->pdata[1] = result1;
    task_sum->pdata[2] = result2;
#endif
    kaapi_stack_push(stack);
    
#ifdef REC 
#  if 1
    fibo_body( task1, stack);
    task1->body = 0;
    fibo_body( task2, stack);
    task2->body = 0;
    sum_body( task_sum, stack);
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
  maintask->state = KAAPI_TASK_INIT;
  maintask->idata[0] = atoi(argv[1]);
  maintask->pdata[1] = &result;
  
  kaapi_stack_push(&stack);
//  kaapi_stack_taskexecall( &stack );

#if 1
  do {
//    fibo_body( maintask, &stack ); err = 1;
    kaapi_stack_taskexec(&stack);
  } while (!kaapi_stack_isempty(&stack));
#endif
  double t1 = kaapi_get_elapsedtime();
  if ((err != 0) && (err != ENOEXEC)) printf("error in executing task: %i, '%s'\n", err, strerror(err) );
  printf("Fibo(%i) = %i *** Time: %e(s)\n", atoi(argv[1]), result, t1-t0 );
  
  kaapi_stack_free(&stack);
  return 0;
}
