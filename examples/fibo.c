#include "kaapi.h"
#include <stdio.h>
#include <stddef.h>


/*#define KAAPI_TRACE_DEBUG */

int fiboseq(int n)
{ return (n<2 ? n : fiboseq(n-1)+fiboseq(n-2) ); }

void sum_body( void* taskarg, kaapi_stack_t* stack );
void fibo_body( void* taskarg, kaapi_stack_t* stack );
void print_body( void* taskarg, kaapi_stack_t* stack );

typedef struct sum_arg_t {
  kaapi_access_t result;
  kaapi_access_t subresult1;
  kaapi_access_t subresult2;
} sum_arg_t;

KAAPI_REGISTER_TASKFORMAT( sum_format,
    "sum",
    -1,
    sum_body,
    sizeof(sum_arg_t),
    3,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_W, KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_R },
    (kaapi_offset_t[])        { offsetof(sum_arg_t, result), offsetof(sum_arg_t, subresult1), offsetof(sum_arg_t, subresult2) },
    (const kaapi_format_t*[]) { &kaapi_int_format, &kaapi_int_format, &kaapi_int_format }
)

void sum_body( void* taskarg, kaapi_stack_t* stack )
{
  sum_arg_t* arg0 = (sum_arg_t*)taskarg;
  *KAAPI_DATA(int, arg0->result) = *KAAPI_DATA(int, arg0->subresult1) + *KAAPI_DATA(int, arg0->subresult2);
}

typedef struct fibo_arg_t {
  int  n;
  kaapi_access_t result;
} fibo_arg_t;

KAAPI_REGISTER_TASKFORMAT( fibo_format,
    "fibo",
    -1,
    fibo_body,
    sizeof(fibo_arg_t),
    2,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_W },
    (kaapi_offset_t[])        { offsetof(fibo_arg_t, n), offsetof(fibo_arg_t, result) },
    (const kaapi_format_t*[]) { &kaapi_int_format, &kaapi_int_format }
)

void fibo_body( void* taskarg, kaapi_stack_t* stack )
{
  fibo_arg_t* arg0 = (fibo_arg_t*)taskarg;
#if defined(KAAPI_TRACE_DEBUG)  
  printf("Fibo(%i)", arg0->n);
#endif
  if (arg0->n < 2)
  {
    *KAAPI_DATA(int, arg0->result) = arg0->n; //fiboseq(arg0->n);
#if defined(KAAPI_TRACE_DEBUG)  
    printf("=@0x%x:%i\n", KAAPI_DATA(int, arg0->result), *KAAPI_DATA(int, arg0->result));
#endif
  }
  else {
#if defined(KAAPI_TRACE_DEBUG)  
    printf("=@0x%x\n", KAAPI_DATA(int, arg0->result));
#endif
#if defined(REC_VER)    
    kaapi_frame_t frame;
#endif
    fibo_arg_t* argf1;
    fibo_arg_t* argf2;
    sum_arg_t*  args;
    kaapi_task_t* task_sum;
    kaapi_task_t* task1;
    kaapi_task_t* task2;

#if defined(REC_VER)    
    kaapi_stack_save_frame(stack, &frame);
#endif
    task1 = kaapi_stack_toptask(stack);
    kaapi_task_initdfg( task1, fibo_body, kaapi_stack_pushdata(stack, sizeof(fibo_arg_t)) );
    argf1 = kaapi_task_getargst( task1, fibo_arg_t );
    argf1->n = arg0->n - 1;
//    argf1->result = kaapi_stack_pushshareddata(stack, sizeof(int));
kaapi_stack_allocateshareddata( &argf1->result, stack, sizeof(int) );
    kaapi_stack_pushtask(stack);

    task2 = kaapi_stack_toptask(stack);
    kaapi_task_initdfg( task2, fibo_body, kaapi_stack_pushdata(stack, sizeof(fibo_arg_t)) );
    argf2 = kaapi_task_getargst( task2, fibo_arg_t);
    argf2->n      = arg0->n - 2;
//    argf2->result = kaapi_stack_pushshareddata(stack, sizeof(int));
kaapi_stack_allocateshareddata( &argf2->result, stack, sizeof(int) );
    kaapi_stack_pushtask(stack);

    task_sum = kaapi_stack_toptask(stack);
    kaapi_task_initdfg( task_sum, sum_body, kaapi_stack_pushdata(stack, sizeof(sum_arg_t)) );
    args = kaapi_task_getargst( task_sum, sum_arg_t);
    args->result.data     = arg0->result.data;
    args->subresult1.data = argf1->result.data;
    args->subresult2.data = argf2->result.data;
    kaapi_stack_pushtask(stack);

#if defined(REC_VER)    
    fibo_body(task1, stack);
    fibo_body(task2, stack);
    sum_body(task_sum, stack);
    kaapi_stack_restore_frame(stack, &frame);
#endif
 }
}

typedef struct print_arg_t {
  double delay;
  int n;
  int niter;
  kaapi_access_t result;
} print_arg_t;

KAAPI_REGISTER_TASKFORMAT( print_format,
    "print",
    -1,
    print_body,
    sizeof(print_arg_t),
    4,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_RW },
    (kaapi_offset_t[])        { offsetof(print_arg_t, delay), offsetof(print_arg_t, n), offsetof(print_arg_t, niter), offsetof(print_arg_t, result) },
    (const kaapi_format_t*[]) { &kaapi_double_format, &kaapi_int_format, &kaapi_int_format, &kaapi_int_format }
)

void print_body( void* taskarg, kaapi_stack_t* stack )
{
  print_arg_t* arg0 = (print_arg_t*)taskarg;
  printf("Fibo(%i)=%i\n", arg0->n, *KAAPI_DATA(int, arg0->result));
  printf("Time: %g\n", arg0->delay/arg0->niter);
}

int main(int argc, char** argv)
{
  kaapi_frame_t frame;
  double t0, t1;
  kaapi_access_t result1;
  long value_result;
  int n;
  int niter;
  int i;
  fibo_arg_t* argf;
  print_arg_t* argp;
  kaapi_task_t* task;
  kaapi_stack_t* stack;
  
  if (argc >1)
    n = atoi(argv[1]);
  else 
    n = 20;
  if (argc >2)
    niter =  atoi(argv[2]);
  else 
    niter = 1;

  stack = kaapi_self_stack();
  kaapi_stack_save_frame(stack, &frame);
  
  for ( i=-1; i<niter; ++i)
  {
    if (i ==0) t0 = kaapi_get_elapsedtime();
  
    kaapi_access_init( &result1, &value_result );

    task = kaapi_stack_toptask(stack);
    kaapi_task_init( task, fibo_body, kaapi_stack_pushdata(stack, sizeof(fibo_arg_t)) );
    argf = kaapi_task_getargst( task, fibo_arg_t );
    argf->n      = n;
    argf->result = result1;
    kaapi_stack_pushtask(stack);
  }
  kaapi_sched_sync( stack );
  t1 = kaapi_get_elapsedtime();

  /* push print task */
  task = kaapi_stack_toptask(stack);
  kaapi_task_init( task, print_body, kaapi_stack_pushdata(stack, sizeof(print_arg_t)) );
  argp = kaapi_task_getargst( task, print_arg_t );
  argp->delay  = t1-t0;
  argp->n      = n;
  argp->niter  = niter;
  argp->result = result1;
  kaapi_stack_pushtask(stack);
  kaapi_stack_pushretn( stack, &frame );

  kaapi_sched_sync( stack );
  
  printf("After sync: Fibo(%i)=%li\n", n, value_result);
  printf("Time Fibo(%i): %f\n", n, t1-t0);
  
  
  return 0;
}
