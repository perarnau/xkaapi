#include "kaapi.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

/** C Adaptive version of fibo
*/
typedef struct fibo_arg_t {
  int n;
  int result;
} fibo_arg_t;


static void fibo_entrypoint(kaapi_task_t*, kaapi_stack_t*);


static void common_entrypoint
(
 kaapi_task_t* task,
 kaapi_stack_t* stack
)
{
  fibo_entrypoint(task, stack);
}

static void thief_entrypoint
(
 kaapi_task_t* task,
 kaapi_stack_t* stack
)
{
  fibo_arg_t* const thief_arg =
    kaapi_task_getargst(task, fibo_arg_t);

  common_entrypoint(task, stack);

  kaapi_return_steal(stack, task, thief_arg, sizeof(*thief_arg));
}


static int fibo_splitter
(
 kaapi_stack_t* victim_stack,
 kaapi_task_t* task,
 int count,
 kaapi_request_t* request
)
{
  const fibo_arg_t* const victim_arg = kaapi_task_getargst(task, fibo_arg_t);

  int i;

  for (i=0; ; ++i)
  {
    if ( kaapi_request_ok(&request[i]) )
    {
      kaapi_stack_t* thief_stack = request[i].stack;
      kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
      fibo_arg_t* thief_arg;

      kaapi_task_initadaptive
	(
	 thief_stack,
	 thief_task,
	 thief_entrypoint,
	 NULL,
	 KAAPI_TASK_ADAPT_DEFAULT
	);

      thief_arg = kaapi_stack_pushdata(thief_stack, sizeof(fibo_arg_t));
      thief_arg->n = victim_arg->n - 1;
      kaapi_task_setargs(thief_task, thief_arg);

      kaapi_stack_pushtask(thief_stack);
      kaapi_request_reply(victim_stack, task, &request[i],
			  thief_stack, sizeof(fibo_arg_t), 1);

      return 1;
    }
  }

  return 0;
}


static int fibo_reducer
(
 kaapi_stack_t* stack,
 kaapi_task_t* task,
 const fibo_arg_t* thief_arg,
 fibo_arg_t* victim_arg
)
{
#if 0
  const int saved_result = victim_arg->result;
#endif

  victim_arg->result +=  thief_arg->result;

#if 0
  printf("[%u] reducing(%d): %d = %d[%d] + %d[%d]\n",
	 (unsigned int)pthread_self(),
	 victim_arg->n,
	 victim_arg->result,
	 saved_result,
	 victim_arg->n,
	 thief_arg->result,
	 thief_arg->n);
#endif

  return 0;
}


static const kaapi_perf_id_t perf_ids[] =
{
  KAAPI_PERF_ID_STEALOP,
  KAAPI_PERF_ID_PAPI_0,
  KAAPI_PERF_ID_PAPI_1
};

static const size_t perf_count =
  sizeof(perf_ids) / sizeof(perf_ids[0]);


static void fibo_entrypoint
(
 kaapi_task_t* task,
 kaapi_stack_t* stack
)
{
  fibo_arg_t* const victim_arg = kaapi_task_getargst(task, fibo_arg_t);

  if (victim_arg->n < 2)
  {
    victim_arg->result = victim_arg->n;
  }
  else {
    kaapi_stealpoint
      (
       stack,
       task,
       fibo_splitter
      );

    /* mute args */
    victim_arg->n -= 2;
    fibo_entrypoint(task, stack);
    victim_arg->n += 2;

    if (!kaapi_preempt_nextthief(stack, task, NULL, fibo_reducer, victim_arg))
    {
      const int result = victim_arg->result;

      /* no thief stole the n-1, compute it */
      victim_arg->n -= 1;
      fibo_entrypoint(task, stack);
      victim_arg->n += 1;

      /* sum fibo(n-1), fibo(n-2) */
      victim_arg->result += result;
    }
    /* else reducer did the sum */
  }
}


int main(int argc, char** argv)
{
  int err = 0;
  double t0, t1;
  int result;

  t0 = kaapi_get_elapsedtime();
  {
    kaapi_stack_t* const stack = kaapi_self_stack();
    kaapi_task_t* const task = kaapi_stack_toptask(stack);

    fibo_arg_t arg;

    arg.result = 0;
    arg.n = atoi(argv[1]);

    kaapi_task_initadaptive
      (
       stack,
       task,
       common_entrypoint,
       &arg,
       KAAPI_TASK_ADAPT_DEFAULT
      );

    kaapi_stack_pushtask(stack);
    kaapi_finalize_steal(stack, task);
    kaapi_sched_sync(stack);

    /* kaapi_perf */
    {
      kaapi_perf_counter_t counter;
      printf("#op: %llu\n", counter);
    }

    result = arg.result;

/*     /\* kaapi_perf, report *\/ */
/*     { */
/*       size_t i; */

/*       kaapi_perf_counter_t counters[KAAPI_PERF_ID_MAX]; */

/*       kaapi_perf_read_counters(KAAPI_PERF_ID_USER(ALL), counters); */

/*       printf("counters: \n"); */
/*       for (i = 0; i < perf_count; ++i) */
/* 	printf(" + %s: %llu\n", */
/* 	       kaapi_perf_id_to_name(perf_ids[i]), */
/* 	       counters[perf_ids[i]]); */
/*     } */

  }
  t1 = kaapi_get_elapsedtime();

  if ((err != 0) && (err != ENOEXEC))
    printf("error in executing task: %i, '%s'\n", err, strerror(err) );
  printf("Fibo(%i) = %i *** Time: %e(s)\n", atoi(argv[1]), result, t1-t0 );
  
  return 0;
}
