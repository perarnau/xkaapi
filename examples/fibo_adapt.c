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


static void thief_entrypoint
(
 kaapi_task_t* task,
 kaapi_stack_t* stack
)
{
  fibo_arg_t* const thief_arg =
    kaapi_task_getargst(task, fibo_arg_t);

  fibo_entrypoint(task, stack);

#if 0
  printf("[%u] thief_entrypoint(%d) == %d\n",
	 (unsigned int)pthread_self(),
	 thief_arg->n,
	 thief_arg->result);
#endif

  kaapi_finalize_steal
    (
     stack,
     task,
     thief_arg,
     sizeof(fibo_arg_t)
    );
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

      /* remove the splitter so that wont be stolen twice */
      task->splitter = NULL;

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
  const int saved_result = victim_arg->result;

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
    /* steal me hardly */
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

#if 0
    printf("-- fibo_entrypoint(%d) %d\n", victim_arg->n - 2, victim_arg->result);
#endif

    if (!kaapi_preempt_nextthief(stack, task, NULL, fibo_reducer, victim_arg))
    {
      const int result = victim_arg->result;

      /* no thief stole the n-1, compute it */
      victim_arg->n -= 1;
      fibo_entrypoint(task, stack);
      victim_arg->n += 1;

#if 0
      printf("!kaapi_preempt_nextthief(%d), result: %d\n",
	     victim_arg->n - 1, victim_arg->result);
#endif

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
       fibo_entrypoint,
       &arg,
       KAAPI_TASK_ADAPT_DEFAULT
      );

    kaapi_stack_pushtask(stack);
    kaapi_finalize_steal(stack, task, 0, 0);
    kaapi_sched_sync(stack);

    result = arg.result;
  }
  t1 = kaapi_get_elapsedtime();

  if ((err != 0) && (err != ENOEXEC))
    printf("error in executing task: %i, '%s'\n", err, strerror(err) );
  printf("Fibo(%i) = %i *** Time: %e(s)\n", atoi(argv[1]), result, t1-t0 );
  
  return 0;
}
