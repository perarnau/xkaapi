#include "kaapi.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>


#define kaapi_trace(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)


typedef void (*kaapi_entry_t)(kaapi_task_t*, kaapi_stack_t*);


/* dfg task */

typedef struct dfg_arg
{
  unsigned int low;
  unsigned int hig;
} dfg_arg_t;


static void dfg_entry(kaapi_task_t* task, kaapi_stack_t* stack)
{
  dfg_arg_t* const arg = kaapi_task_getargst(task, dfg_arg_t);

  kaapi_trace("  > dfg_entry(%u - %u)", arg->low, arg->hig);

  kaapi_trace("  < dfg_entry(%u - %u)", arg->low, arg->hig);
}


KAAPI_REGISTER_TASKFORMAT
(
 dfg_format,
 "dfg",
 dfg_entry,
 sizeof(dfg_arg_t),
 2,
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V },
 (kaapi_offset_t[]){ offsetof(dfg_arg_t, low), offsetof(dfg_arg_t, hig) },
 (const kaapi_format_t*[]) { &kaapi_int_format, &kaapi_int_format }
)


static void start_dfg_task(const dfg_arg_t* arg)
{
  kaapi_stack_t* const self_stack = kaapi_self_frame();

  kaapi_task_t* dfg_task;
  dfg_arg_t* dfg_arg;

  kaapi_trace(" > start_dfg_task(%u - %u)", arg->low, arg->hig);

  dfg_task = kaapi_stack_toptask(self_stack);
  kaapi_task_initdfg( self_stack, dfg_task, dfg_entry, dfg_arg = kaapi_thread_pushdata(self_stack, sizeof(dfg_arg_t)) );
  memcpy(dfg_arg, arg, sizeof(dfg_arg_t));
  kaapi_task_setargs(dfg_task, dfg_arg);
  kaapi_stack_pushtask(self_stack);

  kaapi_sched_sync(self_stack);

  kaapi_trace(" < start_dfg_task(%u - %u)", dfg_arg->low, dfg_arg->hig);
}


/* adaptive task */

static void adaptive_entry(kaapi_task_t* task, kaapi_stack_t* stack);


typedef struct adaptive_arg
{
  /* interval to operate on */
  unsigned int low;
  unsigned int hig;

} adaptive_arg_t;


void fail_requests
(
 kaapi_stack_t* victim_stack,
 kaapi_task_t* task,
 int count,
 kaapi_request_t* requests
)
{
  for (; count; ++requests)
  {
    if (!kaapi_request_ok(requests))
      continue ;
    
    kaapi_request_reply(victim_stack, task, requests, 0, 0);
    
    --count;
  }
}


static int adaptive_splitter
(
 kaapi_stack_t* victim_stack,
 kaapi_task_t* task,
 int count,
 kaapi_request_t* request
)
{
  adaptive_arg_t* const victim_work = kaapi_task_getargst(task, adaptive_arg_t);

  const int replied_count = count;
  adaptive_arg_t* thief_work;

  for (; count > 0; ++request)
    {
      if (victim_work->low == victim_work->hig)
	break;

      if (!kaapi_request_ok(request))
	continue ;

      {
	kaapi_stack_t* const thief_stack = request->stack;
	kaapi_task_t*  const thief_task  = kaapi_stack_toptask(thief_stack);
	void* const stack_data = kaapi_thread_pushdata(thief_stack, sizeof(adaptive_arg_t));

	kaapi_task_initadaptive(thief_stack, thief_task, adaptive_entry, stack_data, KAAPI_TASK_ADAPT_DEFAULT);

	thief_work = kaapi_task_getargst(thief_task, adaptive_arg_t);
	thief_work->low = victim_work->low;
	thief_work->hig = victim_work->low + 1;
	victim_work->low = thief_work->hig;

	kaapi_stack_pushtask(thief_stack);

	kaapi_request_reply(victim_stack, task, request, thief_stack, 1);
      }

      --count; 
    }

  fail_requests(victim_stack, task, count, request);

  return replied_count;
}


static void adaptive_entry(kaapi_task_t* task, kaapi_stack_t* stack)
{
  adaptive_arg_t* const arg = kaapi_task_getargst(task, adaptive_arg_t);

  const adaptive_arg_t saved_arg = *arg;

/*   kaapi_task_t* dfg_task; */
/*   void* dfg_stack; */

  kaapi_trace("> adaptive_entry(%u - %u)", arg->low, arg->hig);

  while (1)
    {
      kaapi_stealpoint(stack, task, adaptive_splitter);

      if (arg->low == arg->hig)
	break;

      /* create a dfg per interval value */
      {
	dfg_arg_t dfg_arg = { saved_arg.low, saved_arg.hig };
	start_dfg_task(&dfg_arg);
      }

      ++arg->low;
    }

  kaapi_finalize_steal(stack, task);

  kaapi_trace("< adaptive_entry(%u - %u)", saved_arg.low, saved_arg.hig);
}


static void start_adaptive_task(const adaptive_arg_t* arg)
{
  kaapi_stack_t* const stack = kaapi_self_frame();
  kaapi_task_t* const task = kaapi_stack_toptask(stack);
  adaptive_arg_t* const data = kaapi_thread_pushdata(stack, sizeof(adaptive_arg_t));

  memcpy(data, arg, sizeof(adaptive_arg_t));

  kaapi_task_initadaptive(stack, task, adaptive_entry, data, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_stack_pushtask(stack);
  kaapi_sched_sync(stack);
}


/* strap first task creation */

static void strap(void)
{
#if 1
  {
    const adaptive_arg_t arg = { 0, 40 };
    start_adaptive_task(&arg);
  }
#else
  {
    const dfg_arg_t dfg_arg = { 0, 3 };
    start_dfg_task(&dfg_arg);
  }
#endif
}


/* main */

int main(int ac, char** av)
{
  strap();

  return 0;
}
