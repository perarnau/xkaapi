#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "kaapi.h"


static void child_entry(void*, kaapi_thread_t*);
static void adaptive_entry(void*, kaapi_thread_t*);

struct work
{
  volatile long l __attribute__((aligned));
  unsigned int i;
  unsigned int j;
};

typedef struct work work_t;


static void init_work(work_t* w)
{
  w->i = 0;
  w->j = 0;
  w->l = 0;
}


static void lock_work(work_t* w)
{
  while (!__sync_bool_compare_and_swap(&w->l, 0, 1))
    ;
}


static void unlock_work(work_t* w)
{
  w->l = 0;
}


static unsigned int pop_work(work_t* a, work_t* b)
{
  lock_work(a);

  b->i = a->i;
  b->j = a->i + ((a->j - a->i) < 10 ? a->j - a->i : 10);
  a->i = b->j;

  unlock_work(a);

  return b->j - b->i;
}


static unsigned int steal_work(work_t* a, work_t* b)
{
  lock_work(a);

  b->j = a->j;
  b->i = a->j - ((a->j - a->i) < 10 ? a->j - a->i : 10);
  a->j = b->i;

  unlock_work(a);

  return b->j - b->i;
}


static void seq_work(work_t* w)
{
  unsigned int q = w->j - w->i;

  printf("seq_work [%u - %u[\n", w->i, w->j);

  for (; q; --q)
    usleep(100000);
}


static int splitter(kaapi_stealcontext_t* sc, int count,
		    kaapi_request_t* requests, void* args)
{
  work_t* const victim_work = args;
  int replied_count = 0;
  int is_done = 0;

  for (; (count > 0) && (!is_done); --count, ++requests)
  {
    kaapi_thread_t* thief_thread;
    kaapi_task_t* thief_task;
    work_t* thief_work = NULL;

    if (!kaapi_request_ok(requests))
      continue ;

    thief_thread = kaapi_request_getthread(requests);
    thief_task = kaapi_thread_toptask(thief_thread);

/*     printf("splittin [%x -> %x]\n", kaapi_self_thread(), thief_thread); */
    thief_work = kaapi_thread_pushdata(thief_thread, sizeof(work_t));

    init_work(thief_work);

    if (steal_work(victim_work, thief_work))
      is_done = 1;

    kaapi_task_init(thief_task, child_entry, thief_work);
    kaapi_thread_pushtask(thief_thread);

/*     kaapi_request_reply_tail(sc, requests, sizeof(work_t)); */
    kaapi_request_reply_head(sc, requests, NULL);

    ++replied_count;
  }

  return replied_count;
}

static void adaptive_entry(void* args, kaapi_thread_t* thread)
{
  work_t* const w = args;

  while (1)
  {
    work_t local_work;

    if (!pop_work(w, &local_work))
      break;

    seq_work(&local_work);
  }
}

static void child_entry(void* args, kaapi_thread_t* thread)
{
  work_t* const w = args;
  kaapi_stealcontext_t* const sc =
    kaapi_thread_pushstealcontext(thread, KAAPI_STEALCONTEXT_DEFAULT, splitter, w);

  adaptive_entry(args, thread);

  kaapi_steal_finalize(sc);
}


static void _root_entry(void* args, kaapi_thread_t* thread)
{
  kaapi_stealcontext_t* const sc =
    kaapi_thread_pushstealcontext(thread, KAAPI_STEALCONTEXT_DEFAULT, splitter, args);

  printf("_root_entry()\n");

  adaptive_entry(args, thread);

  kaapi_steal_finalize(sc);
}


static work_t* create_work(unsigned int q)
{
  static work_t work;
  init_work(&work);
  work.i = 0;
  work.j = q;
  return &work;
}

static void root_entry(unsigned int q)
{
  kaapi_thread_t* thread;
  kaapi_task_t* task;
  kaapi_frame_t frame;

  thread = kaapi_self_thread();

  kaapi_thread_save_frame(thread, &frame);

  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, _root_entry, create_work(q));
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();

  kaapi_thread_restore_frame(thread, &frame);
}


int main(int ac, char** av)
{
  root_entry(50);
  return 0;
}
