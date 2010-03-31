#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "kaapi.h"


/* todos:
   . target stealing
   . check operation ordering
   . original sequence
 */


#define CONFIG_INPUT_SIZE 500
#define CONFIG_ALLOC_RESULT 1
#define CONFIG_STEALABLE_THIEVES 0
#define CONFIG_STATIC_STEAL 0
#define CONFIG_STEAL_SIZE 10
#define CONFIG_POP_SIZE 10


extern void usleep(unsigned long);

static void adaptive_entry(kaapi_stealcontext_t*, void*, kaapi_thread_t*);
static void common_entry(void*, kaapi_thread_t*);

#if !CONFIG_STEALABLE_THIEVES
static void thief_entry(void*, kaapi_thread_t*);
#endif

struct work
{
  /* lock */
  volatile long l __attribute__((aligned));

  /* range */
  unsigned int i;
  unsigned int j;
  unsigned int k;

  /* thief result */
  kaapi_taskadaptive_result_t* r;

  /* debugging */
  unsigned int kid;
};

typedef struct work work_t;


static void init_work(work_t* w)
{
  w->l = 0;

  w->i = 0;
  w->j = 0;
  w->k = 0;

  w->r = NULL;
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


static unsigned int work_remaining_size(work_t* w)
{
  unsigned int size;

  lock_work(w);
  size = w->j - w->k;
  unlock_work(w);

  return size;
}


static void merge_work(work_t* a, work_t* b)
{
  lock_work(a);

  a->j = b->j;
  a->k = b->k;

  unlock_work(a);
}


static unsigned int pop_work(work_t* a, work_t* b)
{
  lock_work(a);

  b->i = a->k;
  b->k = a->k;
  b->j = a->k + ((a->j - a->k) < CONFIG_POP_SIZE ? a->j - a->k : CONFIG_POP_SIZE);

  a->k = b->j;

  unlock_work(a);

  return b->j - b->i;
}


static unsigned int steal_work(work_t* a, work_t* b)
{
  lock_work(a);
  {
#if CONFIG_STATIC_STEAL
    const unsigned int size = CONFIG_STEAL_SIZE;
#else
    const unsigned int size = (a->j - a->k) / 2;
#endif

    b->j = a->j;
    b->i = a->j - ((a->j - a->k) < size ? a->j - a->k : size);
    b->k = b->i;

    a->j = b->i;
  }
  unlock_work(a);

  return b->j - b->i;
}


static void seq_work(work_t* w)
{
  unsigned int q = w->j - w->i;

#if 0
  printf("[%02x,   ] seq [%u - %u[\n", w->kid, w->i, w->j);
#endif

  for (; q; --q)
    usleep(1000);
}


static int reducer
(
 kaapi_stealcontext_t* sc,
 void* thief_arg,
 work_t* thief_work, size_t thief_size,
 work_t* victim_work
)
{
  /* has been reduced */

  if (thief_work->i == thief_work->j)
    return 1;

  printf("[%02x, %02x] red [%04u, %04u[ [%04u, %04u, %04u[ (%lx)\n",
	 victim_work->kid, thief_work->kid,
	 victim_work->i, victim_work->j,
	 thief_work->i, thief_work->k, thief_work->j,
	 (uintptr_t)thief_work);

  merge_work(victim_work, thief_work);

  return 1;
}


static int splitter
(
 kaapi_stealcontext_t* sc,
 int count, kaapi_request_t* requests,
 void* args
)
{
  work_t* const victim_work = args;
  int replied_count = 0;
  int is_done = 0;

  for (; (count > 0) && (!is_done); ++requests)
  {
    kaapi_thread_t* thief_thread;
    kaapi_task_t* thief_task;
    work_t* thief_work = NULL;

    if (!kaapi_request_ok(requests))
      continue ;

    if (work_remaining_size(victim_work) < 10)
      break;

    thief_thread = kaapi_request_getthread(requests);
    thief_task = kaapi_thread_toptask(thief_thread);

    thief_work = kaapi_thread_pushdata(thief_thread, sizeof(work_t));
    init_work(thief_work);

#if CONFIG_ALLOC_RESULT
    thief_work->r = kaapi_allocate_thief_result(sc, sizeof(work_t), NULL);
#else
    thief_work->r = kaapi_allocate_thief_result(sc, sizeof(work_t), thief_work);
#endif

    if (steal_work(victim_work, thief_work))
      is_done = 1;

    printf("red alloc %lx [%u - %u[\n",
	   (uintptr_t)thief_work->r->data,
	   thief_work->i, thief_work->j);

    /* result has to be initialized since from the
       push task the thief may be preempted
     */

    memcpy(thief_work->r->data, thief_work, sizeof(work_t));

    printf("[%02x,   ] spl [%04u, %04u[ [%04u, %04u[ (%lx)\n",
	   victim_work->kid,
	   victim_work->k, victim_work->j,
	   thief_work->i, thief_work->j,
	   (uintptr_t)thief_work->r->data);

#if CONFIG_STEALABLE_THIEVES
    kaapi_task_init(thief_task, common_entry, thief_work);
#else
    kaapi_task_init(thief_task, thief_entry, thief_work);
#endif

    kaapi_thread_pushtask(thief_thread);

/*     kaapi_request_reply_tail(sc, requests, sizeof(work_t)); */
    kaapi_request_reply_head(sc, requests, thief_work->r);

    ++replied_count;
    --count;
  }

  return replied_count;
}


extern unsigned int kaapi_get_current_kid(void);

static void adaptive_entry(kaapi_stealcontext_t* sc, void* args, kaapi_thread_t* thread)
{
  work_t* const w = args;

  w->kid = kaapi_get_current_kid();
  if (w->r != NULL)
    ((work_t*)w->r->data)->kid = w->kid;

  printf("[%02x,   ] ent [%04u, %04u[ (%lx)\n",
	 w->kid, w->i, w->j,
	 w->r != NULL ? (uintptr_t)w->r->data : (uintptr_t)NULL);

  /* sequential work */

 continue_seq:

  while (1)
  {
    work_t local_work;

    /* kaapi_steal_begincritical(sc); */

    if (!pop_work(w, &local_work))
      break;

    /* kaapi_steal_endcritical(sc); */

    local_work.kid = w->kid;
    seq_work(&local_work);

    if (w->r != NULL)
      kaapi_preemptpoint(w->r, sc, NULL, NULL, w, sizeof(work_t), NULL);
  }

  /* retrive thieves results */

  kaapi_taskadaptive_result_t* const ktr =
    kaapi_preempt_getnextthief_head(sc);

  if (ktr != NULL)
  {
#if 0
    printf("[%02x,   ] pre (%lx)\n", w->kid, (uintptr_t)ktr->data);
#endif

    kaapi_preempt_thief(sc, ktr, NULL, reducer, w);
    goto continue_seq;
  }

  if (w->r != NULL)
  {
    /* update results before leaving */
    printf("[%02x,   ] don [%04u, %04u[ (%lx)\n", w->kid, w->i, w->j, (uintptr_t)w->r->data);
    memcpy(w->r->data, w, sizeof(work_t));
  }
}


#if !CONFIG_STEALABLE_THIEVES

static void thief_entry(void* args, kaapi_thread_t* thread)
{
  /* cannot be stolen */

  kaapi_stealcontext_t* const sc =
    kaapi_thread_pushstealcontext(thread, KAAPI_STEALCONTEXT_DEFAULT, NULL, NULL);

  adaptive_entry(sc, args, thread);

  kaapi_steal_finalize(sc);
}

#endif

static void common_entry(void* args, kaapi_thread_t* thread)
{
  work_t* const w = args;
  kaapi_stealcontext_t* const sc =
    kaapi_thread_pushstealcontext(thread, KAAPI_STEALCONTEXT_DEFAULT, splitter, w);

  adaptive_entry(sc, args, thread);

  kaapi_steal_finalize(sc);
}


static work_t* create_work(unsigned int q)
{
  static work_t work;
  init_work(&work);
  work.i = 0;
  work.j = q;
  work.k = 0;
  return &work;
}

static void root_entry(unsigned int q)
{
  kaapi_thread_t* thread;
  kaapi_task_t* task;

  thread = kaapi_self_thread();

  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, common_entry, create_work(q));
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();
}


int main(int ac, char** av)
{
  root_entry(CONFIG_INPUT_SIZE);
  return 0;
}
