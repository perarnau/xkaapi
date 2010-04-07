#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "kaapi.h"


#define CONFIG_INPUT_SIZE 10000
#define CONFIG_ALLOC_RESULT 1
#define CONFIG_STEALABLE_THIEVES 1
#define CONFIG_STATIC_STEAL 0
#define CONFIG_STEAL_SIZE 10
#define CONFIG_POP_SIZE 100
#define CONFIG_CHECK_DATA 1


#if 0 /* kid mark */

static unsigned char marks[16] = {0, };

static void mark_kid(unsigned int kid)
{
  marks[kid] = 1;
}

static void unmark_kid(unsigned int kid)
{
  marks[kid] = 0;
}

static int is_kid_marked(unsigned int kid)
{
  return marks[kid] == 1;
}

#endif

static inline void print_spaces(unsigned int n)
{
#if 0
  for (int i = n; i > 0; --i)
    printf("  ");
#else
#endif
}


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

  /* data pointer */
  unsigned int* data;

  /* thief result */
  kaapi_taskadaptive_result_t* r;

  /* debugging */
  unsigned int kid;
  unsigned int wid;
};

typedef struct work work_t;


#if 0
static void print_work(const void* data)
{
  const work_t* const w = data;
  printf(" :: [%u - %u[\n", w->i, w->j);
}
#endif


static void init_work(work_t* w)
{
  static unsigned int wid __attribute__((aligned)) = 0;

  w->l = 0;

  w->i = 0;
  w->j = 0;
  w->k = 0;

#if CONFIG_CHECK_DATA
  w->data = NULL;
#endif

  w->r = NULL;

  w->kid = (unsigned int)-1;

  w->wid = __sync_add_and_fetch(&wid, 1);
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
  /* assume work locked */
  return w->j - w->k;
}


static void merge_work_safe(work_t* a, work_t* b)
{
  a->j = b->j;
  a->k = b->k;
}


static void __attribute__((unused)) merge_work(work_t* a, work_t* b)
{
  /* assume b safe to access */

  lock_work(a);
  merge_work_safe(a, b);
  lock_work(b);
}


static void __attribute__((unused)) empty_work (work_t* w)
{
  w->i = 0;
  w->j = 0;
  w->k = 0;
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


static unsigned int steal_work_safe(work_t* a, work_t* b)
{
  /* assume a locked */

#if CONFIG_STATIC_STEAL
  const unsigned int size = CONFIG_STEAL_SIZE;
#else
  const unsigned int size = (a->j - a->k) / 2;
#endif

  b->j = a->j;
  b->i = a->j - ((a->j - a->k) < size ? a->j - a->k : size);
  b->k = b->i;

  a->j = b->i;

  return b->j - b->i;
}


static unsigned int __attribute__((unused)) steal_work(work_t* a, work_t* b)
{
  unsigned int res;

  lock_work(a);
  res = steal_work_safe(a, b);
  unlock_work(a);

  return b->j - b->i;
}


static void seq_work(work_t* w)
{
  unsigned int i;

#if 0
  printf("[%02x,   ] seq [%u - %u[\n", w->kid, w->i, w->j);
#endif

  for (i = w->i; i < w->j; ++i)
  {
#if CONFIG_CHECK_DATA
    w->data[i] = (1 << 8) | w->kid;
#else
    usleep(10);
#endif
  }
}


static int reducer
(
 kaapi_stealcontext_t* sc,
 void* thief_arg,
 work_t* thief_work, size_t thief_size,
 work_t* victim_work
)
{
#if 0 /* kid mark */
  mark_kid(thief_work->kid);
#endif

  /* is being reduced */

  print_spaces(victim_work->kid);
  printf("[%02x::%02x, %02x] red [%05u, %05u[ [%05u, %05u, %05u[ (%u)\n",
	 victim_work->kid, victim_work->wid, thief_work->kid,
	 victim_work->i, victim_work->j,
	 thief_work->i, thief_work->k, thief_work->j,
	 thief_work->wid);

  lock_work(victim_work);
  merge_work_safe(victim_work, thief_work);

#if CONFIG_CHECK_DATA
  {
    /* mark as reduced what has been done by the thief */
    unsigned int i;
    for (i = thief_work->i; i < thief_work->k; ++i)
      victim_work->data[i] |= 1 << 9;
  }
#endif

  unlock_work(victim_work);

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

#if 0
  if (is_kid_marked(victim_work->kid))
    return 0;
#endif

  for (; (count > 0) && (!is_done); ++requests)
  {
    kaapi_thread_t* thief_thread;
    kaapi_task_t* thief_task;
    work_t* thief_work = NULL;

    if (!kaapi_request_ok(requests))
      continue ;

    lock_work(victim_work);

    if (work_remaining_size(victim_work) < CONFIG_STEAL_SIZE)
    {
      unlock_work(victim_work);
      break;
    }

    thief_thread = kaapi_request_getthread(requests);
    thief_task = kaapi_thread_toptask(thief_thread);

    thief_work = kaapi_thread_pushdata(thief_thread, sizeof(work_t));

    init_work(thief_work);

#if CONFIG_ALLOC_RESULT
    thief_work->r = kaapi_allocate_thief_result(sc, sizeof(work_t), NULL);
#else
    thief_work->r = kaapi_allocate_thief_result(sc, sizeof(work_t), thief_work);
#endif

#if CONFIG_CHECK_DATA
    thief_work->data = victim_work->data;
#endif

    steal_work_safe(victim_work, thief_work);

    unlock_work(victim_work);

#if 0
    printf("red alloc %lx [%u - %u[\n",
	   (uintptr_t)thief_work->r->data,
	   thief_work->i, thief_work->j);
#endif

    /* result has to be initialized since from the
       push task the thief may be preempted
     */

#if CONFIG_ALLOC_RESULT
    memcpy(thief_work->r->data, thief_work, sizeof(work_t));
#else
    thief_work->r->data = thief_work;
#endif

    print_spaces(victim_work->kid);
    printf("[%02x,   ] spl [%04u, %04u[ [%04u, %04u[ (%lx::%lx)\n",
	   victim_work->kid,
	   victim_work->k, victim_work->j,
	   thief_work->i, thief_work->j,
	   (uintptr_t)thief_task,
	   (uintptr_t)thief_work);

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

static void adaptive_entry
(kaapi_stealcontext_t* sc, void* args, kaapi_thread_t* thread)
{
  work_t* const w = args;

  w->kid = kaapi_get_current_kid();

#if CONFIG_ALLOC_RESULT
  if (w->r != NULL)
    ((work_t*)w->r->data)->kid = w->kid;
#endif

#if 0
  printf("[%02x,   ] ent [%04u, %04u[ (%lx)\n",
	 w->kid, w->i, w->j,
	 w->r != NULL ? (uintptr_t)w->r->data : (uintptr_t)NULL);
#endif

  /* sequential work */

 continue_seq:

#if 0 /* kid mark */
  unmark_kid(w->kid);
#endif

  print_spaces(w->kid);
  printf("[%02x,   ] seq [%04u, %04u[ (%lx)\n", w->kid, w->k, w->j, (uintptr_t)w);

  while (1)
  {
    work_t local_work;

    if (!pop_work(w, &local_work))
      break;

    local_work.data = w->data;
    local_work.kid = w->kid;
    seq_work(&local_work);

    if (w->r != NULL)
    {
      if (kaapi_preemptpoint(w->r, sc, NULL, NULL, w, sizeof(work_t), NULL))
      {
	print_spaces(w->kid);
	printf("[%02x,   ] pre\n", w->kid); fflush(stdout);
	return ;
      }
    }
  }

  /* retrive thieves results */

  kaapi_taskadaptive_result_t* ktr;

  ktr = kaapi_preempt_getnextthief_head(sc);
  if (ktr != NULL)
  {
    print_spaces(w->kid);
    printf("[%02x,   ] preempt_thief\n", w->kid);
    kaapi_preempt_thief(sc, ktr, NULL, reducer, w);
    goto continue_seq;
  }

  /* here no thieves, steal disabled, can leave */

  print_spaces(w->kid);
  printf("[%02x,   ] don [%04u, %04u[ %u\n", w->kid, w->i, w->j, w->wid);

#if CONFIG_ALLOC_RESULT
  if (w->r != NULL)
  {
    /* update results before leaving */
    memcpy(w->r->data, w, sizeof(work_t));
  }
#endif
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
#if CONFIG_CHECK_DATA
  static unsigned int data[CONFIG_INPUT_SIZE];
#endif

  static work_t work;

  init_work(&work);

  work.i = 0;
  work.j = q;
  work.k = 0;

#if CONFIG_CHECK_DATA
  memset(data, 0, sizeof(data));
  work.data = data;
#endif

  return &work;
}

#if CONFIG_CHECK_DATA

static int check_work(const work_t* w)
{
  unsigned int i;

  for (i = 0; i < CONFIG_INPUT_SIZE; ++i)
  {
    const unsigned char kid = w->data[i] & 0xff;

    if (!(w->data[i] & (1 << 8)))
    {
      printf("!!! notProcessed@%u\n", i);
      return -1;
    }

    /* not master, not reduced */
    if (kid && !(w->data[i] & (1 << 9)))
    {
      printf("!!! notReduced@%u, processedBy %u: 0x%08x\n", i, kid, w->data[i]);
      return -1;
    }
  }

  return 0;
}

#endif

static int root_entry(unsigned int q)
{
  kaapi_thread_t* thread;
  kaapi_task_t* task;
  work_t* const work = create_work(q);

  thread = kaapi_self_thread();

  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, common_entry, work);
  kaapi_thread_pushtask(thread);
  kaapi_sched_sync();

#if CONFIG_CHECK_DATA
  if (check_work(work) == -1)
  {
    printf("invalidWork\n");
    return -1;
  }
#endif

  return 0;
}


int main(int ac, char** av)
{
  return root_entry(CONFIG_INPUT_SIZE);
}
