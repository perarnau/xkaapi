#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "kaapi_impl.h"


/* missing decls
 */

extern unsigned long kaapi_numa_get_addr_binding(uintptr_t);


/* numa bound task containers
 */

typedef struct bound_task
{
  struct bound_task* next;

  kaapi_thread_context_t* thread;
  kaapi_task_t* task;
  unsigned int war;

} bound_task_t;

typedef struct bound_tasklist
{
  kaapi_atomic_t lock;
  bound_task_t* volatile head;
  volatile unsigned int count;
  double pad[64 - 3 * 8];
} bound_tasklist_t;

/* at most 8 numa levels */
static bound_tasklist_t bound_tasklists[8] = {{{0}, }, };

static inline void lock_bound_tasklist(bound_tasklist_t* bl)
{
  kaapi_atomic_t* const lock = &bl->lock;

  while (1)
  {
    if ((KAAPI_ATOMIC_READ(lock) == 0) && KAAPI_ATOMIC_CAS(lock, 0, 1))
      break ;
    kaapi_slowdown_cpu();
  }
}

static inline void unlock_bound_tasklist(bound_tasklist_t* bl)
{
  kaapi_atomic_t* const lock = &bl->lock;
  KAAPI_ATOMIC_WRITE(lock, 0);
}

static void push_bound_task
(
 unsigned int binding,
 kaapi_thread_context_t* thread,
 kaapi_task_t* task,
 unsigned int war
)
{
  bound_tasklist_t* const bl = &bound_tasklists[binding];
  bound_task_t* const bt = malloc(sizeof(bound_task_t));

  kaapi_assert_debug(bt);
  bt->next = NULL;
  bt->thread = thread;
  bt->task = task;
  bt->war = war;

  lock_bound_tasklist(bl);

  bt->next = bl->head;
  bl->head = bt;
  ++bl->count;
  __sync_synchronize();

  unlock_bound_tasklist(bl);
}

static int pop_bound_task
(
 unsigned int binding,
 kaapi_thread_context_t** thread,
 kaapi_task_t** task,
 unsigned int* war
)
{
  /* assume kproc locked */

  bound_tasklist_t* const bl = &bound_tasklists[binding];

  bound_task_t* head;

  lock_bound_tasklist(bl);

  head = bl->head;
  if (head == NULL)
  {
    unlock_bound_tasklist(bl);
    return -1;
  }

  bl->head = head->next;
  --bl->count;

  unlock_bound_tasklist(bl);

  *thread = head->thread;
  *task = head->task;
  *war = head->war;

  free(head);

  return 0;
}

static inline unsigned int has_bound_task(unsigned int binding)
{
  /* may not be coherent */
  bound_tasklist_t* const bl = &bound_tasklists[binding];
  return bl->head != NULL;
}


/* exported
 */

#if 0 /* UNIMPLEMENTED */

unsigned int kaapi_task_binding_is_local
(const kaapi_task_binding_t* binding)
{
  unsigned int is_local = 0;

  switch (binding->type)
  {
  default:
  case ANY:
    is_local = 1;
    break ;

  case OCR:
    break ;
  }

  return is_local;
}

#endif /* UNIMPLEMENTED */


#if 0 /* UNUSED */

void kaapi_task_binding_make_ocr
(kaapi_task_binding_t* binding, const uintptr_t* addrs, size_t count)
{
  kaapi_assert_debug(count <= KAAPI_TASK_BINDING_ADDR_COUNT);

  binding->type = OCR;
  memcpy(binding->u.ocr.addrs, addrs, count * sizeof(uintptr_t));
}


void kaapi_task_binding_make_local
(kaapi_task_binding_t* binding)
{
  binding->type = LOCAL;
}

#endif /* UNUSED */


void kaapi_push_bound_task_numaid
(
 unsigned int numaid,
 kaapi_thread_context_t* thread,
 kaapi_task_t* task,
 unsigned int war
)
{
  push_bound_task(numaid, thread, task, war);
}

int kaapi_pop_bound_task_numaid
(
 unsigned int numaid,
 kaapi_thread_context_t** thread,
 kaapi_task_t** task,
 unsigned int* war
)
{
  if (has_bound_task(numaid))
    return pop_bound_task(numaid, thread, task, war);
  return -1;
}


static void count_range_pages
(unsigned int* counts, uintptr_t addr, size_t size)
{
  if (addr & (0x1000 - 1))
  {
    addr &= ~(0x1000 - 1);
    size += 0x1000;
  }

  if (size & (0x1000 - 1))
    size = (size + 0x1000) & ~(0x1000 - 1);

  for (; size; size -= 0x1000, addr += 0x1000)
    ++counts[(size_t)kaapi_numa_get_addr_binding(addr)];
}

unsigned int kaapi_task_binding_numaid
(const kaapi_task_binding_t* binding)
{
  /* temp, assume valid to do so. */

#if 1

  unsigned int page_counts[8] = {0, };
  size_t i;
  size_t max;

  /* count per node pages */
  for (i = 0; i < binding->u.ocr.count; ++i)
  {
    count_range_pages
      (page_counts, binding->u.ocr.addrs[i], binding->u.ocr.sizes[i]);
  }

  /* find the biggest page count */
  max = 0;
  for (i = 1; i < 8; ++i)
  {
    if (page_counts[max] < page_counts[i])
      max = i;
  }

  return (unsigned int)max;

#else
  /* simple strict mapping */

  return (unsigned int)
    kaapi_numa_get_addr_binding(binding->u.ocr.addrs[0]);

#endif
}
