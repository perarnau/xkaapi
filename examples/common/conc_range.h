#ifndef CONC_RANGE_H_INCLUDED
# define CONC_RANGE_H_INCLUDED


/* implement a concurrent range whose semantic is
   similar to a THE queue. this implementation assumes
   there is only one sequential and one parallel
   extractor (resp. pop_front and pop_back). this is
   ensured by the concurrent version of the xkaapi
   runtime (resp. sequential work and splitter).

   locking must be used to solve concurrency issuses
   between pop_back and set since they both write the
   _end pointer. full locking is used for debugging
   and locks everywhere the range is accessed.
 */


#include <limits.h>


#define CONFIG_USE_KAAPI 1
#if CONFIG_USE_KAAPI
#include "kaapi.h"
#endif /* CONFIG_USE_KAAPI */

#define CONFIG_USE_FULL_LOCK 0

typedef long conc_size_t;
#define CONC_SIZE_MIN LONG_MIN
#define CONC_SIZE_MAX LONG_MAX


typedef struct conc_range
{
  volatile long _lock;

  /* separated cache lines since concurrent update */
  volatile conc_size_t _beg __attribute__((aligned(64)));
  volatile conc_size_t _end __attribute__((aligned(64)));

} conc_range_t;


#define CONC_RANGE_INITIALIZER(__beg, __end) { 0L, __beg, __end }

static inline void __full_barrier(void)
{
#if CONFIG_USE_KAAPI
  kaapi_mem_barrier();
#else
  __asm__ __volatile__ ("mfence\n\t":::"memory");
#endif
}

static inline void __write_barrier(void)
{
#if CONFIG_USE_KAAPI
  kaapi_writemem_barrier();
#else
  __asm__ __volatile__ ("sfence\n\t":::"memory");
#endif
}

static inline void __read_barrier(void)
{
#if CONFIG_USE_KAAPI
  kaapi_readmem_barrier();
#else
  __asm__ __volatile__ ("lfence\n\t":::"memory");
#endif
}

static inline void __slowdown(void)
{
#if CONFIG_USE_KAAPI
  kaapi_slowdown_cpu();
#else
  __asm__ __volatile__ ("pause\n\t");
#endif
}


static inline void conc_range_init
(conc_range_t* cr, conc_size_t beg, conc_size_t end)
{
  cr->_lock = 0L;
  cr->_beg = beg;
  cr->_end = end;
}


static inline void conc_range_empty
(conc_range_t* cr)
{
  cr->_beg = CONC_SIZE_MAX;
}


static inline void __lock_range(conc_range_t* cr)
{
  while (1)
  {
    if (__sync_bool_compare_and_swap(&cr->_lock, 0, 1))
      break ;
    __slowdown();
  }
}

static inline void __unlock_range(conc_range_t* cr)
{
  __full_barrier();
  cr->_lock = 0L;
}


#if CONFIG_USE_FULL_LOCK /* full locking */

static inline void conc_range_pop_front
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t max_size)
{
  conc_size_t size;

  __lock_range(cr);

  size = cr->_end - cr->_beg;
  if (max_size > size)
    max_size = size;

  *beg = cr->_beg;
  *end = *beg + max_size;

  cr->_beg = *end;

  __unlock_range(cr);
}

static inline unsigned int conc_range_pop_back
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t size)
{
  unsigned int has_popped = 0;

  /* disable gcc warning */
  *beg = 0;
  *end = 0;

  __lock_range(cr);

  if (size > (cr->_end - cr->_beg))
    goto unlock_return;

  has_popped = 1;

  *end = cr->_end;
  *beg = *end - size;
  cr->_end = *beg;

 unlock_return:
  __unlock_range(cr);

  return has_popped;
}

static inline conc_size_t conc_range_size
(conc_range_t* cr)
{
  conc_size_t size;

  __lock_range(cr);
  size = cr->_end - cr->_beg;
  __unlock_range(cr);

  return size;
}

static inline void conc_range_set
(conc_range_t* cr, conc_size_t beg, conc_size_t end)
{
  __lock_range(cr);
  cr->_beg = beg;
  cr->_end = end;
  __unlock_range(cr);
}

#else /* CONFIG_USE_FULL_LOCK == 0 */

static inline void conc_range_pop_front
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t max_size)
{
  /* concurrency requirements: pop_back */

  /* it is possible the returned range is empty,
     but this function never fails, contrary to
     pop_back (ie. sequential extraction always
     succeeds)
   */

  conc_size_t size = max_size;

  cr->_beg += size;
  __full_barrier();

  if (cr->_beg <= cr->_end)
    goto no_conflict;

  /* handle conflict */

  cr->_beg -= size;

  __lock_range(cr);

  size = cr->_end - cr->_beg;
  if (size > max_size)
    size = max_size;

  cr->_beg += size;

  __unlock_range(cr);

 no_conflict:
  *end = cr->_beg;
  *beg = *end - size;
}

static inline unsigned int conc_range_pop_back
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t size)
{
  /* return value a boolean. in case of a conflict with
     pop_front, this side fails and false is returned
   */

  /* concurrency requirements:
     conc_range_set
     conc_range_pop_front
   */

  unsigned int has_popped = 0;

  /* disable gcc warning */
  *beg = 0;
  *end = 0;

  __lock_range(cr);

  cr->_end -= size;
  __full_barrier();

  if (cr->_end < cr->_beg)
  {
    cr->_end += size;
    goto unlock_return;
  }

  has_popped = 1;

  *beg = cr->_end;
  *end = *beg + size;

 unlock_return:
  __unlock_range(cr);

  return has_popped; /* true */
}

static inline conc_size_t conc_range_size
(conc_range_t* cr)
{
  return cr->_end - cr->_beg;
}

static inline void conc_range_set
(conc_range_t* cr, conc_size_t beg, conc_size_t end)
{
  /* concurrency requirements:
     conc_range_pop_back
   */

  /* NOT calling this routine concurrently
     with pop_front ensures _beg wont move.
     this is required to avoid underflows
   */

  __lock_range(cr);

  cr->_beg = CONC_SIZE_MAX;
  __full_barrier();

  cr->_end = end;
  __full_barrier();

  cr->_beg = beg;

  __unlock_range(cr);
}

#endif /* CONFIG_USE_FULL_LOCK */


#if 0 /* use_case */

#include <stdio.h>
#include "conc_range.h"

int main(int ac, char** av)
{
  conc_range_t cr = CONC_RANGE_INITIALIZER(0, 20);

  while (conc_range_size(&cr) > 0)
  {
    long i, j;

#define MAX_SIZE 3
    if (!conc_range_pop_front(&cr, &i, &j, MAX_SIZE))
      break ;

    printf("%ld - %ld\n", i, j);
  }

  return 0;
}

#endif /* use_case */


#endif /* ! CONC_RANGE_H_INCLUDED */
