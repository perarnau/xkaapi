#ifndef CONC_RANGE_H_INCLUDED
# define CONC_RANGE_H_INCLUDED


/* implement a concurrent range whose
   semantic is similar to a THE queue
 */


#include <limits.h>


#define CONFIG_USE_KAAPI 1
#if CONFIG_USE_KAAPI
#include "kaapi.h"
#endif /* CONFIG_USE_KAAPI */


typedef long conc_size_t;
#define CONC_SIZE_MAX LONG_MAX


typedef struct conc_range
{
  /* separated cache lines since concurrent update */
  volatile conc_size_t _beg __attribute__((aligned(64)));
  volatile conc_size_t _end __attribute__((aligned(64)));
} conc_range_t;


#define CONC_RANGE_INITIALIZER(__beg, __end) { __beg, __end }


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
  __asm__ __volatile__ ("":::"memory");
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

static inline void conc_range_init
(conc_range_t* cr, conc_size_t beg, conc_size_t end)
{
  cr->_beg = beg;
  cr->_end = end;
}

static inline unsigned int conc_range_pop_front
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t size)
{
  /* return value a boolean */

  /* concurrency requirements:
     pop_back
   */

  cr->_beg += size;
    
  __full_barrier();

  if (cr->_beg > cr->_end)
  {
    cr->_beg -= size;
    return 0; /* false */
  }

  *end = cr->_beg;
  *beg = *end - size;

  return 1; /* true */
}

static inline unsigned int conc_range_pop_front_max
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t size)
{
  /* wrapper around the above function with size a maximum size
   */

  for (; size; --size)
    if (conc_range_pop_front(cr, beg, end, size))
      return 1; /* true */
  return 0; /* false */
}

static inline unsigned int conc_range_pop_back
(conc_range_t* cr, conc_size_t* beg, conc_size_t* end, conc_size_t size)
{
  /* return value a boolean */

  /* concurrency requirements:
     conc_range_set
     conc_range_pop_front
   */

  cr->_end -= size;

  __full_barrier();

  if (cr->_end < cr->_beg)
  {
    cr->_end += size;
    return 0; /* false */
  }

  *beg = cr->_end;
  *end = *beg + size;

  return 1; /* true */
}

static inline conc_size_t conc_range_size
(const conc_range_t* cr)
{
  /* warning, can return a negative value */
  return cr->_end - cr->_beg;
}

static inline void conc_range_set
(conc_range_t* cr, conc_size_t beg, conc_size_t end)
{
  /* concurrency requirements:
     conc_range_pop_back
   */

  /* typically in the reducer routine */

  cr->_beg = CONC_SIZE_MAX;

  /* ensure everyone see the previous store */
  __write_barrier();

  cr->_end = end;

  /* ensure everyone see _end before _beg */
  __write_barrier();

  cr->_beg = beg;
}


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
    if (!conc_range_pop_front_max(&cr, &i, &j, MAX_SIZE))
      break ;

    printf("%ld - %ld\n", i, j);
  }

  return 0;
}

#endif /* use_case */


#endif /* ! CONC_RANGE_H_INCLUDED */
