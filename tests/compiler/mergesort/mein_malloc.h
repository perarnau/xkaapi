#ifndef MEIN_MALLOC_H_INCLUDED
# define MEIN_MALLOC_H_INCLUDED


#include <sys/types.h>


#if CONFIG_USE_MEIN_MALLOC

#include <stdint.h>

static uintptr_t mein_base;
static uintptr_t mein_addr;

static inline void mein_init(void* base, size_t size)
{
  mein_base = (uintptr_t)base;
  mein_addr = (uintptr_t)malloc(size);
}

static inline void mein_fini(void)
{
  free((void*)mein_addr);
}

static inline void* mein_malloc(void* id, size_t size)
{
  return (void*)(mein_addr + ((uintptr_t)id - mein_base));
}

static inline void mein_free(void* addr)
{}

#else /* CONFIG_USE_MEIN_MALLOC == 0 */

static inline void mein_init(void* base, size_t size)
{}

static inline void mein_fini(void)
{}

static inline void* mein_malloc(void* id, size_t size)
{ return malloc(size); }

static inline void mein_free(void* fu)
{ free(fu); }

#endif /* CONFIG_USE_MEIN_MALLOC */


#endif /* ! MEIN_MALLOC_H_INCLUDED */
