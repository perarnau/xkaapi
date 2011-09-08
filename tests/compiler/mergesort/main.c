/* warning: include first */
#include "config.h"

#if CONFIG_DO_BENCH
#define __USE_BSD
#include <sys/time.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "kaapi.h"

#include "mein_malloc.h"


#if CONFIG_USE_INPLACE_MERGE /* INPLACE_MERGE */

static inline void swap(int* a, int* b)
{
  const int tmp = *a;
  *a = *b;
  *b = tmp;
}

#pragma kaapi task value(k) write(i[j - i], j[k - j])
static void inplace_merge(int* i, int* j, int* k)
{
 do_merge:
  if (j == k) return ;

  /* find *i > *j and swap */
  for (; (i != j) && (*i <= *j); ++i)
    ;

  if (i == j) return ;

  swap(i, j);

  int* const saved_j = j;

  /* shift j until valid range */
  for (; (j != (k - 1)) && (j[0] > j[1]); ++j)
    swap(j + 0, j + 1);

  /* we are in a valid case, recurse */
  i = i + 1;
  j = saved_j;
  goto do_merge;
}

#else /* INPLACE_MERGE == 0 */

#include <string.h>

static inline void swap_iterators(int** a, int** b)
{
  int* tmp = *a;
  *a = *b;
  *b = tmp;
}

static inline void swap_sizes(unsigned int* a, unsigned int* b)
{
  const unsigned int tmp = *a;
  *a = *b;
  *b = tmp;
}

#pragma kaapi task \
read(ibeg[iend - ibeg], iend) \
read(jbeg[jend - jbeg], jend) \
write(obeg[iend - ibeg + jend - jbeg])
static void ooplace_merge_iter
(int* ibeg, int* iend, int* jbeg, int* jend, int* obeg)
{
  /* iterative outofplace merge */

  /* fill i segment */
 redo_fill:
  for (; (iend != ibeg) && (*ibeg <= *jbeg); ++ibeg, ++obeg)
    *obeg = *ibeg;

  if (ibeg == iend) goto fill_last_j;

  swap_iterators(&ibeg, &jbeg);
  swap_iterators(&iend, &jend);
  goto redo_fill;

  /* fill last j segment */
 fill_last_j:
  for (; jbeg != jend; ++jbeg, ++obeg)
    *obeg = *jbeg;
}

#if CONFIG_USE_PARALLEL_MERGE

static int* binsearch(int* i, int* j, int value)
{
  /* search value in [i, j[ */

  while (i < j)
  {
    int* mi = i + ((j - i) / 2);
    if (value <= *mi) j = mi;
    else i = mi + 1;
  }

  return j;
}

#pragma kaapi task \
read(ibeg[iend - ibeg], iend) \
read(jbeg[jend - jbeg], jend) \
write(obeg[iend - ibeg + jend - jbeg])
static void ooplace_merge_rec
(int* ibeg, int* iend, int* jbeg, int* jend, int* obeg)
{
  /* recursive outofplace merge */

  unsigned int isize = iend - ibeg;
  unsigned int jsize = jend - jbeg;

  int* ipos;
  int* jpos;
  int* opos;

  /* ensure size(i) > size(j) */
  if (isize < jsize)
  {
    swap_iterators(&ibeg, &jbeg);
    swap_iterators(&iend, &jend);
    swap_sizes(&isize, &jsize);
  }

  /* nothing to do */
  if (isize == 0) return ;

  /* sequential merge if threshold reached */
  if ((isize + jsize) <= CONFIG_PARALLEL_MERGE_GRAIN)
  {
    ooplace_merge_iter(ibeg, iend, jbeg, jend, obeg);
    return ;
  }

  /* recursive merge */
  ipos = ibeg + (isize / 2);
  jpos = binsearch(jbeg, jend, *ipos);

  opos = obeg + ((ipos - ibeg) + (jpos - jbeg));

  *opos = *ipos;

  ooplace_merge_rec(ibeg, ipos, jbeg, jpos, obeg);
  ooplace_merge_rec(ipos + 1, iend, jpos, jend, opos + 1);
}

static inline void ooplace_merge
(int* i, int* j, int* k, int* o)
{
  ooplace_merge_rec(i, j, j, k, o);
}

#else /* CONFIG_USE_PARALLEL_MERGE == 0 */

static inline void ooplace_merge(int* i, int* j, int* k, int* o)
{
  ooplace_merge_iter(i, j, j, k, o);
}

#endif /* CONFIG_USE_PARALLEL_MERGE */

#endif /* INPLACE_MERGE */


/* licence: This public-domain C implementation
   from: http://alienryderflex.com/quicksort/
   author: original code by Darel Rex Finley.
   contrib: use int
 */
static void quicksort(int *arr, int elements)
{
#define  MAX_LEVELS  300
  int i = 0;
  int piv, beg[MAX_LEVELS], end[MAX_LEVELS], L, R, swap;

  beg[0]=0; end[0]=elements;
  while (i>=0)
  {
    L=beg[i]; R=end[i]-1;
    if (L<R)
    {
      piv=arr[L];
      while (L<R)
      {
        while (arr[R]>=piv && L<R) R--; if (L<R) arr[L++]=arr[R];
        while (arr[L]<=piv && L<R) L++; if (L<R) arr[R--]=arr[L];
      }

      arr[L]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;

      if (end[i]-beg[i]>end[i-1]-beg[i-1])
      {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
      }
    }
    else
    {
      i--;
    }
  }
}

#if CONFIG_USE_MEMCPY_FREE_TASK

#pragma kaapi task value(n) read(i[n]) write(o[n])
static void memcpy_task(int* o, int* i, unsigned int n)
{
  memcpy(o, i, n * sizeof(int));
}

#pragma kaapi task value(n) readwrite(o[n])
static void free_task(int* o, unsigned int n)
{
  mein_free(o);
}

#pragma kaapi task value(n) read(i[n]) write(o[n])
static void memcpy_and_free
(int* o, int* i, unsigned int n)
{
  size_t k;
  for (k = 0; k < n; k += 0x4000)
  {
    if ((n - k) < 0x4000)
      memcpy_task(o + k, i + k, n - k);
    else
      memcpy_task(o + k, i + k, 0x4000);
  }
  free_task(o, n);
}

#else /* CONFIG_MEMCPY_FREE_TASK */

#pragma kaapi task value(n) read(i[n]) write(o[n])
static void memcpy_and_free(int* o, int* i, unsigned int n)
{
  memcpy(o, i, n * sizeof(int));
  mein_free(o);
}

#endif /* CONFIG_MEMCPY_FREE_TASK */

#pragma kaapi task value(j) readwrite(i[j - i])
static void mergesort(int* i, int* j)
{
  const unsigned int pivot = (j - i) / 2;

  if (pivot <= CONFIG_LEAF_SIZE)
  {
    quicksort(i, (int)(j - i));
    return ;
  }

  mergesort(i, i + pivot);
  mergesort(i + pivot, j);

#if CONFIG_USE_INPLACE_MERGE
  {
    inplace_merge(i, i + pivot, j);
  }
#else /* CONFIG_USE_INPLACE_MERGE == 0 */
  {
    const unsigned int size = (j - i) * sizeof(int);
    int* const o = mein_malloc((void*)i, size);

    ooplace_merge(i, i + pivot, j, o);

    memcpy_and_free(i, o, j - i);
  }
#endif /* CONFIG_USE_INPLACE_MERGE */
}


static void shuffle(int* i, int* j)
{
  for (; i != j; ++i) *i = (int)(rand() % 10000);
}


#if CONFIG_DO_CHECK

static void __attribute__((unused))
print_array(int* i, int* j)
{
  for (; i != j; ++i) printf(" %u", (unsigned int)*i);
  printf("\n");
}

static unsigned int is_sorted(int* i, int* j)
{
  int prev;

  if ((j - i) <= 1) return 1;

  prev = *i;

  for (i = i + 1; i != j; ++i)
  {
    if (prev > *i) return 0;
    prev = *i;
  }

  return 1;
}

#endif /* CONFIG_DO_CHECK */

int main(int ac, char** av)
{
  static const unsigned int count = CONFIG_ARRAY_SIZE;
  int* const arr = (int*)malloc(count * sizeof(int));

#if CONFIG_DO_BENCH
  struct timeval sta, sto, dif;
#endif

#pragma kaapi parallel
  {
    mein_init(arr, count * sizeof(int));

    shuffle(arr, arr + count);

#if CONFIG_DO_BENCH
    gettimeofday(&sta, NULL);
#endif

    mergesort(arr, arr + count);
  } /* #pragma kaapi parallel */

#if CONFIG_DO_BENCH
  gettimeofday(&sto, NULL);
  timersub(&sto, &sta, &dif);
  printf("%u\n", (unsigned int)(dif.tv_sec * 1E6 + dif.tv_usec));
#endif

#if CONFIG_DO_CHECK
  if (is_sorted(arr, arr + count) == 0)
  {
    printf("invalid\n");
    /* print_array(arr, arr + count); */
  }
#endif

  mein_fini();

  return 0;
}
