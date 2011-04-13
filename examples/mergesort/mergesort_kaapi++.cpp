#include "kaapi++"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>


typedef unsigned long elem_type;
typedef elem_type* iterator_type;


#define CONFIG_LEAF_SIZE 0x20
#define CONFIG_ARRAY_SIZE (4 * 1024 * 1024)

#define CONFIG_USE_TIME 1
#define CONFIG_USE_CHECK 1


// sequential code
static void naivesort(iterator_type i, iterator_type j)
{
  iterator_type m, n;
  iterator_type pos;
  elem_type val;

  for (m = i; m != j; ++m)
  {
    pos = m;
    val = *m;

    for (n = m + 1; n != j; ++n)
    {
      if (*n < val)
      {
	val = *n;
	pos = n;
      }
    }

    if (pos != m)
    {
      *pos = *m;
      *m = val;
    }
  }
}

#if 0 // unused
static void mergesort_seq(iterator_type i, iterator_type j)
{
  const size_t d = j - i;

  if (d <= CONFIG_LEAF_SIZE)
  {
    naivesort(i, j);
    return ;
  }

  iterator_type pivot = i + d / 2;
  mergesort_seq(i, pivot);
  mergesort_seq(pivot, j);

  std::inplace_merge(i, pivot, j);
}
#endif // unused


// merge kaapi task

struct mergeTask : public ka::Task<2>::Signature
<
  ka::RW<ka::range1d<elem_type> >,
  ka::RW<ka::range1d<elem_type> >
> {};

template<> struct TaskBodyCPU<mergeTask>
{
  void operator()
  (ka::range1d_rw<elem_type> fu, ka::range1d_rw<elem_type> bar)
  {
    std::inplace_merge(fu.ptr(), bar.ptr(), bar.ptr() + bar.size());
  }
};


// naivesort kaapi task

struct naivesortTask : public ka::Task<1>::Signature
< ka::RW<ka::range1d<elem_type> > > {};

template<> struct TaskBodyCPU<naivesortTask>
{
  void operator()(ka::range1d_rw<elem_type> r)
  {
    naivesort(r.ptr(), r.ptr() + r.size());
  }
};


// naivesort kaapi task

struct mergesortTask : public ka::Task<1>::Signature
< ka::RPWP<ka::range1d<elem_type> > > {};

template<> struct TaskBodyCPU<mergesortTask>
{
  void operator()(ka::range1d_rpwp<elem_type> range)
  {
    const size_t d = range.size();

    if (d <= CONFIG_LEAF_SIZE)
    {
      ka::Spawn<naivesortTask>()(range);
      return ;
    }

    const size_t pivot = d / 2;

    ka::array<1, elem_type> fu((elem_type*)range.ptr(), pivot);
    ka::Spawn<mergesortTask>()(fu);

    ka::array<1, elem_type> bar((elem_type*)range.ptr() + pivot, d - pivot);
    ka::Spawn<mergesortTask>()(bar);

    ka::Spawn<mergeTask>()(fu, bar);
  }
};


// parallel mergesort entrypoint
static void mergesort_par(iterator_type i, iterator_type j)
{
  ka::array<1, elem_type> fubar(i, j - i);
  // ka::Spawn<mergesortTask>(ka::SetStaticSched())(fubar);
  ka::Spawn<mergesortTask>()(fubar);
  ka::Sync();
}


// main code

static bool is_sorted
(iterator_type i, iterator_type j, size_t& k)
{
  if ((j - i) <= 1) return true;

  elem_type prev = *i;

  k = 0;
  for (i = i + 1; i != j; ++i, ++k)
  {
    if (prev > *i) return false;
    prev = *i;
  }

  return true;
}


static void shuffle(iterator_type i, iterator_type j)
{
  for (; i != j; ++i) *i = (elem_type)(rand() % 1000);
}


struct doit
{
  void operator()(int argc, char** argv )
  {
    static const size_t count = CONFIG_ARRAY_SIZE;
    elem_type* const v = (elem_type*)malloc(count * sizeof(elem_type));

#if CONFIG_USE_TIME
    double t0, t1, tt = 0;
#endif

    size_t iter;
    for (iter = 0; iter < 10; ++iter)
    {
      shuffle(v, v + count);

#if CONFIG_USE_TIME
      t0 = kaapi_get_elapsedtime();
#endif

      mergesort_par(v, v + count);
      // mergesort_seq(v, v + count);

#if CONFIG_USE_TIME
      t1 = kaapi_get_elapsedtime();
      // do not time first run
      if (iter == 0) continue ;
      tt += t1 - t0;
#endif

#if CONFIG_USE_CHECK
    size_t k;
    if (is_sorted(v, v + count, k) == false)
      printf("invalid @%lu\n", k);
#endif
    }

#if CONFIG_USE_TIME
    std::cout << tt / (iter - 1) << std::endl; // seconds
#endif
  }
};


int main(int argc, char** argv)
{
  try
  {
    ka::Community com = ka::System::join_community(argc, argv);
    ka::SpawnMain<doit>()(argc, argv); 
    com.leave();
    ka::System::terminate();
  }
  catch (const std::exception& E)
  {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...)
  {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }

  return 0;
}
