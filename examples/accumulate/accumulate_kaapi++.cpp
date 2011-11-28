/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include "kaapi++"
#include <algorithm>

{
template<typename T, typename OP>
class Work {
public:
  Work()
   : _array(0)
  {}
  
  /* cstor */
  Work(T* beg, size_t size, OP op)
   : _op(op), _result(), _array(beg)
  {
    /* initialize work */
    kaapi_atomic_initlock( &_lock );
    kaapi_workqueue_init_with_lock(&_wq, 0, size, &_lock);
  }

  /* cstor */
  Work(T* beg, size_t size, const T& init, OP op)
   : _op(op), _result(init), _array(beg)
  {
    /* initialize work */
    kaapi_atomic_initlock( &_lock );
    kaapi_workqueue_init_with_lock(&_wq, 0, size, &_lock);
  }

  /* */
  ~Work()
  {
    kaapi_atomic_destroylock( &_lock );
  }
  
  /* extract sequential work */
  bool pop( T*& beg, T*& end)
  {
#define CONFIG_SEQ_GRAIN 8
    kaapi_workqueue_index_t b, e;
    if (kaapi_workqueue_pop(&_wq, &b, &e, CONFIG_SEQ_GRAIN)) return false;
    beg = _array+b;
    end = _array+e;
    return true;
  }
  
  /* name of the method should be splitter !!! split work and reply to requests */
  int split (
    ka::StealContext* sc, 
    int nreq, 
    ka::ListRequest::iterator beg,
    ka::ListRequest::iterator end
  );
  
  T* begin() { return _array + kaapi_workqueue_range_begin(&_wq); }
  T* end()   { return _array + kaapi_workqueue_range_end(&_wq); }
  OP& op()   { return _op; }
  T&  result()   { return _result; }

protected:
  /* extract parallel work for nreq. Return the unit size */
  bool helper_split( int& nreq, T*& beg, T*& end)
  {
    kaapi_atomic_lock( &_lock );
#define CONFIG_PAR_GRAIN 8
    kaapi_workqueue_index_t steal_size, i,j;
    kaapi_workqueue_index_t range_size = kaapi_workqueue_size(&_wq);
    if (range_size <= CONFIG_PAR_GRAIN)
    {
      kaapi_atomic_unlock( &_lock );
      return false;
    }

    steal_size = range_size * nreq / (nreq + 1);
    if (steal_size == 0)
    {
      nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
      steal_size = nreq*CONFIG_PAR_GRAIN;
    }

    /* perform the actual steal. if the range
       changed size in between, redo the steal
     */
    if (kaapi_workqueue_steal(&_wq, &i, &j, steal_size))
    {
      kaapi_atomic_unlock( &_lock );
      return false;
    }
    kaapi_atomic_unlock( &_lock );
//    printf("Steal: [%li, %li)\n", i, j);
//    fflush(stdout);
    beg = _array + i;
    end = _array + j;
    return true;
  }
  
protected:
  OP                _op;
  T                 _result;
  T*                _array;
};

}

/** Task for the thief
    CPU implementation: see different implementations
*/
template<typename T, typename OP>
struct TaskAccumulate : public ka::Task<1>::Signature<
  ka::R<ka::range1d<T> >,
  ka::CW<T>, 
  OP 
> {};


/* name of the method should be splitter !!! split work and reply to requests */
template<typename T, typename OP>
int Work<T,OP>::split (
  ka::StealContext* sc, 
  int nreq, 
  ka::ListRequest::iterator beg,
  ka::ListRequest::iterator end
)
{
  /* stolen range */
  T* beg_theft;
  T* end_theft;
  size_t size_theft;

  if (!helper_split( nreq, beg_theft, end_theft )) 
    return 0;
  size_theft = (end_theft-beg_theft)/nreq;

  /* thief work: create a task */
  for (; nreq>1; --nreq, ++beg, beg_theft+=size_theft)
  {
    beg->Spawn<TaskWork<T,OP> >(sc)( 
      new (*beg) Work<T,OP>( beg_theft, size_theft, _op)
    );
    beg->commit();
  }
  beg->Spawn<TaskWork<T,OP> >(sc)(
    new (*beg) Work<T,OP>( beg_theft, end_theft-beg_theft, _op)
  );
  beg->commit();
  ++beg;
  
  return 0;
}



/** Simple task that only do sequential computation
*/
template<typename T, typename OP>
struct TaskBodyCPU<TaskAccumulate<T, OP> > {
  void operator()( ka::StealContext* sc, 
                   ka::range1d_r<T>  range, 
                   ka::pointer_cw<T> result,
                   OP op
                 )
  {
    *result = std::accumulate( range.begin(), range.end(), *result, op );
  }
};



/** Splitter for CPU implementation, call in concurrency with 
    TaskBodyCPU<TaskWork<T, OP> >
*/
template<typename T, typename OP>
struct TaskAdaptative<TaskAccumulate<T, OP> > 
{
  /* public type required */
  struct work_type {
    kaapi_workqueue_t _wq;
    kaapi_lock_t      _lock;
  };
  
  void initialize( 
    work_type* work,
    ka::range1d_r<T> range,
    OP op
  )
  {
    kaapi_atomic_initlock(&work->_lock);
    kaapi_workqueue_init_with_lock(work->_wq, 0, range.size(), &work->_lock );
  }

  void destroy( 
    work_type* work,
  )
  {
    kaapi_atomic_destroylock(&work->_lock);
  }

  
  /* splitter to decompose work in nreq parts */
  int splitter ( ka::StealContext* sc, 
                 int nreq, 
                 ka::ListRequest::iterator begin, 
                 ka::ListRequest::iterator end,
                 work_type* work,
                ) 
  {
    return work->split(sc, nreq, begin, end );
  }

  /* this method pilot the execution */
  void operator() ( ka::range1d_rw<T> input_range,
                    OP op ) 
  {
    ka::Range range;
    
    /* while there is sequential work to do*/
    while (wq->pop(range))
    {
      /* apply w->op foreach item in [range.begin, range.end): call pure DFG task ! */
      TaskBodyCPU<TaskForEach<T, OP> >()( 
          ka::array<1,T>( input_range.begin()+range.begin, 
                          input_range.begin()+range.end
                        ), 
                        op 
      );
    }
  }
  
};



/* For each main function */
template<typename T, class OP>
static T accumulate( T* beg, T* end, T result, OP op )
{
  /* range to process */
  ka::Spawn<TaskAccumulate<T,OP> >()(beg, end, &result, op);
  ka::Sync();
  return result;
}


/**
*/
double myoperator( double x, double y )
{
  return x+y;
}

/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    double t0,t1;
    double sum = 0.f;
    size_t size = 30000;
    if (argc >1) size = atoi(argv[1]);
    
    double* array = new double[size];
    double result;

    for (int iter = 0; iter < 1; ++iter)
    {
      /* initialize, apply, check */
      for (size_t i = 0; i < size; ++i)
        array[i] = 1.f;
        
      t0 = kaapi_get_elapsedns();
      result = accumulate(( array, array+size, 0, myoperator() );
      t1 = kaapi_get_elapsedns();
      sum += (t1-t0)/1000; /* ms */
      kaapi_assert( result == size );
    }

    std::cout << "Done " << sum/100 << " (ms)" << std::endl;
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}



#if 0




///---------------------------------------------------------------

///---------------------------------------------------------------

///---------------------------------------------------------------

///---------------------------------------------------------------

typedef struct work
{
  kaapi_workqueue_t cr;
  const double* array;
  double res;
} work_t;


typedef struct thief_work_t {
  const double* beg;
  const double* end;
  double res;
} thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


/* result reducer */
static int reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* victim work */
  work_t* const vw = (work_t*)varg;

  /* thief work */
  thief_work_t* const tw = (thief_work_t*)tdata;

  /* thief range continuation */
  kaapi_workqueue_index_t beg, end;

  /* accumulate */
  vw->res += tw->res;

  /* retrieve the range */
  beg = (kaapi_workqueue_index_t)(tw->beg - vw->array);
  end = (kaapi_workqueue_index_t)(tw->end - vw->array);

  kaapi_workqueue_set(&vw->cr, beg, end);

  return 0;
}


/* parallel work splitter */
static int splitter
(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* args)
{
  /* victim work */
  work_t* const vw = (work_t*)args;

  /* stolen range */
  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;

  /* reply count */
  int nrep = 0;

  /* size per request */
  kaapi_workqueue_index_t unit_size;

 redo_steal:
  /* do not steal if range size <= PAR_GRAIN */
#define CONFIG_PAR_GRAIN 128
  range_size = kaapi_workqueue_size(&vw->cr);
  if (range_size <= CONFIG_PAR_GRAIN)
    return 0;

  /* how much per req */
  unit_size = range_size / (nreq + 1);
  if (unit_size == 0)
  {
    nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
    unit_size = CONFIG_PAR_GRAIN;
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
   */
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(thief_work_t), NULL);

    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), ktr );
    tw->beg = vw->array+j-unit_size;
    tw->end = vw->array+j;
    tw->res = 0.f;

    /* initialize ktr task may be preempted before entrypoint */
    ((thief_work_t*)ktr->data)->beg = tw->beg;
    ((thief_work_t*)ktr->data)->end = tw->end;
    ((thief_work_t*)ktr->data)->res = 0.f;

    /* reply head, preempt head */
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }

  return nrep;
}


/* seq work extractor */
static int extract_seq(work_t* w, const double** pos, const double** end)
{
  int err;
  /* extract from range beginning */
  kaapi_workqueue_index_t i, j;

#define CONFIG_SEQ_GRAIN 128
  if ((err =kaapi_workqueue_pop(&w->cr, &i, &j, CONFIG_SEQ_GRAIN)) !=0) return 1;

  *pos = w->array + i;
  *end = w->array + j;

  return 0;
}


/* entrypoint */
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* input work */
  thief_work_t* const work = (thief_work_t*)args;

  /* resulting work */
  thief_work_t* const res_work = kaapi_adaptive_result_data(sc);

  /* res += [work->beg, work->end[ */
  while (work->beg != work->end)
  {
    work->res += *work->beg;

    /* update prior reducing */
    ++work->beg;

    const unsigned int is_preempted = kaapi_preemptpoint
      (sc, NULL, NULL, (void*)work, sizeof(thief_work_t), NULL);
    if (is_preempted) return ;
  }

  /* we are finished, update results. */
  res_work->beg = work->beg;
  res_work->end = work->end;
  res_work->res = work->res;
}


/* algorithm main function */
static double accumulate(const double* array, size_t size)
{
  /* self thread, task */
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;

  /* sequential work */
  const double* pos, *end;

  /* initialize work */
  work_t work;
  kaapi_workqueue_init(&work.cr, 0, (kaapi_workqueue_index_t)size);
  work.array = array;
  work.res = 0.f;

  /* push an adaptive task. set the preemption flag. */
  sc = kaapi_task_begin_adaptive(
        thread, 
        KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION, 
        splitter, 
        &work     /* arg for splitter = work to split */
    );

  /* while there is sequential work to do */
 redo_work:
  while (!extract_seq(&work, &pos, &end))
  {
    /* res += [pos, end[ */
    for (; pos != end; ++pos)
      work.res += *pos;
  }

  /* preempt and reduce thieves */
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief(sc, ktr, NULL, reducer, (void*)&work);
    goto redo_work;
  }

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);

  return work.res;
}


/* unit */

int main(int ac, char** av)
{
#define ITEM_COUNT 1000000
  static double array[ITEM_COUNT];

  /* initialize the runtime */
  kaapi_init(1, &ac, &av);

  /* initialize array */
  for (size_t i = 0; i < ITEM_COUNT; ++i)
    array[i] = 2.f;

  for (ac = 0; ac < 100; ++ac)
  {
    double res = accumulate(array, ITEM_COUNT);
    if (res != (double)(2 * ITEM_COUNT))
    {
      printf("invalid: %lf != %lf\n", res, 2.f * ITEM_COUNT);
      break ;
    }
  }

  printf("done\n");

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}

#endif