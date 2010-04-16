/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
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
#ifndef _KASTL_ACCUMULATE_H_
#define _KASTL_ACCUMULATE_H_
#include "kaapi.h"
#include "kastl/kastl_workqueue.h"
#include <algorithm>

#define TRACE_ 0
#ifndef KAAPI_MAX_PROCESSOR
#define KAAPI_MAX_PROCESSOR 16
#endif

#if TRACE_
static pthread_mutex_t lock_cout = PTHREAD_MUTEX_INITIALIZER;
void lockout() 
{
  pthread_mutex_lock( &lock_cout );
}
void unlockout()
{
  pthread_mutex_unlock( &lock_cout );
}
#endif

namespace kastl {
  
namespace impl {

/** Stucture of a work for for_each
*/
template <typename T, typename Iterator, typename Function, typename Inserter, typename Predicate>
class AccumulateWhileWork {
public:
  /* MySelf */
  typedef AccumulateWhileWork<T,Iterator,Function,Inserter,Predicate> Self_t;

  /* cstor */
  AccumulateWhileWork(
    size_t*        retval,
    T&             value,
    Iterator       ibeg,
    Iterator       iend,
    Function&      func,
    Inserter&      insert,
    Predicate&     pred,
    int ws
  ) : _queue(), 
      _returnval(retval), _value(value), 
      _ibeg(ibeg), _iend(iend), _func(func), _accf(insert), _pred(pred), 
      _inputiterator_value(0), _master(0), 
      _pargrain(1)
  { 
  }
  
  ~AccumulateWhileWork()
  {
    delete [] _inputiterator_value;
    delete [] _return_funccall;
  }
  
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef typename Function::result_type result_type;
  
  /* main loop */
  void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    bool isfinish = false;
    impl::range r;
    size_t cntevalfunc =0;
    size_t i, blocsize;
    kaapi_taskadaptive_result_t* thief;

#if TRACE_
    kaapi_uint64_t  t0, t1;
#endif
    
    /*  ---- */
    blocsize = 1*kaapi_getconcurrency();
    _pargrain = 1;

    
    /* input */
    _inputiterator_value = new value_type[blocsize];
    _return_funccall = new ResultElem_t[blocsize];
    
    /*  ---- */
    while ((_ibeg != _iend) && !isfinish)
    {
#if 0
      /* initialize the queue: concurrent operation with respect to steal */
      _queue.set( range(blocsize, blocsize) );      

      /* fill the input iterator and commit it into the queue */
      for (i = 0; (i<blocsize) && (_ibeg != _iend); ++i, ++_ibeg)
      {
        _inputiterator_value[blocsize -1 - i] = *_ibeg;
        _queue.push_front( blocsize -1 - i);
      }      
#else
      /* fill the input iterator and commit it into the queue */
      for (i = 0; (i<blocsize) && (_ibeg != _iend); ++i, ++_ibeg)
      {
        _inputiterator_value[i] = *_ibeg;
      }      

      /* initialize the queue: concurrent operation with respect to steal */
      _queue.set( range(0, i) );      
#endif

      /* do one computation */
      if (_queue.pop(r, 1))
      {
#if TRACE_
lockout();
        std::cout << "Tmaster eval:" << _inputiterator_value[r.first]
                  << " -> " << _return_funccall+r.first
                  << std::endl << std::flush;
unlockout();
#endif
        _func( _return_funccall[r.first].data, _inputiterator_value[r.first] );
        ++cntevalfunc;
        _accf( _value, _return_funccall[r.first].data );
        isfinish = !_pred(_value);
      }

      /* accumulate data from all thieves */
      _queue.clear();

      /* here thief may no steal anymore things */
      thief = kaapi_get_thief_head( sc );
      while (thief !=0)
      {

        /* accumulate data from the thief */
        range r_result;
        ThiefResult_t* thief_result = (ThiefResult_t*)thief->data;

        thief_result->return_queue.pop(r_result, thief_result->return_queue.size());

        if (!r_result.is_empty())
        { /* accumulate data */
          cntevalfunc += r_result.size();
          for (range::index_type i=r_result.first; (i<r_result.last) && !(isfinish = !_pred(_value)); ++i)
            _accf( _value, thief_result->return_funccall[i].data );
        } 
        else 
        {
          /* if thief is finish: remove it from list */
          int err __attribute__((unused)) = kaapi_remove_finishedthief( sc, thief);
        }

        /* next thief ? */
        thief = kaapi_get_nextthief_head( sc, thief );

      } // thief != 0
    } // while pas fini


    /* write the total number of parallel evaluation executed */
    *_returnval = cntevalfunc;
    
    /* send preempt to all thiefs and return */
    _queue.clear();

#if TRACE_
    t0 = kaapi_get_elapsedns();
#endif
    /* only do signal */
    thief = kaapi_get_thief_head( sc );
    while (thief !=0)
    {
      kaapi_preemptasync_thief(sc, thief, 0);
      thief = kaapi_get_nextthief_head( sc, thief );
    }


    /* remove or preempt thief */
    thief = kaapi_get_thief_head( sc );
    while (thief !=0)
    {
      int err = kaapi_remove_finishedthief( sc, thief);
      if (err == EBUSY)
        kaapi_preempt_thief(sc, thief, 0, 0, 0);
      thief = kaapi_get_nextthief_head( sc, thief );
    }
  }

  /* */
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc, count, request );
  }

protected:
  /* result of a thief which is mapped in the kaapi_taskadaptive_result_t data structure
  */
  struct ResultElem_t {
    result_type  data;
  } __attribute__((aligned(64)));

  struct ThiefResult_t {
    work_queue    return_queue;       /* should be correctly aligned ! */
    ResultElem_t* return_funccall;    /* first entry of the array of size cnteval */
  };
  
  /* Thief work
  */
  class ThiefWork_t {
  public:
    ThiefWork_t(  const range& r, 
                  Function& func,
                  kaapi_stealcontext_t* master, 
                  kaapi_taskadaptive_result_t* tr,
                  value_type* inputiterator_value, /* original base pointer */
                  ResultElem_t* return_funccall    /* original base pointer */
    )
     : _range(r), 
       _func(func), 
       _inputiterator_value(inputiterator_value),
       _master(master),
       _tr(tr),
       _thief_result( (ThiefResult_t*)tr->data)
    {
      _thief_result->return_queue.clear();
      /* shift array to begin at 0 when victim pop values */
      _thief_result->return_funccall = return_funccall+r.first;     
    }

    /* main entry point of a thief */
    void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
    {
      range::index_type first = _range.first;
      /* shift to avoid arithmetic in the main loop */
      ResultElem_t* return_funccall = _thief_result->return_funccall - first;
      while ( _range.first != _range.last )
      {
        _func( return_funccall[_range.first].data, _inputiterator_value[_range.first] ); 
        _thief_result->return_queue.push_back( range(_range.first-first, 1+_range.first-first) );

#if TRACE_
lockout();
        std::cout << "Thief " << _thief_result << " eval:" << _inputiterator_value[_range.first]
                  << " -> " << return_funccall + _range.first //].data
                  << std::endl << std::flush;
unlockout();
#endif

        ++_range.first;
        if (_range.is_empty()) break;

        /* test preemption after increment ... */
        if (kaapi_preemptpoint( 
            _tr,                   /* to test preemption */
            sc,                    /* to merge my thieves into list of the victim */
            0,                     /* function to call if preemption signal */
            0,                     /* extra data to pass to victim -> it will get the size of computed value into its reducer */ 
            0, 0,
            0
        )) 
        {
          return;
        }
      } // end while
    }

    /* thief task body */
    static void static_entrypoint( void* arg, kaapi_thread_t* thread )
    {
      /* push a steal context in order to be self stealed, else do not push any think */
      ThiefWork_t* self_work = (ThiefWork_t*)arg;
      kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
        thread,
        KAAPI_STEALCONTEXT_LINKED,
        0, //ThiefWork_t::static_splitter, /* or 0 to avoid steal on thief */
        0, //self_work,
        self_work->_master
      );
      self_work->doit( sc, thread );
      kaapi_steal_thiefreturn( sc );
      self_work->~ThiefWork_t();
    }

  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
    return 0;
  }

  /* */
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    ThiefWork_t* self_work = (ThiefWork_t*)argsplitter;
    return self_work->splitter( sc, count, request );
  }

  protected:
    friend class AccumulateWhileWork<T,Iterator,Function,Inserter,Predicate>;

    range                        _range;               /* first to ensure alignment constraint if change to queue */
    Function&                    _func;
    value_type*                  _inputiterator_value;
    kaapi_stealcontext_t*        _master;              /* for terminaison */
    kaapi_taskadaptive_result_t* _tr;                  /* for preemption */
    ThiefResult_t*               _thief_result;        /* view of tr->data */
  };

  /* splitter: split in bloc of size at most pargraim
  */
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
    size_t size = _queue.size();     /* upper bound */
    if (size < _pargrain) return 0;

    /* take at most count*_pargrain items per thief */
    size_t blocsize = _pargrain;
    size_t size_max = (blocsize * count);
    if (size_max > size) size_max = size-1;
    size_t size_min = 1;         /* min bound */
    range r;

    /* */
    if ( !_queue.steal(r, size_max, size_min )) return 0;
    kaapi_assert_debug (!r.is_empty());
    
    /* size of what the thieves have stolen */
    size = r.size();

    ThiefWork_t* output_work;
    int i = 0;
    int reply_count = 0;

    /* size of thief will get at most blocsize items per thief */    
    blocsize = size / count;

    if (blocsize == 0) 
    { /* reply to less thief... */
      count = size; 
      blocsize = 1; 
    }
#if TRACE_
    std::cout << "Splitter: count=" << count << ", r=[" << r.first << "," << r.last-1 << "]"
              << ", bloc=" << blocsize << ", size=" << size   
              << std::endl << std::flush;
#endif
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_thread_t* thief_thread = kaapi_request_getthread(&request[i]);
        kaapi_task_t* thief_task  = kaapi_thread_toptask(thief_thread);
        kaapi_task_init( thief_task, &ThiefWork_t::static_entrypoint, kaapi_thread_pushdata_align(thief_thread, sizeof(ThiefWork_t), 8) );

        kaapi_assert_debug( !r.is_empty() );
        range rq(r.first, r.last);
        if (count >1) /* adjust for last bloc */
        {
          rq.first = rq.last - blocsize;
          r.last = rq.first;
        }

        output_work = new (kaapi_task_getargst(thief_task, ThiefWork_t))
            ThiefWork_t( 
              rq, 
              _func,
              sc,
              kaapi_allocate_thief_result( sc, sizeof(ThiefResult_t) + rq.size()*sizeof(result_type), 0 ),
              _inputiterator_value,
              _return_funccall
            );
#if TRACE_
        std::cout << "Splitter: reply to=" << i << "[" << rq.first << "," 
                  << rq.last-1 << "]" 
                  << std::endl << std::flush;
#endif

        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request, push it into the tail of the list */
        kaapi_request_reply_tail( sc, &request[i], output_work->_tr );
        --count; 
        ++reply_count;
      }
      ++i;
    }
#if TRACE_
    std::cout << "Splitter: end reply" 
              << std::endl << std::flush;
#endif
    return reply_count; 
  }


protected:
  work_queue                   _queue;     /* first to ensure alignment constraint */
  size_t*                      _returnval; /* number of iteration, output of seq call */
  T&                           _value __attribute__((aligned(64)));
  Iterator                     _ibeg;
  Iterator                     _iend;
  Function&                    _func;
  Inserter&                    _accf;
  Predicate&                   _pred;
  value_type*                  _inputiterator_value __attribute__((aligned(64)));
  ResultElem_t*                _return_funccall     __attribute__((aligned(64)));
  kaapi_stealcontext_t*        _master;
  size_t                       _pargrain;
};

} /* namespace impl */


template <typename T, typename InputIterator, typename Function, typename Inserter, typename Predicate>
size_t accumulate_while( T& value, 
                         InputIterator first, InputIterator last, 
                         Function& func, 
                         Inserter& acc, 
                         Predicate& pred,
                         int windowsize = -1   /*  == default value */
                        )
{
  typedef impl::AccumulateWhileWork<T,InputIterator,Function,Inserter,Predicate> Self_t;
  size_t returnval = 0;
  
  Self_t* work
   = new (kaapi_alloca_align(64, sizeof(Self_t))) 
        Self_t(
          &returnval,
          value, 
          first, 
          last, 
          func, 
          acc, 
          pred, 
          windowsize
        );
  kaapi_assert( (((kaapi_uintptr_t)work) & 0x3F)== 0 );

  kaapi_thread_t* thread =  kaapi_self_thread();
  kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
    thread,
    KAAPI_STEALCONTEXT_DEFAULT,
    Self_t::static_splitter,
    work,
    0
  );
  
  work->doit( sc, thread );
  
  kaapi_steal_finalize( sc );
  kaapi_sched_sync();
  work->~Self_t();

  return returnval;
}

} /* namespace kastl */
#endif
