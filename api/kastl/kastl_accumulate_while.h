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

static pthread_mutex_t lock_cout = PTHREAD_MUTEX_INITIALIZER;
void lockout() 
{
  pthread_mutex_lock( &lock_cout );
}
void unlockout()
{
  pthread_mutex_unlock( &lock_cout );
}


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
      _inputiterator_value(0), _master(0), //_thief_result(0),
      _windowsize(ws), _seqgrain(1), _pargrain(1)
  { 
    for (int i=0; i<KAAPI_MAX_PROCESSOR; ++i)
      _delay[i] =0;
  }
  
  ~AccumulateWhileWork()
  {
    delete [] _inputiterator_value;
  }
  
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef typename Function::result_type result_type;
  
  /* main loop */
  void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    bool isnotfinish = true;
    long thief_nowork = 0;
    long thief_remainwork = 0;
    impl::range r;
    size_t iter;
    size_t i, sz_used, blocsize;
    if (_windowsize == (size_t)-1) 
    {
      blocsize  = 256; //4*kaapi_getconcurrency();
      _pargrain = 4;
    }
    else 
    {
      blocsize = _windowsize;
      _pargrain= blocsize/kaapi_getconcurrency();
      if (_pargrain ==0) _pargrain = 1;
    }
    kaapi_taskadaptive_result_t* thief;
    size_t last_size = blocsize;
    typename Function::result_type return_funccall;
    _inputiterator_value = new value_type[blocsize];

    while ((_ibeg != _iend) && (isnotfinish=_pred(_value)))
    {
#if 0 /* DO NOT CHANGE BLOC SIZE */
      blocsize += _pargrain*thief_nowork-thief_remainwork;

      if (blocsize> last_size) 
      {
        value_type* tmp = _inputiterator_value;
        _inputiterator_value=0;
        delete [] tmp;
        last_size = (2*last_size < (blocsize) ? blocsize: 2*last_size);
        _inputiterator_value = new value_type[last_size];
      }
      /* generate input values into a container with randomiterator */
      size_t pargrain= (blocsize-1)/(kaapi_getconcurrency()-1);
      if (pargrain ==0) _pargrain = 1;
      else {
        if (_pargrain != pargrain) std::cout << "****** NEW PARGRAIN=" << pargrain << std::endl;
        _pargrain = pargrain;
      }
#endif
      for (i = 0; (i<blocsize) && (_ibeg != _iend); ++i, ++_ibeg)
        _inputiterator_value[i] = *_ibeg;
      sz_used = i;

      /* initialize the queue: concurrent operation */
      _cntthief = 0;
{ 
  lockout();      
  std::cout << "New local #work:" << sz_used << " = [" << _inputiterator_value[0] << "," << _inputiterator_value[sz_used-1] << "]" << std::endl << std::flush;      
  unlockout();
}
      kaapi_uint64_t t0 = kaapi_get_elapsedns();
      _queue.set( range(0, sz_used) ); 
      
redo_with_remainding_work:
      iter = 0;
      //while (_queue.pop(r, 1))
      if (_queue.pop(r, 1))
      {
#if TRACE_
{ 
  lockout();      
  std::cout << "Master eval r=[" << _inputiterator_value[r.first] << "," << _inputiterator_value[r.last-1] << "]"
            << std::endl << std::flush;
  unlockout();
}
#endif
        for (; r.first != r.last; ++r.first)
        {
//        std::cout << "Victim eval=" << _inputiterator_value[r.first] << std::endl;
          _func( return_funccall, _inputiterator_value[r.first] );
          ++iter;
          _accf( _value, return_funccall );
          if (!(isnotfinish=_pred(_value))) goto continue_because_predicate_is_false;
        }
      }
continue_because_predicate_is_false:
      *_returnval += iter;
      kaapi_uint64_t t1 = kaapi_get_elapsedns();

//      kaapi_assert_debug( !isnotfinish || _queue.is_empty() );
      long cntthief = _cntthief;
      kaapi_uint64_t delay_thieves = 0;
      for (int i=0; i<KAAPI_MAX_PROCESSOR; ++i)
      {
        if (delay_thieves<_delay[i])
          delay_thieves = _delay[i];
        _delay[i] = 0;
      }
{ 
  lockout();
      std::cout << "End local work, #thief=" << cntthief 
                << ", Max delay (s): " << 1e-9 * delay_thieves
                << ", Master delay (s): " << 1e-9 * double(t1-t0)
                << ", Ratio: " << delay_thieves / double(t1-t0)
                << ", size queue: " << _queue.size()
                << std::endl;
  unlockout();
}
#if 0
      if ((isnotfinish) && (thief !=0) && _queue.is_empty())
      {
        /* generate extra work for thief...I'm preempted... */
        _inputiterator_value[0] = *_ibeg;
        sz_used = 1;
        /* initialize the queue: concurrent operation, a push will be better */
        _queue.set( range(0, sz_used) ); 
      }
#endif
      /* preempt thieves */
      t0 = kaapi_get_elapsedns();
      _queue.clear();
      /* here thief may no steal anymore things */
      thief_nowork = 0;
      thief_remainwork = 0;
      thief = kaapi_preempt_getnextthief_head( sc );
      while (thief !=0)
      {
#if TRACE_
{ 
  lockout();      
  std::cout << kaapi_get_elapsedns() << "::Master preempt Thief:" << thief << std::endl << std::flush;
  unlockout();
}
#endif
        if (kaapi_preempt_thief ( 
            sc, 
            thief,                       /* thief to preempt */
            0,                           /* arg for the thief */
            Self_t::static_mainreducer,  /* my reducer */
            this                         /* extra arg for the reducer */
        ))
        {
          ++thief_remainwork;
#if TRACE_
{ 
  lockout();      
          std::cout << "Remains work the thief:" << thief << std::endl << std::flush;
  unlockout();
}
#endif
        }
        else 
        {
          ++thief_nowork;
#if TRACE_
{ 
  lockout();      
          std::cout << "No work from thief:" << thief << std::endl << std::flush;
  unlockout();
}
#endif
        }
        isnotfinish = _pred(_value);
        
        /* next thief ? */
        thief = kaapi_preempt_getnextthief_head( sc );
      }
      t1 = kaapi_get_elapsedns();

#if 1//TRACE_
{ 
  lockout();      
  std::cout << "Tmaster preempt   :" << double(t1-t0)*1e-9 
            << ", #thiefs end work:" << thief_nowork 
            << ", #work remaining :" << thief_remainwork << std::endl << std::flush;
  unlockout();
}
#endif
      
      if (!_queue.is_empty()) {
        _cntthief = 0;
        lockout();
        std::cout << "Master redo with remainding work" << std::endl << std::flush;
        unlockout();
        goto redo_with_remainding_work;
      }
    }
  }

  /* */
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc, count, request );
  }

protected:
  /* result of a thief: Function::result_type[N]
  */
  
  /* Thief work
  */
  class ThiefWork_t {
  public:
    ThiefWork_t(  range r, Function& func,
                  kaapi_stealcontext_t* master, kaapi_taskadaptive_result_t* tr,
                  size_t windowsize, size_t seqgrain, size_t  pargrain
    )
     : _queue(), _func(func), _master(master), _thief_result(tr),
       _windowsize(windowsize), _seqgrain(seqgrain), _pargrain(pargrain)
    {
      _cnteval             = (long*)_thief_result->data;
      _size                = _cnteval+1;
      _return_funccall     = (result_type*)(_size+1);
      /* next field is properly initialized in splitter */
      _inputiterator_value = (value_type*)(_return_funccall+r.size());
      *_size = r.size();
      *_cnteval = -1; /* for debug */

      /* set queue with initial range to [0, r.last-r.first] because of dedicated result_buffer and inputiterator */
      r.last = r.size();
      r.first = 0;
      _queue.set( r );
    }

    /* main entry point of a thief */
    void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
    {
      range r;
      long iter = 0;
      new (_return_funccall) result_type[*_size];
      
#if TRACE_
{ 
  lockout();      
  std::cout << kaapi_get_elapsedns() << "::Thief " << _thief_result << " eval #size=" << _queue.size()
            << std::endl << std::flush;
  unlockout();
}
#endif
      /* do work */
      while (_queue.pop(r, 2))
      {
        while ( r.first != r.last )
        {
#if TRACE_
          lockout();
          std::cout << kaapi_get_elapsedns() << "::Thief " << _thief_result << " eval:" << r.first << std::endl;
          unlockout();
#endif
          _func( _return_funccall[r.first], _inputiterator_value[r.first] ); 
          ++iter; ++r.first;
          *_cnteval = iter;
          if (r.first == r.last) break;
          int retval = kaapi_preemptpoint( 
              _thief_result,         /* to test preemption */
              sc,                    /* to merge my thieves into list of the victim */
              0,                     /* function to call if preemption signal */
              0,                     /* extra data to pass to victim -> it will get the size of computed value into its reducer */ 
              0, 0,
              0
          );
#if TRACE_
{ 
  lockout();  
  std::cout << kaapi_get_elapsedns() << "::Thief " << _thief_result << " flag preempt:" << _thief_result->req_preempt
            << ", retval preemptpoint: " << retval
            << std::endl << std::flush;
  unlockout();
}
#endif
          if (retval) {
#if TRACE_
{ 
  lockout();      
  std::cout << "Thief " << _thief_result << " preempted at " << r.first 
            << std::endl << std::flush;
  unlockout();
}
#endif
            return;
          }
        }
      }
#if TRACE_
{ 
  lockout();      
  std::cout << kaapi_get_elapsedns() << "::Thief " << _thief_result << " end eval work" 
            << std::endl << std::flush;
  unlockout();
}
#endif
    }

    /* thief task body */
    static void static_entrypoint( void* arg, kaapi_thread_t* thread )
    {
      /* push a steal context in order to be self stealed, else do not push any think */
      ThiefWork_t* self_work = (ThiefWork_t*)arg;
#if TRACE_
      long* cnt = self_work->_cnteval;
#endif
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
#if TRACE_
    std::cout << "Thief " << self_work->_thief_result << " return: " << *cnt << " evaluation(s)" 
              << std::endl << std::flush;
#endif
    }

  protected:
    friend class AccumulateWhileWork<T,Iterator,Function,Inserter,Predicate>;

    int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
    {
      exit(0);
    }
    
    /* */
    static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
    {
      ThiefWork_t* self_work = (ThiefWork_t*)argsplitter;
      return self_work->splitter( sc, count, request );
    }

    work_queue                   _queue;    /* first to ensure alignment constraint */
    Function&                    _func;
    long*                        _cnteval;                             
    long*                        _size;                             
    result_type*                 _return_funccall;
    value_type*                  _inputiterator_value;
    kaapi_stealcontext_t*        _master;       /* for terminaison */
    kaapi_taskadaptive_result_t* _thief_result; /* !=0 only on thief */
    size_t                       _windowsize;
    size_t                       _seqgrain;
    size_t                       _pargrain;
  };

  /* splitter: split in count+1 parts the remainding work
  */
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
    int cntthieves =count;
    /* get some state about the thieves. no concurrency on += */
    _cntthief += cntthieves;
    for (int i=0; cntthieves>0; ++i)
      if (kaapi_request_ok(&request[i]))
      {
        --cntthieves;
#if 0
  lockout();      
        std::cout << "Delay thief[" << i << "]= " << request[i].delay << std::endl<< std::flush;
  unlockout();      
#endif
        if (_delay[i] < request[i].delay) _delay[i] = request[i].delay;
      }
    
    size_t size = _queue.size();     /* upper bound */
    if (size < _pargrain) return 0;

    /* take at most count*_pargrain items per thief */
    size_t blocsize = _pargrain;
    size_t size_max = (blocsize * count);
    if (size_max > size) size_max = size-1;
    size_t size_min = (size_max * 2) / 3;         /* min bound */
    if (size_min ==0) size_min = 1;
    range r;

    /* */
    if ( !_queue.steal(r, size_max, size_min )) return 0;
#if 1//TRACE_
{ 
  lockout();      
    std::cout << "Splitter: count=" << count << ", r=[" << r.first << "," << r.last-1 << "], size=" << size
              << ", steal size=" << r.size()  << ", size_max=" << size_max << ", size_min=" << size_min 
              << std::endl << std::flush;
  unlockout();      
} 
#elif 0
    std::cout << "Splitter: count=" << count 
              << ", input size=" << size << ", get : [" << r.first << "=" << _inputiterator_value[r.first] 
              << "," << r.last-1 << "=" << _inputiterator_value[r.last-1] << "], size_max=" << size_max << ", size_min=" << size_min 
              << std::endl << std::flush;
#endif
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

#if 0
if (r.is_empty())
  std::cout << "**** EMPTY" << r.first << ", " << r.last-1 << "]" << std::endl << std::flush;
#endif
        kaapi_assert_debug( !r.is_empty() );
        range rq(r.first, r.last);
        if (count >1) /* adjust for last bloc */
        {
          rq.first = rq.last - blocsize;
          r.last = rq.first;
        }

#if 0
std::cout << "Replyi[" << count << "] r=" << rq.first << ", " << rq.last-1 << "]" << std::endl << std::flush;
if (r.is_empty())
{
  std::cout << "**** EMPTY" << rq.first << ", " << rq.last -1<< "]" << std::endl << std::flush;
}
#endif
        output_work = new (kaapi_task_getargst(thief_task, ThiefWork_t))
            ThiefWork_t( 
              rq, 
              _func,
              sc,
              kaapi_allocate_thief_result( sc, 2*sizeof(long) + rq.size()*(sizeof(value_type)+sizeof(result_type)), 0 ),
              _windowsize,
              _seqgrain,
              _pargrain
            );
#if TRACE_
        std::cout << "Splitter: reply to=" << i << "[" << rq.first << "," 
                  << rq.last-1 << "]" 
                  << std::endl << std::flush;
#endif
        /* copy the input value for the task */
        int k;
        for ( k=0; rq.first != rq.last; ++rq.first, ++k )
        {
          new (&output_work->_inputiterator_value[k]) value_type(_inputiterator_value[rq.first]);
#if 0
          std::cout << "Splitter: recopy p=" << _inputiterator_value[rq.first] 
              << ", to=" << output_work->_inputiterator_value[k]
              << std::endl << std::flush;
#endif
        }
        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request, push it into the tail of the list */
        kaapi_request_reply_tail( sc, &request[i], output_work->_thief_result );
        --count; 
        ++reply_count;
      }
      ++i;
    }
#if TRACE_
std::cout << "Splitter: end reply" << std::endl;
    std::cout << std::flush;
#endif
    return reply_count; 
  }

  /**/
  int main_reduce(
      kaapi_stealcontext_t* sc, 
      void*                 thief_data,
      long*                 thief_remainwork
  )
  {
    long i, cnteval;
    long* pcnteval                   = (long*)thief_data;
    long* size                       = pcnteval+1;
    result_type* return_funccall     = (result_type*)(size+1);
    /*value_type*  inputiterator_value = (value_type*)(return_funccall+*size);*/
    
    cnteval = *pcnteval;
#if TRACE_
    std::cout << "Victim reduce thief, #eval=" << cnteval << ", #size=" << *size << ", value before:" << _value << std::endl << std::flush;
#endif
    for (i=0; (i<cnteval) && _pred(_value); ++i)
      _accf( _value, return_funccall[i] );

    if ((i==cnteval) && (cnteval < *size))
    {
      std::cout << "Remain work to do #=" << *size - cnteval << std::endl << std::flush;
    } 
#if TRACE_
    {
      std::cout << "Thief has computed no usefull evaluation:" << cnteval-i << std::endl << std::flush;
    }
#endif
    *_returnval += cnteval;
#if TRACE_
    std::cout << "Victim after reduce thief, #eval=" << *_returnval << ", value:" << _value << std::endl;
#endif
    *thief_remainwork += *size-cnteval;
    return (*size != cnteval);
  }
  
  /**/
  static int static_mainreducer( 
      kaapi_stealcontext_t* sc, 
      void*                 thief_arg, 
      void*                 thiefdata, 
      size_t                thiefsize,
      Self_t*               myself,
      long*                 thief_remainwork 
  )
  {
    return myself->main_reduce( sc, thiefdata, thief_remainwork );
  }

protected:
  work_queue                   _queue;     /* first to ensure alignment constraint */
  kaapi_uint64_t               _delay[KAAPI_MAX_PROCESSOR];
  size_t*                      _returnval; /* number of iteration, output of seq call */
  T&                           _value;
  Iterator                     _ibeg;
  Iterator                     _iend;
  Function&                    _func;
  Inserter&                    _accf;
  Predicate&                   _pred;
  value_type*                  _inputiterator_value;
  kaapi_stealcontext_t*        _master;
  long                         _cntthief;
  size_t                       _windowsize;
  size_t                       _seqgrain;
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
