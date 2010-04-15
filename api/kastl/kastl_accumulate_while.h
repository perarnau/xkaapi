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
    bool isfinish = false;
#if TRACE_
    long thief_nowork = 0;
    long thief_remainwork = 0;
#endif
    impl::range r;
    size_t cntevalfunc =0;
    size_t i, sz_used, blocsize;
    kaapi_taskadaptive_result_t* thief;
    size_t last_blocsize;
    kaapi_uint64_t  t0, t1;
    
    /*  ---- */
    blocsize = 3*kaapi_getconcurrency();
    _pargrain = 3;

    last_blocsize = blocsize;
    
    /* input */
    _inputiterator_value = new value_type[blocsize];
    _return_funccall = new ResultElem_t[blocsize];
    
    /*  ---- */
    while ((_ibeg != _iend) && !isfinish)
    {
#if 0 /* DO NOT CHANGE BLOC SIZE */
      {
        blocsize += _pargrain*thief_nowork-thief_remainwork;

        if (blocsize> last_blocsize) 
        {
          value_type* tmp = _inputiterator_value;
          _inputiterator_value=0;
          delete [] tmp;
          last_blocsize = (2*last_blocsize < (blocsize) ? blocsize: 2*last_blocsize);
          _inputiterator_value = new value_type[last_blocsize];
        }
        /* generate input values into a container with randomiterator */
        size_t pargrain= (blocsize-1)/(kaapi_getconcurrency()-1);
        if (pargrain ==0) _pargrain = 1;
        else {
          if (_pargrain != pargrain) std::cout << "****** NEW PARGRAIN=" << pargrain << std::endl;
          _pargrain = pargrain;
        }
      }
#endif
      /* fill the input iterator */
      for (i = 0; (i<blocsize) && (_ibeg != _iend); ++i, ++_ibeg)
        _inputiterator_value[i] = *_ibeg;
      sz_used = i;

      /* initialize the queue: concurrent operation */
#if TRACE_
      {
        lockout();      
        std::cout << "New local #work:" << sz_used << " = [" << _inputiterator_value[0] << "," << _inputiterator_value[sz_used-1] << "]" << std::endl << std::flush;      
        unlockout();
      }
#endif
      _queue.set( range(0, sz_used) ); 
      

#if TRACE_
      { 
        lockout();      
        std::cout << "Master eval r=[" << _inputiterator_value[r.first] << "," << _inputiterator_value[r.last-1] << "]"
                  << std::endl << std::flush;
        unlockout();
      }
#endif
      /* do one computation */
      if (_queue.pop(r, 1))
      {
        _func( _return_funccall[r.first].data, _inputiterator_value[r.first] );
        ++cntevalfunc;
        _accf( _value, _return_funccall[r.first].data );
        isfinish = !_pred(_value);
      }


#if 0 /* display delay max of thief and infos */
      t1 = kaapi_get_elapsedns();
      {
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
#if TRACE_
              lockout();
#endif
                  std::cout << "End local work, #thief=" << cntthief 
                            << ", Max delay (s): " << 1e-9 * delay_thieves
                            << ", Master delay (s): " << 1e-9 * double(t1-t0)
                            << ", Ratio: " << delay_thieves / double(t1-t0)
                            << ", size queue: " << _queue.size()
                            << std::endl;
#if TRACE_
              unlockout();
#endif
            }
      }

      t0 = kaapi_get_elapsedns();
#endif


      /* accumulate data from all thieves */
      _queue.clear();

      /* here thief may no steal anymore things */
      thief = kaapi_get_thief_head( sc );
      while (thief !=0)
      {
#if TRACE_
        { 
          lockout();      
          std::cout << kaapi_get_elapsedns() << "::Master preempt Thief:" << thief << std::endl << std::flush;
          unlockout();
        }
#endif
        /* accumulate data from the thief */
        range r_result;
        ThiefResult_t* thief_result = (ThiefResult_t*)thief->data;

        thief_result->return_queue.pop(r_result, thief_result->return_queue.size());

        if (!r_result.is_empty())
        { /* accumulate data */
#if TRACE_
          lockout();      
          std::cout << "Tmaster pop " << thief << " #result=:" << r_result.size()
                    << std::endl << std::flush;
          unlockout();
#endif
          cntevalfunc += r_result.size();
          for (range::index_type i=r_result.first; (i<r_result.last) && !(isfinish = !_pred(_value)); ++i)
            _accf( _value, thief_result->return_funccall[i].data );
        } 
        else 
        {
          /* if thief is finish: remove it from list */
          int err __attribute__((unused)) = kaapi_remove_finishedthief( sc, thief);
#if TRACE_
          if (err == EBUSY)
          {
            lockout();      
            std::cout << "Tmaster cannot remove thief, it is busy"
                      << std::endl << std::flush;
            unlockout();
          }
          else if (err ==0)
          {
            lockout();      
            std::cout << "Tmaster remove finished thief"
                      << std::endl << std::flush;
            unlockout();
          }
          else
          {
            lockout();      
            std::cout << "ABORT"
                      << std::endl << std::flush;
            unlockout();
            exit(0);
          }
#endif
        }

        /* next thief ? */
        thief = kaapi_get_nextthief_head( sc, thief );
      } // thief != 0
      t1 = kaapi_get_elapsedns();

#if TRACE_
      { 
        lockout();      
        std::cout << "Tmaster reduce   :" << double(t1-t0)*1e-9 
                  << ", #thiefs end work:" << thief_nowork 
                  << ", #work remaining :" << thief_remainwork << std::endl << std::flush;
        unlockout();
      }
#endif      
    }
#if TRACE_
    { 
      lockout();      
      std::cout << "Tmaster reduce   :" << double(t1-t0)*1e-9 
                << ", #thiefs end work:" << thief_nowork 
                << ", #work remaining :" << thief_remainwork << std::endl << std::flush;
      unlockout();
    }
#endif      
    *_returnval = cntevalfunc;
    
    /* send preempt to all thiefs and wait terminaison */
    _queue.clear();
    t0 = kaapi_get_elapsedns();
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
#if TRACE_
    { 
      lockout();      
      std::cout << "Tmaster should have unlinked:" << thief
                << std::endl << std::flush;
      unlockout();
    }
#endif      
      thief = kaapi_get_nextthief_head( sc, thief );
    }
    t1 = kaapi_get_elapsedns();

#if TRACE_
    { 
      lockout();      
      std::cout << "Tmaster reduce   :" << double(t1-t0)*1e-9 
                << ", #thiefs end work:" << thief_nowork 
                << ", #work remaining :" << thief_remainwork << std::endl << std::flush;
      unlockout();
    }
#endif      
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
#if TRACE_
        lockout();
        std::cout << kaapi_get_elapsedns() << "::Thief " << _tr << " eval:" << _range.first << std::endl;
        unlockout();
#endif
        _func( return_funccall[_range.first].data, _inputiterator_value[_range.first] ); 
        _thief_result->return_queue.push_back( range(_range.first-first, 1+_range.first-first) );
#if TRACE_
        lockout();
        std::cout << kaapi_get_elapsedns() << "::Thief " << _tr << " new value pushed:" << _range.first << std::endl;
        unlockout();
#endif
        ++_range.first;
        if (_range.is_empty()) break;

        /* test preemption after increment ... */
        int retval = kaapi_preemptpoint( 
            _tr,                   /* to test preemption */
            sc,                    /* to merge my thieves into list of the victim */
            0,                     /* function to call if preemption signal */
            0,                     /* extra data to pass to victim -> it will get the size of computed value into its reducer */ 
            0, 0,
            0
        );
        if (retval) 
        {
#if TRACE_
          lockout();      
          std::cout << "Thief " << _tr << " preempted at " << _range.first << std::endl << std::flush;
          unlockout();
#endif
          return;
        }
      } // end while
#if TRACE_
      lockout();      
      std::cout << kaapi_get_elapsedns() << "::Thief " << _tr << " end eval work, queue size:" << _thief_result->return_queue.size() 
                << std::endl << std::flush;
      unlockout();
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
    std::cout << "Thief " << self_work->_tr << " return: " << *cnt << " evaluation(s)" 
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

    range                        _range;               /* first to ensure alignment constraint if change to queue */
    Function&                    _func;
    value_type*                  _inputiterator_value;
    kaapi_stealcontext_t*        _master;              /* for terminaison */
    kaapi_taskadaptive_result_t* _tr;                  /* for preemption */
    ThiefResult_t*               _thief_result;        /* view of tr->data */
  };

  /* splitter: split in count+1 parts the remainding work
  */
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
#if 0
    int cntthieves =count;
    /* get some state about the thieves. no concurrency on += */
    _cntthief += cntthieves;
    for (int i=0; cntthieves>0; ++i)
      if (kaapi_request_ok(&request[i]))
      {
        --cntthieves;
  lockout();      
        std::cout << "Delay thief[" << i << "]= " << request[i].delay << std::endl<< std::flush;
  unlockout();      
        if (_delay[i] < request[i].delay) _delay[i] = request[i].delay;
      }
      else
        _delay[i] = 0;
#endif
    
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
#if TRACE_
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

#if TRACE_
std::cout << "Splitter reply [" << count << "] r=" << rq.first << ", " << rq.last-1 << "]" << std::endl << std::flush;
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
std::cout << "Splitter: end reply" << std::endl;
    std::cout << std::flush;
#endif
    return reply_count; 
  }


#if 0 // reduce: not used
{
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
      _accf( _value, return_funccall[i].data );

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
}
#endif

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
  ResultElem_t*                _return_funccall;
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
