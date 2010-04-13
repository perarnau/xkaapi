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
#ifndef _KASTL_FOREACH_H
#define _KASTL_FOREACH_H
#include "kaapi.h"
#include "kastl/kastl_workqueue.h"
#include <algorithm>


namespace kastl {
  
template<class InputIterator, class UnaryOperator>
void for_each( InputIterator begin, InputIterator end, UnaryOperator op );

namespace impl {

/** Stucture of a work for for_each
*/
template<class InputIterator, class UnaryOperator>
class ForeachWork {
public:
  /* MySelf */
  typedef ForeachWork<InputIterator, UnaryOperator> Self_t;

  /* cstor */
  ForeachWork(
    InputIterator ibeg,
    InputIterator iend,
    UnaryOperator& op,
    int sg = 4096,
    int pg = 512
  ) : _queue(), _ibeg(ibeg), _iend(iend), _op(op), _seqgrain(sg), _pargrain(pg)
  { 
    _queue.set( range(0, iend-ibeg) ); 
  }
  
  typedef typename std::iterator_traits<InputIterator>::value_type value_type;

  /* main loop */
  void doit( kaapi_stealcontext_t* sc, kaapi_thread_t* thread )
  {
    /* local iterator for the nano loop */
    impl::range r;

    while (_queue.pop(r, _seqgrain))
    {
#if 1
      std::for_each( _ibeg + r.first, _ibeg + r.last, _op );
#elif 0
      InputIterator pos = _ibeg + r.first;
      InputIterator end = _ibeg + r.last;
      for ( ; pos != end; ++pos)
        _op(*pos);
#else
      impl::range::index_type i;
      for ( i=r.first; i<r.last; ++i)
        _op(_ibeg[i]);
#endif
    }
  }

  /* */
  static int static_splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request, void* argsplitter )
  {
    Self_t* self_work = (Self_t*)argsplitter;
    return self_work->splitter( sc, count, request );
  }

protected:
  /* thief task body */
  static void static_thiefentrypoint( void* arg, kaapi_thread_t* thread )
  {
    Self_t* self_work = (Self_t*)arg;
    kaapi_stealcontext_t* sc = kaapi_thread_pushstealcontext( 
      thread,
      KAAPI_STEALCONTEXT_LINKED,
      Self_t::static_splitter,   /* or 0 to avoid steal on thief */
      self_work,
      self_work->_master
    );
    self_work->doit( sc, thread );
    kaapi_steal_thiefreturn( sc );
  }


  /* splitter: split in count+1 parts the remainding work
  */
  int splitter( kaapi_stealcontext_t* sc, int count, kaapi_request_t* request )
  {
#if 0
int incount = count;
#endif
    size_t size = _queue.size();   /* upper bound */
    if (size < _pargrain) return 0;

/* bug here: cannot steal work if 2 processors !!!! */
    size_t size_max = (size * count) / (2+count); /* max bound */
//    size_t size_max = size;
//    size_t size_max = count*_seqgrain;
    if (size_max ==0) size_max = 1;
    size_t size_min = (size_max * 2) / 3;         /* min bound */
    if (size_min ==0) size_min = 1;
    range r;

    /* */
    if ( (size_min < _seqgrain) || !_queue.steal(r, size_max, size_min )) return 0;
#if 0
std::cout << "Splitter: count=" << count << ", r=[" << r.first << "," << r.last << ")"
          << ", size=" << size  << ", size_max=" << size_max << ", size_min=" << size_min 
          << std::endl;
#endif
    kaapi_assert_debug (!r.is_empty());
    size = r.size();
    
    Self_t* output_work;
    int i = 0;
    int reply_count = 0;
    /* size of each bloc of each thief */    
    size_t bloc = size / count;

    if (bloc < _seqgrain) 
    { /* reply to less thief... */
      count = size/_seqgrain; 
      bloc = _seqgrain; 
    }
#if 0
std::cout << "Splitter: count=" << incount << ", outputcount=" << count << ", r=[" << r.first << "," << r.last << ")"
          << ", bloc=" << bloc << ", size=" << size   
          << std::endl;
#endif
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_thread_t* thief_thread = kaapi_request_getthread(&request[i]);
        kaapi_task_t* thief_task  = kaapi_thread_toptask(thief_thread);
        kaapi_task_init( thief_task, &static_thiefentrypoint, kaapi_thread_pushdata_align(thief_thread, sizeof(Self_t), 8) );
        output_work = new (kaapi_task_getargst(thief_task, Self_t)) 
               Self_t( _ibeg, _iend, _op, _seqgrain, _pargrain );

        kaapi_assert_debug( !r.is_empty() );
        range rq(r.first, r.last);
        if (count ==1)
          output_work->_queue.set( rq );
        else 
        {
          rq.first = rq.last - bloc;
          r.last = rq.first;
          output_work->_queue.set( rq );
        }
        output_work->_master   = sc; /* fonction ici ? */
#if 0
std::cout << "Splitter: reply to=" << i << "[" << rq.first << "," << rq.last << ")" << std::endl;
#endif
        kaapi_thread_pushtask( thief_thread );

        /* reply ok (1) to the request */
        kaapi_request_reply_head( sc, &request[i], 0 );
        --count; 
        ++reply_count;
      }
      ++i;
    }
#if 0
std::cout << std::flush;
#endif
    return reply_count; 
  }

protected:  
  work_queue                   _queue;    /* first to ensure alignment constraint */
  InputIterator                _ibeg;
  InputIterator                _iend;
  UnaryOperator                _op;
  kaapi_stealcontext_t*        _master;
  size_t                       _seqgrain;
  size_t                       _pargrain;
};

} /* namespace impl */


/**
*/
template<class InputIterator, class UnaryOperator>
void for_each ( InputIterator begin, InputIterator end, UnaryOperator op )
{
  typedef impl::ForeachWork<InputIterator, UnaryOperator> Self_t;
  
  Self_t* work = new (kaapi_alloca_align(64, sizeof(Self_t))) Self_t(begin, end, op);
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
}


/**
*/
template<class InputIterator, class UnaryOperator>
void for_each ( InputIterator begin, InputIterator end, UnaryOperator op, int seqgrain, int pargrain = 0 )
{
  typedef impl::ForeachWork<InputIterator, UnaryOperator> Self_t;
  
  Self_t* work = new (kaapi_alloca_align(64, sizeof(Self_t))) Self_t(begin, end, op, seqgrain, pargrain);
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
}


} /* namespace kastl */
#endif
