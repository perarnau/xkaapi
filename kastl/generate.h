/*
 *  test_generate.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_GENERATE_H
#define _CKAAPI_GENERATE_H
#include "kaapi_adapt.h"
#include <algorithm>


template<class RandomAccessIterator, class Generator>
void generate ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end, 
            Generator gen);

/** Stucture of a work for generate
*/
template<class RandomAccessIterator, class Generator>
class GenerateStruct {
public:
  /* cstor */
  GenerateStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Generator gen
  ) : _sc(sc), _ibeg(ibeg), _iend(iend), _gen(gen) 
  {}
  
  /* do generate */
  void doit();

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;

protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Generator _gen;
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    GenerateStruct<RandomAccessIterator, Generator>* w = (GenerateStruct<RandomAccessIterator, 
              Generator>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, 
                        Generator gen
                      )
  {
    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    GenerateStruct<RandomAccessIterator, Generator>* output_work =0;

    /* threshold should be defined (...) */
    if (size < 512) goto reply_failed;
    
    bloc = size / (1+count);
    if (bloc < 128) { count = size/128 -1; bloc = 128; }
    while (count >0)
    {
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext, 
                                              request[i], 
                                              (void**)&output_work, 
                                              sizeof(GenerateStruct<RandomAccessIterator, Generator>) 
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          local_end -= bloc;
          output_work->_gen = gen;
          kaapi_assert( output_work->_iend - output_work->_ibeg >0);

          /* reply ok (1) to the request */
          kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
        }
        else {
          /* reply failed (=last 0 in parameter) to the request */
          kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        }
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  iend  = local_end;
  kaapi_assert( iend - ibeg >0);
  return;
      
reply_failed:
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count; 
      }
      ++i;
    }
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, GenerateStruct<RandomAccessIterator, Generator>* victim_work )
  {
    GenerateStruct<RandomAccessIterator, Generator>* thief_work = 
      (GenerateStruct<RandomAccessIterator, Generator>* )thief_data;
    if (thief_work->_ibeg != thief_work->_iend)
    {
#if defined(SEQ_SUBTRANSFORM)
      std::generate(thief_work->_ibeg, thief_work->_iend, thief_work->_gen);
#else
      GenerateStruct<RandomAccessIterator, Generator> 
        work( sc, 
              thief_work->_ibeg, 
              thief_work->_iend, 
              thief_work->_gen
            );
      work.doit();
#endif
    }
  }
};


/** Adaptive generate
*/
template<class RandomAccessIterator, class Generator>
void GenerateStruct<RandomAccessIterator, Generator>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  int unit_size = 1024;

  while (_iend != _ibeg)
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _gen);

    if (unit_size > _iend-_ibeg) unit_size = _iend-_ibeg;
    nano_iend = _ibeg + unit_size;
    
    /* sequential computation */
     std::generate(_ibeg, nano_iend, _gen);
     _ibeg +=unit_size;

    if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their values which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class Generator>
void generate(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, RandomAccessIterator end,
              Generator gen)
{
  kaapi_steal_context_initpush( stealcontext );
  GenerateStruct<RandomAccessIterator, Generator> work( stealcontext, begin, end, gen);
  work.doit();

}
#endif
