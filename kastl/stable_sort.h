/*
 *  test_stable_sort.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_STABLE_SORT_H
#define _CKAAPI_STABLE_SORT_H
#include "kaapi_adapt.h"
#include <algorithm>
#include<numeric>
#include<functional>

typedef enum { SORTING, MERGE_FINISHED, MERGE_NOT_FINISHED} SORT_STATE;

template<class RandomAccessIterator, class Compare>
void  stable_sort ( kaapi_steal_context_t* stealcontext, 
                    RandomAccessIterator begin, RandomAccessIterator end, Compare comp);

template<class RandomAccessIterator>
void  stable_sort ( kaapi_steal_context_t* stealcontext, 
                    RandomAccessIterator begin, RandomAccessIterator end);

/** Stucture of a work for stable_sort
*/
template<class RandomAccessIterator, class Compare>
class StableSortStruct {
public:
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef val_t* ptr_type;
  /* cstor */
  StableSortStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    ptr_type buf,
    Compare comp) : _sc(sc), _ibeg(ibeg), _init_beg(ibeg), _iend(iend), _buf(buf), _comp(comp)
  {}
  
  /* do stable_sort */
  void doit();

  /* do sort_loop */
  void doit_sort_loop(RandomAccessIterator ibeg, RandomAccessIterator iend, ptr_type buf,
                ptrdiff_t step_size, Compare comp);

 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator _init_beg;
  RandomAccessIterator  _iend;
  ptr_type _buf;
  Compare _comp;
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    StableSortStruct<RandomAccessIterator, Compare>* w = (StableSortStruct<RandomAccessIterator, Compare>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, ptr_type buf, Compare comp
                      )
  {

    int i = 0;

    size_t bloc, size = (iend - ibeg);
    RandomAccessIterator local_end = iend;

    StableSortStruct<RandomAccessIterator, Compare>* output_work =0;

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
                                              sizeof(StableSortStruct<RandomAccessIterator, Compare>)
                                            ) ==0)
        {
          output_work->_iend = local_end;
          output_work->_ibeg = local_end-bloc;
          output_work->_init_beg = output_work->_ibeg;
          local_end -= bloc;
          output_work->_buf = buf + (output_work->_ibeg - ibeg);
          //std::cout << "output_work->_ibeg - ibeg=" << output_work->_ibeg - ibeg << std::endl;
          std::cout << "output_work->_buf-buf=" << output_work->_buf-buf << std::endl;
          output_work->_comp = comp;
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
static void reducer( kaapi_steal_context_t* sc, void* thief_data,
                     StableSortStruct<RandomAccessIterator, Compare>* victim_data)
  {

    StableSortStruct<RandomAccessIterator, Compare>* thief_work =
      (StableSortStruct<RandomAccessIterator, Compare>* )thief_data;

    StableSortStruct<RandomAccessIterator, Compare>* victim_work =
      (StableSortStruct<RandomAccessIterator, Compare>* )victim_data;

  std::cout << "victim_work->_iend-victim_work->_init_beg=" << victim_work->_iend-victim_work->_init_beg << std::endl;
  std::cout << "thief_work->_iend-thief_work->_init_beg=" << thief_work->_iend-thief_work->_init_beg << std::endl;

  const ptrdiff_t len = (victim_work->_iend-victim_work->_init_beg)+(thief_work->_iend-thief_work->_init_beg);
  std::cout << "len=" << len << std::endl;

   RandomAccessIterator victim_middle = victim_work->_iend-(victim_work->_iend-victim_work->_init_beg)/2;
   RandomAccessIterator thief_middle  = thief_work->_iend-(thief_work->_iend-thief_work->_init_beg)/2;  
   ptr_type buf1 = victim_work->_buf;
   ptr_type buf1_last = buf1+(victim_middle-victim_work->_init_beg)+(thief_middle-thief_work->_init_beg);
   ptr_type buf2 = buf1_last;
   ptr_type buf2_last = buf1 + len;
 
   std::merge(victim_work->_init_beg, victim_middle, thief_work->_init_beg, thief_middle, buf1, 
              victim_work->_comp);
   std::merge(victim_middle, victim_work->_iend,  thief_middle, thief_work->_iend, buf2,
              victim_work->_comp);
   std::merge(buf1, buf1_last, buf2, buf2_last, victim_work->_init_beg, victim_work->_comp); 

   victim_work->_iend = thief_work->_iend;

  }

};

/** doit_sort_loop
*/
template<class RandomAccessIterator, class Compare>
void StableSortStruct<RandomAccessIterator, Compare>::doit_sort_loop(RandomAccessIterator ibeg,
                                                                     RandomAccessIterator iend,
                                                                     ptr_type buf,
                                                                     ptrdiff_t step_size,
                                                                     Compare comp)
{

}


/** Adaptive stable_sort
*/
template<class RandomAccessIterator, class Compare>
void StableSortStruct<RandomAccessIterator, Compare>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend ;

  /* amount of work per iteration of the nano loop */
  ptrdiff_t unit_size  = 256;
  ptrdiff_t tmp_size;
  ptrdiff_t step_size = unit_size;
  while ( (_iend != _ibeg) )
  {
    /* definition of the steal point where steal_work may be called in case of steal request
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _buf, _comp);

    tmp_size = _iend-_ibeg;
    if(tmp_size < unit_size ) {
       unit_size = tmp_size; nano_iend = _iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }

    /** Local sorting */
    std::stable_sort(_ibeg, nano_iend, _comp);
    //std::__insertion_sort(_ibeg, nano_iend _comp);
   _ibeg += unit_size; 
  }

  const ptrdiff_t len = _iend - _init_beg;
  const ptr_type  buf_last = _buf + len;  

  while (step_size < len)
  {
     //doit_sort_loop(ibeg, _iend, _buf, step_size, _comp);
     std::__merge_sort_loop(_init_beg, _iend, _buf, step_size, _comp);
     step_size *= 2;
     //doit_sort_loop(_buf, buf_last, ibeg, step_size, _comp);
     std::__merge_sort_loop(_buf, buf_last, _init_beg, step_size, _comp);
     step_size *= 2;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this);

}

/**
*/
template<class RandomAccessIterator, class Compare>
void stable_sort(kaapi_steal_context_t* stealcontext, 
                 RandomAccessIterator begin, RandomAccessIterator end, Compare comp)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::pointer ptr_type;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  ptrdiff_t size = end-begin;
  val_t* buf = new val_t[size];
  StableSortStruct<RandomAccessIterator, Compare> work( stealcontext, begin, end, buf, comp);
  work.doit();
  delete  [] buf;
}

template<class RandomAccessIterator>
void stable_sort(kaapi_steal_context_t* stealcontext, 
                 RandomAccessIterator begin, RandomAccessIterator end)
{
  kaapi_steal_context_initpush( stealcontext );
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  stable_sort(stealcontext, begin, end, std::less<val_t>());
}

#endif
